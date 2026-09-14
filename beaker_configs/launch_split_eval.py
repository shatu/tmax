#!/usr/bin/env python3
"""Launch a harbor eval as TWO named Beaker tasks: a vLLM server and a harbor agent.

Why this exists alongside `launch_eval.sh --split-vllm`
-------------------------------------------------------
`--split-vllm` uses one task with `replicas: 2`. That is the only thing Beaker
gives cross-node *discovery* for, but it has three problems:

1. **No anti-affinity.** Beaker packs both replicas onto one node whenever they
   fit, which silently reverts to the co-located contention the split exists to
   remove. Observed twice in a row on jupiter-cs-aus-141 with `--gpus 1`.
2. **Homogeneous resources.** `resources` is a task-level field, so the agent
   replica is allocated the same `gpuCount` as the server and leaves it idle.
3. **One cluster.** `constraints.cluster` is per-task and a replica group is
   placed as a unit, so the two sides cannot straddle clusters.

Two named tasks fix all three. Discovery is not lost, because it never came from
Beaker in the first place: the two sides rendezvous through a directory on weka
(see `run_eval_in_job.sh`), which is what carries vLLM's randomized port. Giving
the tasks different clusters is the only way to *guarantee* they land on
different nodes.

Trade-off: gantry only ever emits one task, so this builds a raw Beaker spec and
submits it with `beaker experiment create`. That means re-doing what gantry did
for us -- git checkout, secrets, weka mounts, the results dataset. The repo is
public, so the checkout needs no token. (Dropping gantry also sidesteps its
source-dataset name colliding across workspaces, which blocks
`launch_eval.sh` in any workspace but the first one it was used in.)

Usage
-----
    ./beaker_configs/launch_split_eval.py allenai/tmax-9b \\
        --name tmax-9b --job-name tmax-9b-tb21-split \\
        --dataset-path /weka/oe-adapt-default/$USER/datasets/terminal-bench-2-1 \\
        --vllm-cluster ai2/jupiter --agent-cluster ai2/saturn \\
        --tool-call-parser qwen3_xml --language-model-only \\
        --max-model-len 65536 --n-attempts 5 --n-concurrent 8 \\
        --mirror-url host:5000,host2:5000 --min-runtime 8h

Add --dry-run to print the spec without submitting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_URL = "https://github.com/shatu/tmax"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("model_path", help="HF model id or a weka path the image can read")

    # model / vLLM
    p.add_argument("--revision", default="main")
    p.add_argument("--name", default=None, help="served-model-name (default: basename)")
    p.add_argument("--vllm-version", default="0.19.1")
    p.add_argument("--tool-call-parser", default="hermes")
    p.add_argument("--reasoning-parser", default="")
    p.add_argument("--language-model-only", action="store_true")
    p.add_argument("--max-model-len", default="")
    p.add_argument("--gpus", type=int, default=1, help="GPUs for the vLLM task")
    p.add_argument("--tp", type=int, default=None)
    p.add_argument("--dp", type=int, default=1)

    # harbor
    p.add_argument("--dataset", default="terminal-bench@2.0")
    p.add_argument("--dataset-path", default="")
    p.add_argument("--harbor-env", default="docker")
    p.add_argument("--agent", default="Vanillux2Agent:Vanillux2Agent")
    p.add_argument("--model-provider", default="")
    p.add_argument("--n-concurrent", type=int, default=8)
    p.add_argument("--n-attempts", type=int, default=1)
    p.add_argument("--n-tasks", default="")
    p.add_argument("--mirror-url", default=os.environ.get("MIRROR_URL", ""))
    p.add_argument("--agent-gpus", type=int, default=0,
                   help="GPUs for the harbor task (default 0 -- it needs none)")
    p.add_argument("--agent-kwarg", action="append", default=[],
                   help="harbor --agent-kwarg, repeatable (e.g. temperature=0)")
    p.add_argument("--agent-env", action="append", default=[],
                   help="harbor --agent-env, repeatable")

    # placement
    p.add_argument("--cluster", default="ai2/jupiter",
                   help="cluster for both tasks unless overridden per side")
    p.add_argument("--vllm-cluster", default=None)
    p.add_argument("--agent-cluster", default=None)
    # Beaker has no anti-affinity, between tasks OR between experiments. Pinning
    # hostnames is the only way to guarantee a specific layout -- e.g. two
    # concurrent runs whose vLLM servers must not share a node and so contend.
    p.add_argument("--vllm-hostname", default=None,
                   help="pin the vLLM task to this node (overrides --vllm-cluster)")
    p.add_argument("--agent-hostname", default=None,
                   help="pin the harbor task to this node (overrides --agent-cluster)")
    p.add_argument("--priority", default="urgent")
    p.add_argument("--min-runtime", default="")
    p.add_argument("--workspace", default=os.environ.get("BEAKER_WORKSPACE", "ai2/oe-agents"))
    p.add_argument("--budget", default="")
    p.add_argument("--image", default=os.environ.get("BEAKER_IMAGE", "hamishivi/tmax-eval-interactive"))
    p.add_argument("--weka-bucket", default="oe-adapt-default")

    # plumbing
    p.add_argument("--job-name", default=None)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    p.add_argument("--repo-ref", default=None, help="default: current HEAD SHA")
    p.add_argument("--rdv-root", default=None)
    p.add_argument("--hf-token-secret", default="HF_TOKEN")
    p.add_argument("--docker-pat-secret",
                   default=os.environ.get("DOCKER_PAT_SECRET", "shashankg_DOCKER_PAT"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def build_spec(a: argparse.Namespace) -> tuple[str, dict]:
    served = a.name or Path(a.model_path).name
    job_name = a.job_name or f"{served}-{a.dataset.replace('@', '-').replace('.', '-')}"
    exp_name = a.experiment_name or f"eval-{job_name}"
    ref = a.repo_ref or git("rev-parse", "HEAD")
    tp = a.tp if a.tp is not None else a.gpus
    user = os.environ.get("USER", "root")
    weka_mount = f"/weka/{a.weka_bucket}"
    rdv_root = a.rdv_root or f"{weka_mount}/{user}/tmax-eval/rendezvous"

    vllm_cluster = a.vllm_cluster or a.cluster
    agent_cluster = a.agent_cluster or a.cluster

    # Shared env. Every name here is read by scripts/beaker/run_eval_in_job.sh;
    # EVAL_ROLE is the only thing that differs between the two tasks.
    env = {
        "MODEL_PATH": a.model_path,
        "MODEL_REVISION": a.revision,
        "SERVED_MODEL_NAME": served,
        "VLLM_VERSION": a.vllm_version,
        "VLLM_TOOL_CALL_PARSER": a.tool_call_parser,
        "VLLM_REASONING_PARSER": a.reasoning_parser,
        "VLLM_LANGUAGE_MODEL_ONLY": "1" if a.language_model_only else "0",
        "MAX_MODEL_LEN": a.max_model_len,
        "TP_SIZE": str(tp),
        "DP_SIZE": str(a.dp),
        "DATASET": a.dataset,
        "DATASET_PATH": a.dataset_path,
        "HARBOR_ENV": a.harbor_env,
        "AGENT_IMPORT_PATH": a.agent,
        "MODEL_PROVIDER": a.model_provider,
        "N_CONCURRENT": str(a.n_concurrent),
        "N_ATTEMPTS": str(a.n_attempts),
        "N_TASKS": a.n_tasks,
        "EXTRA_AGENT_KWARGS": "\n".join(a.agent_kwarg),
        "EXTRA_AGENT_ENVS": "\n".join(a.agent_env),
        "MIRROR_URL": a.mirror_url,
        "JOB_NAME": job_name,
        "RESULTS_DIR": "/results",
        "VLLM_MODE": "split",
        # RDV_ID is deliberately NOT set: run_eval_in_job.sh defaults it to
        # BEAKER_WORKLOAD_ID, which is the experiment's id and so is shared by
        # both tasks. Both sides log it for confirmation.
        "RDV_ROOT": rdv_root,
        "BEAKER_ALLOW_SUBCONTAINERS": "1",
        "BEAKER_SKIP_DOCKER_SOCKET": "1",
    }

    # gantry normally does the checkout for us; without it the task does its own.
    # The repo is public, so no token is involved.
    run_cmd = (
        "set -euo pipefail\n"
        f"git clone --filter=blob:none {a.repo_url} /workspace/tmax\n"
        "cd /workspace/tmax\n"
        f"git checkout {ref}\n"
        "exec bash scripts/beaker/run_eval_in_job.sh\n"
    )

    def task(name: str, role: str, cluster: str, gpus: int, hostname: str | None) -> dict:
        env_vars = [{"name": k, "value": v} for k, v in sorted(env.items())]
        env_vars.append({"name": "EVAL_ROLE", "value": role})
        env_vars.append({"name": "HF_TOKEN", "secret": a.hf_token_secret})
        env_vars.append({"name": "DOCKER_PAT", "secret": a.docker_pat_secret})
        context: dict = {"priority": a.priority}
        if a.min_runtime:
            context["minRuntime"] = a.min_runtime
        return {
            "name": name,
            "image": {"beaker": a.image},
            "command": ["bash", "-lc", run_cmd],
            "envVars": env_vars,
            "datasets": [{"mountPath": weka_mount, "source": {"weka": a.weka_bucket}}],
            "result": {"path": "/results"},
            "resources": {"gpuCount": gpus},
            "context": context,
            # hostname alone when pinning: Beaker treats each constraint as an
            # allow-list, and a cluster list alongside it is redundant at best.
            "constraints": ({"hostname": [hostname]} if hostname
                            else {"cluster": [cluster]}),
            "hostNetworking": True,
            # Experiment-wide: "if a job for this task fails, all other jobs in
            # the experiment are canceled". So a dead server takes the agent down
            # rather than leaving it to error out every remaining trial.
            "propagateFailure": True,
            "propagatePreemption": True,
        }

    spec = {
        "version": "v2",
        "description": (
            f"Split harbor eval ({a.dataset_path or a.dataset}) of {served} "
            f"[vLLM on {vllm_cluster}, harbor on {agent_cluster}]"
        ),
        "tasks": [
            task("vllm-server", "vllm", vllm_cluster, a.gpus, a.vllm_hostname),
            task("agent-eval", "eval", agent_cluster, a.agent_gpus, a.agent_hostname),
        ],
    }
    if a.budget:
        spec["budget"] = a.budget
    return exp_name, spec


def main(argv: list[str]) -> int:
    a = parse_args(argv)
    exp_name, spec = build_spec(a)

    vllm_cluster = a.vllm_cluster or a.cluster
    agent_cluster = a.agent_cluster or a.cluster
    print("=== split eval (two named Beaker tasks) ===")
    print(f"  experiment:   {exp_name}")
    print(f"  workspace:    {a.workspace}")
    print(f"  vllm-server:  {a.vllm_hostname or vllm_cluster}  gpus={a.gpus}")
    print(f"  agent-eval:   {a.agent_hostname or agent_cluster}  gpus={a.agent_gpus}")
    if a.agent_kwarg:
        print(f"  agent kwargs: {', '.join(a.agent_kwarg)}")
    if (a.vllm_hostname and a.agent_hostname
            and a.vllm_hostname == a.agent_hostname):
        print("  ERROR: both tasks pinned to the SAME host; the in-job guard will")
        print("         fail the run. Pin different hosts.")
        raise SystemExit(2)
    if not (a.vllm_hostname or a.agent_hostname) and vllm_cluster == agent_cluster:
        print("  NOTE: both tasks target one cluster. Beaker has no anti-affinity,")
        print("        so they may land on the SAME node; the in-job guard will")
        print("        fail the run if they do. Use different clusters to guarantee")
        print("        separation, or set REQUIRE_SEPARATE_NODES=0 to allow it.")
    print()

    if a.repo_ref is None:
        ref = git("rev-parse", "HEAD")
        on_remote = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "branch", "-r", "--contains", ref],
            capture_output=True, text=True,
        ).stdout.strip()
        if not on_remote:
            print(f"warning: {ref} is not on any remote branch; the tasks clone from "
                  f"{a.repo_url} and will fail to check it out. Push first.\n",
                  file=sys.stderr)

    if a.dry_run:
        print(json.dumps(spec, indent=2))
        return 0

    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix=f"{exp_name}-",
                                     delete=False) as fh:
        json.dump(spec, fh, indent=2)
        fh.write("\n")
        spec_path = Path(fh.name)
    try:
        result = subprocess.run(
            ["beaker", "experiment", "create", "-n", exp_name,
             "-w", a.workspace, str(spec_path)],
            capture_output=True, text=True,
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode
    finally:
        spec_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
