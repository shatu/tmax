#!/usr/bin/env python3
"""Write <job dir>/eval_meta.json: what eval this is, and which jobs ran it.

Harbor's own config.json records the agent, dataset, k and timeouts. For the
model it records only `openai/<served-name>`, which is a free-form alias. This
file adds the rest, so a batch-downloaded job dir identifies itself:
  * every launch_eval.sh setting passed into the job: model path + revision,
    vLLM version and flags, dataset, agent, k, overrides, mirror;
  * the exact `vllm serve` and `harbor run` commands;
  * the HF commit SHA the revision pointed to (HF ids only; best effort);
  * the tmax git commit and harbor version;
  * Beaker workload/job/node ids;
  * `segments`: one entry per Beaker job that worked on this eval (a new one
    after each preemption), with GPU model, start/finish time, harbor exit code
    and how many trials were already finished when it started;
  * `gpu_types`: every GPU model across segments (more than one = mixed hardware).

Secrets are never written: HF_TOKEN/DOCKER_PAT are not read, and agent env
values whose name looks secret are redacted.

  eval_meta.py start  --job-dir D --vllm-cmd "..." --harbor-cmd "..." --api-port P
  eval_meta.py finish --job-dir D --harbor-rc N
Stdlib only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

SCHEMA_VERSION = 1
META = "eval_meta.json"
SECRETISH = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|APIKEY|PAT\b|CREDENTIAL", re.I)

# launch_eval.sh -> job env vars that define the eval.
EVAL_ENV = [
    "MODEL_PATH", "MODEL_REVISION", "SERVED_MODEL_NAME", "HARBOR_MODEL_NAME", "MODEL_PROVIDER",
    "VLLM_VERSION", "VLLM_TOOL_CALL_PARSER", "VLLM_REASONING_PARSER", "VLLM_LANGUAGE_MODEL_ONLY",
    "MAX_MODEL_LEN", "TP_SIZE", "DP_SIZE",
    "DATASET", "DATASET_PATH", "N_TASKS", "N_ATTEMPTS", "N_CONCURRENT", "JOB_NAME",
    "AGENT_IMPORT_PATH", "EXTRA_AGENT_KWARGS", "HOSTED_VLLM_MODEL_INFO", "EXTRA_UV_PIP_INSTALLS",
    "HARBOR_ENV", "HARBOR_KEEP_TASK_IMAGES",
    "HARBOR_OVERRIDE_CPUS", "HARBOR_OVERRIDE_MEMORY_MB", "HARBOR_OVERRIDE_STORAGE_MB", "HARBOR_OVERRIDE_GPUS",
    "HARBOR_TIMEOUT_MULTIPLIER", "HARBOR_AGENT_TIMEOUT_MULTIPLIER", "HARBOR_VERIFIER_TIMEOUT_MULTIPLIER",
    "HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER", "HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER",
    "HARBOR_AGENT_TIMEOUT_SEC", "MIRROR_URL", "RESULTS_DIR", "EVAL_RESUMABLE",
]
CODE_ENV = ["GITHUB_REPO", "GIT_REF", "GIT_BRANCH", "GANTRY_VERSION"]
BEAKER_ENV = ["BEAKER_WORKLOAD_ID", "BEAKER_WORKSPACE_ID", "BEAKER_TASK_ID", "BEAKER_RESULT_DATASET_ID"]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def env(names) -> dict:
    return {n: os.environ.get(n, "") for n in names}


def redact_agent_envs(raw: str) -> list[str]:
    out = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        k, sep, _v = line.partition("=")
        out.append(f"{k}=<redacted>" if sep and SECRETISH.search(k) else line)
    return out


def redact_cmd(cmd: str) -> str:
    """Hide values of KEY=VALUE tokens whose KEY looks secret (e.g. from --agent-env)."""
    return re.sub(
        r"\b([A-Za-z0-9_]+)=(\S+)",
        lambda m: f"{m.group(1)}=<redacted>" if SECRETISH.search(m.group(1)) else m.group(0),
        cmd,
    )


def sh(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def hf_commit(model: str, revision: str) -> str | None:
    """Commit SHA the revision resolves to on the HF Hub (None for local paths / errors)."""
    if not model or model.startswith("/") or Path(model).exists():
        return None
    url = f"https://huggingface.co/api/models/{model}/revision/{revision or 'main'}"
    req = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r).get("sha")
    except Exception:  # noqa: BLE001
        return None


def gpus() -> list[str]:
    """GPU model per visible device, e.g. ["NVIDIA A100-SXM4-80GB"] (empty if no nvidia-smi).

    Matters for comparing runs: the same eval on A100 vs H100 gives the same
    mean reward but many more AgentTimeoutErrors on time-tight tasks.
    """
    out = sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return [ln.strip() for ln in out.splitlines() if ln.strip()] if out else []


def count_finished(jd: Path) -> int:
    return sum(1 for p in jd.iterdir() if p.is_dir() and (p / "result.json").exists())


def load(jd: Path) -> dict:
    try:
        return json.loads((jd / META).read_text())
    except Exception:  # noqa: BLE001
        return {}


def save(jd: Path, meta: dict) -> None:
    tmp = jd / (META + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    tmp.replace(jd / META)  # atomic: a preemption never leaves a half-written file


def cmd_start(args) -> int:
    jd = Path(args.job_dir)
    jd.mkdir(parents=True, exist_ok=True)
    meta = load(jd)
    eval_env = env(EVAL_ENV)
    eval_env["EXTRA_AGENT_ENVS"] = redact_agent_envs(os.environ.get("EXTRA_AGENT_ENVS", ""))
    meta.update(
        schema_version=SCHEMA_VERSION,
        job_name=os.environ.get("JOB_NAME", jd.name),
        model={
            "path": eval_env["MODEL_PATH"],
            "revision": eval_env["MODEL_REVISION"] or "main",
            # Keep the first resolution: a later segment re-resolving "main"
            # could see a newer commit than the one the eval started on.
            "hf_commit": (meta.get("model") or {}).get("hf_commit")
            or hf_commit(eval_env["MODEL_PATH"], eval_env["MODEL_REVISION"]),
            "served_name": eval_env["SERVED_MODEL_NAME"],
        },
        eval_env=eval_env,
        vllm_cmd=redact_cmd(args.vllm_cmd),
        harbor_cmd=redact_cmd(args.harbor_cmd),
        code={
            **env(CODE_ENV),
            "workdir_commit": sh(["git", "rev-parse", "HEAD"]),
            "harbor_version": sh(["uv", "run", "--no-sync", "python", "-c",
                                  "import importlib.metadata as m; print(m.version('harbor'))"]),
        },
        beaker=env(BEAKER_ENV),
    )
    meta.setdefault("created_at", now())
    segs = meta.setdefault("segments", [])
    seg_gpus = gpus()
    segs.append({
        "beaker_job_id": os.environ.get("BEAKER_JOB_ID", ""),
        "node_hostname": os.environ.get("BEAKER_NODE_HOSTNAME", ""),
        "gpus": os.environ.get("BEAKER_ASSIGNED_GPU_COUNT", ""),
        "gpu_names": seg_gpus,
        "started_at": now(),
        "api_port": args.api_port,
        "finished_trials_at_start": count_finished(jd),
        "finished_at": None,
        "harbor_rc": None,
    })
    # A resumed eval can land on different hardware after a preemption. Keep
    # every GPU type that worked on it, so a mixed run is obvious at a glance.
    meta["gpu_types"] = sorted({g for s in segs for g in s.get("gpu_names") or []})
    save(jd, meta)
    print(f"[eval_meta] wrote {jd / META} (segment {len(segs)})", flush=True)
    return 0


def cmd_finish(args) -> int:
    jd = Path(args.job_dir)
    meta = load(jd)
    if not meta.get("segments"):
        return 0
    seg = meta["segments"][-1]
    seg["finished_at"] = now()
    seg["harbor_rc"] = args.harbor_rc
    seg["finished_trials_at_end"] = count_finished(jd)
    save(jd, meta)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--job-dir", required=True)
    s.add_argument("--vllm-cmd", default="")
    s.add_argument("--harbor-cmd", default="")
    s.add_argument("--api-port", default="")
    s.set_defaults(fn=cmd_start)
    f = sub.add_parser("finish")
    f.add_argument("--job-dir", required=True)
    f.add_argument("--harbor-rc", type=int, required=True)
    f.set_defaults(fn=cmd_finish)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
