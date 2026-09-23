#!/usr/bin/env python3
"""Reproducibility / reliability comparison of harbor runs across sandbox backends.

Given harbor job directories (or Beaker experiment IDs, whose result datasets
are fetched), grouped by backend, this reports per run:

  * completed / errored trials, error rate, errors bucketed by cause
    (infra vs the agent's own 120 s command timeout vs other),
  * pass@1 (mean reward over ALL trials, errors count as 0), pass@1 over
    non-error trials, pass@k,
  * wall-clock per trial (median / p90), environment start time, and total
    job runtime,

and per backend with >= 2 runs, the run-to-run spread:

  * pass@1 standard deviation across runs,
  * per-task agreement: mean |pass count difference| out of k between runs,
    fraction of tasks with identical pass count, and the Jaccard overlap of
    the solved-task sets,
  * fraction of tasks whose ERRORS repeat across runs (a systematic infra
    problem) vs land on different tasks (flakiness).

Usage:
    uv run python scripts/analysis/compare_backend_runs.py \\
        --run opensandbox=jobs/run-a --run opensandbox=jobs/run-b \\
        --run podman=jobs/run-c --run podman=jobs/run-d

    # Beaker experiments: results datasets are fetched into --fetch-dir
    uv run python scripts/analysis/compare_backend_runs.py \\
        --run opensandbox=01M35MBSCBGB0P2EZ37XNPFBD7 --run podman=01KWZPGMH7RACJSYEGCB65YVAQ \\
        --fetch-dir /tmp/harbor-results

    --json out.json writes everything (per-run, per-task, pairwise) for further analysis.

A value that is a directory is used as-is; anything else is treated as a
Beaker experiment ID. The job dir is located inside the fetched dataset by
finding result.json + config.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ── error taxonomy ──────────────────────────────────────────────────────────

_TIMEOUT_RE = re.compile(r"Command timed out after \d+ seconds")


def classify_error(exc: dict | None) -> str | None:
    """Bucket a trial's exception_info. None for a clean trial."""
    if not exc:
        return None
    etype = str(exc.get("exception_type") or "")
    msg = str(exc.get("exception_message") or "")
    if _TIMEOUT_RE.search(msg) or etype == "AgentTimeoutError":
        # The agent's per-command limit. Genuinely slow model commands AND
        # infra-slow compute both land here; see docs/running_evals.md §3b.
        return "command_timeout"
    if "Environment start timed out" in msg or "EnvironmentStartTimeout" in etype:
        return "infra:env_start"
    if "died during exec" in msg or "Sandbox" in msg and "died" in msg:
        return "infra:sandbox_died"
    if "No reward file" in msg or "RewardFileNotFound" in etype or "Verifier" in etype:
        return "infra:verifier"
    if any(s in msg for s in ("Connection", "connection", "vLLM", "APIConnectionError", "Timeout waiting for")):
        return "infra:connection"
    if "AgentTimeout" in etype or "agent timed out" in msg.lower():
        return "agent_timeout"
    return f"other:{etype or 'unknown'}"


def is_infra(bucket: str | None) -> bool:
    return bool(bucket) and bucket.startswith("infra:")


# ── loading ─────────────────────────────────────────────────────────────────


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def find_job_dir(root: Path) -> Path:
    if (root / "result.json").is_file() and (root / "config.json").is_file():
        return root
    hits = [p.parent for p in root.rglob("config.json") if (p.parent / "result.json").is_file()]
    hits = [h for h in hits if any(c.is_dir() and (c / "result.json").is_file() for c in h.iterdir())]
    if not hits:
        raise SystemExit(f"no harbor job dir (result.json + config.json + trial dirs) under {root}")
    return sorted(hits, key=lambda p: len(p.parts))[0]


def fetch_experiment(exp_id: str, fetch_dir: Path) -> Path:
    """Download a Beaker experiment's result dataset (once) and return its dir."""
    out = fetch_dir / exp_id
    if out.is_dir() and any(out.rglob("result.json")):
        return out
    info = json.loads(subprocess.run(["beaker", "experiment", "get", exp_id, "--format", "json"],
                                     check=True, capture_output=True, text=True).stdout)
    info = info[0] if isinstance(info, list) else info
    jobs = info.get("jobs") or []
    if not jobs:
        raise SystemExit(f"{exp_id}: no jobs")
    job = jobs[-1]
    result = (job.get("execution") or {}).get("result") or job.get("result") or {}
    ds = result.get("beaker")
    if not ds:
        raise SystemExit(f"{exp_id}: no result dataset (job still running?)")
    out.mkdir(parents=True, exist_ok=True)
    print(f"fetching result dataset {ds} for {exp_id} -> {out}", file=sys.stderr)
    subprocess.run(["beaker", "dataset", "fetch", ds, "-o", str(out)], check=True)
    return out


def load_run(label: str, source: str, fetch_dir: Path) -> dict:
    p = Path(source)
    root = p if p.is_dir() else fetch_experiment(source, fetch_dir)
    job_dir = find_job_dir(root)
    trials = []
    for tdir in sorted(d for d in job_dir.iterdir() if d.is_dir() and (d / "result.json").is_file()):
        r = json.loads((tdir / "result.json").read_text())
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        exc = r.get("exception_info")
        started, finished = _parse_ts(r.get("started_at")), _parse_ts(r.get("finished_at"))
        env_setup = r.get("environment_setup") or {}
        es, ef = _parse_ts(env_setup.get("started_at")), _parse_ts(env_setup.get("finished_at"))
        trials.append({
            "task": r.get("task_name"),
            "trial": r.get("trial_name") or tdir.name,
            "reward": float(reward) if reward is not None else None,
            "error": classify_error(exc),
            "error_message": (exc or {}).get("exception_message"),
            "wall_s": (finished - started).total_seconds() if started and finished else None,
            "env_start_s": (ef - es).total_seconds() if es and ef else None,
        })
    job_result = json.loads((job_dir / "result.json").read_text())
    js, jf = _parse_ts(job_result.get("started_at")), _parse_ts(job_result.get("finished_at"))
    return {
        "backend": label,
        "source": source,
        "job_dir": str(job_dir),
        "trials": trials,
        "job_runtime_s": (jf - js).total_seconds() if js and jf else None,
        "n_total": job_result.get("n_total_trials"),
    }


# ── per-run metrics ─────────────────────────────────────────────────────────


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:5.1f}%"


def _q(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarize_run(run: dict) -> dict:
    trials = run["trials"]
    n = len(trials)
    errors = [t for t in trials if t["error"]]
    clean = [t for t in trials if not t["error"]]
    by_task: dict[str, list[float]] = defaultdict(list)
    for t in trials:
        by_task[t["task"]].append(t["reward"] if (t["reward"] is not None and not t["error"]) else 0.0)
    k = max((len(v) for v in by_task.values()), default=0)
    pass_at_k = statistics.mean(1.0 if max(v) >= 1.0 else 0.0 for v in by_task.values()) if by_task else None
    solved = {task for task, v in by_task.items() if max(v) >= 1.0}
    pass_counts = {task: sum(1 for r in v if r >= 1.0) for task, v in by_task.items()}
    error_tasks = {t["task"] for t in errors}
    walls = [t["wall_s"] for t in trials if t["wall_s"] is not None]
    envs = [t["env_start_s"] for t in trials if t["env_start_s"] is not None]
    return {
        "backend": run["backend"], "source": run["source"], "n_trials": n, "n_total": run["n_total"], "k": k,
        "n_errors": len(errors), "error_rate": len(errors) / n if n else None,
        "infra_error_rate": sum(1 for t in errors if is_infra(t["error"])) / n if n else None,
        "timeout_error_rate": sum(1 for t in errors if t["error"] == "command_timeout") / n if n else None,
        "error_buckets": dict(Counter(t["error"] for t in errors)),
        "pass1_all": statistics.mean((t["reward"] or 0.0) if not t["error"] else 0.0 for t in trials) if n else None,
        "pass1_clean": statistics.mean(t["reward"] or 0.0 for t in clean) if clean else None,
        "pass1_adj": statistics.mean((t["reward"] or 0.0) if not t["error"] else 0.0
                                     for t in trials if not is_infra(t["error"])) if n else None,
        "pass_at_k": pass_at_k, "solved_tasks": sorted(solved), "pass_counts": pass_counts,
        "error_tasks": sorted(error_tasks),
        "wall_median_s": statistics.median(walls) if walls else None, "wall_p90_s": _q(walls, 0.9),
        "env_start_median_s": statistics.median(envs) if envs else None, "env_start_p90_s": _q(envs, 0.9),
        "job_runtime_s": run["job_runtime_s"],
    }


# ── pairwise agreement ──────────────────────────────────────────────────────


def compare_pair(a: dict, b: dict) -> dict:
    tasks = sorted(set(a["pass_counts"]) & set(b["pass_counts"]))
    if not tasks:
        return {"n_tasks": 0}
    diffs = [abs(a["pass_counts"][t] - b["pass_counts"][t]) for t in tasks]
    sa, sb = set(a["solved_tasks"]), set(b["solved_tasks"])
    ea, eb = set(a["error_tasks"]), set(b["error_tasks"])
    return {
        "n_tasks": len(tasks),
        "pass1_diff": (a["pass1_all"] or 0) - (b["pass1_all"] or 0),
        "mean_abs_pass_count_diff": statistics.mean(diffs),
        "frac_tasks_same_pass_count": sum(1 for d in diffs if d == 0) / len(tasks),
        "solved_jaccard": len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0,
        "solved_only_a": sorted(sa - sb), "solved_only_b": sorted(sb - sa),
        "error_task_jaccard": len(ea & eb) / len(ea | eb) if (ea | eb) else 1.0,
        "errors_in_both": sorted(ea & eb),
    }


# ── report ──────────────────────────────────────────────────────────────────


def _fmt_s(x: float | None) -> str:
    return "-" if x is None else (f"{x / 60:.1f}m" if x >= 120 else f"{x:.0f}s")


def print_report(summaries: list[dict], pairs: list[tuple[dict, dict, dict]]) -> None:
    print("\n== Per run ==")
    hdr = f"{'backend':12s} {'trials':>9s} {'errors':>7s} {'infra':>7s} {'timeout':>8s} {'pass@1':>7s} {'p@1 adj':>8s} {'p@1 clean':>9s} {'pass@k':>7s} {'trial med/p90':>14s} {'env start med/p90':>18s} {'job':>6s}  source"
    print(hdr)
    for s in summaries:
        trials = f"{s['n_trials']}/{s['n_total'] or '?'}"
        print(f"{s['backend']:12s} {trials:>9s} {_pct(s['error_rate']):>7s} {_pct(s['infra_error_rate']):>7s} {_pct(s['timeout_error_rate']):>8s} "
              f"{_pct(s['pass1_all']):>7s} {_pct(s['pass1_adj']):>8s} {_pct(s['pass1_clean']):>9s} {_pct(s['pass_at_k']):>7s} "
              f"{_fmt_s(s['wall_median_s']) + '/' + _fmt_s(s['wall_p90_s']):>14s} "
              f"{_fmt_s(s['env_start_median_s']) + '/' + _fmt_s(s['env_start_p90_s']):>18s} {_fmt_s(s['job_runtime_s']):>6s}  {s['source']}")
    print("\n== Error buckets ==")
    for s in summaries:
        buckets = ", ".join(f"{k}={v}" for k, v in sorted(s["error_buckets"].items(), key=lambda kv: -kv[1])) or "none"
        print(f"{s['backend']:12s} {s['source']}: {buckets}")

    by_backend: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        by_backend[s["backend"]].append(s)
    print("\n== Run-to-run spread within backend ==")
    for backend, runs in by_backend.items():
        if len(runs) < 2:
            print(f"{backend:12s} only {len(runs)} run(s); need >= 2 for spread")
            continue
        p1 = [r["pass1_all"] for r in runs if r["pass1_all"] is not None]
        adj = [r["pass1_adj"] for r in runs if r["pass1_adj"] is not None]
        print(f"{backend:12s} runs={len(runs)}  pass@1 mean={_pct(statistics.mean(p1))} sd={_pct(statistics.pstdev(p1))}  "
              f"pass@1_adj mean={_pct(statistics.mean(adj))} sd={_pct(statistics.pstdev(adj))}  "
              f"error_rate mean={_pct(statistics.mean(r['error_rate'] for r in runs))}")
    print("\n== Pairwise per-task agreement ==")
    for a, b, c in pairs:
        tag = f"{a['backend']} vs {b['backend']}" if a["backend"] != b["backend"] else f"{a['backend']} (2 runs)"
        if not c.get("n_tasks"):
            print(f"{tag}: no common tasks"); continue
        print(f"{tag}: tasks={c['n_tasks']} pass@1 diff={c['pass1_diff']:+.3f} | pass-count mean|Δ|={c['mean_abs_pass_count_diff']:.2f}/k "
              f"same={_pct(c['frac_tasks_same_pass_count'])} | solved Jaccard={c['solved_jaccard']:.2f} "
              f"(only A: {len(c['solved_only_a'])}, only B: {len(c['solved_only_b'])}) | error-task Jaccard={c['error_task_jaccard']:.2f} "
              f"(repeat in both: {len(c['errors_in_both'])})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="BACKEND=DIR_OR_EXPERIMENT",
                    help="label and harbor job dir or Beaker experiment ID (repeatable)")
    ap.add_argument("--fetch-dir", type=Path, default=Path("/tmp/harbor-results"))
    ap.add_argument("--json", type=Path, default=None, help="write full results here")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        if "=" not in spec:
            ap.error(f"--run expects BACKEND=SOURCE, got {spec!r}")
        label, source = spec.split("=", 1)
        runs.append(load_run(label, source, args.fetch_dir))
    summaries = [summarize_run(r) for r in runs]
    pairs = [(a, b, compare_pair(a, b)) for a, b in itertools.combinations(summaries, 2)]
    print_report(summaries, pairs)
    if args.json:
        args.json.write_text(json.dumps({
            "runs": [{**s, "trials": r["trials"]} for s, r in zip(summaries, runs)],
            "pairs": [{"a": a["source"], "b": b["source"], **c} for a, b, c in pairs],
        }, indent=2, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
