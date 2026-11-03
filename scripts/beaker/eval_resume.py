#!/usr/bin/env python3
"""Clean a harbor job dir carried over a Beaker preemption so it resumes.

Eval jobs run with auto_resume on. When Beaker preempts one (SIGTERM, ~10s
grace), it starts a new job in the same task and COPIES the preempted job's
/results into it first. This was verified on exp 01M6F95K86AXSSBJ6JB3ESWMFY:
3 preemptions in a row, and the final result was complete and correct.
Gantry's `--not-preemptible` / preemptible=False turns auto_resume off.

run_eval_in_job.sh therefore has harbor write its job dir straight into
/results/<JOB_NAME>. Before `harbor run`, this script cleans that dir:
  * drops trial dirs whose result.json or config.json don't parse (killed
    mid-write), or whose error is CancelledError;
  * drops an unparseable job-level result.json or lock.json;
  * rewrites the old job's random vLLM port in the agent kwargs
    (api_base=http://localhost:<port>/v1) to this job's port, in config.json,
    lock.json and every trial config.json. Harbor refuses to resume if the
    config differs in any way.

harbor 0.6.6 then resumes on its own. It keeps every trial that has a
result.json, deletes the dirs of trials that were still running, and runs
only the rest. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

API_BASE_RE = re.compile(r"^http://(localhost|127\.0\.0\.1):\d+/v1/?$")


def log(msg: str) -> None:
    print(f"[eval_resume] {msg}", flush=True)


# --------------------------------------------------------------------------- prepare
def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _rewrite_api_base(obj, new: str):
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and API_BASE_RE.match(v):
                if v != new:
                    obj[k] = new
                    n += 1
            else:
                n += _rewrite_api_base(v, new)
    elif isinstance(obj, list):
        for v in obj:
            n += _rewrite_api_base(v, new)
    return n


def _rewrite_file(path: Path, new: str) -> int:
    data = _load(path)
    if data is None:
        return 0
    n = _rewrite_api_base(data, new)
    if n:
        path.write_text(json.dumps(data, indent=4))
    return n


def cmd_prepare(args) -> int:
    jd = Path(args.job_dir)
    if not jd.is_dir() or not any(jd.iterdir()):
        log(f"no earlier state in {jd} — fresh run")
        return 0
    if _load(jd / "config.json") is None:
        # Partial/garbled state we can't resume: start clean rather than let
        # harbor mix old trial dirs into a fresh job.
        log(f"{jd}/config.json missing or unparseable — wiping {jd} for a fresh run")
        for p in jd.iterdir():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        return 0

    drop_types = set(filter(None, args.drop_error_types.split(",")))
    kept = dropped = inflight = 0
    for td in sorted(p for p in jd.iterdir() if p.is_dir()):
        rp, cp = td / "result.json", td / "config.json"
        if not rp.exists():
            inflight += 1  # harbor deletes + reruns these itself
            continue
        res, cfg = _load(rp), _load(cp)
        reason = None
        if res is None or cfg is None:
            reason = "unparseable result/config"
        else:
            et = ((res.get("exception_info") or {}).get("exception_type"))
            if et in drop_types:
                reason = et
        if reason:
            log(f"dropping {td.name}: {reason}")
            shutil.rmtree(td)
            dropped += 1
        else:
            kept += 1

    for name in ("result.json", "lock.json"):
        p = jd / name
        if p.exists() and _load(p) is None:
            log(f"dropping unparseable {name}")
            p.unlink()

    n_rw = 0
    if args.api_base:
        for p in [jd / "config.json", jd / "lock.json", *jd.glob("*/config.json")]:
            if p.exists():
                n_rw += _rewrite_file(p, args.api_base)

    log(
        f"resume state: {kept} finished trials kept, {dropped} dropped, "
        f"{inflight} in-flight (will rerun); rewrote api_base in {n_rw} places "
        f"-> {args.api_base}"
    )
    with open(jd / "resume_log.txt", "a") as fh:
        fh.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"job={os.environ.get('BEAKER_JOB_ID', '?')} prepare kept={kept} "
            f"dropped={dropped} inflight={inflight} api_base={args.api_base}\n"
        )
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--job-dir", required=True)
    p.add_argument("--api-base", default=None)
    p.add_argument("--drop-error-types", default="CancelledError")
    p.set_defaults(fn=cmd_prepare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
