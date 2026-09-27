#!/usr/bin/env python3
"""Per-run pass@1 (mean ± std over the k attempts) from harbor job dirs, raw and
with infra-failed trials removed.

    uv run python scripts/analysis/adjusted_run_stats.py jobs/<job-a> jobs/<job-b> ...

Reports pass@1 = mean over all attempts, ± the std of a single run's mean score,
estimated partition-free from the per-task attempt variance:
std = sqrt(sum_t s_t^2) / T with s_t^2 the unbiased variance of task t's attempts.
(scripts/compute_stats.py instead splits the k attempts into k "runs" in directory
order and takes the std across them; that number depends on the arbitrary split
and swings by 3x between splits, so don't compare it across jobs.)
"raw" counts every errored trial as 0. "adjusted" drops trials whose
exception is infrastructure (image pull rate limit, compose failure, verifier
timeout, ...) from that run's denominator; the model's own 120 s command timeout
(AgentTimeoutError, or RuntimeError "Command timed out after 120 seconds" in newer
harbor) stays a failure. Also prints, per dataset, pass@1 on the tasks that had no
infra error in *any* of the given jobs (the fair common subset).
"""
import collections
import glob
import json
import os
import statistics as st
import sys


def load_job(job_dir):
    trials = collections.defaultdict(list)
    errs = collections.Counter()
    for f in glob.glob(os.path.join(job_dir, "*", "result.json")):
        r = json.load(open(f))
        if "task_name" not in r:
            continue
        vr = r.get("verifier_result") or {}
        reward = (vr.get("rewards") or {}).get("reward", 0.0) if vr else 0.0
        ex = r.get("exception_info") or {}
        et, msg = ex.get("exception_type"), ex.get("exception_message") or ""
        model_fail = et == "AgentTimeoutError" or "Command timed out" in msg
        infra = bool(et) and not model_fail
        if et:
            errs["cmd-timeout" if model_fail else ("pull-rate-limit" if "toomanyrequests" in msg else et)] += 1
        trials[r["task_name"].split("/")[-1]].append((r.get("started_at", ""), reward, infra))
    for v in trials.values():
        v.sort()
    return trials, errs


def score(trials, tasks=None, drop_infra=False):
    """(mean pass@1, std of one run's mean) over the given tasks."""
    tasks = tasks or list(trials)
    means, variances = [], []
    for t in tasks:
        rs = [x[1] for x in trials[t] if not (drop_infra and x[2])]
        if not rs:
            continue  # every attempt of this task was an infra failure
        means.append(st.mean(rs))
        variances.append(st.variance(rs) if len(rs) > 1 else 0.0)
    n = len(means)
    return st.mean(means), (sum(variances) ** 0.5) / n


def fmt(ms):
    return f"{ms[0] * 100:5.1f} ± {ms[1] * 100:4.1f}"


def main(job_dirs):
    jobs = {os.path.basename(d.rstrip("/")): load_job(d) for d in job_dirs}
    print(f"{'job':40s} {'raw pass@1':>14s} {'adjusted':>14s}  infra/total  errors")
    for name, (trials, errs) in jobs.items():
        tot = sum(len(v) for v in trials.values())
        infra = sum(1 for v in trials.values() for x in v if x[2])
        print(f"{name:40s} {fmt(score(trials)):>14s} {fmt(score(trials, drop_infra=True)):>14s}"
              f"  {infra:4d}/{tot}  {dict(errs)}")
    by_ds = collections.defaultdict(dict)
    for name, (trials, _) in jobs.items():
        by_ds[frozenset(trials)][name] = trials
    for task_set, group in by_ds.items():
        bad = {t for trials in group.values() for t, v in trials.items() if any(x[2] for x in v)}
        clean = sorted(task_set - bad)
        if len(group) < 2 or not bad:
            continue
        print(f"\n{len(clean)} of {len(task_set)} tasks with no infra error in any of these {len(group)} jobs:")
        for name, trials in group.items():
            print(f"  {name:40s} {fmt(score(trials, tasks=clean))}")


if __name__ == "__main__":
    main(sys.argv[1:])
