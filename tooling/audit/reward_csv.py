#!/usr/bin/env python3
"""Acceptance reward CSV: steps 1-100, with BOTH global_step and acceptance_step.

Owed deliverable for the sandfleet acceptance arm. Built and validated ahead of
the run against the real (invalidated) steps 1-6 data, so the format can be
signed off before step 100 rather than discovered to be wrong afterwards.

The two step columns are not the same thing and the distinction is the whole
point of the column pair:

  global_step     - the trainer's own optimizer step counter, as recorded in the
                    rollout records. On a FRESH lineage this starts at 1; on a
                    resumed one it does not.
  acceptance_step - 1..N within the acceptance window, i.e. global_step minus the
                    step immediately before the window opened. For a fresh arm
                    the two coincide, and saying so explicitly is worth more than
                    leaving a reader to assume it.

Emits one row per (step, prompt) group plus a per-step aggregate, because group
structure is what shows active sampling accepting rather than filtering.

Usage:
    python3 reward_csv.py ROLLOUTS_DIR [--out reward.csv] [--first-global N]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict


def load(rollouts_dir: str):
    """(step, prompt_idx) -> list of rewards, from every rollouts shard."""
    groups: dict[tuple[int, int], list[float]] = defaultdict(list)
    files = sorted(glob.glob(os.path.join(rollouts_dir, "*_rollouts_*.jsonl")))
    n_rec = n_bad = 0
    for path in files:
        with open(path, errors="replace") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                step = r.get("step")
                pid = r.get("prompt_idx")
                rew = r.get("reward")
                if step is None or pid is None or not isinstance(rew, (int, float)):
                    n_bad += 1
                    continue
                groups[(int(step), int(pid))].append(float(rew))
                n_rec += 1
    return groups, files, n_rec, n_bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--first-global", type=int, default=None,
                    help="global_step of acceptance_step 1 (default: min observed)")
    ap.add_argument("--step-offset", type=int, default=None,
                    help="added to the rollout 'step' field to get global_step "
                         "(default: derived from the trainer-logprob artifacts)")
    args = ap.parse_args()

    groups, files, n_rec, n_bad = load(args.rollouts_dir)
    if not groups:
        print(f"no rollout records found under {args.rollouts_dir}", file=sys.stderr)
        return 0

    # The rollout 'step' field is 0-based while the trainer's own step counter --
    # and the trainer_logprobs_stepNNNNNN filenames, and the "Running training
    # step N" log lines -- are 1-based. Measured on run …__1788205485: rollout
    # steps 0..5 for trainer steps 1..6. Emitting the raw field as global_step
    # would silently misalign the deliverable against every other artifact by
    # one, so derive the offset from the logprob files rather than assume it.
    raw = sorted({s for s, _ in groups})
    lp = sorted({int(m) for f in glob.glob(os.path.join(args.rollouts_dir,
                                                        "*_trainer_logprobs_step*.jsonl"))
                 for m in [os.path.basename(f).split("_step")[1][:6]]})
    if args.step_offset is not None:
        offset = args.step_offset
        how = "explicit --step-offset"
    elif lp:
        offset = max(lp) - max(raw)
        how = f"derived from trainer-logprob steps {min(lp)}..{max(lp)}"
    else:
        offset = 1
        how = "default (+1); no logprob artifacts found to cross-check"
    print(f"step offset       : +{offset}  ({how})")
    if lp and set(s + offset for s in raw) != set(lp):
        print(f"  WARNING: rollout steps {[s + offset for s in raw]} do not match "
              f"logprob steps {lp} — reward rows and logprob artifacts disagree")

    groups = {(s + offset, p): v for (s, p), v in groups.items()}
    steps = sorted({s for s, _ in groups})
    first = args.first_global if args.first_global is not None else min(steps)
    out = args.out or os.path.join(args.rollouts_dir, "acceptance_rewards.csv")

    rows = []
    for s in steps:
        acc = s - first + 1
        per_prompt = {p: v for (st, p), v in groups.items() if st == s}
        allv = [x for v in per_prompt.values() for x in v]
        for p in sorted(per_prompt):
            v = per_prompt[p]
            rows.append({
                "global_step": s,
                "acceptance_step": acc,
                "scope": "group",
                "prompt_idx": p,
                "n_samples": len(v),
                "reward_mean": round(statistics.mean(v), 6),
                "reward_std": round(statistics.pstdev(v), 6) if len(v) > 1 else 0.0,
                "reward_min": min(v),
                "reward_max": max(v),
                "n_zero": sum(1 for x in v if x == 0.0),
                # a group with zero variance contributes no gradient under a
                # group-relative advantage, so degenerate-group count is the
                # signal that matters, not just the mean
                "degenerate": int(len(set(v)) == 1),
            })
        rows.append({
            "global_step": s,
            "acceptance_step": acc,
            "scope": "step",
            "prompt_idx": "",
            "n_samples": len(allv),
            "reward_mean": round(statistics.mean(allv), 6),
            "reward_std": round(statistics.pstdev(allv), 6) if len(allv) > 1 else 0.0,
            "reward_min": min(allv),
            "reward_max": max(allv),
            "n_zero": sum(1 for x in allv if x == 0.0),
            "degenerate": int(len({(p, tuple(v)) for p, v in per_prompt.items()}) == 0),
        })

    cols = ["global_step", "acceptance_step", "scope", "prompt_idx", "n_samples",
            "reward_mean", "reward_std", "reward_min", "reward_max", "n_zero", "degenerate"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    step_rows = [r for r in rows if r["scope"] == "step"]
    grp_rows = [r for r in rows if r["scope"] == "group"]
    print(f"shards            : {len(files)}")
    print(f"records           : {n_rec} parsed, {n_bad} skipped")
    print(f"steps             : {min(steps)}..{max(steps)}  "
          f"(acceptance_step 1..{max(steps) - first + 1}, first_global={first})")
    print(f"groups            : {len(grp_rows)}   degenerate: "
          f"{sum(r['degenerate'] for r in grp_rows)}")
    print(f"wrote             : {out}  ({len(rows)} rows)")
    print()
    print("global acc  n     mean    std     zeros")
    for r in step_rows:
        print(f"{r['global_step']:>6} {r['acceptance_step']:>3} {r['n_samples']:>4} "
              f"{r['reward_mean']:>8.4f} {r['reward_std']:>6.4f} {r['n_zero']:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
