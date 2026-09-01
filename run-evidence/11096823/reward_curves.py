#!/usr/bin/env python3
"""Reward curves for a preserved acceptance bundle.

Requested by hamishivi (tmax-private#1): the reward CSV exists but no plotted
curve had been posted. Emits both the PNG and the tidy per-step series behind
it, so the figure is never the only artifact — a reviewer can recompute every
point from the CSV.

Plots, on shared step axis (global_step == acceptance_step for a fresh lineage):
  1. mean reward per step, with +/-1 std band across the 256 samples
  2. fraction of zero-reward samples per step
  3. per-group mean spread (min/median/max of the 8 group means) — this is the
     one that shows whether active sampling still has gradient signal, since a
     step where all 8 groups collapse to the same mean contributes nothing
  4. degenerate-group count per step (zero-variance groups)

Usage: python3 reward_curves.py REWARD_CSV --out-prefix logs/curves-<job>
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reward_csv")
    ap.add_argument("--out-prefix", default="logs/curves")
    args = ap.parse_args()

    steps = {}
    groups = defaultdict(list)
    for r in csv.DictReader(open(args.reward_csv)):
        gs = int(r["global_step"])
        if r["scope"] == "step":
            steps[gs] = {
                "acceptance_step": int(r["acceptance_step"]),
                "n": int(r["n_samples"]),
                "mean": float(r["reward_mean"]),
                "std": float(r["reward_std"]),
                "n_zero": int(r["n_zero"]),
            }
        else:
            groups[gs].append({"mean": float(r["reward_mean"]),
                               "degenerate": int(r["degenerate"])})

    xs = sorted(steps)
    mean = [steps[s]["mean"] for s in xs]
    std = [steps[s]["std"] for s in xs]
    zfrac = [steps[s]["n_zero"] / steps[s]["n"] for s in xs]
    gmin = [min(g["mean"] for g in groups[s]) for s in xs]
    gmax = [max(g["mean"] for g in groups[s]) for s in xs]
    gmed = [statistics.median([g["mean"] for g in groups[s]]) for s in xs]
    degen = [sum(g["degenerate"] for g in groups[s]) for s in xs]

    fig, ax = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    ax[0].plot(xs, mean, lw=1.6, color="#1f77b4")
    ax[0].fill_between(xs, [m - s for m, s in zip(mean, std)],
                       [m + s for m, s in zip(mean, std)], alpha=0.18, color="#1f77b4")
    ax[0].set_ylabel("mean reward"); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Acceptance rewards, steps {xs[0]}-{xs[-1]} "
                    f"({sum(steps[s]['n'] for s in xs):,} samples)")

    ax[1].plot(xs, zfrac, lw=1.4, color="#d62728")
    ax[1].set_ylabel("zero-reward fraction"); ax[1].set_ylim(0, 1); ax[1].grid(alpha=0.3)

    ax[2].fill_between(xs, gmin, gmax, alpha=0.25, color="#2ca02c", label="group min-max")
    ax[2].plot(xs, gmed, lw=1.4, color="#2ca02c", label="group median")
    ax[2].set_ylabel("per-group mean"); ax[2].legend(loc="upper right", fontsize=8)
    ax[2].grid(alpha=0.3)

    ax[3].plot(xs, degen, lw=1.4, color="#9467bd")
    ax[3].set_ylabel("degenerate groups"); ax[3].set_xlabel("global_step (= acceptance_step)")
    ax[3].set_ylim(-0.5, max(1, max(degen)) + 0.5); ax[3].grid(alpha=0.3)

    png = f"{args.out_prefix}.png"
    fig.tight_layout(); fig.savefig(png, dpi=110)

    series = [{"global_step": s, "acceptance_step": steps[s]["acceptance_step"],
               "n_samples": steps[s]["n"], "reward_mean": steps[s]["mean"],
               "reward_std": steps[s]["std"], "zero_fraction": round(zfrac[i], 6),
               "group_mean_min": gmin[i], "group_mean_median": gmed[i],
               "group_mean_max": gmax[i], "degenerate_groups": degen[i]}
              for i, s in enumerate(xs)]
    js = f"{args.out_prefix}.json"
    json.dump({"source_csv": args.reward_csv, "steps": len(xs), "series": series},
              open(js, "w"), indent=1)

    print(f"steps {xs[0]}..{xs[-1]}  samples {sum(steps[s]['n'] for s in xs):,}")
    print(f"reward mean over run : {statistics.mean(mean):.4f}  "
          f"(first {mean[0]:.3f} -> last {mean[-1]:.3f})")
    print(f"zero-fraction        : {statistics.mean(zfrac):.4f} mean")
    print(f"degenerate groups    : {sum(degen)} across {len(xs)*8} groups")
    print(f"wrote {png} and {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
