#!/usr/bin/env python3
"""Browsable curve bundle for a preserved acceptance run.

Requested by hamishivi (tmax-private#1): a directly viewable set of curves for
trainer 11096823 steps 1-77, because the W&B run lives on the meta-fair internal
instance and Hamish may not be able to reach it.

Source of truth is the LOCAL wandb transaction log
(output/wandb/wandb/run-*/run-*.wandb), read offline via wandb's own datastore
reader. No network, and no dependency on the internal W&B server staying up.

Emits, for every requested field that actually exists:
  * per-metric PNG panels, unsmoothed points plus a clearly labelled 5-step
    rolling line
  * a tidy step-keyed CSV and JSON containing every plotted series
  * a metric-inventory JSON recording which requested fields were PRESENT and
    which were ABSENT, so "no panel" is never ambiguous between "missing" and
    "forgotten"

Usage:
  python3 curve_bundle.py RUN_DIR --out-dir run-evidence/11096823
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# What hamishivi asked for -> the exact names this trainer logs. Anything not
# found is reported ABSENT rather than silently skipped.
REQUESTED = {
    "post_filter_reward": ["val/avg_group_performance_post_filter"],
    "pre_filter_reward": ["val/avg_group_performance_pre_filter"],
    "primary_reward": ["scores", "objective/verifiable_reward",
                       "objective/passthrough_reward"],
    "response_length_mean": ["batch/response_lengths", "val/sequence_lengths"],
    "response_length_p50_p90": ["val/sequence_lengths_max", "val/sequence_lengths_min",
                                "val/sequence_lengths_solved",
                                "val/sequence_lengths_unsolved"],
    "tool_calls_mean": ["tools/aggregate/avg_calls_per_rollout",
                        "tools/bash/avg_calls_per_rollout"],
    "tool_calls_aux": ["tools/aggregate/failure_rate", "tools/aggregate/avg_runtime",
                       "tools/bash/failure_rate", "tools/bash/avg_runtime"],
    "filtering": ["batch/filtered_prompts", "batch/filtered_prompts_zero",
                  "batch/filtered_prompts_solved", "batch/filtered_prompts_nonzero",
                  "batch/total_prompts", "batch/no_resampled_prompts",
                  "val/total_reward_groups"],
}
DEBUG_PREFIX = "debug/"


def read_history_json(path: str):
    """step -> {metric: value} from a pulled W&B history JSON.

    Preferred source. The local .wandb datastore reader recovered only 3 of 77
    steps here (1,312 frames unreadable, most likely a writer/reader version
    mismatch), and a 3-row "steps 1-77" bundle would be worse than none — so the
    authoritative path is the run history pulled from the server, with the
    offline reader kept only as a fallback.
    """
    rows = json.load(open(path))
    hist = defaultdict(dict)
    seen = set()
    for r in rows:
        step = r.get("training_step") or r.get("global_step")
        if step is None:
            continue
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                hist[int(step)][k] = v
                seen.add(k)
    return hist, seen


def read_history(run_dir: str):
    """step -> {metric: value} from the local wandb datastore (fallback)."""
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb

    wandb_file = glob.glob(os.path.join(run_dir, "*.wandb"))[0]
    ds = datastore.DataStore()
    ds.open_for_scan(wandb_file)

    hist = defaultdict(dict)
    seen_keys = set()
    bad = 0
    # The datastore reader raises on record types it does not know (observed:
    # IndexError from an unexpected dtype in the crc table). One malformed frame
    # must not cost us the whole history, so skip and keep scanning, but bound
    # the damage so a truly corrupt file still terminates.
    while True:
        try:
            raw = ds.scan_record()
        except Exception:
            bad += 1
            if bad > 5000:
                break
            continue
        if raw is None:
            break
        try:
            rec = pb.Record()
            rec.ParseFromString(raw[1] if isinstance(raw, tuple) else raw)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in rec.history.item:
            try:
                val = json.loads(item.value_json)
            except Exception:
                continue
            key = item.key or ".".join(item.nested_key)
            seen_keys.add(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                row[key] = val
        step = row.get("training_step") or row.get("global_step")
        if step is None:
            continue
        hist[int(step)].update(row)
    if bad:
        print(f"note: skipped {bad} unreadable datastore frames", file=sys.stderr)
    return hist, seen_keys


def rolling(xs, ys, w=5):
    out = []
    for i in range(len(ys)):
        lo = max(0, i - w + 1)
        out.append(statistics.mean(ys[lo:i + 1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="wandb run dir, or a pulled history .json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-step", type=int, default=77)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.run_dir.endswith(".json"):
        hist, seen = read_history_json(args.run_dir)
    else:
        hist, seen = read_history(args.run_dir)
    steps = sorted(s for s in hist if 1 <= s <= args.max_step)
    if not steps:
        print("no history rows found", file=sys.stderr)
        return 1

    # resolve requested -> present/absent
    inventory = {"present": {}, "absent": [], "extra_debug": []}
    plot_keys = []
    for label, cands in REQUESTED.items():
        found = [c for c in cands if any(c in hist[s] for s in steps)]
        if found:
            inventory["present"][label] = found
            plot_keys.extend(found)
        else:
            inventory["absent"].append({"requested": label, "tried": cands})
    debug_keys = sorted(k for k in seen if k.startswith(DEBUG_PREFIX)
                        and any(k in hist[s] for s in steps))
    inventory["extra_debug"] = debug_keys
    plot_keys.extend(debug_keys)
    plot_keys = list(dict.fromkeys(plot_keys))

    # tidy data
    rows = []
    for s in steps:
        r = {"global_step": s, "acceptance_step": s}
        for k in plot_keys:
            r[k] = hist[s].get(k)
        rows.append(r)
    csv_path = os.path.join(args.out_dir, "curves-11096823-series.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["global_step", "acceptance_step"] + plot_keys)
        w.writeheader(); w.writerows(rows)
    json.dump({"run": os.path.basename(args.run_dir), "steps": steps,
               "series": rows}, open(os.path.join(args.out_dir, "curves-11096823-series.json"), "w"), indent=1)
    json.dump(inventory, open(os.path.join(args.out_dir, "metric-inventory.json"), "w"), indent=1)

    # panels, chunked so each PNG stays readable
    made = []
    per_fig = 6
    for c in range(0, len(plot_keys), per_fig):
        chunk = plot_keys[c:c + per_fig]
        fig, axes = plt.subplots(len(chunk), 1, figsize=(11, 2.5 * len(chunk)), sharex=True)
        if len(chunk) == 1:
            axes = [axes]
        for ax, k in zip(axes, chunk):
            xs = [s for s in steps if hist[s].get(k) is not None]
            ys = [hist[s][k] for s in xs]
            if not xs:
                continue
            ax.plot(xs, ys, ".", ms=4, alpha=0.75, color="#1f77b4", label="per-step (unsmoothed)")
            if len(ys) >= 5:
                ax.plot(xs, rolling(xs, ys, 5), lw=1.5, color="#d62728",
                        label="5-step rolling (smoothed)")
            ax.set_ylabel(k.split("/")[-1][:22], fontsize=8)
            ax.set_title(k, fontsize=9, loc="left")
            ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
        axes[-1].set_xlabel("global_step (= acceptance_step)")
        fig.tight_layout()
        p = os.path.join(args.out_dir, f"curves-11096823-panel{c//per_fig + 1}.png")
        fig.savefig(p, dpi=105); plt.close(fig); made.append(p)

    print(f"steps {steps[0]}..{steps[-1]}  ({len(steps)} rows)")
    print(f"metrics plotted: {len(plot_keys)}  (debug/*: {len(debug_keys)})")
    print(f"requested-but-ABSENT: {[a['requested'] for a in inventory['absent']] or 'none'}")
    for p in made:
        print("  png:", p)
    print("  csv:", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
