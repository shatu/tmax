#!/usr/bin/env python3
"""Reconcile "lost connection to sandbox" rollout ROWS into distinct INCIDENTS.

Requested by hamishivi after two counts of the same phenomenon disagreed:

  * I reported **2** genuine "lost connection to sandbox" while breaking down
    INFRA:transport_error.
  * Rulin's independent scan reported **16** (DPPO) and **20** (SGD).

Both can be correct, because they are counts of different strings. This script
settles it by counting every candidate phrasing SEPARATELY over the same
bounded window rather than picking one and asserting it, and by keeping
"Could not reach Sandfleet endpoint" in its own column as instructed.

The second thing it answers is the one that actually matters operationally: a
rollout ROW is not an incident. One sandbox going away takes out every rollout
that was talking to it, so N rows can be 1 event. Rows are grouped into
incidents by (step, worker/agent identity) where the record carries one, and
the grouping key used for each row is reported so the aggregation can be
checked rather than trusted. Where the saved record cannot establish worker
identity, the row is grouped by step alone and that weaker basis is labelled --
hamishivi's instruction was explicitly not to infer a cause the record cannot
establish, and the same restraint applies to inferring that two rows share an
incident.

Usage:
  python3 lostconn_incidents.py <rollouts_dir> --max-step 200 --tag dppo-11149108 \
      --out-prefix logs/deliverables/steps1-200/lostconn
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool_audit import ANSI, load  # noqa: E402

# Every phrasing counted separately -- no single "the" pattern.
PATTERNS = {
    "lost_connection_to_sandbox_step": re.compile(r"Lost connection to sandbox step", re.I),
    "lost_connection_to_sandbox_any": re.compile(r"lost connection to sandbox", re.I),
    "sandbox_worker_lost": re.compile(r"sandbox worker lost", re.I),
    "could_not_reach_endpoint": re.compile(r"Could not reach Sandfleet endpoint", re.I),
    "sandbox_lost_error_cls": re.compile(r"\bSandboxLostError\b"),
}
# Worker / lease identity, if the saved text carries it at all.
# Loopback is NOT worker identity. The first run grouped one row under
# "127.0.0.1:5000;127.0.0.1:9999" and labelled its basis "step+worker", which
# would have published a task-internal service address as if it identified a
# sandbox worker. Excluded explicitly rather than filtered by eye.
_LOOPBACK = re.compile(r"^(127\.|::1$|localhost$|0\.0\.0\.0$)")
AGENT_URL = re.compile(r"https?://([0-9a-zA-Z_.\-]+):(\d+)")
LEASE_ID = re.compile(r"lease[_ ]?id[\"'=: ]+([0-9a-fA-F-]{8,})", re.I)
SANDBOX_ID = re.compile(r"sandbox[_ ]?id[\"'=: ]+([0-9a-fA-F-]{8,})", re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts_dir")
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-prefix", default="logs/lostconn")
    args = ap.parse_args()

    counts: Counter = Counter()
    rows = []
    n = 0
    for r in load(args.rollouts_dir):
        step = int(r.get("step", -1)) + 1
        if args.max_step is not None and step > args.max_step:
            continue
        n += 1
        ri = r.get("request_info") or {}
        blob = (ANSI.sub("", str(ri.get("tool_outputs") or "")) + "\n"
                + ANSI.sub("", str(ri.get("tool_errors") or "")))
        hit = {k: bool(p.search(blob)) for k, p in PATTERNS.items()}
        for k, v in hit.items():
            if v:
                counts[k] += 1
        if not any(hit.values()):
            continue
        urls = sorted({f"{h}:{p}" for h, p in AGENT_URL.findall(blob)
                       if not _LOOPBACK.match(h)})
        lease = LEASE_ID.search(blob)
        sbx = SANDBOX_ID.search(blob)
        rows.append({
            "step": step,
            "task": r.get("task_id") or r.get("dataset_index") or "",
            "prompt_idx": r.get("prompt_idx", ""),
            "sample_idx": r.get("sample_idx", ""),
            "reward": r.get("reward", ""),
            "num_calls": (ri.get("num_calls") if isinstance(ri, dict) else "") or "",
            "timed_out": bool((ri.get("timeouts") if isinstance(ri, dict) else None)),
            "agent_urls": ";".join(urls),
            "lease_id": lease.group(1) if lease else "",
            "sandbox_id": sbx.group(1) if sbx else "",
            **{f"m_{k}": int(v) for k, v in hit.items()},
        })

    # --- rows -> incidents ---------------------------------------------------
    # Strong key: step + worker identity. Weak key: step alone, labelled.
    incidents: dict = defaultdict(list)
    basis: dict = {}
    for row in rows:
        ident = row["agent_urls"] or row["lease_id"] or row["sandbox_id"]
        if ident:
            key = ("step+worker", row["step"], ident)
        else:
            key = ("step-only", row["step"], "")
        incidents[key].append(row)
        basis[key] = key[0]

    strong = sum(1 for k in incidents if k[0] == "step+worker")
    weak = sum(1 for k in incidents if k[0] == "step-only")

    payload = {
        "tag": args.tag, "records_in_window": n,
        "row_counts_by_pattern": dict(counts),
        "rows_matching_any_pattern": len(rows),
        "incident_count_total": len(incidents),
        "incident_count_by_basis": {"step+worker": strong, "step_only_weak": weak},
        "incidents": [
            {"basis": k[0], "step": k[1], "identity": k[2], "rows": len(v),
             "tasks": sorted({x["task"] for x in v})}
            for k, v in sorted(incidents.items(), key=lambda kv: (kv[0][1], str(kv[0][2])))
        ],
        "note": ("Row counts per pattern are reported separately and deliberately "
                 "not merged: 'Lost connection to sandbox step' and the looser "
                 "'lost connection to sandbox' are different strings and were the "
                 "source of a 2-vs-16 disagreement. 'Could not reach Sandfleet "
                 "endpoint' is kept in its own column. Incidents grouped step-only "
                 "carry no worker identity in the saved record, so that grouping is "
                 "a lower bound on distinctness, not an established fact."),
    }
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    json.dump(payload, open(f"{args.out_prefix}-{args.tag}.json", "w"), indent=1)
    if rows:
        with open(f"{args.out_prefix}-{args.tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"=== {args.tag}  records={n}")
    print("  row counts, per pattern (NOT merged):")
    for k in PATTERNS:
        print(f"    {k:34s} {counts.get(k,0)}")
    print(f"  rows matching any            : {len(rows)}")
    print(f"  distinct INCIDENTS           : {len(incidents)} "
          f"(step+worker {strong}, step-only/weak {weak})")
    for k, v in sorted(incidents.items(), key=lambda kv: (kv[0][1], str(kv[0][2])))[:20]:
        print(f"    step {k[1]:4d}  {k[0]:12s} {str(k[2])[:44]:44s} rows={len(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
