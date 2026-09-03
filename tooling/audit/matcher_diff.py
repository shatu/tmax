#!/usr/bin/env python3
"""Name the rule difference behind the -12 DPPO / -11 SGD verdict residual.

Two independently-written implementations of the same ruled predicate land a
consistent ~11-12 rows apart on both arms. A consistent offset is a rule
difference, not noise, so it is findable by running BOTH implementations over
the same corpus and reading the lines they disagree on -- rather than by
comparing totals and speculating.

Rulin's candidate suspects were their >=2-count-kinds context clause and
duration-form breadth. Reading the two sources side by side, mine looks
*broader* on both of those (I accept "sec"/"seconds", they require a bare "s";
I accept "test session"/"short test summary" as context), which would push my
count UP, not down. So the cause is more likely somewhere I am STRICTER. The
visible candidates are whitespace: they allow "\\d+\\s+passed" and
"\\d+\\s*/\\s*\\d+", I require single literal spaces.

This does not argue the point -- it runs both over real transcripts and groups
every disagreeing line by which side accepted it, so the answer comes from the
corpus.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool_audit import ANSI, TESTS_FAILED, _PASSED_TOKEN, load  # noqa: E402

RULIN = ("/checkpoint/memorization/oscaryinn/tmax/exchange/"
         "rulin_verify_steps1-200_v2/verdict_v3.py")


def load_rulin():
    spec = importlib.util.spec_from_file_location("verdict_v3", RULIN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mine(line: str):
    """'failed' | 'passed' | None under my implementation, per line."""
    if TESTS_FAILED.search(line):
        return "failed"
    if _PASSED_TOKEN.search(line):
        return "passed"
    return None


# why-buckets for a disagreement, checked in order
WHY = [
    ("whitespace_multi_space_count",
     re.compile(r"(?<![\w/.\-])\d+\s{2,}(passed|failed)\b", re.I)),
    ("whitespace_tab_count", re.compile(r"\d+\t+\s*(passed|failed)\b", re.I)),
    ("fraction_spaced_slash", re.compile(r"\d+\s*/\s+\d+\s+passed\b|\d+\s+/\s*\d+\s+passed\b", re.I)),
    ("duration_sec_or_seconds", re.compile(r"\bin\s+\d+(\.\d+)?\s*(sec|seconds)\b", re.I)),
    ("label_prefix_midline", re.compile(r"\S.*[A-Za-z][\w .\-]*:\s*\d+\s+passed\b", re.I)),
    ("test_session_or_summary_ctx", re.compile(r"\btest session\b|\bshort test summary\b", re.I)),
    ("label_prefix_anchored", re.compile(r"^\s*[A-Za-z][\w ()/'\-]{0,40}:\s*\d+(\s*/\s*\d+)?\s+(passed|failed)\b", re.I)),
]


def classify(line: str) -> str:
    for name, pat in WHY:
        if pat.search(line):
            return name
    return "unclassified"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts_dir")
    ap.add_argument("--max-step", type=int, default=200)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rv = load_rulin()
    only_mine: Counter = Counter()
    only_theirs: Counter = Counter()
    why_mine: Counter = Counter()
    why_theirs: Counter = Counter()
    n = 0
    for r in load(args.rollouts_dir):
        step = int(r.get("step", -1)) + 1
        if step > args.max_step:
            continue
        n += 1
        if args.limit and n > args.limit:
            break
        ri = r.get("request_info") or {}
        blob = (ANSI.sub("", str(ri.get("tool_outputs") or "")) + "\n"
                + ANSI.sub("", str(ri.get("tool_errors") or "")))
        for line in blob.splitlines():
            a, b = mine(line), rv.line_verdict(line)
            if a == b:
                continue
            key = line.strip()[:150]
            if a is None and b is not None:
                only_theirs[(b, key)] += 1
                why_theirs[classify(line)] += 1
            elif b is None and a is not None:
                only_mine[(a, key)] += 1
                why_mine[classify(line)] += 1
            else:
                only_theirs[(f"{a}->{b}", key)] += 1
                why_theirs["opposite_verdict"] += 1

    print(f"=== {args.tag}  records scanned {n}")
    print(f"  lines ONLY THEIRS accepts : {sum(only_theirs.values())} "
          f"({len(only_theirs)} distinct)")
    for k, c in why_theirs.most_common():
        print(f"      {k:34s} {c}")
    for (v, line), c in only_theirs.most_common(12):
        print(f"      {c:5d} [{v}] {line!r}")
    print(f"  lines ONLY MINE accepts   : {sum(only_mine.values())} "
          f"({len(only_mine)} distinct)")
    for k, c in why_mine.most_common():
        print(f"      {k:34s} {c}")
    for (v, line), c in only_mine.most_common(12):
        print(f"      {c:5d} [{v}] {line!r}")

    if args.out:
        json.dump({"tag": args.tag, "records": n,
                   "only_theirs_why": dict(why_theirs),
                   "only_mine_why": dict(why_mine),
                   "only_theirs": [[v, l, c] for (v, l), c in only_theirs.most_common(200)],
                   "only_mine": [[v, l, c] for (v, l), c in only_mine.most_common(200)]},
                  open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
