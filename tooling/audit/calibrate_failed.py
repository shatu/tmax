#!/usr/bin/env python3
"""Calibrate the "N failed" test-verdict regex against the real corpus.

`TESTS_FAILED = \\b(?!0\\b)\\d+ failed\\b` treats any digits before the word
"failed" as a test-failure count. Adjudicating the exit0 class surfaced what
that actually matches in this corpus:

    468  '8080 failed'   <- a PORT, from "connection to 8080 failed"
     62  '8443 failed'      likewise
     33  '5432 failed'      likewise (postgres)
     11  '8 failed'      <- the 8 of "UTF-8 failed"
    163  '1 failed'      <- includes "1 out of 1 hunk FAILED" (patch, not tests)

None of those are test verdicts. This matters beyond one class because
_verdict_positions() is line-wise and FAILURE-DOMINANT: one such line anywhere
sets the rollout's terminal verdict to failed, which drives
zero_reward_though_final_tests_passed and full_reward_though_final_tests_failed.

Rather than guess a tighter pattern, this dumps every distinct line that the
current regex matches, with counts, so the replacement is calibrated against
what the corpus actually contains. Candidate patterns are then scored on that
same population and the disagreements printed, so the change can be reviewed as
a diff in behaviour rather than taken on faith.

Usage:
  python3 calibrate_failed.py <rollouts_dir> [--max-step 200] [--limit 20000]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool_audit import ANSI, TESTS_FAILED, load  # noqa: E402

# --- candidate replacement ---------------------------------------------------
# A test verdict is a COUNT in a test-summary context. Require both:
#   * the number is not glued to a preceding word/dot/hyphen  -> kills "UTF-8"
#   * the same line carries another test-summary token, or the count is small
#     and the line is a recognisable pytest/unittest summary
# Ports are excluded by the summary-context requirement, not by digit count, so
# a genuine "1024 failed" in a real summary line still matches.
SUMMARY_CTX = re.compile(
    r"\b\d+\s+(passed|skipped|deselected|xfailed|xpassed|error|errors|warning|warnings)\b"
    r"|\bin\s+\d+(\.\d+)?\s*s(ec|econds)?\b"
    r"|={3,}|\bTests?\b\s*:|\btest session\b|\bshort test summary\b|\bFAILED\s+\S+::",
    re.I,
)
COUNT = re.compile(r"(?<![\w.\-])(?!0\b)\d{1,4} failed\b", re.I)
COLLECTION = re.compile(r"\berror(s)? during collection\b", re.I)
# "1 out of 1 hunk FAILED" / "Hunk #1 FAILED" are patch output, never tests.
PATCH_NOISE = re.compile(r"hunk\b|\bout of \d+ hunks?\b|\.rej\b", re.I)


def candidate(line: str) -> bool:
    if COLLECTION.search(line):
        return True
    if not COUNT.search(line):
        return False
    if PATCH_NOISE.search(line):
        return False
    return bool(SUMMARY_CTX.search(line))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts_dir")
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    old_lines: Counter = Counter()
    only_old: Counter = Counter()   # old matched, candidate rejects
    only_new: Counter = Counter()   # candidate matches, old rejected
    n = 0
    for r in load(args.rollouts_dir):
        step = int(r.get("step", -1)) + 1
        if args.max_step is not None and step > args.max_step:
            continue
        n += 1
        if args.limit and n > args.limit:
            break
        ri = r.get("request_info") or {}
        blob = (ANSI.sub("", str(ri.get("tool_outputs") or "")) + "\n"
                + ANSI.sub("", str(ri.get("tool_errors") or "")))
        for line in blob.splitlines():
            o, c = bool(TESTS_FAILED.search(line)), candidate(line)
            if o:
                old_lines[line.strip()[:160]] += 1
            if o and not c:
                only_old[line.strip()[:160]] += 1
            elif c and not o:
                only_new[line.strip()[:160]] += 1

    print(f"records scanned: {n}")
    print(f"distinct lines matched by CURRENT regex : {len(old_lines)} "
          f"({sum(old_lines.values())} occurrences)")
    print(f"  dropped by candidate : {len(only_old)} distinct "
          f"({sum(only_old.values())} occurrences)")
    print(f"  newly caught by candidate : {len(only_new)} distinct "
          f"({sum(only_new.values())} occurrences)")
    print("\n--- TOP LINES THE CANDIDATE DROPS (should be non-test noise) ---")
    for line, c in only_old.most_common(25):
        print(f"  {c:6d}  {line!r}")
    print("\n--- TOP LINES THE CANDIDATE ADDS (should be real summaries) ---")
    for line, c in only_new.most_common(15):
        print(f"  {c:6d}  {line!r}")
    print("\n--- TOP LINES BOTH AGREE ARE FAILURES ---")
    both = Counter({k: v for k, v in old_lines.items() if k not in only_old})
    for line, c in both.most_common(15):
        print(f"  {c:6d}  {line!r}")

    if args.out:
        json.dump({"records": n,
                   "dropped": only_old.most_common(400),
                   "added": only_new.most_common(200),
                   "kept": both.most_common(200)}, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
