#!/usr/bin/env python3
"""The JOINT verdict semantics ruled by hamishivi at 06:06, as a probe.

Deliberately NOT an edit to tool_audit.py. That file is the one hamishivi is
verifying at commit 2aae548a7 and the exchange copy Rulin reads mirrors it; a
local mutation would leave the tree diverged from the artifact under review.
When the hold clears, this ports across in one edit.

The ruled predicate:

  PASS requires one of
    * a strictly positive, non-fractional count in a runner context
    * a LABEL-PREFIXED positive count -- "Clean configs: 50 passed".
      Rulin's guard is adopted: the count must come IMMEDIATELY after the
      label colon. Without adjacency, "curl: connection to 8443 failed after
      3 retries" regains context through the label rule and the whole port
      false-positive class walks back in through the side door.
    * an EQUAL fraction "N/N passed", N > 0 -- a genuine full pass. My landed
      predicate rejects these, which is the single rule it got wrong.

  NOT a pass
    * zero counts ("0 passed", Rust's "0 passed; 0 failed" = no tests ran)
    * digitless decorated banners ("=== ALL TESTS PASSED ===") -- model echo,
      per the 11096823 precedent
    * UNEQUAL fractions ("2/6 passed") -- a partial result

  Unequal fractions are a FAILURE only under clear summary context, and
  unclassified otherwise, rather than inferred from arbitrary prose. I argued
  for scoring them failed under context on the grounds that
  full_reward_though_final_tests_failed staying 0 means more if it survives a
  rule that could have moved it than if nothing was allowed to count.

Calibrated against the combined corpus rather than hand-picked strings:
Rulin's 87-row his_lines extraction (every line my matcher fired on in the
one-sided disagreement set), plus the ruled boundary cases.
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- pass side --------------------------------------------------------------
_ZERO = r"0+"
# plain positive count, not glued to a word and not part of a fraction
_PASS_PLAIN = re.compile(r"(?<![\w/.\-])(?!%s\b)\d+ passed\b" % _ZERO, re.I)
# equal fraction N/N with N > 0
_PASS_EQFRAC = re.compile(r"(?<![\w/.\-])(?!0+\b)(\d+)/(\d+) passed\b", re.I)
# label-prefixed count: the count must sit IMMEDIATELY after the colon
_PASS_LABEL = re.compile(
    r"[A-Za-z][\w .\-]*:\s*(?!0+\b)(\d+)(?:/(\d+))? passed\b", re.I)
# any fraction at all, to detect the unequal case
_ANY_FRAC = re.compile(r"(?<![\w/.\-])(\d+)/(\d+) passed\b", re.I)
_DIGITLESS_BANNER = re.compile(r"=+[^=\d]*passed[^=\d]*=+", re.I)

_SUMMARY_CTX = re.compile(
    r"\b\d+\s+(failed|skipped|deselected|xfailed|xpassed|error|errors|warning|warnings)\b"
    r"|\bin\s+\d+(\.\d+)?\s*s(ec|econds)?\b"
    r"|={3,}|\bTests?\b\s*:|\bResults?\b\s*:|\btest session\b|\bshort test summary\b",
    re.I,
)


def joint_pass(line: str) -> bool:
    frac = _ANY_FRAC.search(line)
    if frac:
        a, b = int(frac.group(1)), int(frac.group(2))
        # equal and positive -> a genuine full pass; unequal -> never a pass
        return a == b and a > 0
    m = _PASS_LABEL.search(line)
    if m and not m.group(2):
        return True
    # A BARE positive count still needs runner context. Calibrating against
    # Rulin's 87-row extraction caught this: without the requirement the
    # predicate still called 10 DPPO / 22 SGD of the adjudicated FALSE passes
    # a pass, including 'Run 2 passed', '< echo "DEBUG: Pattern 2 passed"'
    # (a heredoc echo), 'Test 4 passed: REJECTED' (a rejection scored as a
    # pass) and 'Evil: 42 blocked (good), 8 passed (bad)'. The failure side
    # has demanded context since defect 2; omitting it here would have
    # rebuilt the very asymmetry this whole correction exists to remove.
    if _PASS_PLAIN.search(line) and _SUMMARY_CTX.search(line):
        return True
    return False


def joint_unequal_fraction_failure(line: str) -> bool:
    """Unequal fraction scores FAILED only under clear summary context."""
    frac = _ANY_FRAC.search(line)
    if not frac:
        return False
    a, b = int(frac.group(1)), int(frac.group(2))
    if a >= b:
        return False
    return bool(_SUMMARY_CTX.search(line))


# --- calibration ------------------------------------------------------------
RULED = [
    # equal fractions -- the rule my landed predicate gets wrong
    ("Clean: 50/50 passed", True, "equal fraction, genuine full pass"),
    ("Random tests: 100/100 passed", True, "equal fraction"),
    ("Clean files: 15/15 passed", True, "equal fraction"),
    ("Clean files: 1/1 passed", True, "equal fraction, N=1"),
    # unequal fractions
    ("=== 2/6 passed ===", False, "partial result"),
    ("Clean: 0/50 passed", False, "zero numerator"),
    # label-prefixed positive counts, and the adjacency guard
    ("Clean configs: 50 passed", True, "label-prefixed positive count"),
    ("curl: connection to 8443 failed after 3 retries", False,
     "Rulin's guard: count not adjacent to the colon, so the label rule must "
     "NOT fire -- otherwise the port false-positive class walks back in"),
    ("Evil: 42 blocked (good), 8 passed (bad)", False,
     "count not adjacent to a label colon"),
    # zero and banners
    ("Evil configs: 0 passed", False, "zero"),
    ("test result: ok. 0 passed; 0 failed; 0 ignored", False, "no tests ran"),
    ("=== ALL TESTS PASSED ===", False, "digitless banner, model echo"),
    ("=== All checks passed! ===", False, "digitless banner"),
    # must-keep positives
    ("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured", True, "Rust"),
    ("Results: 50 passed, 0 failed out of 50 random tests", True, "Results prefix"),
    ("===== 3 passed in 0.04s =====", True, "pytest summary"),
    ("========================= 1 failed, 2 passed in 0.04s ============", True,
     "mixed line; failure dominance is handled separately"),
    # must stay unrecognised (scope, not bugs)
    ("test_server.py::TestX::test_defaults_to_1 PASSED [ 50%]", False,
     "pytest per-test verbose line; old pattern never matched it"),
    ("TEST1 PASSED: x=5", False, "digit glued to an identifier"),
]


def main() -> int:
    bad = 0
    print("=== ruled boundary cases ===")
    for text, want, why in RULED:
        got = joint_pass(text)
        if got != want:
            bad += 1
            print(f"  FAIL want={want} got={got}  {text[:78]!r}\n       ({why})")
    print(f"  {len(RULED)-bad}/{len(RULED)} ruled cases OK")

    # --- against Rulin's 87-row extraction -----------------------------------
    p = ("/checkpoint/memorization/oscaryinn/tmax/exchange/"
         "rulin_verify_steps1-200_v2/his_lines.json")
    if os.path.exists(p):
        h = json.load(open(p))
        print("\n=== Rulin's his_lines extraction (lines my matcher fired on) ===")
        from collections import Counter
        for arm, rows in h.items():
            still = Counter()
            n = 0
            for r in rows:
                v = r.get("his_last_verdict") or []
                if len(v) < 2:
                    continue
                line = v[1]
                n += 1
                if joint_pass(line):
                    still[line.strip()[:96]] += 1
            print(f"  {arm}: {n} rows; joint predicate still calls "
                  f"{sum(still.values())} of them a pass")
            for line, c in still.most_common(8):
                print(f"      {c:3d}  {line!r}")
    else:
        print("\n  his_lines.json not readable -- calibration incomplete")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
