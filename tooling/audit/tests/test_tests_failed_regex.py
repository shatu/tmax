#!/usr/bin/env python3
"""Regression tests for TESTS_FAILED, the test-failure verdict detector.

Two defects have now been found in this one pattern, both of the same shape --
it matched text that is not a test verdict -- and both mattered well beyond the
class they were noticed in, because _verdict_positions() is line-wise and
failure-dominant: one bad line sets a whole rollout's terminal verdict.

  1. "0 failed" scored as a FAILURE ("All tests passed! ... 0 failed").
  2. Any digits before "failed" scored as a count, so ports ("bind() to
     0.0.0.0:8080 failed", 11,385 occurrences on DPPO alone), indices
     ("Drive 3 failed", "Attempt 2 failed"), patch hunks ("1 out of 1 hunk
     FAILED") and even the "8" inside "UTF-8 failed" all read as test failures.

Every case below is real text taken from the steps 1-200 corpus, not invented,
so this pins the behaviour to what the data actually contains.

Run:  python3 test_tests_failed_regex.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_audit import TESTS_FAILED, _PASSED_TOKEN, _verdict_positions  # noqa: E402

# --- the PASS side ----------------------------------------------------------
# Added after Rulin's row-level adjudication of the 87 one-sided
# zero_reward_though_final_tests_passed rows and hamishivi's four-point ruling.
# Same defect family as the failure side, and the same lesson: `0 failed` was
# fixed on one side only, so the asymmetry became the bug.
# Every string is real corpus text from the adjudication set.
PASS_CASES = [
    # must REMAIN recognised
    ("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured", True,
     "Rust cargo summary; a positive count with zeros beside it is still a pass"),
    ("Results: 50 passed, 0 failed out of 50 random tests", True,
     "Results-prefix form Rulin's matcher accepts and mine must too"),
    ("===== 3 passed in 0.04s =====", True, "ordinary pytest summary"),
    ("========================= 1 failed, 2 passed in 0.04s ==================",
     True, "mixed line: the pass token matches, failure dominance is separate"),
    ("Tests: 5 passed, 0 failed", True, "jest-style"),

    # ruling 2: digitless decorated banners are MODEL ECHO, not verdicts
    ("=== ALL TESTS PASSED ===", False,
     "model's own claim banner; 11096823 precedent says echo, not verdict"),
    ("=== All checks passed! ===", False, "same, majority kind in the 87 rows"),

    # ruling 1: strictly positive count
    ("Evil configs: 0 passed", False, "zero is not a pass"),
    ("test result: ok. 0 passed; 0 failed; 0 ignored", False,
     "Rust reporting that NO tests ran; 'ok' here is not a pass verdict"),
    ("00 passed", False, "padded zero is still zero"),

    # ruling 3: fraction forms are partial results
    ("=== 2/6 passed ===", False, "partial result Rulin adjudicated against itself"),
    # RULED 06:06: equal fractions ARE passes. The previous revision rejected
    # them -- that was the superseded 05:42 blanket rule, and these two cases
    # encoded it, so a passing suite was pinning the wrong behaviour.
    ("Clean: 50/50 passed", True, "equal fraction, genuine full pass"),
    ("Random tests: 100/100 passed", True, "equal fraction"),
    ("Clean files: 15/15 passed", True, "equal fraction"),
    ("PROPERTY_TESTS: 10/10 PASSED", True, "equal fraction, label-prefixed"),
    ("Clean: 0/50 passed", False, "zero numerator is not a pass"),
    ("=== 2/6 passed ===", False, "unequal fraction is not a pass"),
    # Rulin's adjacency guard: the count must sit immediately after the colon,
    # or the label rule readmits the entire port false-positive class.
    ("curl: connection to 8443 failed after 3 retries", False,
     "count not adjacent to the label colon"),
    ("Evil: 42 blocked (good), 8 passed (bad)", False,
     "bare count with no runner context"),
    ("Run 2 passed", False, "bare count, no context"),
    ("Test 4 passed: REJECTED", False, "a rejection is not a pass"),
    # inversion markers are ANNOTATION ONLY and must not change the verdict
    ("Evil corpus: 2/2 passed (should reject all)", True,
     "syntactically an equal-fraction pass; inverted task semantics are recorded "
     "in the inversion_marker column, NOT in the predicate"),
]

# (text, should_match, why)
CASES = [
    # --- real pytest summaries: MUST match --------------------------------
    ("============================== 1 failed in 0.04s ===============================",
     True, "bare pytest summary, most common failing line in the corpus"),
    ("========================= 1 failed, 1 passed in 0.04s ==========================",
     True, "mixed summary; failure must dominate the passed count on the same line"),
    ("!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!",
     True, "collection error is a failure even with no count"),
    ("Tests: 1 failed, 2 passed, 3 total", True, "jest-style summary"),
    ("1024 failed, 1 passed in 3s", True,
     "a large but genuine count must survive; ports are excluded by CONTEXT, "
     "not by a digit cap, so this must not be collateral damage"),
    ("2 failed in 1.23s", True, "count plus duration, no banner"),

    # --- defect 1: zero is a pass ----------------------------------------
    ("Results: 50 passed, 0 failed out of 50 random tests", False, "defect 1"),
    ("All tests passed! ... 0 failed", False, "defect 1"),

    # --- defect 2: ports, indices, hunks ----------------------------------
    ("nginx: [emerg] bind() to 0.0.0.0:8080 failed (98: Unknown error)", False,
     "defect 2, the single largest false source: 11,385 occurrences on DPPO"),
    ("nginx: [emerg] bind() to 127.0.0.1:8443 failed (98: Unknown error)", False, "port"),
    ('psql: error: connection to server at "localhost" (::1), port 5432 failed: '
     'FATAL:  password authentication failed for user "postgres"', False, "port"),
    ("* connect to 127.0.0.1 port 8443 failed: Connection refused", False, "port"),
    ("Drive 3 failed.", False, "device index, not a test count"),
    ("Attempt 2 failed: [Errno 98] Address already in use", False, "retry index"),
    ("[2023-10-24 10:16:30] User 5678 failed with ERR_TIMEOUT.", False, "user id"),
    ("1 out of 1 hunk FAILED -- saving rejects to file math_ops.rs.rej", False,
     "patch(1) output, not tests"),
    ("Hunk #1 FAILED at 1.", False, "patch(1) output"),
    ("# If UTF-8 failed or detected as UTF-16 without BOM", False,
     "the '8' belongs to UTF-8; a source comment is not a verdict"),
    ("Rule 4 failed: No path exists between A and Z", False, "rule index"),

    # --- indexed harness verdicts: kept ON PURPOSE ------------------------
    # The old pattern caught these by misreading the index as a count. They are
    # genuine failures, so keeping them is what makes the fix subtractive-only.
    ("Test 8 failed", True, "model-written harness verdict"),
    ("Test 1 FAILED", True, "case-insensitive"),
    ("Property test 1 FAILED", True, "hypothesis-style"),
    ("Check 3 failed: tail", True, "check-style harness"),
]


def main() -> int:
    npass = nfail = 0
    for text, want, why in CASES:
        got = TESTS_FAILED.search(text) is not None
        if got == want:
            npass += 1
        else:
            nfail += 1
            print(f"  FAIL  want={want} got={got}  {text[:88]!r}\n        ({why})")
    print(f"  {npass} passed, {nfail} failed  [failure-side cases]")

    ppass = pfail = 0
    for text, want, why in PASS_CASES:
        got = _PASSED_TOKEN.search(text) is not None
        if got == want:
            ppass += 1
        else:
            pfail += 1
            print(f"  FAIL  want={want} got={got}  {text[:88]!r}\n        ({why})")
    print(f"  {ppass} passed, {pfail} failed  [pass-side cases]")
    nfail += pfail

    # --- the reason it matters: terminal verdict must not flip ------------
    extra = 0
    transcript = (
        "running the suite\n"
        "========================= 2 passed in 0.10s ==========================\n"
        "nginx: [emerg] bind() to 0.0.0.0:8080 failed (98: Unknown error)\n"
    )
    last_pass, last_fail = _verdict_positions(transcript)
    if not (last_pass is not None and last_fail is None):
        print("  FAIL  a trailing nginx port-bind line still flips the terminal "
              "verdict to FAILED — this is the whole reason the pattern matters")
        extra += 1
    else:
        print("  PASS  trailing port-bind noise no longer overrides a passing verdict")

    # a digitless banner must NOT override a real failing summary
    t3 = ("========================= 1 failed, 1 passed in 0.04s ==========\n"
          "=== ALL TESTS PASSED ===\n")
    lp3, lf3 = _verdict_positions(t3)
    if not (lf3 is not None and (lp3 is None or lf3 > lp3)):
        print("  FAIL  a model echo banner still overrides a real failing summary")
        extra += 1
    else:
        print("  PASS  model echo banner no longer overrides a real failing summary")

    transcript2 = (
        "========================= 2 passed in 0.10s ==========================\n"
        "========================= 1 failed, 1 passed in 0.04s ================\n"
    )
    last_pass, last_fail = _verdict_positions(transcript2)
    if not (last_fail is not None and (last_pass is None or last_fail > last_pass)):
        print("  FAIL  a genuine trailing failure summary no longer registers")
        extra += 1
    else:
        print("  PASS  a genuine trailing failure summary still dominates")

    total_fail = nfail + extra
    print()
    if total_fail:
        print(f"  {total_fail} FAILING")
        return 1
    print("TESTS_FAILED_REGEX_TESTS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
