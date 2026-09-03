#!/usr/bin/env python3
"""Apportion SUSPECT:exit0_with_failure_in_same_call into its real sub-causes.

The class name reads like an accusation: "the wrapper reported success while
the command failed". At 5,693 records (DPPO, steps 1-200) that would be a large
silent-corruption surface, and it is the single number most likely to be quoted
out of this audit. It should not be quoted until it is broken down, because the
trigger condition in tool_audit.py is a disjunction:

    if code == "0" and TASK_FAIL_PATTERNS.search(body):
        if TESTS_FAILED.search(body) or re.search(r"Traceback ...", body):

so a bare Python traceback printed inside an exit-0 shell segment fires it, and
that is ordinary agent behaviour: the model runs a snippet in a heredoc, it
raises, the traceback prints, the shell that wrapped it exits 0, and the model
reads the error and fixes it. No test failed and nothing was concealed.

An earlier pass estimated the split from a 400-record draw (79.2% / 20.8%). A
draw was never necessary: which sub-branch fired is a deterministic property of
the segment text, so this runs over the FULL population and the apportionment
carries no sampling error at all.

Two accuracy requirements, both learned the hard way on this audit:

  * Mirror the classifier EXACTLY. tool_audit.py `break`s on the first
    triggering segment, so the class counts records keyed to that segment, not
    all triggering segments. Adjudicating every segment would produce totals
    that do not reconcile with the published class count -- the same shape of
    error as auditing transcript-wide instead of per tool call.
  * Reconciliation is asserted, not hoped for. The sub-bucket totals must sum
    to the class total; if they do not, this prints RECONCILE FAIL and exits
    non-zero rather than emitting a plausible-looking table.

For the genuinely test-failing sub-bucket the interesting question is why the
shell still exited 0, so those are discriminated further into suppressed exit
status (`|| true`, `set +e`, ...), pytest collection errors, and cases where a
later passing verdict in the SAME segment supersedes the failure. What remains
after those is the honest residual: exit 0 with an unexplained real test
failure. That residual is the only part of the 5,693 that deserves the name.

Usage:
  python3 exit0_adjudicate.py <rollouts_dir> --max-step 200 --tag dppo-11149108 \
      [--out-prefix logs/deliverables/steps1-200/exit0]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Import the classifier's own definitions so this cannot drift from the thing it
# is apportioning. Re-declaring the regexes here is exactly how the "0 failed"
# defect would survive a fix in one copy.
from tool_audit import (  # noqa: E402
    ANSI,
    TASK_FAIL_PATTERNS,
    TESTS_FAILED,
    _verdict_positions,
    load,
)

TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
COLLECTION_ERR = re.compile(r"error(s)? during collection", re.I)
# Ways a shell segment legitimately reports 0 after an inner command failed.
SUPPRESSED = re.compile(
    r"\|\|\s*(true|:|echo\b)|;\s*true\b|set\s+\+e\b|--exit-zero\b|"
    r"\bcontinue-on-error\b|2>\s*/dev/null\s*;\s*(true|:)",
)


def first_trigger_segment(blob: str):
    """The exact segment tool_audit.py would have keyed the class to."""
    segs = re.split(r"\(exit_code=(\d+)\)", blob)
    for i in range(1, len(segs), 2):
        code, body = segs[i], segs[i - 1]
        if code == "0" and TASK_FAIL_PATTERNS.search(body):
            if TESTS_FAILED.search(body) or TRACEBACK.search(body):
                return body
    return None


def adjudicate(body: str) -> tuple[str, str]:
    """(bucket, reason) for one triggering segment."""
    tf = TESTS_FAILED.search(body)
    tb = TRACEBACK.search(body)

    if not tf:
        # Traceback alone. The model ran something, it raised, the wrapping
        # shell exited 0. Nothing claims a test passed.
        return "traceback_in_exit0", "traceback_only_no_test_failure"

    # A real "N failed" (or collection error) inside an exit-0 segment.
    # Ask what ACTUALLY matched rather than re-testing the body with a second,
    # looser pattern: an inline `\d+ failed` re-test here would reintroduce the
    # port/index false positives this class was just cleaned of.
    if COLLECTION_ERR.search(tf.group(0)):
        return "tests_failed_in_exit0", "pytest_collection_error"
    if SUPPRESSED.search(body):
        return "tests_failed_in_exit0", "exit_status_explicitly_suppressed"

    # Does a passing verdict supersede the failure inside this same segment?
    last_pass, last_fail = _verdict_positions(body)
    if last_pass is not None and last_fail is not None and last_pass > last_fail:
        return "tests_failed_in_exit0", "later_pass_supersedes_in_same_segment"

    return ("tests_failed_in_exit0",
            "unexplained_exit0_with_test_failure" + ("_with_traceback" if tb else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts_dir")
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-prefix", default="logs/exit0")
    ap.add_argument("--class-total", type=int, default=None,
                    help="published class count; reconciliation is asserted against it")
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()

    buckets: dict[str, int] = {}
    reasons: dict[str, int] = {}
    samples: dict[str, list] = {}
    matched_text: dict[str, dict] = __import__("collections").defaultdict(dict)
    rows = []
    n_records = 0
    n_class = 0

    for r in load(args.rollouts_dir):
        step = int(r.get("step", -1)) + 1
        if args.max_step is not None and step > args.max_step:
            continue
        n_records += 1
        ri = r.get("request_info") or {}
        out = ANSI.sub("", str(ri.get("tool_outputs") or ""))
        err = ANSI.sub("", str(ri.get("tool_errors") or ""))
        blob = out + "\n" + err
        body = first_trigger_segment(blob)
        if body is None:
            continue
        n_class += 1
        bucket, reason = adjudicate(body)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        reasons[reason] = reasons.get(reason, 0) + 1
        rows.append({"step": step, "bucket": bucket, "reason": reason,
                     "segment_chars": len(body)})
        # Record the text that ACTUALLY FIRED, not the segment tail. Sampling
        # tails was useless here: the trigger is often thousands of characters
        # earlier, so the excerpt showed unrelated output and the residual
        # stayed "unexplained" for no better reason than that the evidence was
        # cropped out of view. Capture the match plus a window either side, and
        # tally the distinct matched strings so the residual can be described
        # from what fired rather than from a guess.
        m = TESTS_FAILED.search(body) or TRACEBACK.search(body)
        if m:
            lo, hi = max(0, m.start() - 160), min(len(body), m.end() + 160)
            matched_text[reason][m.group(0).strip().lower()] = \
                matched_text[reason].get(m.group(0).strip().lower(), 0) + 1
            if len(samples.setdefault(reason, [])) < args.samples:
                samples[reason].append({"step": step, "matched": m.group(0),
                                        "context": body[lo:hi]})

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    with open(f"{args.out_prefix}-{args.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "bucket", "reason", "segment_chars"])
        w.writeheader()
        w.writerows(rows)

    total = sum(buckets.values())
    reconciled = (args.class_total is None) or (n_class == args.class_total)
    payload = {
        "tag": args.tag, "max_step": args.max_step,
        "records_in_window": n_records,
        "class_total_recomputed": n_class,
        "class_total_published": args.class_total,
        "reconciles_with_published": reconciled,
        "buckets": buckets, "reasons": reasons,
        "bucket_pct": {k: round(100.0 * v / total, 2) for k, v in buckets.items()} if total else {},
        "reason_pct": {k: round(100.0 * v / total, 2) for k, v in reasons.items()} if total else {},
        "matched_strings_by_reason": {r: dict(sorted(d.items(), key=lambda kv: -kv[1])[:25]) for r, d in matched_text.items()},
        "method": ("Full population, not a sample. Mirrors tool_audit.py's first-"
                   "triggering-segment keying (it breaks on the first hit), so these "
                   "sub-buckets partition the published class count exactly."),
        "samples": samples,
    }
    json.dump(payload, open(f"{args.out_prefix}-{args.tag}.json", "w"), indent=1)

    print(f"=== {args.tag}  window records={n_records}  class={n_class}")
    for k in sorted(buckets, key=lambda x: -buckets[x]):
        print(f"  {k:26s} {buckets[k]:6d}  {100.0*buckets[k]/total:5.1f}%")
    print("  --- reasons ---")
    for k in sorted(reasons, key=lambda x: -reasons[x]):
        print(f"  {k:44s} {reasons[k]:6d}  {100.0*reasons[k]/total:5.1f}%")
    if args.class_total is not None:
        print(f"  RECONCILE {'OK' if reconciled else 'FAIL'}: recomputed {n_class} "
              f"vs published {args.class_total}")
    if total != n_class:
        print(f"  RECONCILE FAIL: buckets sum {total} != class {n_class}")
        return 1
    return 0 if reconciled else 1


if __name__ == "__main__":
    sys.exit(main())
