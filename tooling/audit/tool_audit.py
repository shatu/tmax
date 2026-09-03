#!/usr/bin/env python3
"""Tool-behaviour anomaly audit for a preserved rollout bundle.

Requested by hamishivi on tmax-private#1 (2026-09-01) for trainer 11096823
steps 1-77: not aggregate health, but per-call behaviour — and specifically the
question of whether Sandfleet/TMAX infrastructure faults are being mistaken for
model/task failures, or vice versa.

Design notes that matter for reading the output:

  * INFRA vs TASK is the primary split. A sandbox OOM, a backend loss, a
    transport error or an OCI-conversion fallback is an infrastructure event and
    must not be scored as the model failing the task. A non-zero exit from the
    model's own command, or a test that legitimately fails, is a task outcome.
  * "exit_code=0 alongside failure text" is deliberately separated from
    "non-zero exit": the first is a silent-corruption class (the wrapper claims
    success while the payload says otherwise) and is far more serious.
  * Reward-semantic mismatches are computed against the TERMINAL state of the
    rollout, not against any single call, because a rollout can legitimately
    contain a failed call and still succeed afterwards.

Every class is emitted with counts by step and by image, plus a bounded sample,
so a reviewer can reproduce any row from the source JSONL.

Usage:
  python3 tool_audit.py ROLLOUTS_DIR [--out-prefix logs/tool_audit-<job>]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Infrastructure markers — these indicate the platform failed, not the model.
INFRA_PATTERNS = {
    "sandbox_oom": re.compile(r"OOM reaper|SandboxOOMError|exceeded its Slurm step memory", re.I),
    "backend_lost": re.compile(r"Sandbox worker was lost|SandboxLostError|sandbox_lost", re.I),
    "instance_not_started": re.compile(r"Instance not started", re.I),
    "transport_error": re.compile(r"Could not reach Sandfleet endpoint|Connection refused|"
                                  r"Connection reset by peer|Lost connection to sandbox", re.I),
    "oci_fallback": re.compile(r"Converting OCI blobs|docker://", re.I),
    "reset_failed": re.compile(r"Reset failed after \d+ attempts", re.I),
    "rollout_walltime": re.compile(r"Rollout timed out after", re.I),
    # hamishivi refinement (2): this is a TEXTUAL match and the phrase can come
    # from task content (a model's own program printing "timed out after ..."),
    # so it is not a trustworthy infrastructure count. Labelled TEXTUAL.
}
TEXTUAL_PATTERNS = {
    "tool_call_timeout_textual": re.compile(r"timed out after .*s\b", re.I),
}

# Task-level failure text produced by the model's own commands.
TASK_FAIL_PATTERNS = re.compile(
    r"Traceback \(most recent call last\)|ModuleNotFoundError|SyntaxError|"
    r"command not found|No such file or directory|FAILED|\bERROR\b|assert",
    re.I,
)
# hamishivi refinement (1): pytest summarises as "1 failed, 3 passed" on ONE
# line. Matching "passed" anywhere would score that line as a pass. So a line
# containing "N failed" is a FAILED line regardless of any passed count on it,
# and the passed regex is evaluated only on lines with no failure count.
# NOTE the (?!0\b): "0 failed" is a PASS, not a failure. The original pattern
# (hamishivi refinement (1), to make "1 failed, 3 passed" score as failed) was
# right in intent but matched zero too, so lines like
#   "Results: 50 passed, 0 failed out of 50 random tests"
#   "All tests passed! ... 0 failed"
# were scored as terminal FAILURES. Measured at 0.6% of records in the DPPO
# steps 1-200 window (45 of a 8,000-record sample), and it feeds
# _verdict_positions, so it distorts zero_reward_though_final_tests_passed,
# full_reward_though_final_tests_failed and exit0_with_failure_in_same_call.
#
# SECOND DEFECT, same family, found while apportioning the exit0 class: the
# pattern took ANY digits before "failed", so it was matching ports and indices
# rather than test counts. Calibrated over the full steps 1-200 corpus
# (probes/calibrate_failed.py), the old pattern fired 31,169 times on DPPO and
# 19,727 of those (63%) were not test verdicts at all:
#
#   11,385  nginx: [emerg] bind() to 0.0.0.0:8080 failed (98: Unknown error)
#      578  psql: ... port 5432 failed: FATAL: password authentication failed
#      136  Drive 3 failed.
#       ..  "Attempt 2 failed", "User 5678 failed", "1 out of 1 hunk FAILED",
#           and "8 failed" matched out of the middle of "UTF-8 failed"
#
# Because _verdict_positions is line-wise and FAILURE-DOMINANT, one stray
# "bind() to 0.0.0.0:8080 failed" anywhere in a transcript set that rollout's
# terminal verdict to failed. A test verdict is a COUNT IN A TEST-SUMMARY
# CONTEXT, or an indexed verdict like "Test 8 failed", so require that.
#
# Ports are excluded by the context requirement rather than by capping digits,
# so a real "1024 failed, 1 passed in 3s" still matches. Calibration reported
# ZERO lines newly caught on either arm: the change is purely subtractive
# relative to the old pattern, which is the only shape of change that cannot
# invent new findings.
_FAIL_COUNT = re.compile(r"(?<![\w.\-])(?!0\b)\d{1,4} failed\b", re.I)
_FAIL_SUMMARY_CTX = re.compile(
    r"\b\d+\s+(passed|skipped|deselected|xfailed|xpassed|error|errors|warning|warnings)\b"
    r"|\bin\s+\d+(\.\d+)?\s*s(ec|econds)?\b"
    r"|={3,}|\bTests?\b\s*:|\btest session\b|\bshort test summary\b|\bFAILED\s+\S+::",
    re.I,
)
_FAIL_COLLECTION = re.compile(r"\berror(s)? during collection\b", re.I)
# An indexed verdict from a model-written harness ("Test 8 failed") is a real
# failure, just not a count. The old pattern caught these by accident (reading
# the index as a count); keeping them deliberately is what makes this change
# subtractive-only. 128 occurrences on DPPO, 8 on SGD.
_FAIL_INDEXED = re.compile(
    r"\b(?:property\s+)?(?:test|check|case|assertion)\s*#?\s*\d+\s+failed\b", re.I)
# patch(1) output is never a test verdict.
_FAIL_PATCH_NOISE = re.compile(r"\bhunk\b|\bout of \d+ hunks?\b|\.rej\b", re.I)
# Unequal-fraction pass counts ("2/6 passed") read as a failing run under
# clear summary context; see the ruled boundary in TESTS_FAILED._match_line.
_FAIL_UNEQUAL_FRAC = re.compile(r"(?<![\w/.\-])(\d+)\s*/\s*(\d+)\s+passed\b", re.I)
# Ruled 10:32, deliberately narrow: for UNEQUAL FRACTIONS ONLY, an anchored
# short label immediately followed by "N / M passed" is sufficient summary
# context. Scoped this way on purpose -- broadening plain "N failed" into an
# arbitrary-label rule is what let Rulin's matcher score
#   "Connection to localhost:27017 failed"   (a PORT, read as a count)
#   "Final result: 4 failed downstream nodes" (task-domain quantity)
# as failures. Requiring a FRACTION is what keeps those out: neither is one.
_FRAC_LABEL_CTX = re.compile(
    r"^\s*[A-Za-z][\w .\-]{0,30}:\s*\d+\s*/\s*\d+\s+passed\b", re.I)


class _ShiftedMatch:
    """A match reported in the coordinates of the full text, not of its line."""

    __slots__ = ("_m", "_off")

    def __init__(self, m, off):
        self._m, self._off = m, off

    def start(self):
        return self._m.start() + self._off

    def end(self):
        return self._m.end() + self._off

    def group(self, *a):
        return self._m.group(*a)


class TESTS_FAILED:  # noqa: N801 - kept as a name so call sites are unchanged
    """Line-wise test-failure verdict detector.

    Judged per line because the summary context ("N passed", "in 0.04s", the
    ==== banner) lives on the same line as the count. Offsets are shifted back
    into the caller's coordinates so `.start()`/`.end()` still address the
    original text -- exit0_adjudicate.py slices a context window around them.
    """

    @staticmethod
    def _match_line(line: str):
        if _FAIL_COLLECTION.search(line):
            return _FAIL_COLLECTION.search(line)
        if _FAIL_PATCH_NOISE.search(line):
            return None
        m = _FAIL_INDEXED.search(line)
        if m:
            return m
        m = _FAIL_COUNT.search(line)
        if m and _FAIL_SUMMARY_CTX.search(line):
            return m
        # Ruled boundary: an UNEQUAL fraction ("2/6 passed") is a failing run --
        # four of six did not pass -- but only where the line is clearly a test
        # summary. Outside that, leave it unclassified rather than infer a
        # verdict from arbitrary prose. Scoring it failed under context is the
        # stricter reading on purpose: full_reward_though_final_tests_failed
        # staying 0 means more if it survives a rule that COULD have moved it
        # than if it stays 0 because nothing was allowed to count.
        m = _FAIL_UNEQUAL_FRAC.search(line)
        if m and int(m.group(1)) < int(m.group(2)):
            # Context must come from something OTHER than the fraction itself.
            # "2/6 passed" contains "6 passed", which satisfies the summary
            # context regex, so the fraction would vouch for its own context and
            # every bare fraction would score as a failure. Same circularity as
            # on the pass side; caught by the same kind of unit case. Strip the
            # match, then ask.
            rest = line[:m.start()] + " " + line[m.end():]
            if _FAIL_SUMMARY_CTX.search(rest) or _FRAC_LABEL_CTX.search(line):
                return m
        return None

    @classmethod
    def search(cls, text: str):
        off = 0
        for line in text.splitlines(keepends=True):
            m = cls._match_line(line)
            if m is not None:
                return _ShiftedMatch(m, off)
            off += len(line)
        return None


# The PASS side, corrected to hamishivi's four-point ruling after Rulin's
# row-level adjudication of the zero_reward_though_final_tests_passed
# disagreement set. The old pattern was
#
#     \b\d+ passed\b | =+ .*passed.* =+
#
# and it was the mirror image of the `0 failed` bug: I tightened the FAILURE
# side to require a non-zero count in a summary context and left the PASS side
# bare, so a strict FAILED ran against a loose PASSED and every rollout was
# biased toward final_passed. Adjudication of the 87 one-sided rows found the
# false passes fall into these kinds:
#
#   1. DIGITLESS DECORATED BANNERS, the majority: "=== ALL TESTS PASSED ===",
#      "=== All checks passed! ===". These are the MODEL'S OWN claim banners,
#      not a test runner's verdict. The 11096823 audit already adjudicated this
#      class as model echo; the `=+ .*passed.* =+` alternative contradicted that
#      precedent, so it is removed outright.
#   2. ZERO COUNTS: "Evil configs: 0 passed", and Rust's
#      "test result: ok. 0 passed; 0 failed; ..." which means NO tests ran.
#   3. FRACTION FORMS: "=== 2/6 passed ===" is a partial result, not a pass.
#
# Rule: a terminal pass verdict requires a STRICTLY POSITIVE, NON-FRACTIONAL
# passed count. The lookbehind rejects a count preceded by a digit, "/" or "."
# so the denominator of "2/6 passed" cannot be read as the count.
#
# Documented limitation, shared with Rulin's independent matcher so the two
# agree: blanket fraction exclusion also drops numerator==denominator forms
# such as "Random tests: 100/100 passed", which is arguably a genuine full
# pass. Both matchers are conservative here; the exclusion is NOT lossless and
# PROVENANCE says so rather than letting a reader assume it is.
#
# Purely subtractive by construction: every alternative is strictly narrower
# than the old pattern and no new form is recognised.
# NOTE the lookbehind is [\\w/.-], not [\\d/.]. A first attempt used the
# narrower class and the subtractive check caught it immediately: it ADDED
# 198 lines on DPPO, because dropping \\b let the count match a digit glued to
# an identifier -- "test_empty_multiplier_defaults_to_1 PASSED" fired on the
# "1" of "to_1", and "TEST1 PASSED" on the "1" of "TEST1". Those are pytest
# per-test verbose lines, not summaries, and the old pattern never matched
# them; recognising them would be scope expansion, not a bug fix. This is why
# subtractiveness is MEASURED here and not argued from the regex shape.
# --- the ruled JOINT semantics (hamishivi 06:06, after Rulin's adjudication) --
# The previous revision blanket-rejected every fraction, which was the 05:42
# rule and was superseded while the rerun was in flight. Equal fractions are
# genuine full passes and must be accepted; only UNEQUAL ones are partial.
_PASS_PLAIN = re.compile(r"(?<![\w/.\-])(?!0+\b)\d+ passed\b", re.I)
_ANY_FRAC_PASS = re.compile(r"(?<![\w/.\-])(\d+)\s*/\s*(\d+)\s+passed\b", re.I)
# A label-prefixed count is a verdict, but Rulin's adjacency guard is required:
# the count must sit IMMEDIATELY after the label colon. Without it,
# "curl: connection to 8443 failed after 3 retries" regains context through the
# label rule and the entire port false-positive class walks back in sideways.
_PASS_LABEL = re.compile(
    r"[A-Za-z][\w .\-]*:\s*(?!0+\b)(\d+)(?:\s*/\s*(\d+))?\s+passed\b", re.I)
_PASS_CONTEXT = re.compile(
    r"\b\d+\s+(failed|skipped|deselected|xfailed|xpassed|error|errors|warning|warnings)\b"
    r"|\bin\s+\d+(\.\d+)?\s*s(ec|econds)?\b"
    r"|={3,}|\bTests?\b\s*:|\bResults?\b\s*:|\btest session\b|\bshort test summary\b",
    re.I,
)
# Informational only, NEVER verdict-affecting. Some harnesses invert the sense
# of "passed": "Evil corpus: 2/2 passed (should reject all)" is an equal
# fraction whose task EXPECTED rejection, so a syntactic pass is the task
# failing. I proposed excluding these by vocabulary; Rulin argued that bakes one
# corpus's idiom ("Evil") into the canonical predicate, catching today's rows
# while silently missing tomorrow's "Adversarial:"/"attack corpus" AND looking
# semantics-aware while doing it. That argument is better than mine. So the
# predicate stays purely syntactic per the ruling and the corpus-specific
# knowledge lives here, in an annotation that can be wrong safely.
INVERSION_MARKER = re.compile(
    r"\bevil\b|\bmalicious\b|\badversarial\b|should reject|should fail|"
    r"expected to fail|exit code 1 expected|attack corpus", re.I)


class _PassToken:  # noqa: N801 - name kept so call sites are unchanged
    """Terminal-pass detector under the ruled joint semantics."""

    @staticmethod
    def _match_line(line: str):
        frac = _ANY_FRAC_PASS.search(line)
        if frac:
            a, b = int(frac.group(1)), int(frac.group(2))
            # equal and positive -> genuine full pass; unequal -> never a pass
            return frac if (a == b and a > 0) else None
        m = _PASS_LABEL.search(line)
        if m and not m.group(2):
            return m
        # A BARE positive count still needs runner context. Calibration against
        # Rulin's 87-row extraction showed that without this the predicate still
        # called 10 DPPO / 22 SGD of the adjudicated FALSE passes a pass --
        # 'Run 2 passed', a heredoc 'echo "DEBUG: Pattern 2 passed"',
        # 'Test 4 passed: REJECTED', 'Evil: 42 blocked (good), 8 passed (bad)'.
        # The failure side has demanded context since defect 2; omitting it here
        # rebuilds the very asymmetry this correction exists to remove.
        m = _PASS_PLAIN.search(line)
        if m and _PASS_CONTEXT.search(line):
            return m
        return None

    @classmethod
    def search(cls, text: str):
        off = 0
        for line in text.splitlines(keepends=True):
            m = cls._match_line(line)
            if m is not None:
                return _ShiftedMatch(m, off)
            off += len(line)
        return None


_PASSED_TOKEN = _PassToken


def _deciding_pass_line(text: str):
    """The line that set the terminal PASS verdict, or None.

    Mirrors _verdict_positions' failure-dominant, line-wise walk so the
    annotation describes the same line the verdict came from.
    """
    winner = None
    for line in text.splitlines(keepends=True):
        if TESTS_FAILED.search(line):
            winner = None
        elif _PASSED_TOKEN.search(line):
            winner = line
    return winner


def _verdict_positions(text: str):
    """(last_pass_pos, last_fail_pos) evaluated line-wise, failure-dominant."""
    last_pass = last_fail = None
    off = 0
    for line in text.splitlines(keepends=True):
        if TESTS_FAILED.search(line):
            last_fail = off
        elif _PASSED_TOKEN.search(line):
            last_pass = off
        off += len(line)
    return last_pass, last_fail


class TESTS_PASSED:  # noqa: N801 - kept as a name for the call sites below
    @staticmethod
    def search(text):
        lp, lf = _verdict_positions(text)
        return lp is not None

    @staticmethod
    def finditer(text):
        lp, _ = _verdict_positions(text)
        return iter([type("M", (), {"start": staticmethod(lambda p=lp: p)})()] if lp is not None else [])
EXIT0 = re.compile(r"\(exit_code=0\)")
EXIT_NONZERO = re.compile(r"\(exit_code=(?!0\))(\d+)\)")


def load(rollouts_dir: str):
    for path in sorted(glob.glob(os.path.join(rollouts_dir, "*_rollouts_*.jsonl"))):
        with open(path, errors="replace") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts_dir")
    # Bound the audit to a step range. Without this the audit would cover every
    # record on disk -- production is at 220 while the control is at 200, so an
    # unbounded run would compare 1-220 against 1-201 and call it matched.
    ap.add_argument("--max-step", type=int, default=None,
                    help="inclusive 1-based upper bound on training step")
    ap.add_argument("--out-prefix", default="logs/tool_audit")
    ap.add_argument("--sample", type=int, default=3)
    args = ap.parse_args()

    rows = []
    by_class_step = defaultdict(Counter)
    by_class_task = defaultdict(Counter)
    samples = defaultdict(list)
    totals = Counter()
    # Schema census. Emitted with the results so that any claim made from this
    # audit can be checked against the fields the data actually carries. This
    # is the artifact that would have caught the OOM/sandbox-loss overclaim
    # immediately instead of after review.
    # Census every level, not just the top one: a first pass looked only at
    # r["rollout_state"] and concluded the field did not exist, when in fact it
    # lives at r["request_info"]["rollout_state"]. A census that does not
    # descend can manufacture exactly the false-absence it exists to prevent.
    schema_keys = Counter()
    ri_keys = Counter()
    rs_keys = Counter()
    info_keys = Counter()
    n = 0

    skipped_out_of_range = 0
    for r in load(args.rollouts_dir):
        step = int(r.get("step", -1)) + 1          # rollout step is 0-based
        if args.max_step is not None and step > args.max_step:
            skipped_out_of_range += 1
            continue
        # count AFTER the range filter: n is the denominator for every class
        # percentage, so counting skipped records here would deflate every rate
        # by the size of the excluded tail.
        n += 1
        task = (r.get("ground_truth") or ["?"])[0]
        reward = r.get("reward")
        ri = r.get("request_info") or {}
        rs = ri.get("rollout_state") or {}
        out = ANSI.sub("", str(ri.get("tool_outputs") or ""))
        err = ANSI.sub("", str(ri.get("tool_errors") or ""))
        blob = out + "\n" + err
        stats = ri.get("tool_call_stats") or []
        ncalls = ri.get("num_calls") or len(stats)
        # TERMINAL-STATE SOURCE, corrected after hamishivi's objection.
        # An earlier version consulted rollout_state.info.oom_killed and
        # .sandbox_lost. A key census over all 19,712 records settles it:
        # request_info.rollout_state is present 19,712/19,712 and its `info`
        # sub-dict is present 19,712/19,712 — but `info` carries exactly two
        # keys, env_name and step_count. oom_killed and sandbox_lost appear in
        # ZERO records. So both branches were dead code, and reporting "0
        # occurrences" from a detector that cannot fire reads as "checked and
        # clean" when it means "never checked". They are removed rather than
        # left sitting at zero. Measured terminal claims here are TIMEOUT-ONLY.
        #
        # rollout_state.timeout (2,419 True) and request_info.timeouts agree
        # exactly on this run, so either is a valid source; ri.timeouts is used
        # because it is the flatter of the two.
        timed_out = bool(ri.get("timeouts"))
        schema_keys.update(r.keys())
        ri_keys.update(ri.keys())
        rs_keys.update(rs.keys())
        _info = rs.get("info")
        if isinstance(_info, dict):
            info_keys.update(_info.keys())

        hits = []
        for name, pat in INFRA_PATTERNS.items():
            if pat.search(blob):
                hits.append(("INFRA", name))
        for name, pat in TEXTUAL_PATTERNS.items():
            if pat.search(blob):
                hits.append(("TEXTUAL", name))
        # (structured sandbox_lost / oom_killed hits removed — see the terminal-
        # state note above. The same-named entries in INFRA_PATTERNS remain and
        # are TEXT matches on the transcript, which is a different and weaker
        # claim, labelled as such in the report.)
        if timed_out:
            hits.append(("INFRA", "marked_timeout"))

        nz = EXIT_NONZERO.findall(blob)
        if nz:
            hits.append(("BASELINE", "nonzero_exit"))
        if TASK_FAIL_PATTERNS.search(blob):
            hits.append(("BASELINE", "failure_text"))

        # exit0-with-failure-text must be evaluated PER CALL, not per transcript.
        # A first pass matched any traceback anywhere against any exit_code=0
        # anywhere and flagged 81.5% of rollouts — meaningless, because an agent
        # doing coding work sees tracebacks constantly and then fixes them. The
        # real silent-corruption signal is failure text inside the SAME command
        # segment that reported success, so segment on the exit markers.
        segs = re.split(r"\(exit_code=(\d+)\)", blob)
        # segs = [text0, code0, text1, code1, ...]; text_i is the output that
        # PRECEDES code_i, i.e. that command's own output.
        for i in range(1, len(segs), 2):
            code = segs[i]
            body = segs[i - 1]
            if code == "0" and TASK_FAIL_PATTERNS.search(body):
                # ignore the benign case where the failure text is the model
                # reading a file / echoing an earlier error rather than a result
                if TESTS_FAILED.search(body) or re.search(
                        r"Traceback \(most recent call last\)", body):
                    hits.append(("SUSPECT", "exit0_with_failure_in_same_call"))
                    break
        if not blob.strip() and ncalls:
            hits.append(("SUSPECT", "empty_output_despite_calls"))
        if ncalls == 0:
            hits.append(("SUSPECT", "zero_tool_calls"))
        # --- reward-semantic mismatches -------------------------------------
        # These MUST compare against the rollout's TERMINAL state. A first pass
        # matched any test-failure or infra string anywhere in a multi-turn
        # transcript against the final reward, and flagged 254 "full reward
        # though tests failed" plus 225 "positive reward after infra failure" —
        # nearly all false: the model had a failing run, fixed it, and passed.
        # Sampling those rows showed "N passed" with reward 1.0, i.e. correct.
        # So look only at the LAST test verdict in the transcript.
        last_pass, last_fail = _verdict_positions(blob)
        final_passed = last_pass is not None and (last_fail is None or last_pass > last_fail)
        final_failed = last_fail is not None and (last_pass is None or last_fail > last_pass)

        if final_passed and isinstance(reward, (int, float)) and reward == 0:
            # hamishivi refinement (3): the graded signal is the HIDDEN verifier
            # (/logs/verifier/reward.txt via test.sh), not the model's own test
            # file. "my tests pass but reward is 0" is the expected shape of a
            # hidden-verifier disagreement, not a scoring bug. Measured on the
            # raw records: 30/60 of these pass-signals came from the model's own
            # test_final_state.py and only 1 mentioned the verifier at all.
            if re.search(r"test_final_state|/tmp/test_", blob):
                hits.append(("EXPECTED", "zero_reward_model_own_tests_passed"))
            else:
                hits.append(("SUSPECT", "zero_reward_though_final_tests_passed"))
        if final_failed and isinstance(reward, (int, float)) and reward == 1:
            hits.append(("SUSPECT", "full_reward_though_final_tests_failed"))
        # Infra-vs-reward: only meaningful when the rollout ENDED on the infra
        # event (terminal), not when it hit a transient and then recovered.
        # Timeout is the only terminal state this schema records, so this is a
        # timeout-vs-reward check and is named accordingly.
        if timed_out and isinstance(reward, (int, float)) and reward > 0:
            hits.append(("SUSPECT", "positive_reward_despite_terminal_timeout"))

        for kind, name in set(hits):
            cls = f"{kind}:{name}"
            totals[cls] += 1
            by_class_step[cls][step] += 1
            by_class_task[cls][task] += 1
            if len(samples[cls]) < args.sample:
                samples[cls].append({
                    "class": cls, "task_id": task, "global_step": step,
                    "prompt_idx": r.get("prompt_idx"), "sample_idx": r.get("sample_idx"),
                    "reward": reward, "num_calls": ncalls,
                    "finish_reason": r.get("finish_reason"),
                    "timeout": timed_out, "done": rs.get("done"),
                    "exit_codes_seen": sorted(set(nz))[:5],
                    "excerpt": blob.strip()[-320:],
                })
        rows.append({
            "global_step": step, "acceptance_step": step, "task_id": task,
            "prompt_idx": r.get("prompt_idx"), "sample_idx": r.get("sample_idx"),
            "reward": reward, "num_calls": ncalls, "timeout": int(timed_out),
            # `done` is genuine: 18,043 True / 1,669 False across the run. It
            # was briefly dropped here on the strength of a census that failed
            # to descend into request_info; restored once measured.
            "done": int(bool(rs.get("done"))),
            "finish_reason": r.get("finish_reason"),
            # ANNOTATION, not a verdict input. Flags rollouts whose transcript
            # uses inverted "passed" vocabulary -- "Evil corpus: 2/2 passed
            # (should reject all)" is syntactically an equal-fraction pass but
            # the harness EXPECTED rejection, so the pass means the task failed.
            # Kept out of the predicate deliberately: a vocabulary rule inside
            # the classifier would catch this corpus's "Evil" and silently miss
            # the next one's "Adversarial", while looking semantics-aware. Here
            # it can be wrong safely, and it pre-highlights those rows for the
            # manual inspection this class already requires.
            # Applied to the DECIDING VERDICT LINE only, not the whole
            # transcript. Scanning the full blob flagged 5,666 rows (11% of the
            # arm) because "evil"/"malicious" appear all over these task
            # corpora, and an annotation that fires on a ninth of everything
            # tells the inspection lane nothing. The question is narrow: does
            # the line that DECIDED the pass use inverted vocabulary?
            "inversion_marker": int(bool(
                _deciding_pass_line(blob) is not None
                and INVERSION_MARKER.search(_deciding_pass_line(blob)))),
            "classes": ";".join(sorted(f"{k}:{v}" for k, v in set(hits))),
        })

    csv_path = f"{args.out_prefix}.csv"
    json_path = f"{args.out_prefix}.json"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump({
        "records": n,
        "max_step": args.max_step,
        "records_skipped_out_of_range": skipped_out_of_range,
        "schema_census": {
            "record_keys": dict(schema_keys.most_common()),
            "request_info_keys": dict(ri_keys.most_common()),
            "rollout_state_keys": dict(rs_keys.most_common()),
            "rollout_state_info_keys": dict(info_keys.most_common()),
            "absent_at_every_level": [
                k for k in ("oom_killed", "sandbox_lost")
                if not (schema_keys.get(k) or ri_keys.get(k)
                        or rs_keys.get(k) or info_keys.get(k))
            ],
        },
        "terminal_state_basis": "request_info.timeouts only. rollout_state and "
                                "rollout_state.info are both present in every "
                                "record, but info carries only env_name and "
                                "step_count: oom_killed and sandbox_lost occur "
                                "in zero records at any nesting level, so no "
                                "OOM or sandbox-loss claim is made here",
        # Every class states HOW it was derived. Without this, a reader sees
        # "INFRA:sandbox_oom 234" next to "no OOM claim is made" and cannot tell
        # that the 234 is a regex hit on transcript text, not a platform-
        # reported OOM. Text matches are evidence of a string, not of an event.
        "class_provenance": {
            **{f"INFRA:{k}": "TEXT_MATCH on tool_outputs+tool_errors"
               for k in INFRA_PATTERNS},
            **{f"TEXTUAL:{k}": "TEXT_MATCH on tool_outputs+tool_errors"
               for k in TEXTUAL_PATTERNS},
            "INFRA:marked_timeout": "STRUCTURED from request_info.timeouts",
            "BASELINE:nonzero_exit": "TEXT_MATCH on (exit_code=N) markers",
            "BASELINE:failure_text": "TEXT_MATCH on task-failure vocabulary",
            "SUSPECT:exit0_with_failure_in_same_call":
                "TEXT_MATCH, segmented per tool call",
            "SUSPECT:zero_tool_calls": "STRUCTURED from request_info.num_calls",
            "SUSPECT:empty_output_despite_calls":
                "STRUCTURED num_calls + empty transcript",
            "SUSPECT:zero_reward_though_final_tests_passed":
                "STRUCTURED reward + TEXT_MATCH terminal verdict",
            "SUSPECT:full_reward_though_final_tests_failed":
                "STRUCTURED reward + TEXT_MATCH terminal verdict",
            "SUSPECT:positive_reward_despite_terminal_timeout":
                "STRUCTURED request_info.timeouts + reward",
            "EXPECTED:zero_reward_model_own_tests_passed":
                "STRUCTURED reward + TEXT_MATCH on model-authored test paths",
        },
        "totals": dict(totals),
        "by_step": {k: dict(v) for k, v in by_class_step.items()},
        "by_task_top20": {k: dict(v.most_common(20)) for k, v in by_class_task.items()},
        "samples": {k: v for k, v in samples.items()},
    }, open(json_path, "w"), indent=1)

    print(f"records audited: {n}"
          + (f"  (max_step={args.max_step}; {skipped_out_of_range} skipped beyond range)"
             if args.max_step is not None else ""))
    print(f"wrote {csv_path} and {json_path}\n")
    print(f"{'class':<48} {'count':>7}  {'%':>6}  distinct_tasks")
    for cls, c in totals.most_common():
        print(f"{cls:<48} {c:>7}  {100*c/n:>5.2f}%  {len(by_class_task[cls])}")
    if not totals:
        print("(no anomalies in any class)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
