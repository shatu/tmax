# Classifier corrections applied to this bundle

Two defects were found in `TESTS_FAILED`, the test-failure verdict
detector in `probes/tool_audit.py`. Both are of the same shape -- the
pattern matched text that is not a test verdict -- and both matter beyond
the class they were noticed in, because `_verdict_positions()` is
line-wise and *failure-dominant*: a single bad line sets a whole
rollout's terminal verdict.

**Defect 1 — `0 failed` read as a failure.** `\b(?!0\b)` added.
A clean run reporting `All tests passed! ... 0 failed` scored as a
terminal failure. Present in 0.6% of records.

**Defect 2 — any digits before `failed` read as a count.** Found while
apportioning the exit0 class over the full population. Calibrated against
the whole steps 1-200 corpus, the old pattern fired 31,169 times on DPPO
and **19,727 of those (63%) were not test verdicts**:

```
  11,385  nginx: [emerg] bind() to 0.0.0.0:8080 failed (98: Unknown error)
     578  psql: ... port 5432 failed: FATAL: password authentication failed
     136  Drive 3 failed.
      ..  Attempt 2 failed / User 5678 failed / 1 out of 1 hunk FAILED
      ..  and '8 failed' matched out of the middle of 'UTF-8 failed'
```

A test verdict is now required to be a **count in a test-summary context**
(`N passed`, `in 0.04s`, the `====` banner, ...), a **collection error**, or
an **indexed harness verdict** (`Test 8 failed`). Ports are excluded by the
context requirement rather than by capping digits, so a genuine
`1024 failed, 1 passed in 3s` still matches.

Calibration reported **zero lines newly caught** on either arm: the change
is purely subtractive relative to the old pattern, which is the only shape
of change that cannot invent new findings. Pinned by 25 regression cases in
`probes/patches/test_tests_failed_regex.py`, all drawn from real corpus text.

## production (DPPO) 11149108

| class | as first published | after defect 1 | after defect 2 (this bundle) |
|---|---:|---:|---:|
| `EXPECTED:zero_reward_model_own_tests_passed` | 17 | 19 | **19** |
| `SUSPECT:exit0_with_failure_in_same_call` | 5907 | 5693 | **5003** |
| `SUSPECT:zero_reward_though_final_tests_passed` | 88 | 100 | **120** |

Unchanged by either fix (9 classes): `BASELINE:failure_text`, `BASELINE:nonzero_exit`, `INFRA:marked_timeout`, `INFRA:oci_fallback`, `INFRA:rollout_walltime`, `INFRA:sandbox_oom`, `INFRA:transport_error`, `SUSPECT:zero_tool_calls`, `TEXTUAL:tool_call_timeout_textual`.

## control (SGD) 11151208

| class | as first published | after defect 1 | after defect 2 (this bundle) |
|---|---:|---:|---:|
| `EXPECTED:zero_reward_model_own_tests_passed` | 23 | 30 | **30** |
| `SUSPECT:exit0_with_failure_in_same_call` | 4543 | 4276 | **3831** |
| `SUSPECT:zero_reward_though_final_tests_passed` | 126 | 156 | **158** |

Unchanged by either fix (8 classes): `BASELINE:failure_text`, `BASELINE:nonzero_exit`, `INFRA:marked_timeout`, `INFRA:rollout_walltime`, `INFRA:sandbox_oom`, `INFRA:transport_error`, `SUSPECT:zero_tool_calls`, `TEXTUAL:tool_call_timeout_textual`.

## `exit0_with_failure_in_same_call`, apportioned

Full population, not a sample: which sub-branch fired is a deterministic
property of the segment text. `probes/exit0_adjudicate.py` imports the
classifier's own regexes and mirrors its break-on-first-triggering-segment
keying, so these sub-buckets partition the class count exactly. Both arms
print `RECONCILE OK`.

| arm | class total | `traceback_in_exit0` | `tests_failed_in_exit0` |
|---|---:|---:|---:|
| production (DPPO) 11149108 | 5003 | 4772 (95.4%) | 231 (4.6%) |
| control (SGD) 11151208 | 3831 | 3610 (94.2%) | 221 (5.8%) |

The majority is a plain Python traceback inside an exit-0 shell segment:
the model runs a snippet, it raises, the wrapping shell exits 0, and the
model reads the error and fixes it. That is debugging, not a wrapper
reporting success over a failure. The class name oversells it.

Per-reason breakdown, both arms:

**production (DPPO) 11149108**

| reason | n | % of class |
|---|---:|---:|
| `traceback_only_no_test_failure` | 4772 | 95.4% |
| `unexplained_exit0_with_test_failure` | 162 | 3.2% |
| `pytest_collection_error` | 50 | 1.0% |
| `unexplained_exit0_with_test_failure_with_traceback` | 16 | 0.3% |
| `later_pass_supersedes_in_same_segment` | 3 | 0.1% |

**control (SGD) 11151208**

| reason | n | % of class |
|---|---:|---:|
| `traceback_only_no_test_failure` | 3610 | 94.2% |
| `unexplained_exit0_with_test_failure` | 132 | 3.4% |
| `pytest_collection_error` | 74 | 1.9% |
| `unexplained_exit0_with_test_failure_with_traceback` | 8 | 0.2% |
| `later_pass_supersedes_in_same_segment` | 7 | 0.2% |

## What did not change

The curves. Neither overlay consults `TESTS_FAILED`: the rollout series is
built from reward, response length, tool-call counts and
`request_info.timeouts`, and the trainer series comes from W&B. Every
figure in the two PNGs is identical before and after both fixes.

