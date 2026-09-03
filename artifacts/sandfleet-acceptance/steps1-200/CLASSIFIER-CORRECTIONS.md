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

## Defect 3 — the PASS side, corrected under hamishivi's four-point ruling

Found after Rulin's row-level adjudication of the 87 one-sided
`zero_reward_though_final_tests_passed` rows. Defects 1 and 2 tightened the
FAILURE side; the PASS side was never touched, so a strict FAILED ran against a
loose PASSED and every rollout was biased toward `final_passed`. **The lesson is
the asymmetry, not the individual bug: `0 failed` was fixed on one side only.**

Three rulings, all applied:

1. **strictly positive count** — `0 passed` is not a pass. Rust's
   `test result: ok. 0 passed; 0 failed; ...` means NO tests ran.
2. **digitless decorated banners excluded** — `=== ALL TESTS PASSED ===` is the
   *model's own claim banner*, and the 11096823 audit already adjudicated that
   class as model echo rather than a runner verdict. This was the majority kind
   in the adjudicated rows.
3. **fraction forms excluded** — `=== 2/6 passed ===` is a partial result.

| class | before | after | Δ |
|---|---:|---:|---:|
| `zero_reward_though_final_tests_passed` (DPPO) | 120 | **96** | −24 |
| `zero_reward_though_final_tests_passed` (SGD) | 158 | **104** | −54 |
| `zero_reward_model_own_tests_passed` (DPPO) | 19 | **13** | −6 |
| `zero_reward_model_own_tests_passed` (SGD) | 30 | **19** | −11 |

Every other class is **unchanged**, including both exit0 sub-buckets
(4,772 / 231 and 3,610 / 221) and the class totals 5,003 / 3,831. One SGD row
moved between exit0 *reasons* (`later_pass_supersedes` → `unexplained`) because
its superseding line no longer reads as a pass; buckets unaffected.

**Purely subtractive, measured not argued.** `probes/subtractive_check.py`
compares old and new against the full corpus: **0 lines added** on both arms
(1,101 dropped DPPO, 1,230 SGD). This check earned its place — a first attempt
at the fix used a lookbehind of `[\d/.]` instead of `[\w/.-]`, which silently
**added 198 lines** by matching the `1` in
`test_empty_multiplier_defaults_to_1 PASSED`. Subtractiveness is verified
against data here precisely because the by-construction argument was wrong.

### Convergence with the independent matcher

| | mine | Rulin v3 |
|---|---:|---:|
| zrfp DPPO | 96 | 103 |
| zrfp SGD | **104** | **104** |
| `full_reward_though_final_tests_failed` | 0 / 0 | 0 / 0 |

SGD is now an exact match. DPPO sits 7 below theirs — I have not adjudicated
that residual and am not claiming it is theirs rather than mine.

## Defect 3, corrected again — the ruled JOINT semantics (hamishivi 06:06)

The first pass at defect 3 implemented the **05:42** rule, which excluded every
fraction. That was superseded at **06:06** while the rerun was in flight, and I
committed without re-polling the thread — so `2aae548a7` shipped a predicate and
a matching test suite that pinned the wrong behaviour. hamishivi caught it from
the PROVENANCE prose before the code. The lesson is not the regex: **a
long-running job should re-check its charter at the WRITE, not just the READ.**

Ruled predicate, now implemented:

| form | verdict | why |
|---|---|---|
| `Clean: 50/50 passed` | **pass** | equal fraction, genuine full pass |
| `=== 2/6 passed ===` | **failed** | unequal fraction *under clear summary context* |
| `2/6 passed` (bare) | unclassified | no runner context; do not infer from prose |
| `Clean configs: 50 passed` | pass | label-prefixed positive count |
| `curl: connection to 8443 failed after 3 retries` | no verdict | count not adjacent to the colon |
| `=== ALL TESTS PASSED ===` | no verdict | digitless banner = model echo |
| `Evil configs: 0 passed` | no verdict | zero count |

| class | 05:42 rule | **06:06 ruled** | Rulin job 11186729 |
|---|---:|---:|---:|
| `zero_reward_though_final_tests_passed` DPPO | 96 | **103** | 115 |
| `zero_reward_though_final_tests_passed` SGD | 104 | **122** | 133 |
| `zero_reward_model_own_tests_passed` | 13 / 19 | **12 / 12** | 12 / 13 |
| `exit0_with_failure_in_same_call` | 5,003 / 3,831 | **5,003 / 3,832** | — |

`full_reward_though_final_tests_failed` remains **0 on both arms** — and now it
survives a rule that could have moved it, since unequal fractions under summary
context score as failures. That is worth more than a zero obtained by exclusion.

**Residual against the independent matcher: −12 DPPO, −11 SGD.** A consistent
offset on both arms, so it is systematic rather than noise. It is not
adjudicated and is not claimed to be theirs rather than mine.

Two bugs were caught by measurement before this shipped, not by review:

* the unequal-fraction rule had the **same circularity** just fixed on the pass
  side — `2/6 passed` contains `6 passed`, which satisfies the summary-context
  test, so every bare fraction vouched for its own context. The fraction is now
  stripped before context is evaluated.
* the `inversion_marker` annotation scanned the whole transcript and flagged
  **5,666 rows (11% of the arm)**, because "evil"/"malicious" are everywhere in
  these corpora. It now inspects only the line that DECIDED the verdict: 9 and
  11 rows (0.02%), of which 7 DPPO / 11 SGD are `zrfp` rows.

### `inversion_marker`: annotation, never a verdict input

Some harnesses invert the sense of "passed" — `Evil corpus: 2/2 passed (should
reject all)` is syntactically an equal-fraction pass, but the task EXPECTED
rejection, so the pass means the task failed. I proposed excluding these by
vocabulary. Rulin argued that bakes one corpus's idiom into the canonical
predicate: it catches this dataset's "Evil" and silently misses the next one's
"Adversarial", **while looking semantics-aware**. That argument is better than
mine and the ruling stands as written. The predicate is purely syntactic; the
corpus-specific knowledge lives in a column where it can be wrong safely, and
pre-highlights those rows for the manual inspection this class already needs.

## Final: fraction-scoped label context, and the canonical zrfp figures

Ruled 10:32 after the cross-matcher intersection. **Unequal fractions only**: an
anchored short label immediately followed by `N / M passed` is sufficient
summary context, and whitespace around the slash is tolerated on both sides.

Scoped to fractions **on purpose**. Broadening plain `N failed` into an
arbitrary-label rule is what made the independent matcher score

```
'Connection to localhost:27017 failed: [Errno 111] ...'   <- a PORT read as a count
'Final result: 4 failed downstream nodes'                 <- task-domain quantity
```

as test failures — the exact false-positive class defect 2 removed, re-entering
through a different door. Requiring a *fraction* is what keeps them out: neither
is one. Both are pinned as permanent negative regression cases.

| class | previous | **final** |
|---|---:|---:|
| `zero_reward_though_final_tests_passed` DPPO | 103 | **103** |
| `zero_reward_though_final_tests_passed` SGD | 122 | **120** |
| `zero_reward_model_own_tests_passed` | 12 / 12 | **12 / 13** |
| `exit0_with_failure_in_same_call` | 5,003 / 3,832 | **5,007 / 3,846** |

The two SGD rows that moved are `Results: 10/11 passed` and
`Size 8192: 0/50 passed` — unequal fractions under explicit label context, which
the ruling scores as failures. They were the only ruling-conformance deviations
the intersection found on this side, and they resolve as a *consequence* of the
fraction-label rule rather than needing a separate patch. `exit0` moved because
more lines now register as failures inside exit-0 segments; both arms still
reconcile exactly.

### Canonical figures and the independent ceiling

**Canonical: zrfp 103 (DPPO) / 120 (control).** Rulin's independently-written
matcher reports **115 / 133**, recorded as the independent ceiling.

The entire 12 / 13 gap is named, none of it unexplained:

* the dominant share is the **documented indexed-`Test N failed` difference** —
  this detector accepts an indexed harness verdict, theirs does not, per the
  scope ruling. Where a rollout ends on such a line, this detector flips the
  final verdict to failed and drops the row from `zrfp`;
* 2 rows carry `inversion_marker` (`Evil: 50 passed`) — annotation-lane
  candidates under either matcher;
* 1–2 decorative edges (`--- 8 / 8 passed ---`).

On DPPO the canonical set is a **strict subset** of the independent one: zero
rows flagged here that the other does not flag. The difference is one-directional
and fully attributed.

## What did not change

The curves. Neither overlay consults `TESTS_FAILED`: the rollout series is
built from reward, response length, tool-call counts and
`request_info.timeouts`, and the trainer series comes from W&B. Every
figure in the two PNGs is identical before and after both fixes.

