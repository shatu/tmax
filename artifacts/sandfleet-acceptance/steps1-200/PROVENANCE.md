# Sandfleet acceptance — steps 1–200, immutable bundle

Two 8-node arms differing in **one** thing: the optimizer path. Everything else
— data, seed, image pool, Sandfleet build, node exclusions, rollout shape — is
shared. This directory is the complete evidence for optimizer steps 1–200 and
is not amended in place; a correction lands as a new commit, not an edit.

## Runs

| | production (DPPO) | control (SGD) |
|---|---|---|
| Slurm job | `11149108` | `11151208` |
| W&B run id | `cfb5az1a` | `o55y4lhh` |
| W&B run name | `production-dppo` | `sgd-control` |
| run dir | `oscar-accept8x32-sandfleet__seed42__1788324996` | `oscar-sgd8x32-sandfleet__seed42__1788330989` |
| nodes | 8 | 8 |
| started (UTC) | 2026-09-02T04:59:45 | 2026-09-02T06:38:16 |
| rollout shape | 8 × 32 = 256 episodes/step | 8 × 32 = 256 episodes/step |
| seed | 42 | 42 |

Both arms were still running past step 200 when this bundle was cut. **Nothing
beyond optimizer step 200 is included** — no partial step 201.

## Code under test

| component | SHA |
|---|---|
| TMAX training tree | `a64ea5c4e0507721d1071badec0709cabb09a987` |
| Sandfleet | `86973b4c88ac3157a18b654da0dbec56b2710c3c` (0.5.4) |
| branch this bundle lands on | `geomean_mask` |

## Coverage, verified rather than assumed

Raw rollout shards were staged to an immutable snapshot before analysis,
because analysing a directory the trainer is still writing to can silently
yield a partial step. Full parse of the snapshot:

```
dppo-11149108   61,440 records, unparseable 0, step range 1..240
   steps 1-200 all present: True    records within 1-200: 51,200 = 200 x 256
sgd-11151208    57,088 records, unparseable 0, step range 1..223
   steps 1-200 all present: True    records within 1-200: 51,200 = 200 x 256
steps with != 256 samples: 0 (both arms)
```

W&B history was cross-checked keyed against unkeyed `scan_history()`: 200 = 200
on both arms, no missing steps. That check exists because an unkeyed scan had
previously under-returned without erroring.

## What is in here

| file | what |
|---|---|
| `acceptance-steps1-200-series.{csv,json}` | rollout-derived overlay, one row per (arm, optimizer step) |
| `wandb-steps1-200-series.csv`, `wandb-steps1-200.json` | trainer-side overlay incl. all five `debug/*` series |
| `summary-steps1-200.{csv,json}` | Q1–Q4 and 1–100 vs 101–200 windows, both arms, both overlays |
| `throughput-steps1-200.json` | per-step wall time from the trainer's own `time/total` |
| `tool_audit-*-steps1-200.json` | per-class totals, schema census, class provenance |
| `tool_audit-*-steps1-200.csv.gz` | per-rollout classification, 51,200 rows/arm |
| `exit0-*.{json,csv}` | full-population apportionment of `exit0_with_failure_in_same_call` |
| `lostconn-*.{json,csv}` | `lost connection to sandbox` rows reconciled into incidents |
| `CLASSIFIER-CORRECTIONS.md` | both `TESTS_FAILED` defects, before/after per class per arm |
| `TIMEOUT-PARTITION.md` | six-phase timeout partition and the measured per-tool ceiling |
| `LOST-CONNECTION.md` | the 2-vs-16 reconciliation, rows vs incidents |
| `acceptance-steps1-200-panels.png`, `wandb-steps1-200-panels.png` | the curves |
| `SHA256SUMS` | checksums for every file above |

Per-rollout CSVs are gzipped (4.7 MB → 0.3 MB each) to respect the lean-clone
policy set in `c8a77cf37`.

## Keying and alignment

Rows are keyed on the **optimizer step**. The rollout overlay carries both
`global_step` and `acceptance_step`; over 1–200 they coincide, and both are
emitted so the join is checkable rather than assumed. The trainer overlay is
keyed on the run's own `training_step`.

**No interpolation and no forward-fill.** A step absent from a series is
reported as missing, never filled. This is why some `n` values are 199 rather
than 200: `time/weight_sync` has no value before the first optimizer step. All
five `debug/*` series have full 200/200 coverage on both arms.

## Caveats that constrain what these numbers support

**`post_filter` ≡ `pre_filter`.** `val/avg_group_performance_post_filter` and
`..._pre_filter` are pointwise identical across all 200 steps on both arms.
Rulin traced this to identical call-site arguments in `data_loader.py`, so the
"post-filter" series carries no filtering information. Do not read a
filter-effect from it. The fix is on the hardening ledger, not applied here.

**Text-matched classes are evidence of a string, not of an event.** Every class
in the audit states its derivation in `class_provenance`. `INFRA:sandbox_oom`
is a regex hit on transcript text; the only structurally-derived terminal state
this schema carries is `request_info.timeouts`. `oom_killed` and
`sandbox_lost` appear in **zero** records at any nesting level, so no
OOM-kill or sandbox-loss claim is made from this data.

**`INFRA:oci_fallback` is a task artifact.** It matches a mock apptainer shim
belonging to one task family, is 100% false as an infrastructure signal, and is
absent entirely from the SGD arm.

**`INFRA:transport_error` is mostly task-internal networking.** Breakdown:
1,266 localhost "connection refused" from the task's own services, 296
in-sandbox download resets, **2** genuine "lost connection to sandbox", and
**0** "Could not reach Sandfleet endpoint".

**Classifier corrections applied.** Two defects were found in the
test-failure detector and both are fixed here; see `CLASSIFIER-CORRECTIONS.md`
for the before/after table. Both corrections are subtractive — they remove
false failures — and neither touches the curves, which never consult it.

**A known limitation, left in on purpose.** A bare `FAILED tests/x.py::test_y`
with no count is recognised by neither the old detector nor the new one. Adding
it would be a scope expansion rather than a bug fix, and hamishivi ruled it out
of this acceptance pass; it is recorded here as an open classifier limitation.

**The banner-vs-verdict rule.** A *decorated digitless banner* — `=== ALL TESTS
PASSED ===`, `=== All checks passed! ===` — is **model echo, not a test-runner
verdict**, and does not establish a terminal pass. This follows the precedent
already set in the 11096823 audit, where that class was adjudicated as the
model's own claim rather than a runner's. It was the majority kind among the
adjudicated false passes. A terminal pass requires a strictly positive,
non-fractional count from a runner.

**Fraction exclusion is conservative and NOT lossless.** `2/6 passed` is
correctly rejected as a partial result, but the same rule also rejects
numerator==denominator forms such as `Random tests: 100/100 passed`, which is
arguably a genuine full pass. Both this matcher and Rulin's independent one
share the exclusion, so the two agree — but a reader should not take it as
exact. Named here rather than left to be discovered.

**One figure of mine was wrong and is corrected here.** I previously reported
**2** `lost connection to sandbox` rollouts. For this window the count is **16**
(DPPO) and **20** (SGD), matching Rulin's independent scan; the earlier figure
came from the 11096823 corpus and does not transfer. See `LOST-CONNECTION.md`.

**Independent verification.** Rulin recomputed the headline classes blind, with
a separately-written classifier, and every structural class matched exactly:
`marked_timeout` 5,629 / 5,589, `zero_tool_calls` 600 / 1,194, `sandbox_oom`
272 / 576, `transport_error` 1,502 / 1,195, `rollout_walltime` 840 / 1,390.
Their shard `sha256sum -c` passed 13/13.

## Reproducing

The raw shards are staged world-readable on fair-sc at
`exchange/oscar_raw_steps1-200/` with their own `SHA256SUMS`, so the whole
bundle can be regenerated byte-for-byte from the corpus rather than trusted:

```
python3 probes/tool_audit.py <shard_dir> --max-step 200 --out-prefix <out>
python3 probes/exit0_adjudicate.py <shard_dir> --max-step 200 --tag <tag> \
        --class-total <from the audit> --out-prefix <out>
python3 probes/patches/test_tests_failed_regex.py   # 25 cases, must be OK
```

## Not included, deliberately

No internal URLs, hostnames, or credentials. Two internal identifiers present
in the source artifacts — the W&B instance `base_url` and one compute-node FQDN
that appeared inside a sampled transcript — are replaced with
`<internal-wandb-instance>` / `<internal-compute-node>`. Redaction is
structural (applied to parsed JSON values, then re-serialised) so it cannot
corrupt escaping.

The corpus does contain strings such as `X-Secret-Token:` and `FLAG{...}`.
Those are **synthetic content from a privilege-escalation training task** —
part of the evidence, not secrets — and are deliberately left intact.
