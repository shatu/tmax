# Sandfleet acceptance — steps 1–400 curves snapshot

**Curves only.** Requested by hamishivi as an artifact-only snapshot of the two
live arms, using the same metric definitions, keyed-history checks, alignment
rules and panel layout as the accepted steps 1–200 bundle. This is *not* a
second rollout-classifier audit and makes no classifier claims.

## Runs

| | production (DPPO) | control (SGD) |
|---|---|---|
| Slurm job at cut time | `11149108` | `11151208` |
| W&B run id | `cfb5az1a` | `o55y4lhh` |
| W&B run name | `production-dppo` | `sgd-control` |
| nodes | 8 | 8 |
| rollout shape | 8 × 32 = 256 episodes/step | 8 × 32 = 256 episodes/step |
| seed | 42 | 42 |

Both arms were well past step 400 when this was cut (production at 578, control
at 574), so the window is closed and immutable. **Exactly steps 1–400; no
partial 401.**

Production `11149108` later failed at step 647 and was resumed as `11219522`
from `global_step641`. That is after this window and does not affect anything
here — the W&B run id `cfb5az1a` is unchanged by the resume.

## Code under test

| component | SHA |
|---|---|
| TMAX training tree | `a64ea5c4e0507721d1071badec0709cabb09a987` |
| Sandfleet | `86973b4c88ac3157a18b654da0dbec56b2710c3c` (0.5.4) |

## Coverage, verified rather than assumed

```
production-dppo  cfb5az1a   steps 1..400   400 logged   96 metric keys
sgd-control      o55y4lhh   steps 1..400   400 logged   96 metric keys
keyed scan == unkeyed scan == 400 on both;  steps missing from unkeyed scan: []
all 11 plotted series: 0 missing points, both arms
```

The keyed-vs-unkeyed cross-check exists because an unkeyed `scan_history()` had
previously under-returned without erroring.

## Contents

| file | what |
|---|---|
| `acceptance-steps1-400-panels.png` | rollout-derived curves, both arms |
| `acceptance-steps1-400-series.{csv,json}` | rollout-derived overlay, one row per (arm, optimizer step) |
| `wandb-steps1-400-panels.png` | trainer-side curves, both arms |
| `wandb-steps1-400-series.csv` | trainer-side tidy series |
| `wandb-steps1-400.json` | full trainer-side export incl. per-key missing-point counts |
| `summary-steps1-400.{csv,json}` | Q1–Q4 (100-step quarters) and 1–200 vs 201–400, both overlays |
| `metric-inventory-steps1-400.json` | PRESENT/ABSENT per requested field, plus sparse keys |
| `SHA256SUMS` | checksums for every file above |

## Keying and alignment

Rows are keyed on the **optimizer step**. The rollout overlay carries both
`global_step` and `acceptance_step`. The trainer overlay is keyed on the run's
own `training_step`.

**No interpolation and no forward-fill.** A step absent from a series is
reported as missing, never filled.

## Metric inventory: nothing requested is absent

Every field hamishivi named is PRESENT on both runs — reward /
`post_filter`-labelled series, response length, tool-call count, the trainer
logprob metrics, and **all five `debug/*` series available in both runs**:

```
debug/dppo_mask_frac_kept          debug/vllm_local_reverse_kl
debug/vllm_vs_local_logprob_diff_mean / _max / _std
```

**Sparse, not absent.** Two infrastructure series and the weight-sync family
are logged on a subset of steps and their gaps are reported rather than filled:

| key | production | control |
|---|---:|---:|
| `env/swerl_vanillux_sandbox/infrastructure_failure` | 202 steps carry no value | 272 |
| `env/swerl_vanillux_sandbox/sandbox_lost` | 202 | 272 |
| `time/weight_sync{,_min,_max,_mean,_median}` | 1 | 1 |

`time/weight_sync` is absent for exactly one step on each arm because there is
no weight sync before the first optimizer step.

## Reward, by quarter

| arm | Q1 (1–100) | Q2 (101–200) | Q3 (201–300) | Q4 (301–400) |
|---|---:|---:|---:|---:|
| production (DPPO) | 0.6291 | 0.6502 | 0.6595 | **0.6613** |
| control (SGD) | 0.6243 | 0.6345 | 0.6447 | **0.6476** |

Monotone in both arms across all four quarters, production consistently above
control. Halves: production 0.6397 → 0.6604, control 0.6294 → 0.6461.

**This is a description of two curves, not a claim about the optimizer.** The
arms differ in the optimizer path by construction, but nothing here controls
for run-to-run variance, and a single pair of runs cannot separate an optimizer
effect from seed or scheduling noise.

## Caveat carried forward from the 1–200 bundle

**`post_filter` ≡ `pre_filter`.** `val/avg_group_performance_post_filter` and
`..._pre_filter` remain pointwise identical across all 400 steps on both arms.
Rulin traced this to identical call-site arguments in `data_loader.py`, so the
"post-filter" series carries no filtering information and a filter effect must
not be read from it. Unchanged from the 1–200 window and still unfixed.

## Not included

No internal URLs, hostnames, or credentials. The W&B instance `base_url` is
replaced with `<internal-wandb-instance>`; redaction is structural, applied to
parsed JSON values and re-serialised, so it cannot corrupt escaping.
