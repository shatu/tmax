# PREVIEW — steps 1–200 curves (NOT the final acceptance bundle)

Published early at hamishivi's request so the step-200 curves can be viewed
without waiting on the audit correction. **This directory is a preview.** The
immutable acceptance bundle lands separately and atomically under
`steps1-200/` once the semantic audit is re-run and re-inspected.

## Why these are safe to publish now

The curves are computed **only** from rollout record fields — reward,
`response_tokens` length, `request_info.num_calls`, `request_info.timeouts` —
and the trainer's own W&B series. None of them touch `TESTS_FAILED` or
`_verdict_positions`, which is where the `"0 failed"` classifier defect lives.
That defect affects three *audit* classes and no curve on either panel.

## Files

| file | contents |
|---|---|
| `acceptance-steps1-200-panels.png` | rollout-derived, both arms overlaid, marker at step 100: mean reward, zero-reward fraction, mean and median response length, mean tool calls per rollout, timeout fraction |
| `wandb-steps1-200-panels.png` | trainer-side under exact logged keys: `val/avg_group_performance_post_filter`, `..._pre_filter`, `scores`, `val/sequence_lengths`, `tools/aggregate/avg_calls_per_rollout`, `tools/aggregate/failure_rate`, and **all five** `debug/*` series |

## Runs

| arm | trainer job | run id | steps |
|---|---|---|---|
| production dppo | `11149108` | `cfb5az1a` | 1–200 |
| sgd control | `11151208` | `o55y4lhh` | 1–200 |

TMAX `a64ea5c4e0507721d1071badec0709cabb09a987`, Sandfleet
`86973b4c88ac3157a18b654da0dbec56b2710c3c`, 8×32, fresh lineages so
`global_step == acceptance_step`. No forward-fill, no interpolation; all 11
trainer keys had zero missing points on both arms. Step 201+ excluded.

## Caveat carried with the figures

`val/avg_group_performance_post_filter` is **pointwise identical** to
`..._pre_filter` on both arms — a known trainer metrics bug (identical
arguments at the two call sites in `data_loader.py`), not a property of the
run. Do not read `post_filter` as an independent signal.
