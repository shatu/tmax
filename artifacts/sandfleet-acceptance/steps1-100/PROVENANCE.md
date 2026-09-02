# Sandfleet acceptance — steps 1–100 panels

Two PNG exports, copied verbatim (not regenerated) from the checksummed
acceptance bundle, so they can be rendered directly from this repository by
reviewers without cluster or internal-dashboard access.

## Files

| file | what it is |
|---|---|
| `acceptance-steps1-100-panels.png` | **Rollout-derived.** Six panels computed only from banked rollout records: mean reward, zero-reward fraction, mean and median response length, mean tool calls per rollout, timeout fraction. Every point is recomputable from the reward CSVs. |
| `wandb-steps1-100-panels.png` | **Trainer-side.** Nine panels of the trainer's own logged metrics under their exact keys, including `val/avg_group_performance_post_filter`, `val/avg_group_performance_pre_filter`, `scores`, `val/sequence_lengths`, `tools/aggregate/*` and three `debug/*` series. |

The two are deliberately kept separate: trainer-side length and tool-call
series are **not** renamed into equivalence with the rollout-reconstructed ones,
because they are measured over different populations.

## Runs

| arm | trainer job | experiment run id | steps |
|---|---|---|---|
| production dppo | `11149108` | `cfb5az1a` | 1–100 |
| sgd control | `11151208` | `o55y4lhh` | 1–100 |

Project path `oscaryinn/open-instruct-terminal-rl`. Both arms are fresh
lineages with no resume, so `global_step == acceptance_step` throughout.

Stack: TMAX `a64ea5c4e0507721d1071badec0709cabb09a987` (this branch),
Sandfleet `86973b4c88ac3157a18b654da0dbec56b2710c3c`. 8×32 rollout shape,
`TOTAL_EPISODES=256000` → 1,000 optimizer steps; these panels cover the first
100. Both arms delivered a full 256 samples on every one of those steps, with
zero degenerate (zero-variance) prompt groups across all 800 groups.

## Source checksums

Identical to the exchange bundle they were copied from — a byte-for-byte copy,
verifiable in both places:

```
350dfdfe920bb04e3d99c2f2d18e4c07bcb795cd025394609f762473f73d0353  acceptance-steps1-100-panels.png
c193721482497f742fb501aa4c2ba71974acad8eaaac74c5955767417a078fd0  wandb-steps1-100-panels.png
```

`SHA256SUMS` in this directory repeats them for local verification.

## Caveats carried with these figures

- The trainer-side pull applied **no forward-fill, no interpolation and no
  extrapolation**; all nine plotted keys had zero missing points across both
  runs over steps 1–100.
- `val/avg_group_performance_post_filter` is **pointwise identical** to
  `..._pre_filter` on both arms (0 of 100 steps differ) while filtering is
  demonstrably active. It should not be read as an independent signal. Traced
  to identical arguments at the two call sites in `data_loader.py`; a
  metrics-only fix is on the post-run hardening list.
- The quartile drift visible in the reward panels is **not** claimed as
  learning: it sits inside the per-step standard deviation at this depth.
