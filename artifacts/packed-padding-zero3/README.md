# Distributed padding and scaling validation

Tillicum H200 allocation **287376** completed `0:0` in 48 seconds. All 12 cases passed:

- six GPUs, DP=6 / SP=1;
- eight GPUs, DP=2 / SP=4;
- 1, 7, and 13 real packs with token and sequence loss normalization;
- unequal valid-token counts, masked-only ranks, accumulation, real NCCL collectives, and DeepSpeed ZeRO-3 gradient reduction.

The maximum absolute gradient difference against an unsharded dense reference was 7.46e-9. The runtime was PyTorch 2.10.0+cu128 and DeepSpeed 0.18.4. Each case finishes a zero-learning-rate optimizer step to clear partitioned accumulation buffers without changing the comparison weights.

Negative control **287377** restored the pre-fix trainer scaling statements (from PR parent `a1efba4f`) while retaining the corrected harness and padding. All six SP=4 comparisons failed gradient equivalence as expected; the probe explicitly expected mismatches, so the job completed `0:0` in 28 seconds. This is evidence the comparison detects incorrect scaling, not just successful backward execution.

## Scope

`training/open-instruct/tests/probe_padding_zero3.py` uses production pack collation, tiled DPPO kernels, and the exact scaling statements extracted from `PolicyTrainerRayProcess._compute_tiled_dapo_loss`. It uses a small token-local model to isolate reduction arithmetic. The SP process groups and ZeRO-3 collectives are real. Attention all-to-all, the Qwen GatedDeltaNet kernels, Ray placement, and a complete 24-learner/40-inference training run are **not** covered. The dense loss multiplier also has CPU regression coverage; the GPU probe exercises the tiled path.

The source files used for the successful GPU run and the committed trainer/loss files were checked to have identical Python ASTs (only formatting differs). `source-sha256.json` records committed production file hashes. Summary JSON retains each parameter's maximum gradient error and norm ratio.

Two earlier probe-development jobs are not passing evidence: 287374 called the SP initializer with SP=1 (rejected before loss execution); 287375 failed to finish optimizer steps between cases, leaving ZeRO accumulation buffers uncleared. Both are terminal. The corrected probe calls `engine.step()` with LR=0 after each case. All four owned allocations finished; existing user jobs were untouched.

## Reproduce

Use a compatible CUDA/PyTorch/DeepSpeed environment with pytest and NumPy, from `training/open-instruct`:

```sh
PROBE_SP=1 torchrun --standalone --nproc-per-node=6 tests/probe_padding_zero3.py
PROBE_SP=4 torchrun --standalone --nproc-per-node=8 tests/probe_padding_zero3.py
```

For the negative control, write the old `grpo_fast.py` to a separate path and set `PROBE_SCALE_SOURCE` to that file plus `PROBE_EXPECT_MISMATCH=1`. Never replace the tested working tree's production file in place while jobs are running.
