# Capped DPPO integration validation

Tillicum H200 job 287385 completed 0:0 in 50 seconds. All 12 NCCL/ZeRO-3 comparisons passed: DP6/SP1 and DP2/SP4; 1, 7 and 13 packs; token and sequence normalization; uneven masks and padding-only ranks. Maximum gradient error against an independent dense capped reference: 5.96e-8. Runtime: PyTorch 2.10.0+cu128, DeepSpeed 0.18.4.

The updated `training/open-instruct/tests/probe_padding_zero3.py` alternates rollout logprobs -6 and -1.5, exercising ratios above and below cap=10. Its reference explicitly implements the detached capped coefficient. Run it with `PROBE_SP=1 torchrun --standalone --nproc-per-node=6 tests/probe_padding_zero3.py` and SP=4/nproc=8, from `training/open-instruct`, in the compatible GPU environment. Earlier PR14 receipts describe the earlier uncapped probe at that commit.

This tiny token-local model isolates loss/reduction arithmetic; it does not validate full Qwen attention, Ray placement, Sandfleet, or the production 24/40 training setup. No production training launched.

CPU: 92 tests plus 2 subtests passed across test_dppo_cap_integration, test_grpo_sp_loss_scaling, test_packed_batch_padding, test_reward_packing_loss, test_reward_filtering and test_eval_reward_validity. Changed helpers/tests pass Ruff; grpo_fast retains its 13 pre-existing diagnostics.
