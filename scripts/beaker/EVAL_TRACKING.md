# Terminal-bench eval tracking — keeping the results sheet up to date

One self-contained sheet, one tool. No intermediate per-benchmark sheets.

- **Sheet:** [`terminalbench_combined_evals.csv`](./terminalbench_combined_evals.csv) — one row per
  model/checkpoint, with **TB2.1 and TBlite side by side**. Each row stores its own eval-experiment
  URLs (`tb21_beaker_url`, `tblite_beaker_url`), so scores are (re)extracted straight from Beaker.
- **Tool:** [`combined_evals.py`](./combined_evals.py) — `add` and `refresh`.
- **Launch commands** (how the eval + training jobs themselves are run): see
  [`eval_runs_2026-07-08.md`](./eval_runs_2026-07-08.md).

> Run the tool with the **open-instruct uv env** (it's the fully-provisioned base):
> ```bash
> cd /weka/nora-default/shashankg/code/open-instruct && \
>   uv run python /weka/nora-default/shashankg/code/tmax/scripts/beaker/combined_evals.py <cmd>
> ```

## The two operations

**1. A new model/checkpoint was evaluated → add a row** (auto-extracts scores):
```bash
combined_evals.py add \
  --model <NAME> [--step N] \
  --max-len <32768|65536> --workspace ai2/oe-agents \
  --tb21   <TB2.1_eval_experiment_id> \
  --tblite <TBlite_eval_experiment_id> \
  # train metrics — RL/SFT checkpoints only (carried as-is, not re-extracted):
  [--wandb <wandb_url> --train-exp <train_beaker_id> --grp-w5 X --kl2 X --seq-len N]
```
Pass only the block(s) you have — e.g. TB2.1 now, add `--tblite` later by re-running `add` for the
same `--model`/`--step` (it updates in place). Rows are re-sorted into the canonical order on save.

**2. Evals were still running when added → fill them in later:**
```bash
combined_evals.py refresh            # fills any row that has a URL but no pass@1 yet
combined_evals.py refresh --force    # re-extract every row (idempotent)
```

## End-to-end for a fresh checkpoint
1. **Convert** (open-instruct RL/SFT Qwen3.5 checkpoints save as `Qwen3_5ForCausalLM`, which vLLM
   won't serve) → CG-convert with `convert_qwen35_causallm_to_cg.py` first (SFT epoch checkpoints:
   `convert_sft_epoch_checkpoints.sh` → then CG). Qwen3/base models need no conversion.
2. **Launch** the eval(s) via `../beaker_configs/launch_eval.sh` — see `eval_runs_2026-07-08.md` for the
   exact per-family flags. TB2.1 = `--dataset-path .../terminal-bench-2-1` (89 tasks); TBlite =
   `--dataset openthoughts-tblite@2.0` (100 tasks). Verify the registry **mirror is live** first.
3. **Track**: `combined_evals.py add ... --tb21 <exp> --tblite <exp>` → then `refresh` once they finish.

## Row order (applied automatically on every save)
raw Qwen (Qwen3 → Qwen3.5) · published Tmax SFT · local SFT (small → big; Qwen3 → Qwen3.5) ·
published Tmax RL · RL checkpoints by step.

## Columns
`model_name, step, max_model_len` ·
`tb21_{pass@1, pass@5, pass1_adj, err_rate, beaker_url}` ·
`tblite_{pass@1, pass@5, pass1_adj, err_rate, beaker_url}` ·
`train_grp_perf_w5, train_kl2, train_seq_len, wandb_url, train_beaker_url` · `workspace`.

## How scores/errors are computed
From each experiment's scoring-job log: `pass@1`, `pass@5`, the exception-count table, and `X/Y trials`.
- **Error split:** `AgentTimeoutError` = the model's own 120s bash timeout (real failure); everything
  else (`RuntimeError` = vLLM/connection crash, `Verifier*`, `RewardFileNotFound`, …) = **infra**.
- `err_rate = total_errors / n_trials` (TB2.1 = 89 tasks × k, TBlite = 100 × k; read from the log).
- `pass1_adj` = pass@1 with infra-failed trials removed from the denominator — the fair cross-run
  number when infra error rates differ. Blank when a log has no exception table (can't split infra).

## Notes
- Eval **names are unreliable** (a TB2.1 run can be named `...terminal-bench-2-0`) — identity is the
  launch config, not the name. Always pass the real experiment IDs.
- The old per-benchmark sheets + builders (`track_evals.py`, `build_tblite_sheet.py`,
  `build_combined.py`, `dppo9b_4n64k_tb21_evals.csv`, `sft_evals.csv`, `tblite_evals.csv`) were
  **retired** — this sheet + tool supersede them. Don't recreate them.


## Backend reliability / reproducibility study (2026-09-22)

Question: does the OpenSandbox backend (`--harbor-env opensandbox`) give more reliable evals than
in-job podman? Protocol: same model, dataset, agent and flags, two runs per backend, launched the same
night on ai2/jupiter; compare error rate by cause, `pass1_adj`, and run-to-run per-task agreement with
`scripts/analysis/compare_backend_runs.py` (takes job dirs or experiment IDs and fetches results).

Model `hamishivi/Qwen3.5-9B`, TB 2.1 (`--dataset-path .../terminal-bench-2-1`), Vanillux2Agent,
`openai` provider, `qwen3_xml`, 64k context, k=5, n-concurrent 8, 1 GPU.

| backend | run | experiment | notes |
|---|---|---|---|
| opensandbox | 1 | https://beaker.org/ex/01M35MBSCBGB0P2EZ37XNPFBD7 | commit cfe46a65 (before the exec-output-dir hardening and mirror default) |
| opensandbox | 2 | https://beaker.org/ex/01M35V7TMRB9V7FNKRZEPB6CX9 | commit 46595e3a |
| podman | 1 | https://beaker.org/ex/01M35VAB8NA18B2P8F22RKRNQJ | `--keep-task-images`, no Docker Hub mirror (registry-mirror workload was down) |
| podman | 2 | https://beaker.org/ex/01M35VAF0ZZ53531S1QDW8DQQ0 | same |

```bash
uv run python scripts/analysis/compare_backend_runs.py \
  --run opensandbox=01M35MBSCBGB0P2EZ37XNPFBD7 --run opensandbox=01M35V7TMRB9V7FNKRZEPB6CX9 \
  --run podman=01M35VAB8NA18B2P8F22RKRNQJ --run podman=01M35VAF0ZZ53531S1QDW8DQQ0 \
  --json backend_repro.json
```

Reading the output: `infra` errors (env start, sandbox died, verifier, connection) are the backend's
fault; `timeout` (the agent's 120 s command limit) is mostly the model's, **but on OpenSandbox part of
it is infra-induced** — sandbox nodes give ~1–2 effective cores vs the unthrottled GPU node under
podman, so compute-heavy commands trip the limit more often (see docs/running_evals.md §3b). Compare
`pass1_adj` and the per-task pass-count agreement across the two runs of each backend; lower spread
and a higher error-task Jaccard (errors repeating on the same tasks rather than random ones) mean a
more reliable backend.
