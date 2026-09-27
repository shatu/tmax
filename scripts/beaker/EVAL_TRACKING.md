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

### Results (all four runs complete, 2026-09-23; full JSON in `backend_repro_2026-09-23.json`)

| backend | run | errors | infra | cmd timeout | pass@1 | pass@1 adj | pass@5 | env start med/p90 | job |
|---|---|---|---|---|---|---|---|---|---|
| opensandbox | 1 | 27.4% | 4.7% | 22.7% | 18.7% | 19.6% | 28.1% | 12 s / 45 s | 6.4 h |
| opensandbox | 2 | 29.4% | 5.4% | 23.8% | 16.2% | 17.1% | 25.8% | 10 s / 41 s | 6.2 h |
| podman | 1 | 15.1% | 0.4% | 14.6% | 18.2% | 18.3% | 29.2% | 2 s / 5 s | 6.5 h |
| podman | 2 | 18.0% | 0.4% | 17.5% | 20.0% | 20.1% | 34.8% | 2 s / 8 s | 6.0 h |
| podman (July, Qwen/Qwen3.5-9B) | — | 16.2% | 0.4% | 15.7% | 19.1% | 19.2% | 36.0% | 1 s / 5 s | 6.6 h |

Run-to-run (2 runs each): pass@1 sd 1.2 pts (opensandbox) vs 0.9 pts (podman); tasks with identical
pass count out of 5: 78.7% vs 76.4%; solved-set Jaccard 0.71 vs 0.73; error-task Jaccard **0.81 vs
0.45** — OpenSandbox's errors repeat on the same tasks (systematic: compute-heavy tasks + three images
that never pull within 600 s), podman's land on different tasks (flaky timeouts).

OpenSandbox infra errors (~5%/run), root-caused afterwards (2026-09-25):
- 15/run env-start timeouts on `caffe-cifar-10`, `mcmc-sampling-stan`, `rstan-to-pystan`: **not** image
  pulls (images are 50–240 MB) — those tasks declare `cpus = 4`, and a 4-vCPU pod never schedules on
  sandbox-standard's 4-vCPU node pool (control plane `POD_READY_TIMEOUT` after 180 s, HTTP 504).
  Fixed: the env caps the CPU request at 2 (`max_cpus`); requests aren't enforced as limits there anyway.
- 4–7/run "peer closed connection" + 1 proxy parse error: the server proxy cuts every exec stream at
  **300 s** (measured: silent and heartbeat commands alike die at 300 s). All three traced cases were
  verifiers 304 s in. Fixed: commands now run detached and are polled (see docs/running_evals.md §3b).
- 1/run sandbox died + 1/run verifier download failed, three of four on `bn-fit-modify` (memory-hungry,
  declares 2 GB): consistent with node memory-pressure eviction on the shared 16 GB nodes; pods were gone
  before Kubernetes events could be read. Not fixed; would need a bigger memory request or dedicated nodes.
Podman: 2 verifier "no reward file" per run, nothing else.

**Verdict:** for this workload OpenSandbox is *not* more reliable than in-job podman: ~5% infra errors
vs 0.4%, and ~1.6× the command-timeout rate because sandbox nodes give ~1–2 effective cores where
podman containers were unthrottled — which depresses pass@5 by 6–9 pts while pass@1 stays within
noise. Repeatability of the score itself is comparable. What OpenSandbox does buy: no nested-container
privileges or podman patch stack, clean teardown (0 leaked sandboxes across 4 jobs), and running on
clusters that block nested containers. Closing the gap is a deployment change (bigger nodes / enforced
CPU limits, longer image-pull budget), not a client fix.

Reading the output: `infra` errors (env start, sandbox died, verifier, connection) are the backend's
fault; `timeout` (the agent's 120 s command limit) is mostly the model's, **but on OpenSandbox part of
it is infra-induced** — sandbox nodes give ~1–2 effective cores vs the unthrottled GPU node under
podman, so compute-heavy commands trip the limit more often (see docs/running_evals.md §3b). Compare
`pass1_adj` and the per-task pass-count agreement across the two runs of each backend; lower spread
and a higher error-task Jaccard (errors repeating on the same tasks rather than random ones) mean a
more reliable backend.


## OpenSandbox-trained 9B DPPO checkpoints — 5-run pass@1 (2026-09-26)

Four checkpoints from the 9B DPPO runs that used the OpenSandbox backend *for training*
(evaluated here on the standard in-job podman backend so the numbers are comparable to the
`swerl-qwen35-9b-dppo-4n64k` sweep above). All were DeepSpeed *state* saves only
(`global_stepN/`), converted with `convert_ds_state_to_cg.py` (see `README.md`) into
`/weka/oe-adapt-default/pradeepd/checkpoints/<exp>_step<N>_cg`:

| row in the sheet | checkpoint state | init model |
|---|---|---|
| `swerl-qwen35-9b-dppo-osb-nomask-4n64k` / 232 | `..._opensandbox_no_mask_infra/global_step232` | hamishivi/Qwen3.5-9B |
| `swerl-qwen35-9b-dppo-osb-mask-4n64k` / 233 | `..._opensandbox_mask_infra_backup/global_step233` | hamishivi/Qwen3.5-9B |
| `swerl-qwen35-9b-dppo-osb-mask-4n64k` / 504 | `..._opensandbox_mask_infra/global_step504` | hamishivi/Qwen3.5-9B |
| `swerl-qwen35-9b-glm52sft-dppo-osb-mask-4n64k` / 502 | `..._glm52_sft_dppo_4node_64k_opensandbox_mask_infra/global_step502` | `pradeepd/checkpoints/hamish_qwen35_9b_tmax_glm52_all_sft_composite` |

Eval config (one job per checkpoint × dataset, launched from commit `90a79420` of `eval_opensandbox`):
Vanillux2Agent, `openai` provider, `qwen3_xml`, 64k context, 1 GPU, `--n-concurrent 8`,
`--n-attempts 5`, podman backend with `--keep-task-images` (no mirror), ai2/jupiter+saturn.
TB2.1 via `--dataset-path /weka/oe-adapt-default/shashankg/datasets/terminal-bench-2-1`,
TBLite via `--dataset openthoughts-tblite@2.0`. Experiment IDs are in the sheet rows.

**pass@1 ± std over 5 runs**: `-k 5` gives 5 trials per task; `scripts/compute_stats.py`
(run by the job, output in the result dataset's `metrics.json`) treats the i-th trial of every
task as run i and reports `mean`/`std` over the 5 runs. Fetch with
`beaker dataset fetch <result-dataset> --prefix metrics.json -o <dir>` or read the scoring
lines in the job log; `combined_evals.py refresh` fills the sheet's pass@1/pass@5.

### Results (all 8 jobs finished 2026-09-27; sheet rows refreshed)

pass@1 in %, ± the std of one run's mean score (partition-free estimate from
`scripts/analysis/adjusted_run_stats.py`, run on the downloaded job dirs). *raw* = errored trials
count as 0 (what the sheet's `pass@1` is); *adjusted* = trials that died on image pulls removed
from the denominator. Reference: the original podman-trained sweep `swerl-qwen35-9b-dppo-4n64k`
step 500 is 22.9 / 51.6 (adj 25.1 / 54.2) and `Qwen/Qwen3.5-9B` is 19.3 / 43.2 (adj 21.9 / 45.6).

| checkpoint | TB2.1 raw | TB2.1 adj | TBLite raw | TBLite adj |
|---|---|---|---|---|
| no-mask, step 232 | 18.4 ± 2.5 | 19.6 ± 2.5 | 40.6 ± 3.4 | 40.8 ± 3.3 |
| mask, step 233 | 17.1 ± 2.8 | 19.9 ± 2.6 | 43.7 ± 3.5 | 43.9 ± 3.5 |
| mask, step 504 | 20.0 ± 3.1 | 22.5 ± 2.9 | 43.4 ± 3.4 | 43.6 ± 3.4 |
| GLM5.2-SFT init, mask, step 502 | 9.4 ± 2.4 | 10.9 ± 2.5 | 21.7 ± 2.9 | 21.7 ± 2.9 |

The three base-init checkpoints are within noise of each other and of the Qwen3.5-9B base on
TB2.1; on TBLite the masked runs are ~3 pts above no-mask, which is about one std. All are well
below the podman-trained step-500 model on TBLite (43–44 vs 51.6). The GLM5.2-SFT-init run is far
worse on both (its SFT init `hamish_qwen35_9b_tmax_glm52_all_sft_composite` was 16-attempt-evaluated
on 2026-09-18, see `01M2SD0M91TMFEK4RBSKARJCSR`); ~15% of its TBLite trials hit the 120 s command
timeout vs ~10% for the others.

**Infra failure in this batch — Docker Hub pull rate limit on TB2.1.** 39–65 of 445 trials per
TB2.1 job (9–15%; 2–3 of 500 on TBLite) errored with `toomanyrequests: You have reached your
unauthenticated pull rate limit` during `docker compose up`, spread over 24–34 tasks per job (never
all 5 attempts of a task). Two causes: (1) five of the eight jobs were scheduled onto the same
jupiter node, and with `--host-networking` they share one IP for the anonymous cap; (2) the pulls
were *anonymous* despite `docker login` + `podman login` both reporting OK in the job log, so the
podman service is not reading the credentials the script writes (the 2026-09-22 study got away with
it: 2 jobs, 0 rate-limit errors). The 34-task subset with no pull failure in any job ranks the
checkpoints the same way (no-mask 25.9, mask-504 23.5, mask-233 22.9, GLM-SFT 12.4). To fill the
holes without a full re-run: `harbor job resume -p jobs/<job> --filter-error-type RuntimeError`
re-runs only the errored trials (needs the job dir from the result dataset restored into a new
eval job, and a fix for the anonymous pulls or one job per node). Command-timeout `RuntimeError`s
("Command timed out after 120 seconds", 47–77 per job) are the model's own failures, not infra —
newer harbor reports them as `RuntimeError` rather than `AgentTimeoutError`, so the sheet's
`pass1_adj` (which treats every `RuntimeError` as infra) over-corrects for these rows.
