# `rl_data/` — Terminal-environment generation for RL/SFT training

This package **synthesizes brand-new Terminal/CLI tasks** (containerized sandbox
scenarios + automated pass/fail verifiers) and the machinery to **solve, score,
analyze, decontaminate, and publish** them as agentic RL/SFT training data.

> **The full guide lives in [`../docs/rl_data.md`](../docs/rl_data.md).** Read
> that first — it has the end-to-end pipeline, task anatomy, the taxonomy/sampler,
> container base images, the `comparison/` branch, a glossary, a "which script do
> I run" cheat sheet, the full ordered runbook, cost/smoke tests, and gotchas.
> This README is just the in-folder map.

> **One-line mental model:** an LLM (Gemini by default) invents a task + its
> verifier → we bake it into an Apptainer container → we run *other* LLM agents
> against it to measure difficulty (pass@k) → filter, dedup vs. eval benchmarks,
> upload to Hugging Face.

## Layout

```
rl_data/
├── generate_tasks.py      # Stage 1: synthesize tasks (templates → tests → container def → fixtures)
├── generate_solutions.py  # Stage 2: run agents N× per task, score pass@k
├── analyze.py             # Stage 3: corpus stats + difficulty/quality plots
├── estimate_cost.py       # project LLM cost before a run
├── upload_to_hf.py        # Stage 4: push solved corpus to the Hub
├── generator/             # the task-generation engine (taxonomy, env, solvers, fixtures)
├── comparison/            # SIDE BRANCH: benchmark our corpus vs external datasets
├── decontamination/       # n-gram overlap vs eval benchmarks
├── containers/            # Apptainer base-image defs (9 per-domain + base_intricate)
└── scripts/               # runnable shell wrappers for every stage
```

## The pipeline

```
generate_tasks.py     → task_*/  (task.json, container.def, setup.sh, test_*.py, fixtures/)
                          ↑ runs on containers/base_*.sif (shared base + per-task setup.sh delta)
generate_solutions.py → solutions/<MODEL>[_<harness>][_thinking]_summary.json  (pass@k)
analyze.py / classify_difficulty.py / combine_corpora.py / decontamination/ / upload_to_hf.py
```

The solved trajectories feed the SFT stack ([`../sft/`](../sft) — see
[`../docs/sft.md`](../docs/sft.md)) via `convert_trajectories.py`.

**Everything else — every stage, the taxonomy, base images, the full ordered
runbook, and gotchas — is in [`../docs/rl_data.md`](../docs/rl_data.md).**
