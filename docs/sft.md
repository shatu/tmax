# SFT: Data Generation & Training Guide

This document explains the **Supervised Fine-Tuning (SFT)** stack in the `tmax`
repository: how training data is generated from raw agent trajectories, what
scripts to run at each stage, and how to launch single- and multi-node training.

Everything lives under [`sft/`](../sft):

```
sft/
├── preprocessing/      # raw trajectories -> unified SFT parquet (data generation)
│   ├── pipeline.py         # orchestrator (Terminus-2 + Sera + passthrough sources)
│   ├── convert.py          # Terminus-2 trace -> unified messages
│   ├── convert_sera.py     # Sera SWE-agent trace -> unified messages
│   ├── convert_trajectories.py  # rl_data solve outputs -> unified messages
│   ├── builders.py         # helpers: tool_calls, tool_results, reasoning, submit
│   ├── json_extraction.py  # 5-strategy JSON extractor for assistant turns
│   ├── filters.py          # mandatory / optional quality filters
│   ├── filter_bad_tool_call.py  # post-pass: drop literal "<tool_call>" rows
│   ├── harness.py          # vanillux / tassie system-prompt + tool specs
│   ├── report.py           # rich conversion reports
│   └── config/             # sources.yaml, tool_schemas.json, system_prompt.txt
├── data.py             # load converted parquet -> HF Dataset (+ tool injection)
├── pre_tokenize.py     # tokenize messages -> input_ids (+ assistant_masks)
├── train.py            # TRL SFTTrainer w/ DeepSpeed Ulysses sequence parallel
├── configs/            # accelerate + deepspeed configs (sp4/sp8, 1-4 nodes)
├── scripts/            # runnable shell wrappers for every stage
└── tests/              # pytest for json_extraction / builders / convert
```

## The pipeline at a glance

```
 raw trajectories                unified SFT parquet            tokenized dataset           trained model
 (HF Hub / rl_data)              messages|tools|source|meta     input_ids[, asst_masks]     checkpoint
        │                               │                              │                         │
        │  run_conversion*.sh           │  run_pretokenize_*.sh        │  run_sft*.sh            │
        └──► preprocessing.pipeline ────┴──► pre_tokenize.py ──────────┴──► train.py ────────────┘
             (or convert_trajectories)      (assistant-only loss masks)     (TRL + Ulysses SP)
                     │
                     └─(optional)─► filter_bad_tool_call.py ─► upload_data_to_hf.sh
```

There are **three stages**. Each writes a concrete on-disk artifact that the
next stage consumes:

1. **Conversion / data generation** — turn raw, source-specific agent traces
   into a single unified schema (`messages | tools | source | metadata`).
2. **Pre-tokenization** — apply the model's chat template, produce `input_ids`
   and (optionally) assistant-only loss masks.
3. **Training** — TRL `SFTTrainer` with DeepSpeed Ulysses sequence parallelism
   for long (65K-token) agent trajectories.

You can also tokenize on-the-fly inside `train.py` and skip stage 2, but
pre-tokenizing once is the recommended path for repeated/large runs.

---

## The unified data schema

Every conversion writes parquet with **exactly four columns**, so datasets from
different sources can be concatenated without type conflicts:

| Column     | Type        | Meaning                                                       |
| ---------- | ----------- | ------------------------------------------------------------- |
| `messages` | `list[dict]`| The conversation, one normalized message struct per turn.     |
| `tools`    | `str`       | JSON-encoded OpenAI tool definitions (kept as a string for lossless HF round-trip). |
| `source`   | `str`       | Provenance label, e.g. `nvidia/Nemotron-Terminal-Corpus/skill_based_easy`. |
| `metadata` | `dict`      | Conversion bookkeeping (model, task, turn counts, warnings…). |

Each message is normalized to a **fixed 5-key struct** (see
[`convert.py:48`](../sft/preprocessing/convert.py) `_normalise_message`). Keeping
the keys identical across all converters lets HF Datasets infer a single type:

```python
{
    "content": str,              # "" if originally None
    "reasoning_content": str,    # model analysis/plan, or text inside <think>...</think>
    "role": str,                 # "system" | "user" | "assistant" | "tool"
    "tool_call_ids": list[str],  # IDs of the tool calls a tool message answers
    "tool_calls": list[dict],    # OpenAI-style tool calls (bash command per turn)
}
```

The agent uses a **single `bash` tool** (persistent shell — working dir and env
vars carry across calls), defined in
[`config/tool_schemas.json`](../sft/preprocessing/config/tool_schemas.json). Task
completion is represented by the assistant emitting
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (the `SUBMIT_COMMAND`).

### Harnesses (system prompt + instance framing)

The same raw trace can be rendered into two framings, selected with
`--harness` ([`harness.py`](../sft/preprocessing/harness.py)):

- **`vanillux`** (default) — short system prompt + mini-swe-agent v2 instance
  template, loaded from `rl_data/generator/vanillux_prompts.yaml`. This matches
  exactly what `rl_data.generator.vanillux_solver` sends at RL solve time, so SFT
  and RL see the same prompt distribution.
- **`tassie`** (legacy) — persistent-bash system prompt from
  [`config/system_prompt.txt`](../sft/preprocessing/config/system_prompt.txt),
  bare task in the user turn. Reproduces the older `tmax-sft-full-20260409`
  format byte-for-byte.

Both write the identical four-column schema and the same single-bash tool spec.

---

## Stage 1 — Data generation (conversion)

### Registered sources

Sources are declared in
[`config/sources.yaml`](../sft/preprocessing/config/sources.yaml). Each entry says
how to load the raw data and which converter to use:

| Source                                   | Loader (`type`)         | Converter (`format`) | Notes                                       |
| ---------------------------------------- | ----------------------- | ----------------- | ------------------------------------------- |
| `open-thoughts/OpenThoughts-Agent-v1-SFT`| `huggingface`           | terminus2 (default) | `conversations` column                      |
| `nvidia/Nemotron-Terminal-Corpus`        | `huggingface_parquet`   | terminus2         | 4 subsets: `dataset_adapters`, `skill_based_easy/medium/mixed` (each a glob `pattern`) |
| `allenai/Sera-4.6-Lite-47000`            | `huggingface`           | **`sera`**        | already SWE-agent format; normalized to bash-only |
| `m-a-p/TerminalTraj`                      | `huggingface`           | terminus2         | `messages` column                           |
| `osieosie/tmax-sft-skill-tax-…-thinking` | `huggingface_passthrough`| (none)           | already in target schema; copied verbatim (source label `skill_tax_20260505_2.2k_combined_balanced_thinking_all`) |

**Two dispatch keys in `sources.yaml` drive the converter choice:**

- `type` selects the *loader*: `huggingface` (Datasets `load_dataset`),
  `huggingface_parquet` (glob a `pattern` of parquet files), or
  `huggingface_passthrough` (rows are *already* in the unified schema — see below).
- `format` selects the *converter* for loadable sources: `terminus2` (default,
  `convert.py`) or `sera` (`convert_sera.py`). The pipeline branches on this at
  [`pipeline.py:307`](../sft/preprocessing/pipeline.py).

**`huggingface_passthrough` sources skip conversion entirely.** The skill-tax /
rl_data corpus is produced **offline** by `convert_trajectories.py` (see
[Stage 1 → skill-tax](#skill-tax--rl_data-trajectories-the-link-to-the-rl_data-pipeline)),
uploaded to the Hub already in the 4-column schema, and then ingested verbatim
by `_process_passthrough` ([`pipeline.py:460`](../sft/preprocessing/pipeline.py)) —
which copies the rows and only backfills a missing `tools` column from the
selected harness. No convert/filter/JSON-extraction runs on these rows.

### What conversion does

For each row, the pipeline ([`pipeline.py`](../sft/preprocessing/pipeline.py)
`process_source`) runs:

1. **Convert** the raw trace to the unified schema:
   - **Terminus-2** ([`convert.py`](../sft/preprocessing/convert.py)
     `convert_trace`): parse the task description from message 0, emit
     system+user, then walk `(assistant, user)` pairs. Assistant turns are JSON
     blobs (`{"analysis", "plan", "commands", "task_complete"}`); the
     [`json_extraction.py`](../sft/preprocessing/json_extraction.py) 5-strategy
     cascade robustly recovers them even when malformed or wrapped in
     `<think>` tags. Commands become a single bash tool call; the following
     terminal output becomes a `tool` message. Reasoning-only turns are buffered
     into the next tool call.
   - **Sera** ([`convert_sera.py`](../sft/preprocessing/convert_sera.py)
     `convert_sera_trace`): already SWE-agent format; converts
     `str_replace_editor`/`submit` tool calls into equivalent **bash** commands
     so everything is bash-only, strips `<think>` tags into `reasoning_content`.
   - **rl_data trajectories** ([`convert_trajectories.py`](../sft/preprocessing/convert_trajectories.py)
     `convert`): reads per-task `*_summary.json` solve outputs, strips LiteLLM
     artifacts, optionally keeps only successful trajectories.
2. **Filter** ([`filters.py`](../sft/preprocessing/filters.py)):
   - **Mandatory drops:** `conversion_failed`, `json_extraction_failed`,
     `too_few_turns` (`<1`), `no_task_complete` (unless `--include-partial`),
     `contains_ctrl_c`.
   - **Optional drops:** exceeds `--max-turns` (`exceeds_<N>_turns`). *(A
     `trivial_only` drop exists in [`filters.py`](../sft/preprocessing/filters.py)
     but is gated by `drop_trivial_only`, which `process_source` never passes —
     so it never fires in practice.)*
   - **Warning flags** (recorded, not dropped): `no_task_complete`,
     `task_delim_missing`, `prose_outside_json`, `missing_tool_result`.
3. **Record stats + examples** and write outputs. The JSON report also records
   the chosen `harness` and the per-strategy JSON-extraction distribution.
   (Note: the *text* report only prints the strategy-1/2/3/0 counts — `<think>`
   strategies 4 & 5 are counted but not displayed; check `conversion_report.json`
   for those.)

Passthrough sources skip steps 1–2 entirely (rows copied verbatim).

Outputs per run (under `--output-dir`):
- one parquet per source (`messages | tools | source | metadata`)
- `dropped_*.jsonl` (one record per dropped row, for diagnosis)
- `conversion_report.json` + `conversion_report.txt` (yields, drop reasons, JSON
  extraction strategy distribution, turn-count stats, sampled examples)

### Running conversion

**Quick teaser (1% of every source, ~1–5 min)** — use this to sanity-check the
pipeline before a full run:

```bash
bash sft/scripts/run_conversion_teaser.sh
# -> output/preprocessing/terminus2_sweagent_1pct
```

**Full conversion (all sources):**

```bash
bash sft/scripts/run_conversion.sh
# default harness = vanillux, --max-turns 999
# -> output/preprocessing/terminus2_vanillux_full_<date>
# uploads to HF (osieosie/tmax-sft-full-<date>) by default
# (run_conversion.sh has no --no-upload; the skill-tax and filter scripts do)
```

To reproduce the legacy framing: `bash sft/scripts/run_conversion.sh --harness tassie`.

Under the hood these call:

```bash
python -m preprocessing.pipeline \
  --num-workers $(nproc) \
  --output-dir output/preprocessing/<name> \
  --harness vanillux \
  --max-turns 999 \
  --num-examples 3
```

Useful `pipeline.py` flags:

| Flag                               | Purpose                                              |
| ---------------------------------- | ---------------------------------------------------- |
| `--sources LABEL [LABEL ...]`      | Restrict to specific source labels (default: all).   |
| `--sample N` / `--sample-frac F`   | Subsample N traces / fraction per source.            |
| `--max-turns N`                    | Drop traces longer than N turns.                     |
| `--include-partial`                | Keep traces without a final submit (flag, not drop). |
| `--harness {vanillux,tassie}`      | Choose the prompt framing.                           |
| `--shard-index I --num-shards K`   | Process one shard (for distributed conversion).      |
| `--merge-shards DIR [DIR ...]`     | Merge previously produced shard outputs.             |

**Sharded conversion (SLURM array, for the very large corpora):**

```bash
bash sft/scripts/run_conversion_sharded.sh
# runs NUM_SHARDS shards (SLURM array or sequential loop), then --merge-shards
```

### Skill-tax / rl_data trajectories (the link to the rl_data pipeline)

This is how solved **terminal tasks generated by the `rl_data/` pipeline** become
SFT data. `rl_data.generate_solutions` runs an agent N times against each task and
writes `task_*/solutions/<summary>.json`; `convert_trajectories.py`
([`sft/preprocessing/convert_trajectories.py`](../sft/preprocessing/convert_trajectories.py))
reads those summaries and emits the unified 4-column parquet — same schema as the
other converters, so the trainer ingests it unchanged.

```bash
bash sft/scripts/run_conversion_skill_tax_sft.sh
# runs `uv run python -m preprocessing.convert_trajectories` TWICE:
#   --tasks-dir  <rl_data .../tasks_skill_tax_...>
#   --model-tag  hosted_vllm_Qwen_Qwen3.5-27B   # solve-time model id, '/'→'_'
#   --harness    bash|vanillux                  # solve-time harness
#   [--thinking]                                # if solved with reasoning traces
#   --output-dir output/preprocessing/skill_tax_...
#   --name <NAME_ALL>                           # → <NAME_ALL>.parquet  (all trajectories)
# 2nd run adds --filter-success → <NAME_ONLY_SUCCESS>.parquet (success-only)
# then uploads both as two configs in ONE HF repo (unless --no-upload)
```

**The filename contract (must match rl_data exactly).**
`convert_trajectories.py` reconstructs the summary filename from
`--model-tag` / `--harness` / `--thinking` using the **same `_summary_basename`
logic** as `rl_data/generate_solutions.py`. So those three flags must equal the
values used at *solve* time:

| solve-time (`generate_solutions`) | summary file read | converter flags |
|---|---|---|
| model `M`, `bash`, no thinking | `<M>_summary.json` | `--model-tag <M>` |
| model `M`, `vanillux` | `<M>_vanillux_summary.json` | `--model-tag <M> --harness vanillux` |
| model `M`, `vanillux`, thinking | `<M>_vanillux_thinking_summary.json` | `--model-tag <M> --harness vanillux --thinking` |

`--model-tag` is the solve-time model id with `/`→`_` (e.g.
`hosted_vllm/Qwen/Qwen3.5-27B` → `hosted_vllm_Qwen_Qwen3.5-27B`). If any of the
three differ, the scan finds **zero** matching summaries and silently writes an
empty parquet. The converter strips LiteLLM artifacts (`function_call`,
`thinking_blocks`, `provider_specific_fields`, …), folds reasoning into
`reasoning_content`, and rewrites each `bash` tool call into the canonical
`{command: …}` shape.

> ⚠️ **`--harness` means two different things in this repo.** In
> `convert_trajectories.py` it selects **which solve-time summary file to read**
> (`bash` vs `vanillux` — the *solver* that produced the trajectories). In
> `pipeline.py` / `run_conversion.sh` it selects the **output prompt framing**
> (`vanillux` vs `tassie` — see [Harnesses](#harnesses-system-prompt--instance-framing)).
> They share the word but are unrelated axes; `bash` is valid only for the
> trajectory converter, `tassie` only for the pipeline.

### Post-conversion: filter bad tool-call tokens

Some upstream traces leak the literal string `<tool_call>` into message text
(instead of using the structured `tool_calls` field), which would teach the model
to hallucinate that tag. Drop those rows:

```bash
bash sft/scripts/run_filter_bad_tool_call.sh
# python -m preprocessing.filter_bad_tool_call
#   --input-dir  output/preprocessing/<name>
#   --output-dir output/preprocessing/<name>_bad_tool_call_filtered
#   --needle "<tool_call>"
# writes filter_report.json with per-source kept/removed counts
```

### Uploading data to the Hub

```bash
bash sft/scripts/upload_data_to_hf.sh   # needs HF_TOKEN
# each parquet becomes a separate config/subset under data/<config>/train-*.parquet
```

---

## Stage 2 — Pre-tokenization

[`pre_tokenize.py`](../sft/pre_tokenize.py) applies the model chat template and
saves a HF Dataset to disk containing `input_ids` and, with
`--assistant_only_loss`, an `assistant_masks` column.

Key behaviors:

- **`reasoning_content` handling** (`_adapt_messages`): if the tokenizer's chat
  template natively supports `reasoning_content` (e.g. Qwen3.5), it's passed
  through; otherwise (e.g. Qwen3-4B-Instruct-2507) it's wrapped into
  `<think>…</think>` and merged into `content`.
- **Assistant-only loss masks** (`_build_assistant_masks`): Qwen chat templates
  lack the `{% generation %}` tag, so HF's `return_assistant_tokens_mask` returns
  all zeros. This function instead scans the tokenized stream for ChatML role
  boundaries (`<|im_start|>assistant\n … <|im_end|>`) and sets the mask to `1`
  on assistant content **including** the closing `<|im_end|>` (so the model
  learns to stop).
- **Truncation** to `--max_length` (default 65536) and optional **sharding**
  (`--num_shards`/`--shard_index`) for parallel tokenization.
- **Diagnostics**: writes `tokenization_diagnostics.jsonl` with decoded samples.

Data loading is handled by [`data.py`](../sft/data.py) `load_converted_corpus`,
which reads the converted parquet, optionally filters by `--sources` / subsamples
by `--sample_frac`, and injects the constant `tools` column.

### Running pre-tokenization

```bash
# Qwen3.5-4B, full SWE-agent corpus, assistant-only loss
bash sft/scripts/run_pretokenize_sweagent_full_qwen3.5.sh
# -> tokenized_tbmax_terminus2_sweagent_full_<date>_qwen3.5_42

# Qwen3-4B-Instruct-2507 variant
bash sft/scripts/run_pretokenize_sweagent_full_qwen3.sh

# Nemotron-Terminal subset, 5% sample
bash sft/scripts/run_pretokenize_nemotron-terminal_qwen3.5.sh
```

These wrap:

```bash
python pre_tokenize.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --data_dir   output/preprocessing/terminus2_sweagent_full_<date> \
  --output_path tokenized_tbmax_terminus2_sweagent_full_<date>_qwen3.5_42 \
  --max_length 65536 \
  --num_proc $(nproc) \
  --seed 42 \
  --assistant_only_loss
```

---

## Stage 3 — Training

[`train.py`](../sft/train.py) runs TRL's `SFTTrainer`. The headline feature is
**DeepSpeed Ulysses Sequence Parallelism (SP)**, which shards a single 65K-token
sequence across several GPUs so long agent trajectories fit in memory.

### How parallelism is configured

The accelerate config YAML sets env vars that `train.py` reads:

```python
sp_size       = int(os.environ.get("PARALLELISM_CONFIG_SP_SIZE", "1"))  # GPUs per SP group
dp_world_size = max(1, args.num_gpus // sp_size)                        # data-parallel replicas
grad_accum    = args.global_batch_size // (dp_world_size * args.per_device_train_batch_size)
```

The invariant is **`sp_size × dp_shard_size = processes per node`**. The provided
configs cover SP=4 and SP=8 across 1/2/4 nodes:

| Config                                  | Nodes | GPUs | SP | DP  |
| --------------------------------------- | ----- | ---- | -- | --- |
| `accelerate_ds_z3_sp4_8xh200.yaml`      | 1     | 8    | 4  | 2   |
| `accelerate_ds_z3_sp8_8xh200.yaml`      | 1     | 8    | 8  | 1   |
| `accelerate_ds_z3_sp4_2x8xh200.yaml`    | 2     | 16   | 4  | 4   |
| `accelerate_ds_z3_sp8_2x8xh200.yaml`    | 2     | 16   | 8  | 2   |
| `accelerate_ds_z3_sp4_4x8xh200.yaml`    | 4     | 32   | 4  | 8   |
| `accelerate_ds_z3_sp8_4x8xh200.yaml`    | 4     | 32   | 8  | 4   |
| `accelerate_fsdp_8xh200.yaml`           | 1     | 8    | —  | FSDP FULL_SHARD (no SP) |

SP traffic (all-to-all) is kept within a node (NVLink); ZeRO-3 model sharding
spans the DP dimension (and nodes). The raw DeepSpeed JSON configs
(`ds_z3_sp8.json`, `ds_z2_sp8.json`, `ds_z3_offload_sp8.json`) are alternatives
for direct-DeepSpeed setups; `ds_z3_offload_sp8.json` offloads optimizer+params
to CPU for memory-constrained runs.

Notable training internals:
- `per_device_train_batch_size` **must be 1** with Ulysses SP.
- Custom `SFTTrainerSP.compute_loss` aggregates a token-weighted mean loss across
  the SP group (and safely skips ranks whose tokens are all masked to `-100`).
- If `assistant_masks` is present in the tokenized dataset, TRL's collator
  automatically masks non-assistant labels — so **assistant-only loss is driven
  by stage 2**, not a training flag.
- `gradient_checkpointing`, `bf16`, `flash_attention_2`, and `packing` are on.
- A `CollatorWithPositionIds` wrapper injects global `position_ids` so each SP
  rank gets correct rotary positions for its shard.
- Several Qwen3.5-specific SP fixes are applied at startup ([`train.py:189`](../sft/train.py)):
  pointing `deepspeed.utils.groups.mpu` at the SP parallel-state module, patching
  `UlyssesSPAttentionHF.forward` to handle 3-D MROPE `position_ids`, and copying
  attention dims out of `config.text_config` to the top level so DeepSpeed can
  find them. These are why SP training works on Qwen3.5 out of the box.

### Running training

**Single node (8× H200):**

```bash
# Qwen3.5-4B
bash sft/scripts/run_sft_qwen3.5-4b.sh
# Qwen3-4B-Instruct-2507
bash sft/scripts/run_sft_qwen3-4b.sh
```

These run locally via `accelerate launch`:

```bash
accelerate launch --config_file configs/accelerate_ds_z3_sp4_8xh200.yaml \
  train.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --tokenized_dataset_path tokenized_tbmax_terminus2_sweagent_full_<date>_qwen3.5_42 \
  --num_gpus 8 \
  --global_batch_size 128 \
  --max_length 65536 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --packing
```

**Multi-node (SLURM):**

```bash
bash sft/scripts/run_sft_multinode_qwen3.5-4b.sh     # 2 nodes / 16 GPUs
bash sft/scripts/run_sft_multinode_qwen3-4b.sh       # 2 nodes / 16 GPUs (epochs=1, lr=2e-6, 100k samples)
bash sft/scripts/run_sft_multinode_qwen3-4b_4node.sh # 4 nodes / 32 GPUs
```

The multinode scripts are SLURM batch scripts: they set `MASTER_ADDR` from
SLURM, generate a per-node launcher, and `srun --ntasks-per-node=1` an
`accelerate launch` with `--machine_rank $SLURM_NODEID` and
`--deepspeed_multinode_launcher standard`.

Key `train.py` arguments:

| Arg                            | Default              | Notes                                              |
| ------------------------------ | -------------------- | -------------------------------------------------- |
| `--model_name_or_path`         | `Qwen/Qwen3.5-4B`    | HF id or local path.                               |
| `--tokenized_dataset_path`     | —                    | Pre-tokenized dataset(s); multiple are concatenated. |
| `--data_dir`                   | terminus2_sweagent   | On-the-fly path (used when no tokenized dataset).  |
| `--num_gpus`                   | 8                    | Used to compute grad accumulation.                 |
| `--global_batch_size`          | 128                  | Effective batch; grad-accum derived from it.       |
| `--per_device_train_batch_size`| 1                    | Must be 1 under Ulysses SP.                         |
| `--max_length`                 | 65536                | Sequence length.                                   |
| `--num_train_epochs`           | 2                    |                                                    |
| `--learning_rate`              | 2e-5                 |                                                    |
| `--packing`                    | off                  | Pack multiple samples per sequence.                |
| `--max_train_samples`          | —                    | Subsample to N examples.                           |
| `--wandb_project` / `--run_name`| —                   | W&B logging.                                        |

### Uploading the trained model

```bash
bash sft/scripts/upload_model_to_hf.sh   # needs HF_TOKEN
# uploads safetensors/config/tokenizer; ignores checkpoint-*/wandb/logs
```

---

## End-to-end example

```bash
cd sft

# 1. Generate data (full corpus, vanillux framing) and drop bad tool-call rows
bash scripts/run_conversion.sh
bash scripts/run_filter_bad_tool_call.sh

# 2. Pre-tokenize for Qwen3.5-4B with assistant-only loss
bash scripts/run_pretokenize_sweagent_full_qwen3.5.sh

# 3. Train on 8 GPUs (single node)
bash scripts/run_sft_qwen3.5-4b.sh

# 4. Publish
bash scripts/upload_model_to_hf.sh
```

For a fast smoke test, swap step 1 for `bash scripts/run_conversion_teaser.sh`
(1% of each source).

---

## Tips & gotchas

- **Schema parity is load-bearing.** All three converters
  (`convert.py`, `convert_sera.py`, `convert_trajectories.py`) must emit the
  identical 5-key message struct, or HF Datasets fails to infer a single type
  when concatenating sources.
- **Assistant-only loss is set at tokenization time**, via
  `--assistant_only_loss` in `pre_tokenize.py` (produces `assistant_masks`).
  Training picks it up automatically — there is no separate training flag.
- **`per_device_train_batch_size` must stay 1** with Ulysses SP; scale via
  `--global_batch_size` (which sets grad accumulation) and the SP/DP layout.
- **Pick the harness deliberately.** Use `vanillux` to match RL solve-time
  prompts; use `tassie` only to reproduce legacy `tmax-sft-full-20260409` runs.
  Remember the two harness namespaces: `pipeline.py` uses `{vanillux, tassie}`
  (output framing), `convert_trajectories.py` uses `{bash, vanillux}` (which
  solve-time summary to read).
- **The rl_data → SFT link is a filename contract.** When converting skill-tax
  trajectories, `--model-tag` / `--harness` / `--thinking` must match the values
  used at `rl_data.generate_solutions` solve time, or you get a silent empty
  parquet. See [the skill-tax section](#skill-tax--rl_data-trajectories-the-link-to-the-rl_data-pipeline).
- **Passthrough sources are pre-converted.** The skill-tax HF dataset is ingested
  verbatim (no convert/filter) — regenerate it with `convert_trajectories.py`, not
  by tweaking pipeline flags.
- **Always run the teaser first** on a new source or after pipeline changes —
  it's minutes, not hours, and the `conversion_report.txt` surfaces drop reasons
  and JSON-extraction failures before you commit to a full run.
- **Tests** for the trickiest logic live in [`sft/tests/`](../sft/tests)
  (`test_json_extraction.py`, `test_builders.py`, `test_convert.py`) — run them
  after touching the converters.
```
