"""Enumerate the exact task/image order a run will consume, without executing it.

Mirrors grpo_fast's data path: get_cached_dataset_tulu (same transforms) then the
DataPreparationActor's prompt stream — HFDataLoader(batch_size=1, seed, dp 0/1,
automatic_reshuffle=True). Each optimizer step consumes NUM_UNIQUE prompts in
stream order, so the first STEPS*NUM_UNIQUE rows are the run's task set.

Usage:
  python enumerate_task_images.py --dataset <jsonl> --seed 42 --num-unique 4 --steps 12
"""

import argparse
import json
import tempfile

from datasets import Dataset, load_from_disk

from open_instruct import data_loader as data_loader_lib
from open_instruct.dataset_transformation import TokenizerConfig, get_cached_dataset_tulu
from open_instruct.environments.swerl_vanillux_sandbox import _BASH_TOOL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=None,
        help="Raw jsonl; rebuilds a cache (row count may differ from a real run's cache — prefer --cache-dir).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Path to the RUN'S OWN cached transformed dataset "
            "(from its 'Found cached dataset at' log line). Refuses to rebuild."
        ),
    )
    parser.add_argument("--model", default="hamishivi/Qwen3.5-9B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-unique", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--max-prompt-token-length", type=int, default=2048)
    parser.add_argument("--out", default="/dev/stdout")
    args = parser.parse_args()

    if args.cache_dir:
        dataset = load_from_disk(args.cache_dir)
        print(f"loaded run cache {args.cache_dir}: {len(dataset)} rows")
        _enumerate(dataset, args)
        return
    if not args.dataset:
        raise SystemExit("provide --cache-dir (preferred: the run's own cache) or --dataset")
    tc = TokenizerConfig(tokenizer_name_or_path=args.model, chat_template_name="tulu")
    transform_fn_args = [
        {"system_prompt_override": None, "tool_definitions": [_BASH_TOOL], "pass_tools_to_chat_template": True},
        {"max_prompt_token_length": args.max_prompt_token_length},
    ]
    dataset = get_cached_dataset_tulu(
        dataset_mixer_list=[args.dataset, "1.0"],
        dataset_mixer_list_splits=["train"],
        tc=tc,
        dataset_transform_fn=["rlvr_tokenize_v1", "rlvr_max_length_filter_v1"],
        transform_fn_args=transform_fn_args,
        dataset_cache_mode="local",
        hf_entity=None,
        dataset_local_cache_dir="local_dataset_cache",
        dataset_skip_cache=False,
        system_prompt_override=None,
    )
    print(
        f"transformed dataset rows: {len(dataset)} — "
        "WARNING: a rebuilt cache can differ from the run's own cache; prefer --cache-dir"
    )
    _enumerate(dataset, args)


def _enumerate(dataset: Dataset, args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory() as work_dir:
        loader = data_loader_lib.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=args.seed,
            dp_rank=0,
            dp_world_size=1,
            work_dir=work_dir,
            automatic_reshuffle=True,
            collator=data_loader_lib.single_example_collator,
        )
        rows = []
        it = iter(loader)
        for step in range(1, args.steps + 1):
            for _ in range(args.num_unique):
                example = next(it)
                env_config = example.get("env_config")
                if isinstance(env_config, str):
                    env_config = json.loads(env_config)
                image = None
                if isinstance(env_config, dict):
                    image = env_config.get("image") or (env_config.get("kwargs") or {}).get("image")
                idx = example.get("index")
                if hasattr(idx, "item"):
                    idx = int(idx.reshape(-1)[0])
                rows.append({"step": step, "index": idx, "image": image})

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    uniq = sorted({r["image"] for r in rows if r["image"]})
    print(f"steps={args.steps} prompts={len(rows)} unique_images={len(uniq)}")
    for img in uniq:
        print(img)


if __name__ == "__main__":
    main()
