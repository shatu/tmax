"""Convert passing GLM solve trajectories into the tmax SFT dataset format.

Reads per-task ``hosted_vllm_<model>_summary.json`` files (as written by
rl_data.generate_solutions and mirrored to the weka summary cache), keeps only
successful rollouts, and emits rows matching the schema of the reference mix
``hamishivi/tmax-sft-skill-tax-...-thinking-no-tool-call``:

    messages: [
      {role: system, content},
      {role: user, content: "Please solve this task:\\n\\n<description>"},
      {role: assistant, content, reasoning_content, tool_calls: [
          {id, type, function: {name, arguments: <DICT, not JSON string>}}],
       tool_call_ids: []},
      {role: tool, tool_call_id, content},
      ...
    ]
    tools:    the bash tool schema (sample_solutions.TOOL_SCHEMAS)
    source:   provenance string
    metadata: {task, episode, date, model, num_success, pass_at_1}

``reasoning_content`` is kept as a separate field — the student tokenizer's
chat template renders it into <think> blocks at training time, so thinking is
included without any inlining here.

Usage::

    uv run python -m rl_data.scripts.sft.convert_glm_trajectories \\
        --solutions-dir /weka/oe-adapt-default/pradeepd/tmax_solutions/tasks_sft_16.5k \\
        --model-tag hosted_vllm_glm-5.2-fp8 \\
        --out glm52_sft_trajectories.jsonl \\
        [--tokenizer Qwen/Qwen3.5-9B --max-seq-length 65536]

Then create the HF dataset from the JSONL (``datasets.Dataset.from_json`` +
``push_to_hub``) and point the open-instruct SFT script's dataset_mixer at it.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rl_data.generator.sample_solutions import TOOL_SCHEMAS

USER_PREFIX = "Please solve this task:\n\n"

# litellm Message.model_dump() keys that must not enter the dataset.
_ASSISTANT_DROP_KEYS = {"function_call", "provider_specific_fields", "annotations", "audio"}


def _convert_assistant(msg: dict) -> dict:
    out = {
        "role": "assistant",
        "content": msg.get("content") or "",
        "reasoning_content": msg.get("reasoning_content") or "",
        "tool_call_ids": [],
        "tool_calls": [],
    }
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"command": args}  # keep malformed calls inspectable
        out["tool_calls"].append({
            "id": tc.get("id"),
            "type": tc.get("type") or "function",
            "function": {"name": fn.get("name"), "arguments": args},
        })
    return out


def _convert_messages(messages: list[dict]) -> list[dict] | None:
    """Transform one rollout's stored messages; None if structurally unusable."""
    out: list[dict] = []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "system":
            out.append({"role": "system", "content": m.get("content") or ""})
        elif role == "user":
            content = m.get("content") or ""
            if i <= 1 and not content.startswith(USER_PREFIX):
                content = USER_PREFIX + content
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            out.append(_convert_assistant(m))
        elif role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id"),
                "content": m.get("content") or "",
            })
        else:
            return None
    # A trainable sample must end with the model speaking.
    while out and out[-1]["role"] != "assistant":
        out.pop()
    if sum(1 for m in out if m["role"] == "assistant") == 0:
        return None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--solutions-dir", required=True,
                    help="Root holding task_*/solutions/<model-tag>_summary.json")
    ap.add_argument("--model-tag", default="hosted_vllm_glm-5.2-fp8",
                    help="Summary filename prefix (default: hosted_vllm_glm-5.2-fp8)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--tokenizer", default=None,
                    help="Optional HF tokenizer for a training-time length check "
                         "(e.g. Qwen/Qwen3.5-9B; requires `transformers`)")
    ap.add_argument("--max-seq-length", type=int, default=65536,
                    help="Drop samples longer than this under --tokenizer (default: 65536)")
    ap.add_argument("--include-failures", action="store_true",
                    help="DEBUG ONLY: keep non-passing rollouts too")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer  # optional heavy dep
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    root = Path(args.solutions_dir)
    summaries = sorted(root.glob(f"task_*/solutions/{args.model_tag}_summary.json"))
    if not summaries:
        raise SystemExit(f"no {args.model_tag}_summary.json under {root}")

    now = datetime.now(timezone.utc).isoformat()
    n_rows = n_skipped_fail = n_skipped_len = n_skipped_struct = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for sp in summaries:
            task = sp.parent.parent.name
            s = json.loads(sp.read_text(encoding="utf-8"))
            k = {int(a): b for a, b in s.get("pass_at_k", {}).items()}
            for ep, r in enumerate(s.get("results", [])):
                if not r.get("success") and not args.include_failures:
                    n_skipped_fail += 1
                    continue
                msgs = _convert_messages(r.get("messages", []))
                if msgs is None:
                    n_skipped_struct += 1
                    continue
                if tok is not None:
                    ids = tok.apply_chat_template(msgs, tools=TOOL_SCHEMAS, tokenize=True)
                    if len(ids) > args.max_seq_length:
                        n_skipped_len += 1
                        continue
                row = {
                    "messages": msgs,
                    "tools": TOOL_SCHEMAS,
                    "source": f"tmax-glm-trajectories/{root.name}/{args.model_tag}",
                    "metadata": {
                        "task": task,
                        "episode": f"sol-{ep}",
                        "date": now,
                        "model": args.model_tag,
                        "num_success": s.get("num_success"),
                        "pass_at_1": k.get(1),
                        "success": bool(r.get("success")),
                        "over_budget": bool(r.get("over_budget", False)),
                    },
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"tasks scanned    : {len(summaries)}")
    print(f"rows written     : {n_rows} -> {args.out}")
    print(f"skipped: failures={n_skipped_fail} over-length={n_skipped_len} "
          f"malformed={n_skipped_struct}")


if __name__ == "__main__":
    main()
