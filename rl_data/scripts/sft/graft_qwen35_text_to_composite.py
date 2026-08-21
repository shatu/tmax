"""Graft a text-only Qwen3.5 SFT checkpoint back into the composite format.

Why: transformers >= 5.4 loads the composite ``Qwen/Qwen3.5-*`` models
(``Qwen3_5ForConditionalGeneration``, text+vision) as a causal LM and saves
the TEXT SLICE (``Qwen3_5ForCausalLM`` + ``Qwen3_5TextConfig``). No released
vLLM can serve that shape: the native registry only knows the composite
(text-arch support is on vllm main, unreleased), and vLLM's generic
transformers backend rejects the class as incompatible. Grafting the trained
text weights back into the composite produces a checkpoint every vLLM since
0.19 serves natively — exactly like the base model (use --language_model_only).

Key mapping (verified against transformers 5.4 on meta device — 427/427 text
keys map, the other 333 composite keys are the vision tower, taken from base):

    model.<X>    -> model.language_model.<X>
    lm_head.*    -> lm_head.*   (unchanged)

Usage (CPU-only; needs ~40 GB RAM for two bf16 9B models)::

    uv run python rl_data/scripts/sft/graft_qwen35_text_to_composite.py \\
        --checkpoint /weka/.../checkpoints/qwen35_9b_tmax_glm52_successful_sft \\
        --out        /weka/.../checkpoints/qwen35_9b_tmax_glm52_successful_sft_composite

The tokenizer/chat template are copied from the TRAINED checkpoint (that's
what SFT baked in); processor/vision files come from the base repo.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="Trained text-only checkpoint dir")
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B",
                    help="Composite base model the checkpoint was trained from")
    ap.add_argument("--out", required=True, help="Output dir for the composite checkpoint")
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

    print(f"loading base composite: {args.base}")
    comp = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cpu"
    )
    print(f"loading trained text checkpoint: {args.checkpoint}")
    text = Qwen3_5ForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map="cpu"
    )

    remapped = {}
    for k, v in text.state_dict().items():
        if k.startswith("model."):
            remapped["model.language_model." + k[len("model."):]] = v
        else:
            remapped[k] = v  # lm_head.*

    missing, unexpected = comp.load_state_dict(remapped, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys after remap (mapping drifted?): {unexpected[:5]}")
    non_vision_missing = [k for k in missing if not k.startswith("model.visual")]
    if non_vision_missing:
        raise SystemExit(f"missing non-vision keys (mapping drifted?): {non_vision_missing[:5]}")
    print(f"grafted {len(remapped)} tensors; base retains {len(missing)} vision tensors")

    # Sanity: a trained tensor must have actually replaced the base one.
    probe = "model.language_model.layers.0.input_layernorm.weight"
    assert torch.equal(
        comp.state_dict()[probe], text.state_dict()["model.layers.0.input_layernorm.weight"]
    ), "probe tensor mismatch — graft did not take"

    out = Path(args.out)
    print(f"saving composite checkpoint -> {out}")
    comp.save_pretrained(out, safe_serialization=True)

    # Tokenizer + chat template from the TRAINED checkpoint (what SFT used).
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    tok.save_pretrained(out)
    # Vision/processor plumbing from the base repo so the composite is complete.
    try:
        AutoProcessor.from_pretrained(args.base).save_pretrained(out)
        # save_pretrained(processor) may overwrite tokenizer files with the
        # base ones — restore the trained tokenizer's files on top.
        tok.save_pretrained(out)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: processor copy failed ({e}); vLLM --language_model_only may not need it")

    # generation_config from the trained checkpoint if present.
    gen_cfg = Path(args.checkpoint) / "generation_config.json"
    if gen_cfg.exists():
        shutil.copy(gen_cfg, out / "generation_config.json")

    print("done. Serve with the NATIVE vLLM path, e.g.:")
    print(f"  vllm serve {out} --language_model_only ...")


if __name__ == "__main__":
    main()
