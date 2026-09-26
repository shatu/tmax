#!/usr/bin/env python
"""Convert a DeepSpeed ZeRO checkpoint *state* (``global_stepN/``) written by
open-instruct's ``--checkpoint_state_dir`` into a vLLM-loadable
``Qwen3_5ForConditionalGeneration`` (CG) checkpoint, in one CPU-only pass.

Why
---
open-instruct saves two kinds of RL checkpoints: HF-format ``step_N/`` saves
(``--save_freq``) and DeepSpeed *state* saves (``--checkpoint_state_freq``,
``global_stepN/`` under ``--checkpoint_state_dir``). The latter are the only
copy of the weights when a run was preempted between HF saves, and they are
not loadable by anything but DeepSpeed: the params live as fp32 flat
partitions inside ``bf16_zero_pp_rank_*_optim_states.pt`` (one file per
learner). DeepSpeed drops a ``zero_to_fp32.py`` helper next to them that
reassembles a full fp32 state dict; this script drives that helper and then
does what ``convert_qwen35_causallm_to_cg.py`` does for HF saves:

  1. ``zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(state_dir, tag)``
     -> fp32 state dict with the *in-memory* parameter names of the trained
     ``Qwen3_5ForCausalLM`` (``model.layers.*``, ``model.embed_tokens.*``,
     ``model.norm.*``, ``lm_head.weight``);
  2. cast to bf16 and rename to the CG on-disk layout
     (``model.language_model.*`` + ``lm_head.weight``);
  3. graft those tensors onto the donor's vision tower / mtp weights, copy the
     donor's CG config + processor files and the *init model's* tokenizer +
     chat template (RL does not change the tokenizer), write sharded
     safetensors following the donor's index.

The state dir is treated as READ-ONLY. Output goes to ``--out``.

Usage (0-GPU Beaker job; needs ~3x the fp32 model size in RAM, so ask for
>= 256 GiB for a 9B model)::

    python scripts/beaker/convert_ds_state_to_cg.py \
        --state-dir /weka/.../deletable_checkpoint_states/<exp_name> \
        --tag global_step504 \
        --init-model hamishivi/Qwen3.5-9B \
        --donor hamishivi/Qwen3.5-9B \
        --out /weka/.../<exp_name>_step504_cg
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# CG on-disk layout (what vLLM's Qwen3_5ForConditionalGeneration loader expects).
TEXT_PREFIX = "model.language_model."
TEXT_EXACT = ("lm_head.weight",)
CRITICAL_DIMS = ("hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads")
DONOR_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
)
INIT_MODEL_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)
COMPLETE_MARKER = "CONVERSION_COMPLETE"

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_text_key(key: str) -> bool:
    return key in TEXT_EXACT or key.startswith(TEXT_PREFIX)


def to_cg_key(key: str) -> str:
    """Map an in-memory Qwen3_5ForCausalLM param name to the CG on-disk name."""
    if key in TEXT_EXACT or key.startswith(TEXT_PREFIX):
        return key
    if key.startswith("model.visual.") or key.startswith("mtp."):
        return key
    if key.startswith("model."):
        return TEXT_PREFIX + key[len("model."):]
    return key


def resolve_model_dir(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download

    log(f"'{name_or_path}' is not a local dir; fetching via HF hub (HF_HOME={os.environ.get('HF_HOME', '<default>')})")
    return snapshot_download(name_or_path)


def load_zero_to_fp32(state_dir: str):
    path = os.path.join(state_dir, "zero_to_fp32.py")
    if not os.path.exists(path):
        sys.exit(f"!!! {path} not found; is --state-dir the directory that holds global_step*/ and 'latest'?")
    spec = importlib.util.spec_from_file_location("zero_to_fp32", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:
        # The helper is written by the DeepSpeed version that trained the run and
        # imports constants from it. A slightly older DeepSpeed lacks the newer
        # (AutoEP / MoE-only) constant names, which never occur in a dense
        # checkpoint, so stub them and retry rather than require an exact match.
        m = re.search(r"cannot import name '(\w+)'", str(e))
        if not m:
            raise
        import deepspeed.checkpoint.constants as c
        names = re.findall(r"\b(AUTOEP_[A-Z0-9_]+)\b", open(path).read())
        for n in sorted(set(names)):
            if not hasattr(c, n):
                setattr(c, n, f"__stub_{n.lower()}__")
                log(f"stubbed missing deepspeed constant {n} (installed deepspeed is older than the run's)")
        spec.loader.exec_module(mod)
    return mod


def check_configs(init_dir: str, donor_dir: str) -> dict:
    init_cfg = json.load(open(os.path.join(init_dir, "config.json")))
    donor_cfg = json.load(open(os.path.join(donor_dir, "config.json")))
    init_text = init_cfg.get("text_config", init_cfg)
    donor_text = donor_cfg.get("text_config", donor_cfg)
    mismatches = [(k, init_text.get(k), donor_text.get(k)) for k in CRITICAL_DIMS
                  if init_text.get(k) is not None and donor_text.get(k) is not None
                  and init_text.get(k) != donor_text.get(k)]
    if mismatches:
        for k, s, d in mismatches:
            print(f"!!! config dim mismatch {k}: init={s} donor={d}")
        sys.exit("!!! donor is not the same base as the init model; aborting.")
    log(f"config dims match init/donor: {{ {', '.join(f'{k}={donor_text.get(k)}' for k in CRITICAL_DIMS)} }}")
    return init_text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", required=True, help="open-instruct --checkpoint_state_dir (holds global_step*/, latest, zero_to_fp32.py)")
    ap.add_argument("--tag", default=None, help="which global_stepN to convert (default: contents of 'latest')")
    ap.add_argument("--init-model", required=True, help="the run's --model_name_or_path: source of tokenizer + chat template (local dir or HF id)")
    ap.add_argument("--donor", default="hamishivi/Qwen3.5-9B", help="canonical CG checkpoint of the same base (local dir or HF id)")
    ap.add_argument("--out", required=True, help="output dir for the converted checkpoint")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = ap.parse_args()

    state_dir = os.path.realpath(args.state_dir)
    out = os.path.realpath(args.out)
    if out == state_dir or out.startswith(state_dir + os.sep):
        sys.exit("!!! --out must not be inside --state-dir")
    if os.path.exists(out):
        if os.path.exists(os.path.join(out, COMPLETE_MARKER)) and not args.force:
            log(f"{out} already has {COMPLETE_MARKER}; nothing to do (use --force to redo)")
            return
        if not args.force:
            sys.exit(f"!!! --out exists (incomplete): {out} (use --force to overwrite).")
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    tag = args.tag
    if tag is None:
        with open(os.path.join(state_dir, "latest")) as f:
            tag = f.read().strip()
    step_dir = os.path.join(state_dir, tag)
    if not os.path.isdir(step_dir):
        sys.exit(f"!!! {step_dir} does not exist")
    n_optim = len(glob.glob(os.path.join(step_dir, "*optim_states.pt")))
    n_model = len(glob.glob(os.path.join(step_dir, "*model_states.pt")))
    log(f"state  : {step_dir}  ({n_optim} optim_states, {n_model} model_states files)")
    try:
        import deepspeed
        ms = sorted(glob.glob(os.path.join(step_dir, "*model_states.pt")))[0]
        ds_version = torch.load(ms, map_location="cpu", weights_only=False).get("ds_version")
        log(f"deepspeed: checkpoint written by {ds_version}, installed {deepspeed.__version__}")
    except Exception as e:  # noqa: BLE001
        log(f"could not read ds_version: {e}")
    if n_optim == 0:
        sys.exit("!!! no *optim_states.pt files: not a ZeRO checkpoint (or incomplete save)")

    init_dir = resolve_model_dir(args.init_model)
    donor_dir = resolve_model_dir(args.donor)
    log(f"init   : {init_dir}")
    log(f"donor  : {donor_dir}")
    log(f"out    : {out}")
    init_text_cfg = check_configs(init_dir, donor_dir)

    donor_index_path = os.path.join(donor_dir, "model.safetensors.index.json")
    if not os.path.exists(donor_index_path):
        sys.exit(f"!!! donor has no model.safetensors.index.json ({donor_dir})")
    weight_map = json.load(open(donor_index_path))["weight_map"]
    shards = sorted(set(weight_map.values()))
    donor_text_keys = {k for k in weight_map if is_text_key(k)}
    log(f"donor: {len(weight_map)} tensors in {len(shards)} shards, {len(donor_text_keys)} text tensors")

    # ---- 1. reassemble fp32 params from the ZeRO partitions -------------------
    z2f = load_zero_to_fp32(state_dir)
    log(f"running zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(tag={tag}) ...")
    t0 = time.time()
    fp32_sd = z2f.get_fp32_state_dict_from_zero_checkpoint(state_dir, tag=tag)
    log(f"reassembled {len(fp32_sd)} tensors in {time.time() - t0:.0f}s")
    sample = list(fp32_sd.keys())[:3]
    log(f"sample param names: {sample}")

    # ---- 2. cast + rename into the CG layout --------------------------------------
    dtype = DTYPES[args.dtype]
    ours: dict[str, torch.Tensor] = {}
    for k in list(fp32_sd.keys()):
        t = fp32_sd.pop(k)
        if hasattr(t, "contiguous") and not isinstance(t, torch.Tensor):  # lazy GatheredTensor in newer deepspeed
            t = t.contiguous()
        ours[to_cg_key(k)] = t.to(dtype).contiguous()
        del t
    del fp32_sd

    if "lm_head.weight" not in ours:
        embed_key = TEXT_PREFIX + "embed_tokens.weight"
        if init_text_cfg.get("tie_word_embeddings") and embed_key in ours:
            log("lm_head.weight absent and tie_word_embeddings=true: reusing embed_tokens as lm_head")
            ours["lm_head.weight"] = ours[embed_key]
    missing = sorted(donor_text_keys - set(ours))
    extra = sorted(set(ours) - donor_text_keys)
    if missing:
        sys.exit(f"!!! converted state is missing {len(missing)} text tensors the donor expects, e.g. {missing[:5]}")
    if extra:
        log(f"note: {len(extra)} tensors not in the donor text set are dropped, e.g. {extra[:5]}")
        for k in extra:
            del ours[k]

    # ---- 3. write shards: our text weights, donor vision/mtp weights ---------------
    swapped = kept = 0
    for shard in shards:
        out_tensors = {}
        with safe_open(os.path.join(donor_dir, shard), framework="pt") as st:
            for k in st.keys():
                if k in ours:
                    dshape = tuple(st.get_slice(k).get_shape())
                    if tuple(ours[k].shape) != dshape:
                        sys.exit(f"!!! shape mismatch for {k}: ours={tuple(ours[k].shape)} donor={dshape}")
                    out_tensors[k] = ours.pop(k)
                    swapped += 1
                else:
                    out_tensors[k] = st.get_tensor(k).contiguous()
                    kept += 1
        save_file(out_tensors, os.path.join(out, shard), metadata={"format": "pt"})
        log(f"wrote {shard}: {len(out_tensors)} tensors")
        del out_tensors
    if swapped != len(donor_text_keys):
        sys.exit(f"!!! expected to swap {len(donor_text_keys)} text tensors, swapped {swapped}")
    log(f"swapped(trained text)={swapped}  kept(donor vision/mtp)={kept}")

    shutil.copy(donor_index_path, os.path.join(out, "model.safetensors.index.json"))
    for f in DONOR_FILES:
        p = os.path.join(donor_dir, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out, f))
            log(f"copied donor:{f}")
    for f in INIT_MODEL_FILES:
        p = os.path.join(init_dir, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out, f))
            log(f"copied init-model:{f}")

    # ---- 4. sanity + provenance ----------------------------------------------------
    out_cfg = json.load(open(os.path.join(out, "config.json")))
    arch = out_cfg.get("architectures")
    log(f"output architectures: {arch}  model_type: {out_cfg.get('model_type')}")
    if arch != ["Qwen3_5ForConditionalGeneration"]:
        log("!!! WARNING: output architecture is not Qwen3_5ForConditionalGeneration; vLLM may not load it")
    n_out = 0
    for shard in shards:
        with safe_open(os.path.join(out, shard), framework="pt") as st:
            n_out += len(list(st.keys()))
    assert n_out == len(weight_map), f"tensor count {n_out} != donor index {len(weight_map)}"
    info = {
        "state_dir": state_dir, "tag": tag, "init_model": args.init_model, "donor": args.donor,
        "dtype": args.dtype, "n_optim_partitions": n_optim, "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                     cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None,
    }
    json.dump(info, open(os.path.join(out, "conversion_info.json"), "w"), indent=2)
    open(os.path.join(out, COMPLETE_MARKER), "w").write(info["converted_at"] + "\n")
    log(f"DONE. Converted checkpoint at: {out}")
    log("serve with a --name WITHOUT 'ada' in it and WITHOUT --language-model-only (it is a CG checkpoint).")


if __name__ == "__main__":
    main()
