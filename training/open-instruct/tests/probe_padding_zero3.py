"""torchrun GPU probe: real NCCL/SP groups and ZeRO-3 on a tiny token-local model.

Uses production pack collation, tiled DPPO kernels and the trainer's exact scale
statements. The token-local model isolates loss reduction; it is not a Qwen/Ray
or attention-all-to-all integration test.
"""

import __future__

import ast
import copy
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from deepspeed.runtime.sequence_parallel import parallel_state_sp as mpu
from deepspeed.utils import safe_get_full_grad
from test_packed_batch_padding import production_namespace


def trainer_scale(ns):
    path = Path(os.environ.get("PROBE_SCALE_SOURCE", Path(__file__).parents[1] / "open_instruct/grpo_fast.py"))
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "PolicyTrainerRayProcess"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_compute_tiled_dapo_loss")
    start = next(
        i
        for i, n in enumerate(method.body)
        if (isinstance(n, ast.If) and ast.unparse(n.test) == "loss_denominator_mode == 'sequence'")
        or (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "scale")
    )
    end = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "tiled_outputs"
    )
    fn = ast.parse(
        "def scale(self, response_mask, loss_denominator, loss_denominator_mode, rollout_sample_ids): pass"
    ).body[0]
    fn.body = method.body[start:end] + [ast.Return(ast.Name("scale", ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, str(path), "exec", __future__.annotations.compiler_flag), ns)
    return ns["scale"]


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 4)
        self.lm_head = torch.nn.Linear(4, 5, bias=False)

    def forward(self, x):
        return self.backbone(x)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    sp = int(os.environ.get("PROBE_SP", "1"))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=4))
    deepspeed.init_distributed()
    if sp > 1:
        mpu.initialize_sequence_parallel(sequence_parallel_size=sp)
    group = mpu.get_sequence_parallel_group() if sp > 1 else None
    dp = world // sp
    ns = production_namespace()
    ns["dist"] = dist
    # Pull the already-reviewed scaling helpers when present on the tested tree.
    path = Path(__file__).parents[1] / "open_instruct/grpo_utils.py"
    helpers = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"tiled_grpo_loss_scale", "deepspeed_gradient_reduction_divisor"}
    ]
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), str(path), "exec", __future__.annotations.compiler_flag), ns
    )
    ns["grpo_utils"] = SimpleNamespace(**ns)
    scale_fn = trainer_scale(ns)
    torch.manual_seed(8)
    model = Tiny().to(device)
    reference = copy.deepcopy(model)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": dp,
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": False,
            "reduce_bucket_size": 10000,
            "stage3_param_persistence_threshold": 0,
        },
        "sequence_parallel_size": sp,
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "steps_per_print": 1000000,
    }
    engine, _, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=config, mpu=mpu if sp > 1 else None
    )
    context = SimpleNamespace(
        args=SimpleNamespace(world_size=world, sequence_parallel_size=sp, deepspeed_stage=3), _sp_group=group
    )
    results = []
    for count in (1, 7, 13):
        for mode in ("token", "sequence"):
            engine.zero_grad()
            reference.zero_grad()
            # No optimizer update: every case compares the same fixed weights.
            reference.load_state_dict(initial)
            np.random.seed(11)
            packed = ns["pack_sequences"](
                queries=[[1]] * count,
                responses=[[2, 3, 4, 2, 3, 4, 2, 3]] * count,
                masks=[[1] * 8] * count,
                pack_length=9,
                pad_token_id=0,
                vllm_logprobs=[[-6.0, -1.5] * 4] * count,
                rollout_sample_ids=list(range(count)),
            )
            packed.advantages = [(m.float() * 0.03) for m in packed.response_masks]
            # Uneven real response masks, including SP shards with no responses.
            for i, mask in enumerate(packed.response_masks):
                mask[1 + (i % 8 + 1) :] = 0
            workers = ns["prepare_collated_data_for_workers"](packed, dp, 1, 0, pin_memory=False)
            full_weights = []
            for i, mask in enumerate(packed.response_masks):
                valid = mask[1:].bool().to(device)[None]
                ids = packed.rollout_sample_ids[i][1:].to(device)[None]
                weights = valid.float() if mode == "token" else ns["_sequence_loss_weights"](valid, ids)[0]
                full_weights.append(weights)
            denominator = sum(w.sum().item() for w in full_weights)
            expected_loss = torch.zeros((), device=device)
            for i, weights in enumerate(full_weights):
                x = torch.arange(24, device=device).reshape(1, 8, 3).float() / 24 + i / 100
                logits = reference.lm_head(reference(x))
                labels = packed.query_responses[i][1:].to(device)[None]
                lp = logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
                adv = packed.advantages[i][1:].to(device)[None]
                old = packed.vllm_logprobs[i][1:].to(device)[None]
                coefficient = (lp - old).clamp(-20, 20).exp().clamp(max=10).detach()
                expected_loss = expected_loss + (-adv * coefficient * lp * weights).sum() / denominator
            expected_loss.backward()
            worker = workers[rank // sp]
            # Extract a pack's source identity before slicing so token-local features match reference.
            for i in range(len(worker)):
                ids = worker.rollout_sample_ids[i][:, 1:].to(device)
                origin = max(int(ids.max()), 0)
                x = torch.arange(24, device=device).reshape(1, 8, 3).float() / 24 + origin / 100
                sl = slice((rank % sp) * (8 // sp), (rank % sp + 1) * (8 // sp))
                x = x[:, sl]
                mask = worker.response_masks[i][:, 1:].bool().to(device)[:, sl]
                ids = ids[:, sl]
                labels = worker.query_responses[i][:, 1:].to(device)[:, sl]
                adv = worker.advantages[i][:, 1:].to(device)[:, sl]
                old = worker.vllm_logprobs[i][:, 1:].to(device)[:, sl]
                engine.set_gradient_accumulation_boundary(i == len(worker) - 1)
                hidden = engine(x)
                scale = scale_fn(context, mask, denominator, mode, ids)
                loss, *_ = ns["tiled_grpo_lm_head_loss"](
                    lm_head=engine.module.lm_head,
                    hidden_states=hidden,
                    selected_token_ids=labels,
                    response_mask=mask,
                    advantages=adv,
                    old_logprobs=old,
                    ref_logprobs=None,
                    temperature=1.0,
                    beta=0.0,
                    clip_lower=0.2,
                    clip_higher=0.2,
                    shards=2,
                    loss_scale=scale,
                    loss_fn="dppo",
                    dppo_divergence_threshold=100,
                    loss_denominator=mode,
                    rollout_sample_ids=ids,
                    sequence_process_group=group,
                )
                assert torch.isfinite(loss), (count, mode, rank, loss)
                engine.backward(loss)
            errors = {}
            ratios = {}
            for (name, param), refparam in zip(engine.module.named_parameters(), reference.parameters(), strict=True):
                grad = safe_get_full_grad(param)
                assert torch.isfinite(grad).all(), (count, mode, rank, name)
                errors[name] = (grad - refparam.grad).abs().max().item()
                ratios[name] = (grad.norm() / refparam.grad.norm()).item()
            record = dict(
                count=count,
                sp=sp,
                dp=dp,
                mode=mode,
                max_errors=errors,
                norm_ratios=ratios,
                passed=max(errors.values()) < 2e-5,
            )
            # Complete the ZeRO step to reset partitioned accumulation buffers.
            engine.step()
            results.append(record)
            if rank == 0:
                print("PADDING_RESULT " + json.dumps(record), flush=True)
    if rank == 0:
        print(
            "PADDING_SUMMARY "
            + json.dumps(
                dict(
                    torch=torch.__version__,
                    deepspeed=deepspeed.__version__,
                    world=world,
                    sp=sp,
                    passed=all(r["passed"] for r in results),
                    results=results,
                )
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    passed = all(r["passed"] for r in results)
    expect_mismatch = os.environ.get("PROBE_EXPECT_MISMATCH") == "1"
    if passed == expect_mismatch:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
