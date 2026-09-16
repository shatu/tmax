#!/bin/bash

# TailSFT (arXiv:2608.25756) variant of sft_qwen35_9b_tmax_sft.sh.
# 1 node x 8 GPUs, 65536 seq len, effective batch 256 (1 per device x 32 accum x 8).
#
# TailSFT needs per-example losses, which with the liger fused CE only exist at
# per_device_train_batch_size=1 — so the 4x8 batch/accum of the standard script
# becomes 1x32 here (same effective batch). Filtering operates on the 8-example
# selection batch across ranks at each micro-step, relative to base-model losses
# pre-scored (and cached in output_dir) before training starts.
#
# Usage:
#   ./sft_qwen35_9b_tmax_sft_tailsft.sh [BEAKER_IMAGE] [FILTER_FRACTION] [FILTER_SCHEDULE]
# Paper-recommended configs: static 0.25, or ramp 0.5 (their math-domain winner).

BEAKER_IMAGE="${1:-shashankg/open-instruct-integration-test-omni_agent_cuda13-cuda13}"
FILTER_FRACTION="${2:-0.25}"
FILTER_SCHEDULE="${3:-static}"
MODEL="hamishivi/Qwen3.5-9B"
DATASET="allenai/tmax-sft-glm-52"
DATASET_CONFIG="all"
FRACTION_TAG="${FILTER_FRACTION//./}"

# Jupiter (H100, sm_90): matches this repo's Dockerfile (CUDA 12.8 + FA3 cu128
# wheels). The B300s on ai2/holmes need the separate cuda13 image lineage.
uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/oe-agents \
    --priority high \
    --image "$BEAKER_IMAGE" \
    --description "TMax TailSFT (${FILTER_SCHEDULE} f=${FILTER_FRACTION}) on hamishivi/Qwen3.5-9B with GLM 5.2 ${DATASET_CONFIG} rollouts" \
    --pure_docker_mode \
    --preemptible \
    --min_runtime 5h \
    --num_nodes 1 \
    --env BEAKER_ALLOW_SUBCONTAINERS=1 \
    --env BEAKER_SKIP_DOCKER_SOCKET=1 \
    --gpus 8 \
    --no_auto_dataset_cache \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name hamish_qwen35_9b_tmax_glm52_${DATASET_CONFIG}_tailsft_${FILTER_SCHEDULE}${FRACTION_TAG} \
    --wandb_project_name oe-general-agents \
    --model_name_or_path $MODEL \
    --tokenizer_name $MODEL \
    --use_liger_kernel \
    --max_seq_length 65536 \
    --sequence_parallel_size 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --tailsft_filter_fraction $FILTER_FRACTION \
    --tailsft_filter_schedule $FILTER_SCHEDULE \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list \
        $DATASET 1.0 \
    --dataset_mixer_list_config_names \
        $DATASET_CONFIG \
    --dataset_mixer_list_splits \
        train \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 42 \
    --output_dir /weka/oe-adapt-default/pradeepd/checkpoints \
    --checkpointing_steps 20 \
    --keep_last_n_checkpoints 1 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
