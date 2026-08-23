#!/bin/bash

# SFT for Qwen3.5-9B on all splits of the TMAX skill-tax no-tool-call SFT mix.
# 4 nodes x 8 GPUs = 32 GPUs, SP=2, 32k seq len.

BEAKER_IMAGE="${1:-shashankg/open-instruct-integration-test-omni_agent_cuda13-cuda13}"
MODEL="Qwen/Qwen3.5-9B"
DATASET="allenai/tmax-sft-glm-52"
DATASET_CONFIG="successful"

uv run python mason.py \
    --cluster ai2/holmes \
    --workspace ai2/oe-agents-holmes \
    --priority high \
    --image "$BEAKER_IMAGE" \
    --description "TMax SFT with GLM 5.2 successful rollouts" \
    --pure_docker_mode \
    --preemptible \
    --min_runtime 8h \
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
    --exp_name qwen35_9b_tmax_glm52_successful_sft \
    --wandb_project_name oe-general-agents \
    --model_name_or_path $MODEL \
    --tokenizer_name $MODEL \
    --use_liger_kernel \
    --max_seq_length 65536 \
    --sequence_parallel_size 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
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
