#!/usr/bin/env bash
# DPPO training of Qwen3.5-9B on swerl-tmax-15k with the VMVM remote sandbox.
#
# Port of hamishivi/tmax scripts/tmax/RL/qwen35_9b.sh (Beaker) to this Slurm
# cluster, using the VMVM backend (vacli-leased VMs + podman) instead of
# Beaker/Docker. Sandbox images are pulled inside the leased VM from
# vmvm-registry.fbinfra.net/rulin-swerl-tmax-v3/<hash>:latest (mirrored by
# scripts in /checkpoint/comem/rulin/vmvm_upload/).
#
# Submit with sbatch so the run survives a Cursor/SSH logout:
#   sbatch scripts/train/debug/envs/swerl_vanillux_tmax15k_vmvm_dppo_slurm.sh
# Override knobs at submit time, e.g.:
#   sbatch --nodes=8 --export=ALL,WITH_X2P=1,POOL_SIZE=256 \
#     scripts/train/debug/envs/swerl_vanillux_tmax15k_vmvm_dppo_slurm.sh
#
#SBATCH --job-name=swerl-tmax15k-vmvm-dppo-9b
#SBATCH --account=comem
#SBATCH --qos=h100_comem_high
#SBATCH --partition=h100
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600g

#SBATCH --time=7-00:00:00
# NOTE: intentionally NO `#SBATCH --export=...`. After the cgroup-v2 upgrade, any explicit
# --export (even `--export=ALL,VAR=..`) makes Slurm run user-env retrieval, which fails and
# holds the job (Reason=user_env_retrieval_failed_requeued_held). Submit instead from a shell
# that has `export`ed the needed vars (WITH_X2P=1, RUN_ID, EXP_NAME, hyperparams, ...); with no
# --export, sbatch propagates the submit-shell env by default (the working path).
#SBATCH --output=/checkpoint/comem/rulin/logs/swerl_tmax15k_vmvm_dppo/%x-%j.out
#SBATCH --error=/checkpoint/comem/rulin/logs/swerl_tmax15k_vmvm_dppo/%x-%j.err

set -euo pipefail

# sbatch copies this script to /var/spool/slurmd, so BASH_SOURCE no longer points
# at the repo. Pin REPO_ROOT explicitly and source helpers by absolute path.
export REPO_ROOT="${REPO_ROOT:-/checkpoint/comem/rulin/open-instruct}"
SCRIPT_DIR="${REPO_ROOT}/scripts/train/debug/envs"
source "${SCRIPT_DIR}/checkpoint_env.sh"

mkdir -p /checkpoint/comem/rulin/logs/swerl_tmax15k_vmvm_dppo

# Drop any inherited HTTP proxy. sbatch propagates the submit shell's environment by
# default (we deliberately pass no --export, see the note above), so a proxy that is only
# reachable from wherever the job was submitted -- a dev container, a tunnel, an agent
# sandbox -- follows the job onto the compute nodes and breaks egress there. The symptom
# is a run that starts, prints TOOL_CONFIGS, then hangs forever in
# "wandb: Network error (ProxyError)" retries with no further output. Compute nodes reach
# wandb/HF directly, so unsetting is always correct here. scripts/eval_vmvm/serve_and_eval.sh
# already does the same.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy 2>/dev/null || true

# ---------------------------------------------------------------------------
# Offline HF cache that holds the model, prompt dataset, and per-task data.
# (cache/huggingface, not the checkpoint_env.sh default hf-cache.)
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME_OVERRIDE:-/checkpoint/comem/rulin/cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# ---------------------------------------------------------------------------
# Run configuration (override via sbatch --export=ALL,KEY=VALUE,...)
# ---------------------------------------------------------------------------
NODES="${SLURM_NNODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

DATASET_JSONL="${DATASET_JSONL:-/checkpoint/comem/rulin/vmvm_upload/swerl_tmax_15k_vmvm.jsonl}"
TASK_DATA_HF_REPO="${TASK_DATA_HF_REPO:-hamishivi/swerl-tmax-15k}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-hamishivi/Qwen3.5-9B}"

EXP_NAME="${EXP_NAME:-swerl_qwen35_9b_fp32lm_dppo_vmvm}"
RUN_ID="${RUN_ID:-${EXP_NAME}__seed42__$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/runs/${RUN_ID}}"
ROLLOUTS_SAVE_PATH="${ROLLOUTS_SAVE_PATH:-${REPO_ROOT}/output/rollouts/${RUN_ID}}"
CHECKPOINT_STATE_DIR="${CHECKPOINT_STATE_DIR:-${REPO_ROOT}/output/checkpoint_states/${RUN_ID}}"
CHECKPOINT_STATE_FREQ="${CHECKPOINT_STATE_FREQ:-10}"
KEEP_LAST_N_CHECKPOINTS="${KEEP_LAST_N_CHECKPOINTS:-3}"

# Reference DPPO hyper-parameters (qwen35_9b.sh).
MAX_PROMPT_TOKEN_LENGTH="${MAX_PROMPT_TOKEN_LENGTH:-2048}"
PER_TURN_MAX_TOKENS="${PER_TURN_MAX_TOKENS:-16384}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-65536}"
PACK_LENGTH="${PACK_LENGTH:-67584}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
NUM_UNIQUE_PROMPTS_ROLLOUT="${NUM_UNIQUE_PROMPTS_ROLLOUT:-8}"
NUM_SAMPLES_PER_PROMPT_ROLLOUT="${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-32}"
ASYNC_STEPS="${ASYNC_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-adamw}"
SGD_MOMENTUM="${SGD_MOMENTUM:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
TOTAL_EPISODES="${TOTAL_EPISODES:-128000}"
DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-3}"
SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
GRPO_NUM_LEARNERS_PER_NODE="${GRPO_NUM_LEARNERS_PER_NODE:-8 8}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-48}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
BETA="${BETA:-0.0}"
USE_VLLM_LOGPROBS="${USE_VLLM_LOGPROBS:-true}"
TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP="${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP:-0.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-64}"
VERIFICATION_REWARD="${VERIFICATION_REWARD:-1.0}"
TOOL_PARSER_TYPE="${TOOL_PARSER_TYPE:-vllm_qwen3_xml}"
BACKEND_TIMEOUT="${BACKEND_TIMEOUT:-1200}"
SAVE_FREQ="${SAVE_FREQ:-20}"
LOCAL_EVAL_EVERY="${LOCAL_EVAL_EVERY:-10}"

# vLLM / Qwen3.5 state-model + fp32-head optimizations.
VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
LM_HEAD_FP32="${LM_HEAD_FP32:-true}"
LIGER_GRPO_LOSS_CHUNK_SIZE="${LIGER_GRPO_LOSS_CHUNK_SIZE:-8}"
LOSS_FN="${LOSS_FN:-dppo}"
DPPO_DIVERGENCE_TYPE="${DPPO_DIVERGENCE_TYPE:-tv}"
DPPO_DIVERGENCE_THRESHOLD="${DPPO_DIVERGENCE_THRESHOLD:-0.1}"
# PPO/CISPO clip bounds + per-token TIS mask (config defaults: 0.2 / 0.272 / 0 / 0).
CLIP_LOWER="${CLIP_LOWER:-0.2}"
CLIP_HIGHER="${CLIP_HIGHER:-0.272}"
TIS_MASK_LOWER="${TIS_MASK_LOWER:-0.0}"
TIS_MASK_UPPER="${TIS_MASK_UPPER:-0.0}"
ADVANTAGE_NORMALIZATION_TYPE="${ADVANTAGE_NORMALIZATION_TYPE:-centered}"
# Geometric-mean sequence ratio mask (geomean_mask feature). Empty = disabled.
# When set, whole responses whose geomean(π_θ/π_rollout) falls outside
# [lower, upper] are masked in both the pg loss and KL. Requires TIS cap = 0.
SEQUENCE_TIS_MASK_LOWER="${SEQUENCE_TIS_MASK_LOWER:-}"
SEQUENCE_TIS_MASK_UPPER="${SEQUENCE_TIS_MASK_UPPER:-}"
# Reference-policy KL constraint (TR-DPO-style). Empty = use code defaults / disabled.
# beta>0 turns on the KL penalty (needs load_ref_policy=True, the default). alpha is the
# polyak coefficient for ref updates (alpha=1.0 => hard copy of current weights).
# ref_policy_update_freq = update the ref every N steps.
ALPHA="${ALPHA:-}"
REF_POLICY_UPDATE_FREQ="${REF_POLICY_UPDATE_FREQ:-}"
# Entropy monitoring + policy-checkpoint capture window (empty = off).
# Entropy tracking is ON by default for all runs; disable by exporting RECORD_ENTROPY= (empty).
RECORD_ENTROPY="${RECORD_ENTROPY-1}"
CAPTURE_CHECKPOINT_WINDOW="${CAPTURE_CHECKPOINT_WINDOW:-}"

# VMVM sandbox knobs. pool_size is the number of concurrent env actors -> one
# leased VM each. Must be >= the rollout batch (num_unique_prompts_rollout *
# num_samples_per_prompt_rollout = 256 here) plus async headroom, or rollouts
# starve waiting for a free actor (acquire_reset_pools dominates and they time
# out). 512 = 64 leases/node on 8 nodes (top of the vmvm sweet spot) and matches
# the reference DPPO run.
POOL_SIZE="${POOL_SIZE:-512}"
SWERL_VMVM_TENANT="${SWERL_VMVM_TENANT:-async_2881758}"
SWERL_VMVM_TTL="${SWERL_VMVM_TTL:-1200s}"
SWERL_VMVM_NETWORK="${SWERL_VMVM_NETWORK:-host}"
SWERL_SANDBOX_TIMEOUT="${SWERL_SANDBOX_TIMEOUT:-120}"
SWERL_SANDBOX_TEST_TIMEOUT="${SWERL_SANDBOX_TEST_TIMEOUT:-120}"

RAY_NUM_CPUS="${RAY_NUM_CPUS:-192}"
GRPO_TRAINER_CPUS_PER_GPU="${GRPO_TRAINER_CPUS_PER_GPU:-8}"
SWERL_SANDBOX_TIMING_LOGS="${SWERL_SANDBOX_TIMING_LOGS:-1}"
SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S="${SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S:-1.0}"

WANDB_PROJECT="${WANDB_PROJECT:-open-instruct-terminal-rl}"
WANDB_ENTITY="${WANDB_ENTITY:-rulin}"

if [ ! -f "${DATASET_JSONL}" ]; then
    echo "Missing VMVM dataset jsonl: ${DATASET_JSONL}" >&2
    echo "Generate it with scripts/data/build_swerl_tmax_vmvm_jsonl.py" >&2
    exit 1
fi
if [ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]; then
    echo "Missing training Python environment at ${UV_PROJECT_ENVIRONMENT}" >&2
    exit 1
fi

echo "=== swerl-tmax-15k VMVM DPPO launch ==="
echo "nodes=${NODES} gpus/node=${GPUS_PER_NODE} learners/node='${GRPO_NUM_LEARNERS_PER_NODE}' vllm_engines=${VLLM_NUM_ENGINES} sp=${SEQUENCE_PARALLEL_SIZE}"
echo "dataset=${DATASET_JSONL}"
echo "task_data_hf_repo=${TASK_DATA_HF_REPO}"
echo "model=${MODEL_NAME_OR_PATH}"
echo "run_id=${RUN_ID} output_dir=${OUTPUT_DIR}"
echo "loss_fn=${LOSS_FN} dppo(${DPPO_DIVERGENCE_TYPE}, thr=${DPPO_DIVERGENCE_THRESHOLD}) lm_head_fp32=${LM_HEAD_FP32}"
echo "pool_size=${POOL_SIZE} backend=${SWERL_SANDBOX_BACKEND:-unset}"
echo "WITH_X2P=${WITH_X2P:-unset}"

srun \
    --ntasks-per-node 1 \
    --gpus-per-node "${GPUS_PER_NODE}" \
    --cpus-per-task "${RAY_NUM_CPUS}" \
    --kill-on-bad-exit=1 \
    bash -lc '
        set -euo pipefail
        cd "'"${REPO_ROOT}"'"
        source scripts/train/debug/envs/checkpoint_env.sh
        export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

        export HF_HOME="'"${HF_HOME}"'"
        export HF_HUB_CACHE="'"${HF_HUB_CACHE}"'"
        export HF_DATASETS_CACHE="'"${HF_DATASETS_CACHE}"'"
        export HF_HUB_OFFLINE="'"${HF_HUB_OFFLINE}"'"
        export TRANSFORMERS_OFFLINE="'"${TRANSFORMERS_OFFLINE}"'"
        export HF_DATASETS_OFFLINE="'"${HF_DATASETS_OFFLINE}"'"
        # NCCL 2.27.5 RAS -- the out-of-band diagnostics monitor -- has a memory-safety
        # bug: when any peer leaves the RAS network it can race its own timeout handlers
        # against a bsearch over the peers array and SIGSEGV
        # [ras/peers.cc:862 ncclSocketsCompare <- bsearch <- ras/client_support.cc:984].
        # In job 10820355 one vLLM engine of 48 died and that segfaulted ~13 learner
        # processes across all 8 nodes: one engine failure took down the whole fleet.
        # RAS is diagnostics only, so disabling it costs training nothing and removes
        # the blast radius.
        # SWERL_RESET_FAILURE_ZERO_REWARD is deliberately LEFT OFF.
        # It converts a failed sandbox reset into a zero-reward rollout instead of
        # crashing the job, which is right for an isolated transient -- but catastrophic
        # during a registry outage. Enabling it on 2026-08-21 did exactly that: while
        # vmvm-registry was down, run 10833021 logged 4140 "marking rollout as zero
        # reward" events (the pre-outage run logged 0), percent_solved_mean collapsed
        # 0.70 -> 0.14 the instant it resumed, and the policy trained on ~garbage reward
        # for 24 steps. Crash-fast + watchdog restart loses an hour; this loses the run
        # silently, which is far worse. Re-enable only with an outage guard (e.g. abort
        # if the zero-reward-reset rate over a window exceeds a few percent).
        export NCCL_RAS_ENABLE=0
        # Lets swerl_vanillux_sandbox._prefer_local_sif() map a docker-hub image ref to
        # the prebuilt .sif pool instead of pulling. Harmless under the vmvm backend.
        export SWERL_APPTAINER_SIF_DIR="${SWERL_APPTAINER_SIF_DIR:-/checkpoint/comem/rulin/cache/apptainer-sifs}"
        export SWERL_SANDBOX_BACKEND="${SWERL_SANDBOX_BACKEND:-vmvm}"
        export WITH_X2P=1
        export SEQUENCE_TIS_MASK_LOWER="'"${SEQUENCE_TIS_MASK_LOWER}"'"
        export SEQUENCE_TIS_MASK_UPPER="'"${SEQUENCE_TIS_MASK_UPPER}"'"
        export ALPHA="'"${ALPHA}"'"
        export REF_POLICY_UPDATE_FREQ="'"${REF_POLICY_UPDATE_FREQ}"'"
        export RECORD_ENTROPY="'"${RECORD_ENTROPY}"'"
        export CAPTURE_CHECKPOINT_WINDOW="'"${CAPTURE_CHECKPOINT_WINDOW}"'"
        export SWERL_VMVM_TENANT="'"${SWERL_VMVM_TENANT}"'"
        export SWERL_SANDBOX_TIMING_LOGS="'"${SWERL_SANDBOX_TIMING_LOGS}"'"
        export SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S="'"${SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S}"'"
        export RAY_NUM_CPUS="'"${RAY_NUM_CPUS}"'"
        export WANDB_PROJECT="'"${WANDB_PROJECT}"'"
        export WANDB_ENTITY="'"${WANDB_ENTITY}"'"
        export WANDB_BASE_URL="'"${WANDB_BASE_URL:-https://meta-fair.wandb.io}"'"
        export RAY_TMPDIR="/tmp/ray-${USER}-${SLURM_JOB_ID}"
        export TRITON_CACHE_DIR="/tmp/triton-${USER}-${SLURM_JOB_ID}-${SLURM_NODEID:-0}"
        export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-${USER}-${SLURM_JOB_ID}-${SLURM_NODEID:-0}"
        export CUDA_CACHE_PATH="/tmp/cuda-cache-${USER}-${SLURM_JOB_ID}-${SLURM_NODEID:-0}"
        mkdir -p "${RAY_TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${CUDA_CACHE_PATH}"

        # vacli/x509 preflights removed (tmax-private#1 review): VMVM-only checks; the
        # vmvm backend hard-fails in emit_tool_configs.py, apptainer/sandfleet need
        # neither, and the x509 check killed cross-account (non-VMVM) hosts.

        # Resolve the model to a local snapshot path (offline cache).
        MODEL_NAME_OR_PATH="'"${MODEL_NAME_OR_PATH}"'"
        if [ ! -d "${MODEL_NAME_OR_PATH}" ]; then
            MODEL_NAME_OR_PATH="$(MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" "${UV_PROJECT_ENVIRONMENT}/bin/python" - <<PY
import os
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ["MODEL_NAME_OR_PATH"], local_files_only=True))
PY
)"
        fi
        echo "Using local model path: ${MODEL_NAME_OR_PATH}"

        source scripts/train/debug/envs/ray_node_setup_slurm.sh

        NODE_ID="${SLURM_NODEID:-${SLURM_PROCID:-0}}"
        if [ "${NODE_ID}" != "0" ]; then
            exit 0
        fi

        # Tool config emission + schema assertion live in ONE shared script
        # (tmax-private#1): keys asserted against dataclasses.fields of the pinned
        # config; the dead vmvm backend hard-fails here, and the orphaned data
        # knobs (SWERL_SKIP_DATA_BATCHES_ON_RESUME / SWERL_EXCLUDE_*) fail fast
        # at grpo_fast startup.
        TOOL_CONFIGS="$(
            TASK_DATA_HF_REPO="'"${TASK_DATA_HF_REPO}"'" \
            SWERL_SANDBOX_TEST_TIMEOUT="'"${SWERL_SANDBOX_TEST_TIMEOUT}"'" \
            SWERL_SANDBOX_TIMEOUT="'"${SWERL_SANDBOX_TIMEOUT}"'" \
            PYTHONPATH="'"${REPO_ROOT}"'" \
            "'"${UV_PROJECT_ENVIRONMENT}"'/bin/python" "'"${REPO_ROOT}"'/scripts/train/debug/envs/emit_tool_configs.py"
        )"
        if [ -z "${TOOL_CONFIGS}" ]; then
            echo "FATAL: emit_tool_configs.py refused (invalid input / unsupported backend / schema drift). Aborting before training." >&2
            exit 1
        fi
        export TOOL_CONFIGS
        echo "TOOL_CONFIGS=${TOOL_CONFIGS}"

        set +e
        "${UV_PROJECT_ENVIRONMENT}/bin/python" open_instruct/grpo_fast.py \
            --dataset_mixer_list "'"${DATASET_JSONL}"'" 1.0 \
            --dataset_mixer_list_splits train \
            --max_prompt_token_length "'"${MAX_PROMPT_TOKEN_LENGTH}"'" \
            --per_turn_max_tokens "'"${PER_TURN_MAX_TOKENS}"'" \
            --response_length "'"${RESPONSE_LENGTH}"'" \
            --pack_length "'"${PACK_LENGTH}"'" \
            --per_device_train_batch_size "'"${PER_DEVICE_TRAIN_BATCH_SIZE}"'" \
            --num_unique_prompts_rollout "'"${NUM_UNIQUE_PROMPTS_ROLLOUT}"'" \
            --num_samples_per_prompt_rollout "'"${NUM_SAMPLES_PER_PROMPT_ROLLOUT}"'" \
            --async_steps "'"${ASYNC_STEPS}"'" \
            --model_name_or_path "${MODEL_NAME_OR_PATH}" \
            --temperature "'"${TEMPERATURE}"'" \
            --learning_rate "'"${LEARNING_RATE}"'" \
            --optimizer_type "'"${OPTIMIZER_TYPE}"'" \
            --sgd_momentum "'"${SGD_MOMENTUM}"'" \
            --max_grad_norm "'"${MAX_GRAD_NORM}"'" \
            --total_episodes "'"${TOTAL_EPISODES}"'" \
            --lr_scheduler_type constant \
            --deepspeed_stage "'"${DEEPSPEED_STAGE}"'" \
            --sequence_parallel_size "'"${SEQUENCE_PARALLEL_SIZE}"'" \
            --num_epochs "'"${NUM_EPOCHS}"'" \
            --num_learners_per_node '"${GRPO_NUM_LEARNERS_PER_NODE}"' \
            --vllm_num_engines "'"${VLLM_NUM_ENGINES}"'" \
            --vllm_tensor_parallel_size "'"${VLLM_TENSOR_PARALLEL_SIZE}"'" \
            --beta "'"${BETA}"'" \
            --use_vllm_logprobs "'"${USE_VLLM_LOGPROBS}"'" \
            --truncated_importance_sampling_ratio_cap "'"${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP}"'" \
            --clip_lower "'"${CLIP_LOWER}"'" \
            --clip_higher "'"${CLIP_HIGHER}"'" \
            --tis_mask_lower "'"${TIS_MASK_LOWER}"'" \
            --tis_mask_upper "'"${TIS_MASK_UPPER}"'" \
            ${SEQUENCE_TIS_MASK_LOWER:+--sequence_tis_mask_lower ${SEQUENCE_TIS_MASK_LOWER}} \
            ${SEQUENCE_TIS_MASK_UPPER:+--sequence_tis_mask_upper ${SEQUENCE_TIS_MASK_UPPER}} \
            ${ALPHA:+--alpha ${ALPHA}} \
            ${REF_POLICY_UPDATE_FREQ:+--ref_policy_update_freq ${REF_POLICY_UPDATE_FREQ}} \
            ${RECORD_ENTROPY:+--record_entropy} \
            ${CAPTURE_CHECKPOINT_WINDOW:+--capture_checkpoint_window ${CAPTURE_CHECKPOINT_WINDOW}} \
            --seed "'"${SEED}"'" \
            --gradient_checkpointing \
            --vllm_enable_prefix_caching \
            --push_to_hub false \
            --with_tracking \
            --wandb_project "'"${WANDB_PROJECT}"'" \
            --wandb_entity "'"${WANDB_ENTITY}"'" \
            --save_traces \
            --save_trainer_logprobs true \
            --tools swerl_vanillux_sandbox \
            --tool_configs "${TOOL_CONFIGS}" \
            --pool_size "'"${POOL_SIZE}"'" \
            --max_steps "'"${MAX_STEPS}"'" \
            --verification_reward "'"${VERIFICATION_REWARD}"'" \
            --tool_parser_type "'"${TOOL_PARSER_TYPE}"'" \
            --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
            --active_sampling \
            --backend_timeout "'"${BACKEND_TIMEOUT}"'" \
            --vllm_gdn_prefill_backend "'"${VLLM_GDN_PREFILL_BACKEND}"'" \
            --checkpoint_state_freq "'"${CHECKPOINT_STATE_FREQ}"'" \
            --checkpoint_state_dir "'"${CHECKPOINT_STATE_DIR}"'" \
            --keep_last_n_checkpoints "'"${KEEP_LAST_N_CHECKPOINTS}"'" \
            --log_momentum_diagnostics "'"${LOG_MOMENTUM_DIAGNOSTICS:-false}"'" \
            --positive_advantage_only "'"${POSITIVE_ADVANTAGE_ONLY:-false}"'" \
            --leaky_negative_momentum "'"${LEAKY_NEGATIVE_MOMENTUM:-false}"'" \
            --leaky_full_v "'"${LEAKY_FULL_V:-false}"'" \
            --robust_momentum "'"${ROBUST_MOMENTUM:-false}"'" \
            --robust_momentum_c "'"${ROBUST_MOMENTUM_C:-4.0}"'" \
            --adam_epsilon "'"${ADAM_EPSILON:-1e-8}"'" \
            --grad_spike_clip_k "'"${GRAD_SPIKE_CLIP_K:-0.0}"'" \
            --grad_spike_ema_beta "'"${GRAD_SPIKE_EMA_BETA:-0.98}"'" \
            --inflight_updates true \
            --lm_head_fp32 "'"${LM_HEAD_FP32}"'" --use_liger_grpo_loss --liger_grpo_loss_chunk_size "'"${LIGER_GRPO_LOSS_CHUNK_SIZE}"'" \
            --advantage_normalization_type "'"${ADVANTAGE_NORMALIZATION_TYPE}"'" \
            --loss_fn "'"${LOSS_FN}"'" \
            --dppo_divergence_type "'"${DPPO_DIVERGENCE_TYPE}"'" \
            --dppo_divergence_threshold "'"${DPPO_DIVERGENCE_THRESHOLD}"'" \
            --rollouts_save_path "'"${ROLLOUTS_SAVE_PATH}"'" \
            --output_dir "'"${OUTPUT_DIR}"'" \
            --exp_name "'"${EXP_NAME}"'" \
            --local_eval_every "'"${LOCAL_EVAL_EVERY}"'" \
            --save_freq "'"${SAVE_FREQ}"'" \
            --try_launch_beaker_eval_jobs_on_weka False
        train_exit=$?
        ray stop --force >/dev/null 2>&1 || true
        exit "${train_exit}"
    '
