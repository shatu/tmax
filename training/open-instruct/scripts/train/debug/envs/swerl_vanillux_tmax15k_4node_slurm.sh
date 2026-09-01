#!/usr/bin/env bash
# Launch SWERL Vanillux GRPO on Slurm using a checkpoint-local uv environment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/checkpoint_env.sh"

NODES="${NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ACCOUNT="${ACCOUNT:-comem}"
PARTITION="${PARTITION:-h200}"
QOS="${QOS:-h200_comem_high}"
TIME_LIMIT="${TIME_LIMIT:-7-00:00:00}"
MEM_PER_NODE="${MEM_PER_NODE:-0}"
DATASET_ID="${DATASET_ID:-hamishivi/swerl-tmax-15k}"
TASK_DATA_HF_REPO="${TASK_DATA_HF_REPO-${DATASET_ID}}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_SLUG="${DATASET_SLUG:-${DATASET_ID##*/}}"
DATASET_SLUG="${DATASET_SLUG//-/_}"
EXP_NAME="${EXP_NAME:-swerl_qwen3_8b_our_sft_${DATASET_SLUG}_grpo}"
RUN_ID="${RUN_ID:-${EXP_NAME}__seed42__$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_STATE_DIR="${CHECKPOINT_STATE_DIR:-${REPO_ROOT}/output/checkpoint_states/${RUN_ID}}"
CHECKPOINT_STATE_FREQ="${CHECKPOINT_STATE_FREQ:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/runs/${RUN_ID}}"
ROLLOUTS_SAVE_PATH="${ROLLOUTS_SAVE_PATH:-${REPO_ROOT}/output/rollouts/${RUN_ID}}"
TOTAL_EPISODES="${TOTAL_EPISODES:-128000}"
POOL_SIZE="${POOL_SIZE:-768}"
SAVE_FREQ="${SAVE_FREQ:-20}"
LOCAL_EVAL_EVERY="${LOCAL_EVAL_EVERY:-10}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-hamishivi/sft_qwen3_8b_our_sft}"
MAX_PROMPT_TOKEN_LENGTH="${MAX_PROMPT_TOKEN_LENGTH:-2048}"
PER_TURN_MAX_TOKENS="${PER_TURN_MAX_TOKENS:-16384}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-32768}"
PACK_LENGTH="${PACK_LENGTH:-35840}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
NUM_UNIQUE_PROMPTS_ROLLOUT="${NUM_UNIQUE_PROMPTS_ROLLOUT:-32}"
NUM_SAMPLES_PER_PROMPT_ROLLOUT="${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-8}"
ASYNC_STEPS="${ASYNC_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-3}"
SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
GRPO_NUM_LEARNERS_PER_NODE="${GRPO_NUM_LEARNERS_PER_NODE:-8}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
BETA="${BETA:-0.0}"
USE_VLLM_LOGPROBS="${USE_VLLM_LOGPROBS:-true}"
TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP="${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP:-0.0}"
SAVE_TRACES="${SAVE_TRACES:-1}"
SAVE_TRAINER_LOGPROBS="${SAVE_TRAINER_LOGPROBS:-false}"
MAX_STEPS="${MAX_STEPS:-64}"
VERIFICATION_REWARD="${VERIFICATION_REWARD:-1.0}"
TOOL_PARSER_TYPE="${TOOL_PARSER_TYPE:-vllm_hermes}"
VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-}"
LM_HEAD_FP32="${LM_HEAD_FP32:-}"
USE_LIGER_GRPO_LOSS="${USE_LIGER_GRPO_LOSS:-}"
LIGER_GRPO_LOSS_CHUNK_SIZE="${LIGER_GRPO_LOSS_CHUNK_SIZE:-}"
LOSS_FN="${LOSS_FN:-}"
DPPO_DIVERGENCE_TYPE="${DPPO_DIVERGENCE_TYPE:-}"
DPPO_DIVERGENCE_THRESHOLD="${DPPO_DIVERGENCE_THRESHOLD:-}"
TOOL_CONFIG_IMAGE="${TOOL_CONFIG_IMAGE:-}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-24}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-192}"
GRPO_TRAINER_CPUS_PER_GPU="${GRPO_TRAINER_CPUS_PER_GPU:-8}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-${RAY_NUM_CPUS}}"
SWERL_ENV_ACTOR_CREATE_BATCH_SIZE="${SWERL_ENV_ACTOR_CREATE_BATCH_SIZE:-16}"
SWERL_ENV_ACTOR_CREATE_BATCH_SLEEP_S="${SWERL_ENV_ACTOR_CREATE_BATCH_SLEEP_S:-1}"
SWERL_ENV_PREWARM_MAX_PREPARED="${SWERL_ENV_PREWARM_MAX_PREPARED:-0}"
SWERL_ENV_PREWARM_IN_VLLM="${SWERL_ENV_PREWARM_IN_VLLM:-0}"
SWERL_SANDBOX_TIMING_LOGS="${SWERL_SANDBOX_TIMING_LOGS:-1}"
SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S="${SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S:-0.5}"
DRY_RUN="${DRY_RUN:-0}"

CPU_BUDGET_SUMMARY="$(
    python - <<PY
import math

nodes = int("${NODES}")
gpus_per_node = int("${GPUS_PER_NODE}")
ray_num_cpus = float("${RAY_NUM_CPUS}")
pool_size = int("${POOL_SIZE}")
env_actor_num_cpus = 1.0
vllm_num_engines = int("${VLLM_NUM_ENGINES}")
vllm_tensor_parallel_size = 1
num_learners_per_node = sum(int(x) for x in "${GRPO_NUM_LEARNERS_PER_NODE}".split())
trainer_cpus_per_gpu = float("${GRPO_TRAINER_CPUS_PER_GPU}")

env_pool_cpus = pool_size * env_actor_num_cpus
model_pg_cpus = num_learners_per_node * trainer_cpus_per_gpu
policy_actor_cpus = num_learners_per_node * 4
vllm_pg_cpus = vllm_num_engines * vllm_tensor_parallel_size
misc_cpus = 8
total_ray_cpus = nodes * ray_num_cpus
estimated_needed = env_pool_cpus + model_pg_cpus + vllm_pg_cpus + misc_cpus

print(f"CPU budget: Ray advertises {total_ray_cpus:g} CPUs ({ray_num_cpus:g}/node)")
print(f"CPU budget: env pool reserves ~{env_pool_cpus:g} CPUs ({pool_size} actors * {env_actor_num_cpus:g})")
print(f"CPU budget: model placement group needs {model_pg_cpus:g} CPUs on one node ({trainer_cpus_per_gpu:g}/GPU); policy actors use ~{policy_actor_cpus:g} inside it")
print(f"CPU budget: vLLM placement group needs ~{vllm_pg_cpus:g} CPUs total")
print(f"CPU budget: estimated total demand before overhead ~{estimated_needed:g}/{total_ray_cpus:g} CPUs")
if ray_num_cpus < model_pg_cpus:
    print(f"CPU budget WARNING: RAY_NUM_CPUS={ray_num_cpus:g} is lower than model PG per-node need {model_pg_cpus:g}")
if estimated_needed > total_ray_cpus:
    print("CPU budget WARNING: estimated CPU demand exceeds advertised Ray CPUs")
PY
)"

if [ ! -x "${CHECKPOINT_ROOT}/tools/uv/bin/uv" ]; then
    echo "Missing checkpoint-local uv at ${CHECKPOINT_ROOT}/tools/uv/bin/uv" >&2
    exit 1
fi

if [ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]; then
    echo "Missing training Python environment at ${UV_PROJECT_ENVIRONMENT}" >&2
    echo "Run uv sync on a GPU node first." >&2
    exit 1
fi

echo "Launching ${NODES} node(s), ${GPUS_PER_NODE} GPU(s)/node"
echo "Repo: ${REPO_ROOT}"
echo "Python: ${UV_PROJECT_ENVIRONMENT}/bin/python"
echo "Partition/QOS: ${PARTITION}/${QOS}"
echo "Sandbox images: per-sample env_config.image from the dataset"
echo "Dataset: ${DATASET_ID} (${DATASET_SPLIT})"
echo "Task data HF repo: ${TASK_DATA_HF_REPO}"
echo "Run ID: ${RUN_ID}"
echo "Checkpoint state dir: ${CHECKPOINT_STATE_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Rollouts save path: ${ROLLOUTS_SAVE_PATH}"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Total episodes: ${TOTAL_EPISODES}"
echo "Pool size: ${POOL_SIZE}"
echo "Response length: ${RESPONSE_LENGTH}"
echo "Pack length: ${PACK_LENGTH}"
echo "Sequence parallel size: ${SEQUENCE_PARALLEL_SIZE}"
echo "Learners per node: ${GRPO_NUM_LEARNERS_PER_NODE}"
echo "Loss fn: ${LOSS_FN:-default}"
echo "vLLM enforce eager: ${VLLM_ENFORCE_EAGER}"
echo "vLLM engines: ${VLLM_NUM_ENGINES}"
echo "Ray CPUs per node: ${RAY_NUM_CPUS}"
echo "Trainer CPUs per GPU: ${GRPO_TRAINER_CPUS_PER_GPU}"
echo "Slurm CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Env actor CPUs: Ray default (1)"
echo "Env actor create batch: ${SWERL_ENV_ACTOR_CREATE_BATCH_SIZE} sleep=${SWERL_ENV_ACTOR_CREATE_BATCH_SLEEP_S}s"
echo "Env prewarm max prepared: ${SWERL_ENV_PREWARM_MAX_PREPARED}"
echo "Env prewarm in vLLM: ${SWERL_ENV_PREWARM_IN_VLLM}"
echo "Sandbox timing logs: ${SWERL_SANDBOX_TIMING_LOGS} threshold=${SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S}s"
echo "${CPU_BUDGET_SUMMARY}"

if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1; not submitting Slurm job."
    exit 0
fi

SRUN_EXCLUDE_ARGS=()
if [ -n "${SLURM_EXCLUDE:-}" ]; then
    SRUN_EXCLUDE_ARGS+=(--exclude "${SLURM_EXCLUDE}")
    echo "Excluding Slurm nodes: ${SLURM_EXCLUDE}"
fi

srun \
    --account "${ACCOUNT}" \
    --partition "${PARTITION}" \
    --qos "${QOS}" \
    "${SRUN_EXCLUDE_ARGS[@]}" \
    --nodes "${NODES}" \
    --ntasks-per-node 1 \
    --cpus-per-task "${SLURM_CPUS_PER_TASK}" \
    --gpus-per-node "${GPUS_PER_NODE}" \
    --exclusive \
    --kill-on-bad-exit=1 \
    --mem "${MEM_PER_NODE}" \
    --time "${TIME_LIMIT}" \
    bash -lc '
        set -euo pipefail
        cd "'"${REPO_ROOT}"'"
        source scripts/train/debug/envs/checkpoint_env.sh
        export EXP_NAME="'"${EXP_NAME}"'"
        export DATASET_ID="'"${DATASET_ID}"'"
        export TASK_DATA_HF_REPO="'"${TASK_DATA_HF_REPO}"'"
        export DATASET_SPLIT="'"${DATASET_SPLIT}"'"
        export RUN_ID="'"${RUN_ID}"'"
        export CHECKPOINT_STATE_DIR="'"${CHECKPOINT_STATE_DIR}"'"
        export CHECKPOINT_STATE_FREQ="'"${CHECKPOINT_STATE_FREQ}"'"
        export KEEP_LAST_N_CHECKPOINTS="'"${KEEP_LAST_N_CHECKPOINTS:-3}"'"
        export OUTPUT_DIR="'"${OUTPUT_DIR}"'"
        export ROLLOUTS_SAVE_PATH="'"${ROLLOUTS_SAVE_PATH}"'"
        export TOTAL_EPISODES="'"${TOTAL_EPISODES}"'"
        export MODEL_NAME_OR_PATH="'"${MODEL_NAME_OR_PATH}"'"
        export MAX_PROMPT_TOKEN_LENGTH="'"${MAX_PROMPT_TOKEN_LENGTH}"'"
        export PER_TURN_MAX_TOKENS="'"${PER_TURN_MAX_TOKENS}"'"
        export RESPONSE_LENGTH="'"${RESPONSE_LENGTH}"'"
        export PACK_LENGTH="'"${PACK_LENGTH}"'"
        export PER_DEVICE_TRAIN_BATCH_SIZE="'"${PER_DEVICE_TRAIN_BATCH_SIZE}"'"
        export NUM_UNIQUE_PROMPTS_ROLLOUT="'"${NUM_UNIQUE_PROMPTS_ROLLOUT}"'"
        export NUM_SAMPLES_PER_PROMPT_ROLLOUT="'"${NUM_SAMPLES_PER_PROMPT_ROLLOUT}"'"
        export ASYNC_STEPS="'"${ASYNC_STEPS}"'"
        export LEARNING_RATE="'"${LEARNING_RATE}"'"
        export DEEPSPEED_STAGE="'"${DEEPSPEED_STAGE}"'"
        export SEQUENCE_PARALLEL_SIZE="'"${SEQUENCE_PARALLEL_SIZE}"'"
        export NUM_EPOCHS="'"${NUM_EPOCHS}"'"
        export GRPO_NUM_LEARNERS_PER_NODE="'"${GRPO_NUM_LEARNERS_PER_NODE}"'"
        export VLLM_TENSOR_PARALLEL_SIZE="'"${VLLM_TENSOR_PARALLEL_SIZE}"'"
        export BETA="'"${BETA}"'"
        export USE_VLLM_LOGPROBS="'"${USE_VLLM_LOGPROBS}"'"
        export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP="'"${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP}"'"
        export SAVE_TRACES="'"${SAVE_TRACES}"'"
        export SAVE_TRAINER_LOGPROBS="'"${SAVE_TRAINER_LOGPROBS}"'"
        export MAX_STEPS="'"${MAX_STEPS}"'"
        export VERIFICATION_REWARD="'"${VERIFICATION_REWARD}"'"
        export TOOL_PARSER_TYPE="'"${TOOL_PARSER_TYPE}"'"
        export VLLM_GDN_PREFILL_BACKEND="'"${VLLM_GDN_PREFILL_BACKEND}"'"
        export LM_HEAD_FP32="'"${LM_HEAD_FP32}"'"
        export USE_LIGER_GRPO_LOSS="'"${USE_LIGER_GRPO_LOSS}"'"
        export LIGER_GRPO_LOSS_CHUNK_SIZE="'"${LIGER_GRPO_LOSS_CHUNK_SIZE}"'"
        export LOSS_FN="'"${LOSS_FN}"'"
        export DPPO_DIVERGENCE_TYPE="'"${DPPO_DIVERGENCE_TYPE}"'"
        export DPPO_DIVERGENCE_THRESHOLD="'"${DPPO_DIVERGENCE_THRESHOLD}"'"
        export TOOL_CONFIG_IMAGE="'"${TOOL_CONFIG_IMAGE}"'"
        export POOL_SIZE="'"${POOL_SIZE}"'"
        export SAVE_FREQ="'"${SAVE_FREQ}"'"
        export LOCAL_EVAL_EVERY="'"${LOCAL_EVAL_EVERY}"'"
        export VLLM_ENFORCE_EAGER="'"${VLLM_ENFORCE_EAGER}"'"
        export VLLM_NUM_ENGINES="'"${VLLM_NUM_ENGINES}"'"
        export RAY_NUM_CPUS="'"${RAY_NUM_CPUS}"'"
        export GRPO_TRAINER_CPUS_PER_GPU="'"${GRPO_TRAINER_CPUS_PER_GPU}"'"
        export SWERL_ENV_ACTOR_CREATE_BATCH_SIZE="'"${SWERL_ENV_ACTOR_CREATE_BATCH_SIZE}"'"
        export SWERL_ENV_ACTOR_CREATE_BATCH_SLEEP_S="'"${SWERL_ENV_ACTOR_CREATE_BATCH_SLEEP_S}"'"
        export SWERL_ENV_PREWARM_MAX_PREPARED="'"${SWERL_ENV_PREWARM_MAX_PREPARED}"'"
        export SWERL_ENV_PREWARM_IN_VLLM="'"${SWERL_ENV_PREWARM_IN_VLLM}"'"
        export SWERL_SANDBOX_TIMING_LOGS="'"${SWERL_SANDBOX_TIMING_LOGS}"'"
        export SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S="'"${SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S}"'"
        export SWERL_APPTAINER_SIF_DIR="'"${SWERL_APPTAINER_SIF_DIR:-/checkpoint/comem/rulin/cache/apptainer-sifs}"'"
        export HF_HUB_OFFLINE="'"${HF_HUB_OFFLINE:-1}"'"
        export TRANSFORMERS_OFFLINE="'"${TRANSFORMERS_OFFLINE:-1}"'"
        export HF_DATASETS_OFFLINE="'"${HF_DATASETS_OFFLINE:-1}"'"
        export SWERL_SKIP_HF_LOGIN="'"${SWERL_SKIP_HF_LOGIN:-1}"'"
        export WANDB_PROJECT="'"${WANDB_PROJECT:-open-instruct-terminal-rl}"'"
        export WANDB_ENTITY="'"${WANDB_ENTITY:-rulin}"'"
        export WANDB_BASE_URL="'"${WANDB_BASE_URL:-https://meta-fair.wandb.io}"'"
        export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
        export RAY_TMPDIR="/tmp/ray-${USER}-${SLURM_JOB_ID}"
        export APPTAINER_TMPDIR="/tmp/apptainer-${USER}-${SLURM_JOB_ID}"
        export TRITON_CACHE_DIR="/tmp/triton-${USER}-${SLURM_JOB_ID}-${SLURM_NODEID:-0}"
        export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-${USER}-${SLURM_JOB_ID}-${SLURM_NODEID:-0}"
        export CUDA_CACHE_PATH="/tmp/cuda-cache-${USER}-${SLURM_JOB_ID}-${SLURM_NODEID:-0}"
        mkdir -p "${RAY_TMPDIR}" "${APPTAINER_TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${CUDA_CACHE_PATH}"

        if ! command -v apptainer >/dev/null 2>&1; then
            echo "apptainer is not available on $(hostname)" >&2
            exit 1
        fi

        if [ -n "${APPTAINER_DOCKER_PAT:-}" ]; then
            APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:-hamishivi740}"
            mkdir -p "$(dirname "${APPTAINER_AUTHFILE}")"
            chmod 700 "$(dirname "${APPTAINER_AUTHFILE}")"
            printf "%s" "${APPTAINER_DOCKER_PAT}" | apptainer registry login \
                --authfile "${APPTAINER_AUTHFILE}" \
                --username "${APPTAINER_DOCKER_USERNAME}" \
                --password-stdin \
                docker://docker.io >/dev/null
            chmod 600 "${APPTAINER_AUTHFILE}"
            unset APPTAINER_DOCKER_PAT
        fi

        if [ "${SWERL_SKIP_HF_LOGIN:-0}" != "1" ] && [ "${HF_HUB_OFFLINE:-0}" != "1" ] && [ -n "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]; then
            if ! "${UV_PROJECT_ENVIRONMENT}/bin/python" - <<PY
import os
from huggingface_hub import login

login(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"), add_to_git_credential=False)
PY
            then
                echo "WARNING: huggingface_hub.login failed; continuing with HF_TOKEN from environment" >&2
            fi
        fi

        if [ ! -d "${MODEL_NAME_OR_PATH}" ]; then
            MODEL_NAME_OR_PATH="$(
                "${UV_PROJECT_ENVIRONMENT}/bin/python" - <<PY
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_NAME_OR_PATH"]
revision = os.environ.get("MODEL_REVISION") or None
offline = os.environ.get("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes", "on"} or os.environ.get(
    "TRANSFORMERS_OFFLINE", "0"
).lower() in {"1", "true", "yes", "on"}
print(snapshot_download(repo_id, revision=revision, local_files_only=offline))
PY
            )"
            export MODEL_NAME_OR_PATH
        fi
        echo "Using local model path: ${MODEL_NAME_OR_PATH}"

        source scripts/train/debug/envs/ray_node_setup_slurm.sh

        NODE_ID="${SLURM_NODEID:-${SLURM_PROCID:-0}}"
        if [ "${NODE_ID}" != "0" ]; then
            exit 0
        fi

        # Tool config emission + schema assertion live in ONE shared script
        # (tmax-private#1). pwd/cache_dir/tmp_dir/authfile were removed from
        # SWERLVanilluxSandboxEnvConfig at tmax@61b5a85d (the APPTAINER_* dirs still
        # steer the runtime via the environment, just not via tool config).
        TOOL_CONFIGS=""
        TOOL_CONFIGS="$(
            SWERL_SANDBOX_BACKEND=apptainer \
            TASK_DATA_HF_REPO="${TASK_DATA_HF_REPO}" \
            SWERL_SANDBOX_TEST_TIMEOUT="${SWERL_SANDBOX_TEST_TIMEOUT:-120}" \
            SWERL_SANDBOX_TIMEOUT="${SWERL_SANDBOX_TIMEOUT:-120}" \
            PYTHONPATH="${REPO_ROOT}" \
            "${UV_PROJECT_ENVIRONMENT}/bin/python" "${REPO_ROOT}/scripts/train/debug/envs/emit_tool_configs.py"
        )"
        if [ -z "${TOOL_CONFIGS}" ]; then
            echo "FATAL: emit_tool_configs.py refused (invalid input / unsupported backend / schema drift). Aborting before training." >&2
            exit 1
        fi
        export TOOL_CONFIGS

        GRPO_OPTIONAL_ARGS=()
        if [ "${SAVE_TRACES}" = "1" ] || [ "${SAVE_TRACES}" = "true" ]; then
            GRPO_OPTIONAL_ARGS+=(--save_traces)
        fi
        if [ -n "${VLLM_GDN_PREFILL_BACKEND}" ]; then
            GRPO_OPTIONAL_ARGS+=(--vllm_gdn_prefill_backend "${VLLM_GDN_PREFILL_BACKEND}")
        fi
        if [ -n "${LM_HEAD_FP32}" ]; then
            GRPO_OPTIONAL_ARGS+=(--lm_head_fp32 "${LM_HEAD_FP32}")
        fi
        if [ "${USE_LIGER_GRPO_LOSS}" = "1" ] || [ "${USE_LIGER_GRPO_LOSS}" = "true" ]; then
            GRPO_OPTIONAL_ARGS+=(--use_liger_grpo_loss)
        fi
        if [ -n "${LIGER_GRPO_LOSS_CHUNK_SIZE}" ]; then
            GRPO_OPTIONAL_ARGS+=(--liger_grpo_loss_chunk_size "${LIGER_GRPO_LOSS_CHUNK_SIZE}")
        fi
        if [ -n "${LOSS_FN}" ]; then
            GRPO_OPTIONAL_ARGS+=(--loss_fn "${LOSS_FN}")
        fi
        if [ -n "${DPPO_DIVERGENCE_TYPE}" ]; then
            GRPO_OPTIONAL_ARGS+=(--dppo_divergence_type "${DPPO_DIVERGENCE_TYPE}")
        fi
        if [ -n "${DPPO_DIVERGENCE_THRESHOLD}" ]; then
            GRPO_OPTIONAL_ARGS+=(--dppo_divergence_threshold "${DPPO_DIVERGENCE_THRESHOLD}")
        fi

        set +e
        "${UV_PROJECT_ENVIRONMENT}/bin/python" open_instruct/grpo_fast.py \
            --dataset_mixer_list "${DATASET_ID}" 1.0 \
            --dataset_mixer_list_splits "${DATASET_SPLIT}" \
            --max_prompt_token_length "${MAX_PROMPT_TOKEN_LENGTH}" \
            --per_turn_max_tokens "${PER_TURN_MAX_TOKENS}" \
            --response_length "${RESPONSE_LENGTH}" \
            --pack_length "${PACK_LENGTH}" \
            --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
            --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS_ROLLOUT}" \
            --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}" \
            --async_steps "${ASYNC_STEPS}" \
            --model_name_or_path "${MODEL_NAME_OR_PATH}" \
            --temperature 1.0 \
            --learning_rate "${LEARNING_RATE}" \
            --total_episodes "${TOTAL_EPISODES}" \
            --lr_scheduler_type constant \
            --deepspeed_stage "${DEEPSPEED_STAGE}" \
            --sequence_parallel_size "${SEQUENCE_PARALLEL_SIZE}" \
            --num_epochs "${NUM_EPOCHS}" \
            --num_learners_per_node ${GRPO_NUM_LEARNERS_PER_NODE} \
            --vllm_num_engines "${VLLM_NUM_ENGINES}" \
            --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
            --vllm_enforce_eager "${VLLM_ENFORCE_EAGER}" \
            --beta "${BETA}" \
            --use_vllm_logprobs "${USE_VLLM_LOGPROBS}" \
            --truncated_importance_sampling_ratio_cap "${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP}" \
            --seed 42 \
            --gradient_checkpointing \
            --vllm_enable_prefix_caching \
            --push_to_hub false \
            --with_tracking \
            --wandb_project "${WANDB_PROJECT}" \
            --wandb_entity "${WANDB_ENTITY}" \
            --save_trainer_logprobs "${SAVE_TRAINER_LOGPROBS}" \
            --tools swerl_vanillux_sandbox \
            --tool_configs "${TOOL_CONFIGS}" \
            --pool_size "${POOL_SIZE}" \
            --max_steps "${MAX_STEPS}" \
            --verification_reward "${VERIFICATION_REWARD}" \
            --tool_parser_type "${TOOL_PARSER_TYPE}" \
            --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
            --active_sampling \
            --backend_timeout 1200 \
            --checkpoint_state_freq "${CHECKPOINT_STATE_FREQ}" \
            --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}" \
            --keep_last_n_checkpoints "${KEEP_LAST_N_CHECKPOINTS:-3}" \
            --log_momentum_diagnostics "${LOG_MOMENTUM_DIAGNOSTICS:-false}" \
            --inflight_updates true \
            --advantage_normalization_type centered \
            --rollouts_save_path "${ROLLOUTS_SAVE_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --exp_name "${EXP_NAME}" \
            --local_eval_every "${LOCAL_EVAL_EVERY}" \
            --save_freq "${SAVE_FREQ}" \
            --try_launch_beaker_eval_jobs_on_weka False \
            "${GRPO_OPTIONAL_ARGS[@]}" \
            ${EXTRA_GRPO_ARGS:-}
        train_exit=$?
        ray stop --force >/dev/null 2>&1 || true
        exit "${train_exit}"
    '
