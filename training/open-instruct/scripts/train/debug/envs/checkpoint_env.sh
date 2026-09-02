#!/usr/bin/env bash
# Checkpoint-local runtime environment for open-instruct on Slurm nodes.

set -euo pipefail

export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/checkpoint/comem/rulin}"
if [ -z "${REPO_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    export REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)"
fi

# Load local secrets/config for non-interactive Slurm jobs. This is intentionally
# silent so tokens do not appear in logs. Override with CHECKPOINT_DOTENV_FILE.
if [ -n "${CHECKPOINT_DOTENV_FILE:-}" ] && [ -f "${CHECKPOINT_DOTENV_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${CHECKPOINT_DOTENV_FILE}"
    set +a
elif [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
elif [ -f "${CHECKPOINT_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${CHECKPOINT_ROOT}/.env"
    set +a
fi

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi

export PATH="${CHECKPOINT_ROOT}/tools/uv/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CHECKPOINT_ROOT}/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${CHECKPOINT_ROOT}/uv-python}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${REPO_ROOT}/.venv}"

export HF_HOME="${HF_HOME:-${CHECKPOINT_ROOT}/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CHECKPOINT_ROOT}/xdg-cache}"
export TMPDIR="${TMPDIR:-${CHECKPOINT_ROOT}/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-${CHECKPOINT_ROOT}/ray-tmp}"
export APPTAINER_CONFIGDIR="${APPTAINER_CONFIGDIR:-${CHECKPOINT_ROOT}/apptainer-config}"
export SINGULARITY_CONFIGDIR="${SINGULARITY_CONFIGDIR:-${APPTAINER_CONFIGDIR}}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${CHECKPOINT_ROOT}/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${CHECKPOINT_ROOT}/apptainer-tmp}"
export APPTAINER_AUTHFILE="${APPTAINER_AUTHFILE:-${CHECKPOINT_ROOT}/apptainer-auth/docker-auth.json}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CHECKPOINT_ROOT}/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CHECKPOINT_ROOT}/torchinductor-cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${CHECKPOINT_ROOT}/cuda-cache}"

export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/output/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${CHECKPOINT_ROOT}/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${CHECKPOINT_ROOT}/wandb-config}"
if [ -z "${WANDB_MODE:-}" ]; then
    if [ -n "${WANDB_API_KEY:-}" ]; then
        export WANDB_MODE="online"
    else
        export WANDB_MODE="offline"
    fi
fi
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://meta-fair.wandb.io}"
export WANDB_ENTITY="${WANDB_ENTITY:-rulin}"
export WANDB_PROJECT="${WANDB_PROJECT:-open-instruct-terminal-rl}"

case "${WANDB_DIR}:${WANDB_CACHE_DIR}:${WANDB_CONFIG_DIR}" in
    *"${HOME:-/home/rulin}"*)
        export WANDB_DIR="${REPO_ROOT}/output/wandb"
        export WANDB_CACHE_DIR="${CHECKPOINT_ROOT}/wandb-cache"
        export WANDB_CONFIG_DIR="${CHECKPOINT_ROOT}/wandb-config"
        ;;
esac

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

mkdir -p \
    "${UV_CACHE_DIR}" \
    "${UV_PYTHON_INSTALL_DIR}" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${XDG_CACHE_HOME}" \
    "${TMPDIR}" \
    "${RAY_TMPDIR}" \
    "${APPTAINER_CONFIGDIR}" \
    "${APPTAINER_CACHEDIR}" \
    "${APPTAINER_TMPDIR}" \
    "$(dirname "${APPTAINER_AUTHFILE}")" \
    "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}" \
    "${WANDB_CONFIG_DIR}"
