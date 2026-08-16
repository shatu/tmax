#!/usr/bin/env bash
#
# Launch a single beaker task that:
#   1. downloads a task corpus (compact tasks.zip) from the HF Hub
#   2. installs apptainer and builds any missing base SIFs in-job
#   3. serves a model with vLLM on the local GPUs (GLM-5.2 recipe on B300,
#      generic profile on H100)
#   4. runs rl_data.generate_solutions against it (the SFT warm-start rollouts)
#   5. syncs solutions/ summaries to the gantry results dataset as it goes
#
# Sibling of launch_eval.sh; the inner script is scripts/beaker/run_gen_solutions_in_job.sh.
#
# Usage:
#   ./beaker_configs/launch_gen_solutions.sh [options]
#
# Examples:
#   # GLM-5.2-FP8 on 8xB300 (ai2/holmes), full 16.5K SFT corpus, 8 rollouts/task
#   ./beaker_configs/launch_gen_solutions.sh --hardware b300
#
#   # Smoke test: 10 tasks
#   ./beaker_configs/launch_gen_solutions.sh --hardware b300 --num-tasks 10
#
#   # A model that fits on H100s (ai2/jupiter), e.g. served TP=1 DP=8
#   ./beaker_configs/launch_gen_solutions.sh --hardware h100 \
#       --vllm-model Qwen/Qwen3.5-9B --name qwen3.5-9b
#
# NOTE: GLM-5.2 does NOT fit on a single 8xH100 node (~743 GB FP8 weights vs
# 640 GB HBM), and the NVFP4 variant needs Blackwell. GLM-5.2 therefore
# requires --hardware b300; the launcher enforces this.
#
# Uses Beaker Gantry to submit the current git HEAD. The SHA must be pushed to
# the remote; local dirty changes are not included in the remote job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- defaults ----------------------------------------------------------------
HARDWARE="b300"
CLUSTER=""                       # default derived from HARDWARE below
BUILD_SIFS_ONLY=0
GPU_COUNT_SET=0
VLLM_MODEL="zai-org/GLM-5.2-FP8"
SERVED_MODEL_NAME=""
VLLM_VERSION="0.23.0"
VLLM_PORT=8008
GPU_COUNT=8
TP_SIZE=""
DP_SIZE=""
MAX_MODEL_LEN=131072
VLLM_GPU_UTIL=""
VLLM_MAX_NUM_SEQS=32
VLLM_TOOL_CALL_PARSER=""
VLLM_REASONING_PARSER=""
VLLM_ENABLE_MTP=0
VLLM_USE_DEEP_GEMM=0
VLLM_EXTRA_ARGS=""
TASKS_HF_DATASET="allenai/TMax-SFT-16.5K"
TASKS_DIR_NAME="tasks_sft_16.5k"
NUM_SOLUTIONS=8
MAX_ACTIONS=16
MAX_TOKENS=""
NUM_TASKS=999999
START_AT=0
WORKERS=12
NUM_POOL_WORKERS=16
SOLUTION_TEMPERATURE=0.7
SAMPLE_SIZE=0
SAMPLE_SEED=0
FORCE_RERUN=0
JOB_NAME=""
MIN_RUNTIME=""
APPTAINER_FLAVOR="${APPTAINER_FLAVOR:-}"   # default derived from HARDWARE below
PRIORITY="high"
BUDGET="ai2/oe-omai"
HF_TOKEN_SECRET_NAME="pradeepd_HF_TOKEN"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/oe-agents-holmes}"
# Default image is per-hardware (see below): the cuda13.0 image is meant for
# the B300 cluster, where the toolkit must target sm_103 (--use-deep-gemm);
# h100 keeps the cuda12.8 image. Preset $BEAKER_IMAGE or pass --image to override.
BEAKER_IMAGE="${BEAKER_IMAGE:-}"
BEAKER_DOCKER_IMAGE="${BEAKER_DOCKER_IMAGE:-}"
REPO_GIT_REF=""
WEKA_MOUNT="oe-adapt-default:/weka/oe-adapt-default"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --hardware HW          b300 | h100 (default: b300).
                         b300 -> GLM-5.2 recipe (FP8 KV cache, TP=8).
                         h100 -> generic profile for models that fit on H100s.
  --cluster CLUSTER      beaker cluster (default: ai2/holmes for b300,
                         ai2/jupiter for h100)
  --vllm-model MODEL     HF repo to serve (default: zai-org/GLM-5.2-FP8)
  --name NAME            served-model-name (default: lowercased basename of model)
  --vllm-version VER     vLLM version for uvx (default: 0.23.0 — the GLM-5.2
                         recipe requires >= 0.23.0)
  --gpus N               GPUs (default: 8)
  --tp N                 tensor-parallel-size (default: 8 on b300, 1 on h100)
  --dp N                 data-parallel-size (default: 1 on b300, gpus/tp on h100)
  --max-model-len LEN    context window (default: 131072)
  --gpu-util F           --gpu-memory-utilization (default: vllm default)
  --max-num-seqs N       --max-num-seqs (default: 32, per GLM recipe)
  --tool-call-parser P   override tool parser (default: glm47 for GLM, hermes else)
  --reasoning-parser P   override reasoning parser (default: glm45 for GLM, none else)
  --enable-mtp           add the recipe's MTP speculative-decoding flags
                         (default off: tool-calling + MTP need vllm main)
  --use-deep-gemm        enable DeepGEMM JIT FP8 kernels (default off: needs a
                         cu129+ toolchain for B300 sm_103; CUTLASS fallback
                         is precompiled and JIT-free)
  --vllm-extra-args S    free-form extra args for vllm serve
  --tasks-dataset DS     HF dataset with tasks.zip (default: allenai/TMax-SFT-16.5K)
  --tasks-dir-name NAME  corpus dir under rl_data/output/ (default: tasks_sft_16.5k)
  --num-solutions N      rollouts per task (default: 8)
  --max-actions N        max agent turns (default: 16)
  --max-tokens N         per-turn generation cap (default: max-model-len/4)
  --num-tasks N          cap task count (default: all)
  --start-at N           task offset (default: 0)
  --workers N            parallel tasks (default: 12)
  --num-pool-workers N   parallel LLM calls per turn (default: 16)
  --temperature F        solution temperature (default: 0.7)
  --sample-size N        fixed random subset (default: 0 = off)
  --sample-seed N        subset seed (default: 0)
  --force-rerun          re-run tasks that already have a summary for this model
  --job-name NAME        beaker experiment name (default: gen-sol-<served-name>)
  --min-runtime DUR      minimum guaranteed runtime before the job can be
                         preempted, e.g. '1h', '30m' (default: server default,
                         i.e. preemptible at any time)
  --apptainer-flavor F   plain | suid (default: suid on b300, plain on h100)
  --build-sifs-only      CPU-only job: build missing base SIFs into the weka
                         cache and exit (no vLLM/solver; defaults --gpus 0).
                         Run on a userns-capable cluster (e.g. --hardware h100
                         -> ai2/jupiter); holmes cannot build SIFs at all.
  --priority PRI         beaker priority (default: urgent)
  --budget BUDGET        beaker budget (default: workspace default)
  --workspace WS         beaker workspace (default: \$BEAKER_WORKSPACE or ai2/tmax)
  --image IMAGE          beaker image (default: ai2/cuda13.0-ubuntu22.04-torch2.11.0
                         on b300, ai2/cuda12.8-dev-ubuntu22.04-torch2.10.0 on h100)
  --docker-image IMAGE   public Docker image instead of a beaker image
  --weka MOUNT           weka mount as src:dst (default: $WEKA_MOUNT);
                         'none' to disable (HF cache then stays job-local!)
  --repo-ref REF         git SHA/branch to run (default: current HEAD SHA)
EOF
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --hardware)          HARDWARE="$2"; shift 2 ;;
        --cluster)           CLUSTER="$2"; shift 2 ;;
        --vllm-model)        VLLM_MODEL="$2"; shift 2 ;;
        --name)              SERVED_MODEL_NAME="$2"; shift 2 ;;
        --vllm-version)      VLLM_VERSION="$2"; shift 2 ;;
        --gpus)              GPU_COUNT="$2"; GPU_COUNT_SET=1; shift 2 ;;
        --build-sifs-only)   BUILD_SIFS_ONLY=1; shift ;;
        --tp)                TP_SIZE="$2"; shift 2 ;;
        --dp)                DP_SIZE="$2"; shift 2 ;;
        --max-model-len)     MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-util)          VLLM_GPU_UTIL="$2"; shift 2 ;;
        --max-num-seqs)      VLLM_MAX_NUM_SEQS="$2"; shift 2 ;;
        --tool-call-parser)  VLLM_TOOL_CALL_PARSER="$2"; shift 2 ;;
        --reasoning-parser)  VLLM_REASONING_PARSER="$2"; shift 2 ;;
        --enable-mtp)        VLLM_ENABLE_MTP=1; shift ;;
        --use-deep-gemm)     VLLM_USE_DEEP_GEMM=1; shift ;;
        --vllm-extra-args)   VLLM_EXTRA_ARGS="$2"; shift 2 ;;
        --tasks-dataset)     TASKS_HF_DATASET="$2"; shift 2 ;;
        --tasks-dir-name)    TASKS_DIR_NAME="$2"; shift 2 ;;
        --num-solutions)     NUM_SOLUTIONS="$2"; shift 2 ;;
        --max-actions)       MAX_ACTIONS="$2"; shift 2 ;;
        --max-tokens)        MAX_TOKENS="$2"; shift 2 ;;
        --num-tasks)         NUM_TASKS="$2"; shift 2 ;;
        --start-at)          START_AT="$2"; shift 2 ;;
        --workers)           WORKERS="$2"; shift 2 ;;
        --num-pool-workers)  NUM_POOL_WORKERS="$2"; shift 2 ;;
        --temperature)       SOLUTION_TEMPERATURE="$2"; shift 2 ;;
        --sample-size)       SAMPLE_SIZE="$2"; shift 2 ;;
        --sample-seed)       SAMPLE_SEED="$2"; shift 2 ;;
        --force-rerun)       FORCE_RERUN=1; shift ;;
        --job-name)          JOB_NAME="$2"; shift 2 ;;
        --min-runtime)       MIN_RUNTIME="$2"; shift 2 ;;
        --apptainer-flavor)  APPTAINER_FLAVOR="$2"; shift 2 ;;
        --priority)          PRIORITY="$2"; shift 2 ;;
        --budget)            BUDGET="$2"; shift 2 ;;
        --workspace)         BEAKER_WORKSPACE="$2"; shift 2 ;;
        --image)             BEAKER_IMAGE="$2"; BEAKER_DOCKER_IMAGE=""; shift 2 ;;
        --docker-image)      BEAKER_DOCKER_IMAGE="$2"; BEAKER_IMAGE=""; shift 2 ;;
        --weka)              WEKA_MOUNT="$2"; shift 2 ;;
        --repo-ref)          REPO_GIT_REF="$2"; shift 2 ;;
        -h|--help)           usage ;;
        *) echo "unknown option: $1"; usage ;;
    esac
done

# --- derive + validate ---------------------------------------------------------
case "$HARDWARE" in
    b300) CLUSTER="${CLUSTER:-ai2/holmes}" ;;
    h100) CLUSTER="${CLUSTER:-ai2/jupiter}" ;;
    *) echo "error: --hardware must be b300 or h100, got '$HARDWARE'" >&2; exit 1 ;;
esac

# Per-hardware default image (only when neither --image nor --docker-image
# nor $BEAKER_IMAGE was given). The cuda13.0 image is for the B300 cluster.
if [ -z "$BEAKER_IMAGE" ] && [ -z "$BEAKER_DOCKER_IMAGE" ]; then
    case "$HARDWARE" in
        b300) BEAKER_IMAGE="ai2/cuda13.0-ubuntu22.04-torch2.11.0" ;;
        h100) BEAKER_IMAGE="ai2/cuda12.8-dev-ubuntu22.04-torch2.10.0" ;;
    esac
fi

# Per-hardware apptainer flavor default: holmes (b300) denies userns mapping
# writes, and 'apptainer build' %post cannot run without them in the plain
# (non-setuid) flavor — runs 5-6 proved base-SIF builds fail there. The suid
# starter sidesteps userns entirely, so it's the working default on b300.
if [ -z "$APPTAINER_FLAVOR" ]; then
    case "$HARDWARE" in
        b300) APPTAINER_FLAVOR="suid" ;;
        *)    APPTAINER_FLAVOR="plain" ;;
    esac
fi

# Build-only jobs never serve a model: no GPUs needed, model guard moot.
if [ "$BUILD_SIFS_ONLY" = "1" ] && [ "$GPU_COUNT_SET" = "0" ]; then
    GPU_COUNT=0
fi

if [[ "$BUILD_SIFS_ONLY" != "1" && "$HARDWARE" == "h100" && "$VLLM_MODEL" == *GLM-5.2* ]]; then
    cat >&2 <<'EOF'
error: GLM-5.2 cannot be served on a single 8xH100 node:
  * FP8 weights are ~743 GB vs 640 GB of HBM on 8xH100
  * the ~465 GB NVFP4 variant requires Blackwell FP4 tensor cores
Use --hardware b300 for GLM-5.2, or pass a smaller --vllm-model for h100.
EOF
    exit 1
fi

if [ -z "$SERVED_MODEL_NAME" ]; then
    SERVED_MODEL_NAME="$(basename "$VLLM_MODEL" | tr '[:upper:]' '[:lower:]')"
fi
if [ -z "$MAX_TOKENS" ]; then
    # Reserve 3/4 of the context for the (growing) agent history; cap each
    # turn's generation at the remaining 1/4 so vLLM never rejects with
    # prompt_tokens + max_tokens > max_model_len.
    MAX_TOKENS=$(( MAX_MODEL_LEN / 4 ))
fi
if [ -z "$TP_SIZE" ]; then
    if [ "$HARDWARE" = "b300" ]; then TP_SIZE=8; else TP_SIZE=1; fi
fi
if [ -z "$DP_SIZE" ]; then
    if [ "$HARDWARE" = "b300" ]; then DP_SIZE=1; else DP_SIZE=$(( GPU_COUNT / TP_SIZE )); fi
fi

if [ -z "$REPO_GIT_REF" ]; then
    REPO_GIT_REF="$(git -C "$REPO_ROOT" rev-parse HEAD)"
fi
if [ "$BUILD_SIFS_ONLY" = "1" ]; then
    JOB_NAME="${JOB_NAME:-build-base-sifs}"
else
    JOB_NAME="${JOB_NAME:-gen-sol-${SERVED_MODEL_NAME}}"
fi
BEAKER_NAME="$JOB_NAME"

cat <<EOF
=== Launching SFT solution generation on Beaker ===
  Hardware:     ${HARDWARE}  (cluster ${CLUSTER}, ${GPU_COUNT} GPUs, TP=${TP_SIZE} DP=${DP_SIZE})
  Model:        ${VLLM_MODEL} (served as ${SERVED_MODEL_NAME}, vLLM ${VLLM_VERSION})
  Context:      max_model_len=${MAX_MODEL_LEN}, per-turn max_tokens=${MAX_TOKENS}
  Parsers:      tool=${VLLM_TOOL_CALL_PARSER:-<auto>} reasoning=${VLLM_REASONING_PARSER:-<auto>}  MTP=${VLLM_ENABLE_MTP}
  Tasks:        ${TASKS_HF_DATASET} -> rl_data/output/${TASKS_DIR_NAME}
  Solver:       num_solutions=${NUM_SOLUTIONS} max_actions=${MAX_ACTIONS} workers=${WORKERS}
                num_tasks=${NUM_TASKS} start_at=${START_AT} sample_size=${SAMPLE_SIZE} force_rerun=${FORCE_RERUN}
  Weka:         ${WEKA_MOUNT}
  Image:        ${BEAKER_IMAGE:+beaker:${BEAKER_IMAGE}}${BEAKER_DOCKER_IMAGE:+docker:${BEAKER_DOCKER_IMAGE}}
  Apptainer:    flavor=${APPTAINER_FLAVOR}  sif_cache=${SIF_CACHE_DIR:-<in-job default>}
  Job:          ${BEAKER_NAME}  workspace=${BEAKER_WORKSPACE}  priority=${PRIORITY}
  Repo ref:     ${REPO_GIT_REF}
EOF

# Sanity check: the SHA must be reachable on the remote.
if ! git -C "$REPO_ROOT" branch -r --contains "$REPO_GIT_REF" 2>/dev/null | grep -q .; then
    echo
    echo "warning: $REPO_GIT_REF doesn't appear to be on any remote branch."
    echo "         the beaker job will fail to clone it. push first or pass --repo-ref."
fi

# --- launch via Gantry -----------------------------------------------------------
GANTRY_CMD=(
    uvx --from beaker-gantry gantry --quiet run
    --yes
    --allow-dirty
    --workspace "$BEAKER_WORKSPACE"
    --name "$BEAKER_NAME"
    --description "SFT solution generation (${TASKS_HF_DATASET}) with ${SERVED_MODEL_NAME} via vLLM on ${HARDWARE}"
    --ref "$REPO_GIT_REF"
    --cluster "$CLUSTER"
    --gpus "$GPU_COUNT"
    --priority "$PRIORITY"
    --env-secret "HF_TOKEN=${HF_TOKEN_SECRET_NAME}"
    --env "HARDWARE=${HARDWARE}"
    --env "VLLM_MODEL=${VLLM_MODEL}"
    --env "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
    --env "VLLM_VERSION=${VLLM_VERSION}"
    --env "VLLM_PORT=${VLLM_PORT}"
    --env "GPU_COUNT=${GPU_COUNT}"
    --env "TP_SIZE=${TP_SIZE}"
    --env "DP_SIZE=${DP_SIZE}"
    --env "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
    --env "VLLM_GPU_UTIL=${VLLM_GPU_UTIL}"
    --env "VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
    --env "VLLM_TOOL_CALL_PARSER=${VLLM_TOOL_CALL_PARSER}"
    --env "VLLM_REASONING_PARSER=${VLLM_REASONING_PARSER}"
    --env "VLLM_ENABLE_MTP=${VLLM_ENABLE_MTP}"
    --env "VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM}"
    --env "VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}"
    --env "TASKS_HF_DATASET=${TASKS_HF_DATASET}"
    --env "TASKS_DIR_NAME=${TASKS_DIR_NAME}"
    --env "NUM_SOLUTIONS=${NUM_SOLUTIONS}"
    --env "MAX_ACTIONS=${MAX_ACTIONS}"
    --env "MAX_TOKENS=${MAX_TOKENS}"
    --env "NUM_TASKS=${NUM_TASKS}"
    --env "START_AT=${START_AT}"
    --env "WORKERS=${WORKERS}"
    --env "NUM_POOL_WORKERS=${NUM_POOL_WORKERS}"
    --env "SOLUTION_TEMPERATURE=${SOLUTION_TEMPERATURE}"
    --env "SAMPLE_SIZE=${SAMPLE_SIZE}"
    --env "SAMPLE_SEED=${SAMPLE_SEED}"
    --env "FORCE_RERUN=${FORCE_RERUN}"
    --env "APPTAINER_FLAVOR=${APPTAINER_FLAVOR}"
    --env "SIF_CACHE_DIR=${SIF_CACHE_DIR:-}"
    --env "BUILD_SIFS_ONLY=${BUILD_SIFS_ONLY}"
    # Run the task with the privileges nested containers need (user
    # namespaces for apptainer build/%post). Same setting the harbor/podman
    # eval pipeline relies on; without it holmes blocks userns creation.
    --env BEAKER_ALLOW_SUBCONTAINERS=1
    --env BEAKER_SKIP_DOCKER_SOCKET=1
    --host-networking
    --propagate-failure
    --no-python
)

if [ "$WEKA_MOUNT" != "none" ]; then
    GANTRY_CMD+=(--weka "$WEKA_MOUNT")
fi
if [ -n "$BEAKER_IMAGE" ]; then
    GANTRY_CMD+=(--beaker-image "$BEAKER_IMAGE")
elif [ -n "$BEAKER_DOCKER_IMAGE" ]; then
    GANTRY_CMD+=(--docker-image "$BEAKER_DOCKER_IMAGE")
else
    echo "error: either --image or --docker-image must be set" >&2
    exit 1
fi
if [ -n "$BUDGET" ]; then
    GANTRY_CMD+=(--budget "$BUDGET")
fi
if [ -n "$MIN_RUNTIME" ]; then
    GANTRY_CMD+=(--min-runtime "$MIN_RUNTIME")
fi

GANTRY_CMD+=(--upload "$REPO_ROOT/scripts/beaker:/uploaded-beaker-scripts")
GANTRY_CMD+=(-- bash /uploaded-beaker-scripts/run_gen_solutions_in_job.sh)

echo
printf 'Launching with:'
printf ' %q' "${GANTRY_CMD[@]}"
printf '\n\n'

"${GANTRY_CMD[@]}"
