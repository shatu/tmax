#!/usr/bin/env bash
#
# Inner script invoked inside a beaker task. Downloads a task corpus from the
# HF Hub, installs apptainer, spins up a vLLM server on the local GPUs, then
# runs rl_data.generate_solutions against it. Sibling of run_eval_in_job.sh
# (which drives harbor/podman instead of the rl_data apptainer harness).
#
# Driven entirely by env vars (set by beaker_configs/launch_gen_solutions.sh):
#
#   -- vLLM serving --
#   HARDWARE                 b300 | h100 — selects the serving profile (required)
#   VLLM_MODEL               HF repo to serve (required, e.g. zai-org/GLM-5.2-FP8)
#   SERVED_MODEL_NAME        --served-model-name (required, e.g. glm-5.2-fp8)
#   VLLM_VERSION             vLLM package version for uvx (default: 0.23.0)
#   VLLM_PORT                port for vLLM (default: 8008)
#   TP_SIZE                  --tensor-parallel-size (default: 8 for b300, 1 for h100)
#   DP_SIZE                  --data-parallel-size (default: 1 for b300,
#                            GPU_COUNT/TP for h100)
#   GPU_COUNT                GPUs allocated to the job (default: 8)
#   MAX_MODEL_LEN            --max-model-len (default: 131072)
#   VLLM_GPU_UTIL            --gpu-memory-utilization (default: vllm's default)
#   VLLM_MAX_NUM_SEQS        --max-num-seqs (default: 32, per the GLM recipe)
#   VLLM_TOOL_CALL_PARSER    tool parser (default: glm47 for GLM models, hermes otherwise)
#   VLLM_REASONING_PARSER    reasoning parser (default: glm45 for GLM models, empty otherwise)
#   VLLM_ENABLE_MTP          1 to add the recipe's MTP speculative-decoding flags.
#                            Default 0: the vLLM recipe notes tool-calling + MTP
#                            only work together on vllm main, and our solver
#                            harness is tool-calling-based.
#   VLLM_EXTRA_ARGS          free-form extra args appended to vllm serve
#   VLLM_READY_TIMEOUT       seconds to wait for /v1/models (default: 3600 —
#                            a ~700 GB checkpoint takes a while to load)
#
#   -- task corpus --
#   TASKS_HF_DATASET         HF dataset repo with a compact tasks.zip
#                            (default: allenai/TMax-SFT-16.5K)
#   TASKS_DIR_NAME           dir name under rl_data/output/ (default: tasks_sft_16.5k)
#
#   -- solver --
#   NUM_SOLUTIONS            rollouts per task (default: 8)
#   MAX_ACTIONS              max agent turns (default: 16)
#   MAX_TOKENS               per-turn generation cap (default: MAX_MODEL_LEN/4)
#   NUM_TASKS                task-count cap (default: 999999 = all)
#   START_AT                 task offset (default: 0)
#   WORKERS                  parallel tasks (default: 12)
#   NUM_POOL_WORKERS         parallel LLM calls within a turn (default: 16)
#   SOLUTION_TEMPERATURE     default 0.7
#   COMMAND_TIMEOUT          per-command timeout in containers (default: 600)
#   SHELL_INIT_TIMEOUT       default 240
#   SHELL_INIT_ATTEMPTS      default 3
#   BUILD_WORKERS            concurrent base-SIF builds (default: 4)
#   BUILD_RETRIES            default 3
#   SAMPLE_SIZE / SAMPLE_SEED  optional fixed random subset (default: 0 = off)
#   FORCE_RERUN              1 to re-run tasks that already have a summary (default: 0)
#
#   -- plumbing --
#   RESULTS_DIR              where to sync solutions/ summaries (default: /results,
#                            persisted by gantry as the result dataset)
#   SYNC_INTERVAL            seconds between periodic result syncs (default: 300)
#   APPTAINER_VERSION        apptainer release to install (default: 1.4.4)
#   HF_CACHE_DIR             HF_HOME override. Default: a weka path if
#                            /weka/oe-adapt-default is mounted (so the ~700 GB
#                            GLM download survives across jobs), else ~/.cache.

set -euo pipefail

log() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$*"; }

: "${HARDWARE:?set HARDWARE=b300|h100}"
: "${VLLM_MODEL:?set VLLM_MODEL}"
: "${SERVED_MODEL_NAME:?set SERVED_MODEL_NAME}"
: "${VLLM_VERSION:=0.23.0}"
: "${VLLM_PORT:=8008}"
: "${GPU_COUNT:=8}"
: "${MAX_MODEL_LEN:=131072}"
: "${VLLM_MAX_NUM_SEQS:=32}"
: "${VLLM_ENABLE_MTP:=0}"
: "${VLLM_READY_TIMEOUT:=3600}"
: "${TASKS_HF_DATASET:=allenai/TMax-SFT-16.5K}"
: "${TASKS_DIR_NAME:=tasks_sft_16.5k}"
: "${NUM_SOLUTIONS:=8}"
: "${MAX_ACTIONS:=16}"
: "${MAX_TOKENS:=$(( MAX_MODEL_LEN / 4 ))}"
: "${NUM_TASKS:=999999}"
: "${START_AT:=0}"
: "${WORKERS:=12}"
: "${NUM_POOL_WORKERS:=16}"
: "${SOLUTION_TEMPERATURE:=0.7}"
: "${COMMAND_TIMEOUT:=600}"
: "${SHELL_INIT_TIMEOUT:=240}"
: "${SHELL_INIT_ATTEMPTS:=3}"
: "${BUILD_WORKERS:=4}"
: "${BUILD_RETRIES:=3}"
: "${SAMPLE_SIZE:=0}"
: "${SAMPLE_SEED:=0}"
: "${FORCE_RERUN:=0}"
: "${RESULTS_DIR:=/results}"
: "${SYNC_INTERVAL:=300}"
: "${APPTAINER_VERSION:=1.4.4}"

# Parser defaults keyed off the model family.
if [[ "$VLLM_MODEL" == *GLM* || "$VLLM_MODEL" == *glm* ]]; then
    : "${VLLM_TOOL_CALL_PARSER:=glm47}"
    : "${VLLM_REASONING_PARSER:=glm45}"
else
    : "${VLLM_TOOL_CALL_PARSER:=hermes}"
    : "${VLLM_REASONING_PARSER:=}"
fi

# TP/DP defaults per profile: b300 serves one big TP=8 engine; h100 is the
# generic small-model profile where DP over TP wins.
if [[ "$HARDWARE" == "b300" ]]; then
    : "${TP_SIZE:=8}"
    : "${DP_SIZE:=1}"
else
    : "${TP_SIZE:=1}"
    : "${DP_SIZE:=$(( GPU_COUNT / TP_SIZE ))}"
fi

TASKS_DIR="rl_data/output/${TASKS_DIR_NAME}"

# --- 1. Base tooling ---------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
log "running uv sync"
uv sync

log "installing apt deps (rsync, squashfs-tools, ...)"
export DEBIAN_FRONTEND=noninteractive
# Some AI2 CUDA images ship stale third-party apt sources (e.g.
# packages.lunarg.com/vulkan) whose Release file has been taken down; that
# alone makes `apt-get update` exit non-zero under set -e even though the
# Ubuntu repos refreshed fine. Drop known-dead lists and tolerate a partial
# index refresh — the install step below will fail loudly if anything we
# actually need is missing.
rm -f /etc/apt/sources.list.d/*lunarg* /etc/apt/sources.list.d/*vulkan* 2>/dev/null || true
apt-get update -qq || log "WARN: apt-get update failed for some repos (continuing)"
apt-get install -y -qq rsync curl ca-certificates squashfs-tools uidmap

# --- 1b. CUDA build toolchain (nvcc) ------------------------------------------
# On Blackwell, vLLM's FlashInfer backend JIT-compiles its TRT-LLM fused-MoE
# kernels for sm_103a at engine start — that needs a real nvcc, and the ai2
# runtime images ship CUDA libs but no dev packages ("nvcc: not found").
# Install NVIDIA's minimal build metapackage (nvcc + cudart headers) matching
# the CUDA 13.0 runtime when nvcc is missing. gcc comes via build-essential
# (nvcc needs a host compiler).
: "${CUDA_BUILD_PKG:=cuda-minimal-build-13-0}"
if ! command -v nvcc >/dev/null 2>&1; then
    log "nvcc missing — installing ${CUDA_BUILD_PKG} from NVIDIA's apt repo"
    if ! apt-get install -y -qq "$CUDA_BUILD_PKG" build-essential 2>/dev/null; then
        . /etc/os-release
        _ubu="ubuntu$(echo "$VERSION_ID" | tr -d .)"
        curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${_ubu}/x86_64/cuda-keyring_1.1-1_all.deb" -o /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb
        apt-get update -qq || true
        apt-get install -y -qq "$CUDA_BUILD_PKG" build-essential
    fi
fi
_cuda_dir="$(ls -d /usr/local/cuda-13.* 2>/dev/null | sort -V | tail -1 || true)"
if [ -n "$_cuda_dir" ]; then
    export CUDA_HOME="${CUDA_HOME:-$_cuda_dir}"
    export PATH="$CUDA_HOME/bin:$PATH"
    [ -e /usr/local/cuda ] || ln -s "$_cuda_dir" /usr/local/cuda
fi
nvcc --version 2>/dev/null | grep release || log "WARN: nvcc still unavailable — FlashInfer JIT will fail"

# --- 2. Apptainer -------------------------------------------------------------
if ! command -v apptainer >/dev/null 2>&1; then
    log "installing apptainer ${APPTAINER_VERSION}"
    _arch="$(dpkg --print-architecture)"
    curl -fsSL \
        "https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer_${APPTAINER_VERSION}_${_arch}.deb" \
        -o /tmp/apptainer.deb
    apt-get install -y -qq /tmp/apptainer.deb
fi
apptainer --version

# env.py always passes --fakeroot; fakeroot resolves subuid/subgid ranges for
# the calling user (root inside the job container), so make sure they exist.
grep -q '^root:' /etc/subuid 2>/dev/null || echo 'root:100000:65536' >> /etc/subuid
grep -q '^root:' /etc/subgid 2>/dev/null || echo 'root:100000:65536' >> /etc/subgid

# Preflight: apptainer's --fakeroot/--userns flags need unprivileged user
# namespaces. Where the kernel/runtime blocks them (as on holmes Beaker jobs),
# drop both flags via the env.py opt-out — we run as real root inside the job
# container, so they add nothing anyway.
if ! unshare -U true 2>/dev/null; then
    log "user namespaces unavailable — running apptainer as plain root (APPTAINER_NO_USERNS=1)"
    export APPTAINER_NO_USERNS=1
fi

# Cache + scratch on local disk (fast, ephemeral).
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/tmp/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/apptainer_tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" /tmp/apptainer_instances
mkdir -p "$HOME/.apptainer"
if [ ! -L "$HOME/.apptainer/instances" ]; then
    rm -rf "$HOME/.apptainer/instances"
    ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

# --- 3. HF cache on weka when available (GLM-5.2-FP8 is ~700 GB) -------------
if [ -z "${HF_CACHE_DIR:-}" ] && [ -d /weka/oe-adapt-default ]; then
    HF_CACHE_DIR="/weka/oe-adapt-default/${USER:-$(whoami)}/hf_cache"
fi
if [ -n "${HF_CACHE_DIR:-}" ]; then
    export HF_HOME="$HF_CACHE_DIR"
    mkdir -p "$HF_HOME"
fi
export HF_HUB_ENABLE_HF_TRANSFER=1
log "HF_HOME=${HF_HOME:-<default>}"

# --- 4. Download + extract the task corpus ------------------------------------
if [ -d "$TASKS_DIR" ] && [ "$(find "$TASKS_DIR" -maxdepth 1 -type d -name 'task_*' | wc -l)" -gt 0 ]; then
    log "task corpus already present at $TASKS_DIR — skipping download"
else
    log "downloading tasks.zip from $TASKS_HF_DATASET"
    TASKS_ZIP="$(uv run python - <<PY
from huggingface_hub import hf_hub_download
print(hf_hub_download("${TASKS_HF_DATASET}", "tasks.zip", repo_type="dataset"))
PY
)"
    log "extracting $TASKS_ZIP -> $TASKS_DIR"
    mkdir -p "$TASKS_DIR"
    (cd "$TASKS_DIR" && python3 -m zipfile -e "$TASKS_ZIP" .)
fi
N_TASK_DIRS="$(find "$TASKS_DIR" -maxdepth 1 -type d -name 'task_*' | wc -l)"
log "task corpus ready: $N_TASK_DIRS task dirs"

# --- 5. Start vLLM in the background ------------------------------------------
VLLM_LOG=/tmp/vllm.log
VLLM_LOG_TAIL_LINES="${VLLM_LOG_TAIL_LINES:-300}"

VLLM_CMD=( uvx "vllm==${VLLM_VERSION}" serve "$VLLM_MODEL"
           --served-model-name "$SERVED_MODEL_NAME"
           --port "$VLLM_PORT"
           --tensor-parallel-size "$TP_SIZE"
           --max-model-len "$MAX_MODEL_LEN"
           --max-num-seqs "$VLLM_MAX_NUM_SEQS"
           --enable-auto-tool-choice
           --tool-call-parser "$VLLM_TOOL_CALL_PARSER"
           --enable-prefix-caching )
if [ "$DP_SIZE" -gt 1 ]; then
    VLLM_CMD+=( --data-parallel-size "$DP_SIZE" )
fi
if [ -n "${VLLM_REASONING_PARSER:-}" ]; then
    VLLM_CMD+=( --reasoning-parser "$VLLM_REASONING_PARSER" )
fi
if [ -n "${VLLM_GPU_UTIL:-}" ]; then
    VLLM_CMD+=( --gpu-memory-utilization "$VLLM_GPU_UTIL" )
fi
if [[ "$HARDWARE" == "b300" ]]; then
    # Per the recipes.vllm.ai GLM-5.2 B200/B300 config: FP8 KV cache.
    VLLM_CMD+=( --kv-cache-dtype fp8_e4m3 )
    # DeepGEMM (JIT-compiled FP8 GEMMs) is disabled by default: its JIT needs
    # a CUDA toolchain new enough for the B300's sm_103 (>= 12.9; the ai2
    # images ship 12.8) and its shared JIT cache corrupts under 8 TP workers
    # compiling concurrently ("Corrupted JIT cache directory"). The CUTLASS
    # fallback kernels are precompiled into the vLLM wheel — slower, but no
    # JIT. Set VLLM_USE_DEEP_GEMM=1 to opt back in (e.g. on a cu129+ image).
    : "${VLLM_USE_DEEP_GEMM:=0}"
    if [ "$VLLM_USE_DEEP_GEMM" = "1" ]; then
        # DeepGEMM JITs with the image's nvcc; B300 (sm_103) needs CUDA >= 12.9.
        # Downgrade to the CUTLASS fallback instead of crashing at weight load.
        NVCC_VER="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
        NVCC_NUM=0
        if [ -n "$NVCC_VER" ]; then
            NVCC_NUM=$(( $(echo "$NVCC_VER" | cut -d' ' -f1) * 100 + $(echo "$NVCC_VER" | cut -d' ' -f2) ))
        fi
        if [ "$NVCC_NUM" -lt 1209 ]; then
            log "WARN: nvcc missing or < 12.9 (got '${NVCC_VER:-none}') — DeepGEMM can't target sm_103; falling back to CUTLASS (VLLM_USE_DEEP_GEMM=0)"
            VLLM_USE_DEEP_GEMM=0
        else
            export VLLM_DEEP_GEMM_WARMUP=skip
            rm -rf "$HOME/.cache/deep_gemm"   # clear partial entries from prior crashes
        fi
    fi
    export VLLM_USE_DEEP_GEMM
fi
if [ "$VLLM_ENABLE_MTP" = "1" ]; then
    VLLM_CMD+=( --speculative-config.method mtp
                --speculative-config.num_speculative_tokens 5 )
fi
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    VLLM_CMD+=( ${VLLM_EXTRA_ARGS} )
fi

log "launching vllm: ${VLLM_CMD[*]}"
"${VLLM_CMD[@]}" >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# --- 6. Result sync (periodic + on exit) ---------------------------------------
mkdir -p "$RESULTS_DIR"
sync_results() {
    # Solutions summaries + run logs, preserving the task_*/solutions/ tree.
    rsync -am \
        --include='task_*/' --include='task_*/solutions/***' \
        --include='logs/***' --exclude='*' \
        "$TASKS_DIR/" "$RESULTS_DIR/$TASKS_DIR_NAME/" 2>/dev/null || true
}

cleanup() {
    log "cleanup: killing vllm pid $VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    log "cleanup: final result sync -> $RESULTS_DIR"
    sync_results
    tail -n "$VLLM_LOG_TAIL_LINES" "$VLLM_LOG" > "$RESULTS_DIR/vllm_tail.log" 2>/dev/null || true
}
trap cleanup EXIT

(
    while true; do
        sleep "$SYNC_INTERVAL"
        sync_results
        # Count only THIS run's summaries — the published corpus ships with
        # summaries from other models in the same solutions/ dirs.
        n_done="$(find "$TASKS_DIR" -name "hosted_vllm_${SERVED_MODEL_NAME}_summary.json" 2>/dev/null | wc -l)"
        log "progress: periodic sync done ($n_done/$N_TASK_DIRS tasks have a ${SERVED_MODEL_NAME} summary)"
    done
) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; cleanup' EXIT

# --- 7. Wait for vLLM ----------------------------------------------------------
log "waiting for vllm on :$VLLM_PORT (up to ${VLLM_READY_TIMEOUT}s)"
_deadline=$(( SECONDS + VLLM_READY_TIMEOUT ))
until curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "vllm process died — tail of $VLLM_LOG:"
        tail -n "$VLLM_LOG_TAIL_LINES" "$VLLM_LOG" || true
        exit 1
    fi
    if [ "$SECONDS" -ge "$_deadline" ]; then
        log "vllm not ready after ${VLLM_READY_TIMEOUT}s — tail of $VLLM_LOG:"
        tail -n "$VLLM_LOG_TAIL_LINES" "$VLLM_LOG" || true
        exit 1
    fi
    sleep 10
done
log "vllm ready"

# --- 8. Run the solver ----------------------------------------------------------
export MODEL="hosted_vllm/${SERVED_MODEL_NAME}"
export HOSTED_VLLM_API_BASE="http://localhost:${VLLM_PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

_MODEL_TAG="$(echo "$MODEL" | tr '/' '_')"
_RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
TERMINAL_LOG="$(pwd)/${TASKS_DIR}/logs/${_MODEL_TAG}_${_RUN_TS}.log"
mkdir -p "$(dirname "$TERMINAL_LOG")"

EXTRA_ARGS=( --base-sifs-dir rl_data/containers --terminal-log "$TERMINAL_LOG" )
if [ "$FORCE_RERUN" = "1" ]; then
    EXTRA_ARGS+=( --force-rerun )
fi
if [ "$SAMPLE_SIZE" != "0" ]; then
    EXTRA_ARGS+=( --sample-size "$SAMPLE_SIZE" --sample-seed "$SAMPLE_SEED" )
fi

log "running generate_solutions: MODEL=$MODEL WORKERS=$WORKERS NUM_SOLUTIONS=$NUM_SOLUTIONS (containers: $(( WORKERS * NUM_SOLUTIONS )))"
uv run python -m rl_data.generate_solutions \
    --tasks-dir "$TASKS_DIR" \
    --model "$MODEL" \
    --num-solutions "$NUM_SOLUTIONS" \
    --max-actions "$MAX_ACTIONS" \
    --max-tokens "$MAX_TOKENS" \
    --num-tasks "$NUM_TASKS" \
    --start-at "$START_AT" \
    --workers "$WORKERS" \
    --num-pool-workers "$NUM_POOL_WORKERS" \
    --solution-temperature "$SOLUTION_TEMPERATURE" \
    --command-timeout "$COMMAND_TIMEOUT" \
    --shell-init-timeout "$SHELL_INIT_TIMEOUT" \
    --shell-init-attempts "$SHELL_INIT_ATTEMPTS" \
    --build-workers "$BUILD_WORKERS" \
    --build-retries "$BUILD_RETRIES" \
    --verbose \
    "${EXTRA_ARGS[@]}"

log "solver finished — final sync"
sync_results
log "done"
