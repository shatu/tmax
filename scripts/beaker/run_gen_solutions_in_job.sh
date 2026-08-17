#!/usr/bin/env bash
#
# Inner script invoked inside a beaker task. Downloads a task corpus from the
# HF Hub, sets up podman, spins up a vLLM server on the local GPUs, then runs
# rl_data.generate_solutions (--container-runtime podman) against it.
# Sibling of run_eval_in_job.sh (harbor evals) — same podman-on-Beaker
# foundations, different workload.
#
# Why podman and not apptainer: AI2 Beaker jobs run inside a user namespace
# whose uid/gid mapping writes are host-denied (holmes) and whose /proc masks
# are kernel-locked (everywhere) — apptainer's build engine AND runtime are
# both unusable there. Rootful podman with userns=host + a /proc:/proc volume
# needs neither; every primitive the solver uses was validated by probe jobs
# on ai2/holmes (2026-08-16). See scripts/beaker/README.md.
#
# Driven entirely by env vars (set by beaker_configs/launch_gen_solutions.sh):
#
#   -- vLLM serving --
#   HARDWARE                 b300 | h100 — selects the serving profile (required)
#   VLLM_MODEL               HF repo to serve (required, e.g. zai-org/GLM-5.2-FP8)
#   SERVED_MODEL_NAME        --served-model-name (required, e.g. glm-5.2-fp8)
#   VLLM_VERSION             vLLM package version for uvx (default: 0.23.0)
#   VLLM_PORT                port for vLLM (default: 8008)
#   TP_SIZE / DP_SIZE        parallelism (defaults: TP=8 DP=1 on b300; TP=1
#                            DP=GPU_COUNT on h100)
#   GPU_COUNT                GPUs allocated to the job (default: 8)
#   MAX_MODEL_LEN            --max-model-len (default: 131072)
#   VLLM_GPU_UTIL            --gpu-memory-utilization (default: vllm's default)
#   VLLM_MAX_NUM_SEQS        --max-num-seqs (default: 32, per the GLM recipe)
#   VLLM_TOOL_CALL_PARSER    tool parser (default: glm47 for GLM, hermes otherwise)
#   VLLM_REASONING_PARSER    reasoning parser (default: glm45 for GLM, empty otherwise)
#   VLLM_ENABLE_MTP          1 to add MTP speculative decoding (default 0:
#                            tool-calling + MTP only coexist on vllm main)
#   VLLM_USE_DEEP_GEMM       1 to enable DeepGEMM JIT FP8 kernels (default 0;
#                            needs nvcc >= 12.9 for B300 sm_103)
#   VLLM_EXTRA_ARGS          free-form extra args appended to vllm serve
#   VLLM_READY_TIMEOUT       seconds to wait for /v1/models (default: 5400 —
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
#   MAX_TRAJECTORY_TOKENS    stop rollouts whose est. training-format length
#                            (history + completions incl. reasoning) exceeds
#                            this; set to the SFT max_seq_length (default: 0 = off)
#   SAMPLE_SIZE / SAMPLE_SEED  optional fixed random subset (default: 0 = off)
#   FORCE_RERUN              1 to re-run tasks that already have a summary (default: 0)
#
#   -- plumbing --
#   BUILD_IMAGES_ONLY        1 to only ensure the 10 base images exist in the
#                            weka tar cache, then exit (CPU-only; no vLLM)
#   IMAGE_CACHE_DIR          podman-save tar cache. Default:
#                            /weka/oe-adapt-default/pradeepd/tmax_base_images
#                            when that weka mount is present.
#   SUMMARY_CACHE_DIR        shared cross-job summary store. Default:
#                            /weka/oe-adapt-default/pradeepd/tmax_solutions/<TASKS_DIR_NAME>
#                            when weka is mounted. Jobs RESTORE this model's
#                            summaries from it at startup (so completed tasks
#                            are skipped across jobs/shards/preemptions) and
#                            sync new ones back every SYNC_INTERVAL.
#   RESULTS_DIR              where to sync solutions/ summaries (default: /results,
#                            persisted by gantry as the PER-JOB result dataset)
#   SYNC_INTERVAL            seconds between periodic result syncs (default: 300)
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
: "${VLLM_USE_DEEP_GEMM:=0}"
: "${VLLM_READY_TIMEOUT:=5400}"
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
: "${MAX_TRAJECTORY_TOKENS:=0}"
: "${SAMPLE_SIZE:=0}"
: "${SAMPLE_SEED:=0}"
: "${FORCE_RERUN:=0}"
: "${BUILD_IMAGES_ONLY:=0}"
: "${RESULTS_DIR:=/results}"
: "${SYNC_INTERVAL:=300}"

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

log "installing apt deps (podman, rsync, ...)"
export DEBIAN_FRONTEND=noninteractive
# Some AI2 CUDA images ship stale third-party apt sources (e.g.
# packages.lunarg.com/vulkan) whose Release file has been taken down; that
# alone makes `apt-get update` exit non-zero under set -e even though the
# Ubuntu repos refreshed fine. Drop known-dead lists and tolerate a partial
# index refresh — the install step below will fail loudly if anything we
# actually need is missing.
rm -f /etc/apt/sources.list.d/*lunarg* /etc/apt/sources.list.d/*vulkan* 2>/dev/null || true
apt-get update -qq || log "WARN: apt-get update failed for some repos (continuing)"
apt-get install -y -qq rsync curl ca-certificates \
    podman crun uidmap fuse-overlayfs slirp4netns

# --- 2. Podman config ----------------------------------------------------------
# The exact configuration validated by the runtime-smoke probes on holmes:
# rootful podman, host namespaces everywhere (userns=host needs none of the
# uid-mapping writes Beaker hosts deny), and a /proc:/proc volume so
# containers get a bound proc instead of a fresh procfs mount (which the
# kernel refuses while the job's locked /proc masks exist).
log "writing /etc/containers/containers.conf"
mkdir -p /etc/containers
cat > /etc/containers/containers.conf <<'CONF'
[containers]
netns="host"
userns="host"
ipcns="host"
utsns="host"
cgroupns="host"
cgroups="disabled"
log_driver = "k8s-file"
volumes = [
        "/proc:/proc",
]
default_sysctls = []
[engine]
cgroup_manager = "cgroupfs"
events_logger="file"
runtime="crun"
CONF

# --- 3. Weka-backed caches ------------------------------------------------------
if [ -z "${IMAGE_CACHE_DIR:-}" ] && [ -d /weka/oe-adapt-default ]; then
    IMAGE_CACHE_DIR="/weka/oe-adapt-default/pradeepd/tmax_base_images"
fi
if [ -z "${HF_CACHE_DIR:-}" ] && [ -d /weka/oe-adapt-default ]; then
    HF_CACHE_DIR="/weka/oe-adapt-default/${USER:-$(whoami)}/hf_cache"
fi
if [ -n "${HF_CACHE_DIR:-}" ]; then
    export HF_HOME="$HF_CACHE_DIR"
    mkdir -p "$HF_HOME"
fi
export HF_HUB_ENABLE_HF_TRANSFER=1
log "HF_HOME=${HF_HOME:-<default>}  IMAGE_CACHE_DIR=${IMAGE_CACHE_DIR:-<none>}"

# Ensure the 10 shared base images exist (present > cache tar > podman build).
# Runs generate_solutions' own ensure step so job and solver can't drift.
ensure_base_images() {
    TMAX_IMAGE_CACHE_DIR="${IMAGE_CACHE_DIR:-}" uv run python - <<'PY'
import os
from rl_data.generate_solutions import SolutionConfig, _ensure_podman_base_images
cfg = SolutionConfig(
    tasks_dir="unused",
    container_runtime="podman",
    image_cache_dir=os.environ.get("TMAX_IMAGE_CACHE_DIR") or None,
)
_ensure_podman_base_images(cfg)
PY
}

# --- 3b. Build-images-only mode ---------------------------------------------------
if [ "$BUILD_IMAGES_ONLY" = "1" ]; then
    log "BUILD_IMAGES_ONLY=1 — ensuring base images in the cache, then exiting"
    ensure_base_images
    log "build-images-only done"
    exit 0
fi

# --- 4. CUDA build toolchain (nvcc) ------------------------------------------
# On Blackwell, vLLM's FlashInfer backend JIT-compiles its TRT-LLM fused-MoE
# kernels for sm_103a at engine start — that needs a real nvcc, and the ai2
# runtime images ship CUDA libs but no dev packages. cuda-libraries-dev is
# the same metapackage the official nvidia/cuda devel images install: it
# covers ALL the CUDA library dev headers FlashInfer includes (cublasLt.h,
# curand_kernel.h, nvrtc.h, ...).
: "${CUDA_BUILD_PKGS:=cuda-minimal-build-13-0 cuda-libraries-dev-13-0}"
if ! command -v nvcc >/dev/null 2>&1; then
    log "nvcc missing — installing ${CUDA_BUILD_PKGS} from NVIDIA's apt repo"
    # shellcheck disable=SC2086
    if ! apt-get install -y -qq $CUDA_BUILD_PKGS build-essential 2>/dev/null; then
        . /etc/os-release
        _ubu="ubuntu$(echo "$VERSION_ID" | tr -d .)"
        curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${_ubu}/x86_64/cuda-keyring_1.1-1_all.deb" -o /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb
        apt-get update -qq || true
        # shellcheck disable=SC2086
        apt-get install -y -qq $CUDA_BUILD_PKGS build-essential
    fi
fi
_cuda_dir="$(ls -d /usr/local/cuda-13.* 2>/dev/null | sort -V | tail -1 || true)"
if [ -n "$_cuda_dir" ]; then
    export CUDA_HOME="${CUDA_HOME:-$_cuda_dir}"
    export PATH="$CUDA_HOME/bin:$PATH"
    [ -e /usr/local/cuda ] || ln -s "$_cuda_dir" /usr/local/cuda
fi
nvcc --version 2>/dev/null | grep release || log "WARN: nvcc still unavailable — FlashInfer JIT will fail"

# --- 5. Download + extract the task corpus ------------------------------------
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
    (cd "$TASKS_DIR" && uv run python -m zipfile -e "$TASKS_ZIP" .)
fi
N_TASK_DIRS="$(find "$TASKS_DIR" -maxdepth 1 -type d -name 'task_*' | wc -l)"
log "task corpus ready: $N_TASK_DIRS task dirs"

# --- 5b. Cross-job summary cache on weka ----------------------------------------
# The gantry results dataset is per-job, and the workdir is ephemeral — without
# a shared store, every new job (or preemption resume) would redo completed
# tasks. Restore this model's summaries from weka so generate_solutions' skip
# logic sees prior work; sync_results writes new ones back continuously.
if [ -z "${SUMMARY_CACHE_DIR:-}" ] && [ -d /weka/oe-adapt-default ]; then
    SUMMARY_CACHE_DIR="/weka/oe-adapt-default/pradeepd/tmax_solutions/${TASKS_DIR_NAME}"
fi
_SUMMARY_GLOB="hosted_vllm_${SERVED_MODEL_NAME}*"
if [ -n "${SUMMARY_CACHE_DIR:-}" ] && [ -d "$SUMMARY_CACHE_DIR" ]; then
    _n_restored="$(find "$SUMMARY_CACHE_DIR" -name "$_SUMMARY_GLOB" 2>/dev/null | wc -l)"
    if [ "$_n_restored" -gt 0 ]; then
        log "restoring $_n_restored ${SERVED_MODEL_NAME} summarie(s) from $SUMMARY_CACHE_DIR"
        rsync -am \
            --include='task_*/' --include='task_*/solutions/' \
            --include="task_*/solutions/${_SUMMARY_GLOB}" --exclude='*' \
            "$SUMMARY_CACHE_DIR/" "$TASKS_DIR/"
    fi
fi
mkdir -p "${SUMMARY_CACHE_DIR:-/tmp/tmax_solutions}"

# --- 6. Start vLLM in the background ------------------------------------------
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
    # DeepGEMM JIT is opt-in: needs nvcc >= 12.9 for sm_103, and its shared
    # JIT cache corrupts under 8 TP workers on a cold start. The CUTLASS
    # fallback kernels are precompiled into the wheel.
    export VLLM_USE_DEEP_GEMM
    if [ "$VLLM_USE_DEEP_GEMM" = "1" ]; then
        export VLLM_DEEP_GEMM_WARMUP=skip
        rm -rf "$HOME/.cache/deep_gemm"
    fi
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

# --- 6b. Ensure base images while vLLM loads -----------------------------------
# Image ensure (cache load or first-time podman build) overlaps the ~30-50 min
# weight load instead of serializing after it.
ENSURE_LOG=/tmp/ensure_images.log
ensure_base_images >"$ENSURE_LOG" 2>&1 &
ENSURE_PID=$!

# --- 7. Result sync (periodic + on exit) ---------------------------------------
mkdir -p "$RESULTS_DIR"
sync_results() {
    rsync -am \
        --include='task_*/' --include='task_*/solutions/***' \
        --include='logs/***' --exclude='*' \
        "$TASKS_DIR/" "$RESULTS_DIR/$TASKS_DIR_NAME/" 2>/dev/null || true
    # Shared weka store: only THIS model's summaries (the shipped baseline
    # summaries ride along in the corpus already), so shards/preemptions/
    # future jobs skip completed tasks and conversion reads one location.
    if [ -n "${SUMMARY_CACHE_DIR:-}" ]; then
        rsync -am \
            --include='task_*/' --include='task_*/solutions/' \
            --include="task_*/solutions/${_SUMMARY_GLOB}" --exclude='*' \
            "$TASKS_DIR/" "$SUMMARY_CACHE_DIR/" 2>/dev/null || true
    fi
}

cleanup() {
    log "cleanup: killing vllm pid $VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    log "cleanup: final result sync -> $RESULTS_DIR"
    sync_results
    tail -n "$VLLM_LOG_TAIL_LINES" "$VLLM_LOG" > "$RESULTS_DIR/vllm_tail.log" 2>/dev/null || true
    cp "$ENSURE_LOG" "$RESULTS_DIR/ensure_images.log" 2>/dev/null || true
}
trap cleanup EXIT

(
    while true; do
        sleep "$SYNC_INTERVAL"
        sync_results
        n_done="$(find "$TASKS_DIR" -name "hosted_vllm_${SERVED_MODEL_NAME}_summary.json" 2>/dev/null | wc -l)"
        log "progress: periodic sync done ($n_done/$N_TASK_DIRS tasks have a ${SERVED_MODEL_NAME} summary)"
    done
) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; cleanup' EXIT

# --- 8. Wait for image ensure + vLLM --------------------------------------------
if ! wait "$ENSURE_PID"; then
    log "base-image ensure FAILED — tail of $ENSURE_LOG:"
    tail -40 "$ENSURE_LOG" || true
    exit 1
fi
log "base images ready:"
grep -E "ready:|Cached|Loading|Building" "$ENSURE_LOG" | tail -12 || true

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

# --- 9. Run the solver ----------------------------------------------------------
export MODEL="hosted_vllm/${SERVED_MODEL_NAME}"
export HOSTED_VLLM_API_BASE="http://localhost:${VLLM_PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

_MODEL_TAG="$(echo "$MODEL" | tr '/' '_')"
_RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
TERMINAL_LOG="$(pwd)/${TASKS_DIR}/logs/${_MODEL_TAG}_${_RUN_TS}.log"
mkdir -p "$(dirname "$TERMINAL_LOG")"

EXTRA_ARGS=( --container-runtime podman --terminal-log "$TERMINAL_LOG" )
if [ -n "${IMAGE_CACHE_DIR:-}" ]; then
    EXTRA_ARGS+=( --image-cache-dir "$IMAGE_CACHE_DIR" )
fi
if [ "$FORCE_RERUN" = "1" ]; then
    EXTRA_ARGS+=( --force-rerun )
fi
if [ "$MAX_TRAJECTORY_TOKENS" != "0" ]; then
    EXTRA_ARGS+=( --max-trajectory-tokens "$MAX_TRAJECTORY_TOKENS" )
fi
if [ "$SAMPLE_SIZE" != "0" ]; then
    EXTRA_ARGS+=( --sample-size "$SAMPLE_SIZE" --sample-seed "$SAMPLE_SEED" )
fi

log "running generate_solutions: MODEL=$MODEL WORKERS=$WORKERS NUM_SOLUTIONS=$NUM_SOLUTIONS (containers: $(( WORKERS * NUM_SOLUTIONS )), runtime: podman)"
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
    --verbose \
    "${EXTRA_ARGS[@]}"

log "solver finished — final sync"
sync_results
log "done"
