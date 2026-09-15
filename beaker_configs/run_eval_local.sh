#!/usr/bin/env bash
#
# LOCAL smoke-test mirror of scripts/beaker/run_eval_in_job.sh.
#
# Runs the same harbor eval pipeline (vLLM + harbor `--env docker` + VanilluxAgent)
# on the local machine against the REAL Docker daemon, so you can validate the
# flow on a couple of tasks before paying for a full Beaker job via
# beaker_configs/launch_eval.sh.
#
# Differences vs. the beaker inner script (scripts/beaker/run_eval_in_job.sh):
#   * No apt-get / podman / uidmap install — we use the host Docker daemon.
#   * No `podman system service` — DOCKER_HOST stays at the default.
#   * Of the harbor source patches, only `network_mode: host` is applied. The
#     `:U` bind-mount and chmod-0o777 patches exist purely to work around
#     podman's user-namespace remapping; rootful Docker writes bind mounts
#     directly, so they're unnecessary. network_mode: host IS still required:
#     the SWE-agent runs INSIDE the task container and calls vLLM at
#     localhost:$VLLM_PORT, which only resolves to the host vLLM when the
#     container shares the host network namespace.
#   * Defaults to a tiny run: TP=1 on one GPU, 2 tasks, 2 concurrent.
#   * Defaults to harbor `--env docker`. Pass `--harbor-env modal` to run each
#     task container as a Modal cloud sandbox instead: no local Docker daemon,
#     no Docker Hub PAT, and no compose patch is needed, but the agent must be
#     an in-process one (see the --harbor-env note below).
#
# Usage:
#   ./beaker_configs/run_eval_local.sh [model_path] [options]
#
# Example (defaults: Qwen/Qwen3.5-4B, mini-swe-agent, terminal-bench@2.0, 2 tasks):
#   ./beaker_configs/run_eval_local.sh
#   ./beaker_configs/run_eval_local.sh Qwen/Qwen3.5-4B --n-concurrent 1 --task fix-git
#   ./beaker_configs/run_eval_local.sh Qwen/Qwen3.5-4B --agent swe-agent --n-tasks 1
#
# Modal smoke test (containers in Modal's cloud, model served locally):
#   export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
#   ./beaker_configs/run_eval_local.sh Qwen/Qwen3.5-4B \
#       --harbor-env modal --agent Vanillux2Agent:Vanillux2Agent \
#       --model-provider openai --tool-call-parser qwen3_xml \
#       --max-model-len 32768 --n-tasks 2
#
# Fastest possible Modal wiring check — no GPU, no vLLM, hosted API model:
#   export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... ANTHROPIC_API_KEY=...
#   ./beaker_configs/run_eval_local.sh --harbor-env modal --skip-vllm \
#       --harbor-model-name anthropic/claude-sonnet-4-5 \
#       --agent Vanillux2Agent:Vanillux2Agent --n-tasks 1

set -euo pipefail

log() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- defaults ----------------------------------------------------------------
MODEL_PATH="Qwen/Qwen3.5-4B"
REVISION="main"
SERVED_MODEL_NAME=""
GPU_DEVICES="0"            # CUDA_VISIBLE_DEVICES for vLLM
TP_SIZE=1
DP_SIZE=1
VLLM_PORT=8008
VLLM_VERSION="0.19.1"
VLLM_TOOL_CALL_PARSER="hermes"
VLLM_REASONING_PARSER=""
MODEL_PROVIDER=""
VLLM_LANGUAGE_MODEL_ONLY=0
MAX_MODEL_LEN=""
GPU_MEM_UTIL="0.85"
DATASET="terminal-bench@2.0"
DATASET_PATH=""            # local dataset/task dir (harbor --path); overrides --dataset when set
# Default to harbor's built-in mini-swe-agent: it works against the harbor
# version this repo locks (0.6.6) and drives a litellm openai/ model.
# NOTE: the Beaker default, VanilluxAgent:VanilluxAgent, imports
# `ExecInput` / `create_run_agent_commands` which DO NOT EXIST in harbor 0.6.6
# (nor any released harbor, nor harbor main) — it targets an unreleased/fork
# harbor, so `--agent VanilluxAgent:VanilluxAgent` fails to import here. Pass
# it explicitly only once the harbor pin is fixed.
AGENT_IMPORT_PATH="mini-swe-agent"
N_CONCURRENT=2
N_ATTEMPTS=1
N_TASKS=2                  # harbor -l / --n-tasks (small for a smoke test)
SINGLE_TASK=""             # harbor --include-task-name (overrides N_TASKS when set)
JOB_NAME=""
RESULTS_DIR=""
HARBOR_ENV="docker"        # harbor --env; "modal" runs containers as Modal sandboxes
HARBOR_ENV_KWARGS=""       # newline-separated harbor --environment-kwarg values
HARBOR_MODEL_NAME=""       # full litellm model id; overrides <provider>/<served-name>
SKIP_VLLM=0                # skip serving a local model (needs --harbor-model-name)

# first positional arg (if it doesn't start with --) is the model path
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then MODEL_PATH="$1"; shift; fi

while [ $# -gt 0 ]; do
    case "$1" in
        --revision)         REVISION="$2"; shift 2 ;;
        --name)             SERVED_MODEL_NAME="$2"; shift 2 ;;
        --gpu-devices)      GPU_DEVICES="$2"; shift 2 ;;
        --tp)               TP_SIZE="$2"; shift 2 ;;
        --dp)               DP_SIZE="$2"; shift 2 ;;
        --port)             VLLM_PORT="$2"; shift 2 ;;
        --vllm-version)     VLLM_VERSION="$2"; shift 2 ;;
        --tool-call-parser) VLLM_TOOL_CALL_PARSER="$2"; shift 2 ;;
        --reasoning-parser) VLLM_REASONING_PARSER="$2"; shift 2 ;;
        --model-provider)   MODEL_PROVIDER="$2"; shift 2 ;;
        --language-model-only|--language_model_only) VLLM_LANGUAGE_MODEL_ONLY=1; shift ;;
        --max-model-len)    MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-mem-util)     GPU_MEM_UTIL="$2"; shift 2 ;;
        --dataset)          DATASET="$2"; shift 2 ;;
        --dataset-path)     DATASET_PATH="$2"; shift 2 ;;
        --agent)            AGENT_IMPORT_PATH="$2"; shift 2 ;;
        --n-concurrent)     N_CONCURRENT="$2"; shift 2 ;;
        --n-attempts)       N_ATTEMPTS="$2"; shift 2 ;;
        --n-tasks)          N_TASKS="$2"; shift 2 ;;
        --task)             SINGLE_TASK="$2"; shift 2 ;;
        --job-name)         JOB_NAME="$2"; shift 2 ;;
        --results-dir)      RESULTS_DIR="$2"; shift 2 ;;
        --harbor-env)       HARBOR_ENV="$2"; shift 2 ;;
        --env-kwarg)        HARBOR_ENV_KWARGS+="${HARBOR_ENV_KWARGS:+$'\n'}$2"; shift 2 ;;
        --harbor-model-name) HARBOR_MODEL_NAME="$2"; shift 2 ;;
        --skip-vllm)        SKIP_VLLM=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

if [ "$SKIP_VLLM" = "1" ] && [ -z "$HARBOR_MODEL_NAME" ]; then
    echo "FATAL: --skip-vllm needs --harbor-model-name (e.g. anthropic/claude-sonnet-4-5),"
    echo "       since there is no locally served model to point the agent at."; exit 1
fi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
DATASET_SLUG="${DATASET//[^A-Za-z0-9]/-}"
JOB_NAME="${JOB_NAME:-local-${SERVED_MODEL_NAME}-${DATASET_SLUG}}"

cat <<EOF
=== Local tmax eval smoke test ===
  Model:        ${MODEL_PATH}@${REVISION}
  Served name:  ${SERVED_MODEL_NAME}
  vLLM version: ${VLLM_VERSION}  (parser=${VLLM_TOOL_CALL_PARSER})
  GPUs:         CUDA_VISIBLE_DEVICES=${GPU_DEVICES} (TP=${TP_SIZE}, DP=${DP_SIZE})
  Dataset:      ${DATASET}
  Tasks:        ${SINGLE_TASK:-first ${N_TASKS}}  (n_concurrent=${N_CONCURRENT}, k=${N_ATTEMPTS})
  Agent:        ${AGENT_IMPORT_PATH}
  Harbor env:   ${HARBOR_ENV}
  Env kwargs:   ${HARBOR_ENV_KWARGS:-<none>}
  Local vLLM:   $([ "$SKIP_VLLM" = "1" ] && echo "skipped (model: ${HARBOR_MODEL_NAME:-<unset!>})" || echo "yes")
  Job name:     ${JOB_NAME}
  DOCKER_HOST:  ${DOCKER_HOST:-(default: local docker daemon)}
EOF

# Everything below is specific to running task containers on THIS machine's
# Docker daemon. Remote-sandbox backends (--harbor-env modal) need none of it:
# no daemon, no compose plugin, no Docker Hub PAT — Modal pulls and builds the
# task image on its own side.
if [ "$HARBOR_ENV" = "docker" ]; then
    # --- 0. Preconditions -------------------------------------------------------
    command -v docker >/dev/null 2>&1 || { echo "docker CLI not found"; exit 1; }
    docker info >/dev/null 2>&1 || { echo "docker daemon not reachable"; exit 1; }
    docker compose version >/dev/null 2>&1 || {
        log "installing docker compose v2 plugin (harbor shells out to 'docker compose')"
        mkdir -p /root/.docker/cli-plugins
        curl -fsSL \
            "https://github.com/docker/compose/releases/download/v2.39.4/docker-compose-linux-$(uname -m)" \
            -o /root/.docker/cli-plugins/docker-compose
        chmod +x /root/.docker/cli-plugins/docker-compose
    }

    # --- 0b. Docker Hub auth (mirrors scripts/beaker/run_eval_in_job.sh) ---------
    # harbor pulls task images from Docker Hub. Authenticate so pulls don't hit the
    # unauthenticated cap, VERIFY with `docker login`, and HARD-ABORT on failure —
    # no anonymous fallback (deterministic, matching the Beaker path). The PAT comes
    # from $DOCKER_PAT, else is read from the beaker secret via the beaker CLI.
    DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-shashankg209}"
    DOCKER_PAT_SECRET="${DOCKER_PAT_SECRET:-shashankg_DOCKER_PAT}"
    AUTH_WORKSPACE="${BEAKER_WORKSPACE:-ai2/oe-agents}"
    if [ -z "${DOCKER_PAT:-}" ] && command -v beaker >/dev/null 2>&1; then
        DOCKER_PAT="$(beaker secret read "$DOCKER_PAT_SECRET" --workspace "$AUTH_WORKSPACE" 2>/dev/null || true)"
    fi
    [ -n "${DOCKER_PAT:-}" ] || {
        echo "FATAL: no Docker Hub PAT. Set DOCKER_PAT, or grant beaker access to secret '$DOCKER_PAT_SECRET' in '$AUTH_WORKSPACE'."; exit 1; }
    # A broken credsStore (e.g. the VS Code dev-containers credential helper) makes
    # `docker login` fail to persist the auth — drop it (keep other keys) so login
    # writes a plain auth entry.
    if [ -f "$HOME/.docker/config.json" ] && grep -q '"credsStore"' "$HOME/.docker/config.json" 2>/dev/null; then
        log "neutralizing docker credsStore (backup: config.json.bak) so login persists"
        cp "$HOME/.docker/config.json" "$HOME/.docker/config.json.bak"
        python3 -c "import json,os;p=os.path.expanduser('~/.docker/config.json');d=json.load(open(p));d.pop('credsStore',None);d.pop('credHelpers',None);json.dump(d,open(p,'w'),indent=2)" \
            || { echo "FATAL: failed to rewrite ~/.docker/config.json"; exit 1; }
    fi
    log "docker login as '$DOCKERHUB_USERNAME'"
    if printf '%s' "$DOCKER_PAT" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin docker.io >/dev/null 2>&1; then
        log "Docker Hub login OK ($DOCKERHUB_USERNAME)"
    else
        echo "FATAL: Docker Hub login failed for '$DOCKERHUB_USERNAME'. Check DOCKERHUB_USERNAME and DOCKER_PAT / secret '$DOCKER_PAT_SECRET'."; exit 1
    fi
fi

log "uv sync"
uv sync

if [ "$HARBOR_ENV" = "modal" ]; then
    # Install the SDK directly, not `harbor[modal]`: the extra re-resolves
    # harbor itself and would upgrade it off the version uv.lock pins.
    log "installing modal SDK"
    uv pip install 'modal>=1.4.0'
    # Credentials, in precedence order: the ambient env, then the beaker secrets
    # (same fallback the DOCKER_PAT block above uses), then ~/.modal.toml. The
    # beaker path means a dev box needs no `modal token new` of its own, and it
    # keeps the token pair in exactly one place for both the local and the
    # Beaker leg.
    MODAL_WORKSPACE="${BEAKER_WORKSPACE:-ai2/oe-agents}"
    MODAL_TOKEN_ID_SECRET="${MODAL_TOKEN_ID_SECRET:-shashankg_MODAL_TOKEN_ID}"
    MODAL_TOKEN_SECRET_SECRET="${MODAL_TOKEN_SECRET_SECRET:-shashankg_MODAL_TOKEN_SECRET}"
    if [ -z "${MODAL_TOKEN_ID:-}" ] && command -v beaker >/dev/null 2>&1; then
        log "reading modal token from beaker secret '$MODAL_TOKEN_ID_SECRET' ($MODAL_WORKSPACE)"
        MODAL_TOKEN_ID="$(beaker secret read "$MODAL_TOKEN_ID_SECRET" --workspace "$MODAL_WORKSPACE" 2>/dev/null || true)"
        export MODAL_TOKEN_ID
    fi
    if [ -z "${MODAL_TOKEN_SECRET:-}" ] && command -v beaker >/dev/null 2>&1; then
        MODAL_TOKEN_SECRET="$(beaker secret read "$MODAL_TOKEN_SECRET_SECRET" --workspace "$MODAL_WORKSPACE" 2>/dev/null || true)"
        export MODAL_TOKEN_SECRET
    fi
    if [ ! -f "$HOME/.modal.toml" ] && { [ -z "${MODAL_TOKEN_ID:-}" ] || [ -z "${MODAL_TOKEN_SECRET:-}" ]; }; then
        echo "FATAL: --harbor-env modal needs credentials. Export MODAL_TOKEN_ID and"
        echo "       MODAL_TOKEN_SECRET, or grant beaker access to secrets"
        echo "       '$MODAL_TOKEN_ID_SECRET' / '$MODAL_TOKEN_SECRET_SECRET' in '$MODAL_WORKSPACE',"
        echo "       or run 'modal token new'."; exit 1
    fi
    log "modal credentials OK"
    # Modal injects its own client runtime into every sandbox image, and HOW it
    # does that depends on the workspace's image builder version. The oldest
    # (2023.12, still the default on a fresh workspace) runs
    # `pip install -r /modal_requirements.txt` WITH transitive deps and supports
    # only Python 3.10-3.12. Terminal-Bench task images are python:3.13-slim, so
    # they fail it twice over: unsupported interpreter, and aiohttp compiled from
    # source in an image with no gcc ("error: [Errno 2] ... 'gcc'"), surfacing as
    # a bare ImageBuildError per trial. 2024.10+ switched to
    # `uv pip install --system --no-deps`, which builds nothing. Pin it here so a
    # run does not depend on a dashboard setting at modal.com/settings/image-config.
    export MODAL_IMAGE_BUILDER_VERSION="${MODAL_IMAGE_BUILDER_VERSION:-2025.06}"
    log "modal image builder version: $MODAL_IMAGE_BUILDER_VERSION"
    log "patching harbor modal env onto Modal's current filesystem API"
    uv run python scripts/patch_harbor_modal.py
fi

if [ "$HARBOR_ENV" = "docker" ]; then
    # --- 1. Patch harbor compose: network_mode: host ----------------------------
    # Only patch needed for rootful Docker. Lets the in-container SWE-agent reach
    # the host's vLLM at localhost:$VLLM_PORT.
    log "patching harbor docker-compose-base.yaml (network_mode: host)"
    uv run python - <<'PY'
import pathlib, harbor
hdir = pathlib.Path(harbor.__file__).parent
compose = hdir / "environments/docker/docker-compose-base.yaml"
text = compose.read_text()
if "network_mode: host" not in text:
    text = text.replace(
        "  main:\n    volumes:",
        "  main:\n    network_mode: host\n    volumes:",
    )
    compose.write_text(text)
    print("patched: added network_mode: host")
else:
    print("already patched")
PY
fi

# Serving a model locally is optional: --skip-vllm + --harbor-model-name runs
# the eval against a hosted API instead, which is the quickest way to validate a
# sandbox backend on its own (no GPU, no 10-minute model load).
if [ "$SKIP_VLLM" != "1" ]; then
    # --- 2. Start vLLM in the background ----------------------------------------
    # Pin fastapi < 0.137: fastapi 0.137 changed the router internals and breaks
    # prometheus-fastapi-instrumentator (which vLLM mounts on every route), so the
    # API server 500s on every request including /v1/models — the readiness probe
    # then never passes. (Same pin as the Beaker run_eval_in_job.sh path.)
    VLLM_LOG=/tmp/vllm_local.log
    VLLM_CMD=( uvx --with "fastapi<0.137" "vllm==${VLLM_VERSION}" serve "$MODEL_PATH"
               --revision "$REVISION"
               --tokenizer-revision "$REVISION"
               --served-model-name "$SERVED_MODEL_NAME"
               --enable-auto-tool-choice
               --tool-call-parser "$VLLM_TOOL_CALL_PARSER"
               --port "$VLLM_PORT"
               --gpu-memory-utilization "$GPU_MEM_UTIL"
               --tensor-parallel-size "$TP_SIZE"
               --data-parallel-size "$DP_SIZE" )
    [ -n "$MAX_MODEL_LEN" ] && VLLM_CMD+=( --max-model-len "$MAX_MODEL_LEN" )
    # Reasoning models (e.g. Qwen3) emit <think>...</think>; --reasoning-parser
    # splits that into reasoning_content so tool-calls/content parse cleanly.
    [ -n "$VLLM_REASONING_PARSER" ] && VLLM_CMD+=( --reasoning-parser "$VLLM_REASONING_PARSER" )
    [ "$VLLM_LANGUAGE_MODEL_ONLY" = "1" ] && VLLM_CMD+=( --language_model_only )

    log "launching vllm (CUDA_VISIBLE_DEVICES=$GPU_DEVICES): ${VLLM_CMD[*]}"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "${VLLM_CMD[@]}" >"$VLLM_LOG" 2>&1 &
    VLLM_PID=$!

    cleanup() {
        log "cleanup: killing vllm pid $VLLM_PID"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    }
    trap cleanup EXIT

    # Gate readiness on a real completion: a 200 on /v1/models can precede the
    # engine being able to GENERATE (the first request then fails "model does not
    # exist", notably for the 9B). Probe /v1/chat/completions instead.
    vllm_can_generate() {
        curl -sf -X POST "http://localhost:$VLLM_PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
            >/dev/null 2>&1
    }
    log "waiting for vllm to serve completions on :$VLLM_PORT (up to 30 min) — tail $VLLM_LOG"
    VLLM_READY=0
    for _ in $(seq 1 360); do
        if vllm_can_generate; then
            log "vllm ready (completion probe ok)"; VLLM_READY=1; break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            log "vllm died — tail of $VLLM_LOG:"; tail -200 "$VLLM_LOG" || true; exit 1
        fi
        sleep 5
    done
    [ "$VLLM_READY" -eq 1 ] || {
        log "vllm not ready in 30 min — tail of $VLLM_LOG:"; tail -200 "$VLLM_LOG" || true; exit 1
    }
fi

# --- 3. Run harbor ----------------------------------------------------------
# CRITICAL networking note (local vs. Beaker):
#   On Beaker, run_eval_in_job.sh runs podman INSIDE the job container, so the
#   task container's netns == the job container's netns == where vLLM listens;
#   localhost:$VLLM_PORT works there.
#   Locally we use the HOST Docker daemon, so harbor's task containers are
#   SIBLINGS of this container. `network_mode: host` puts them on the real
#   host's netns — NOT this container's — so localhost does NOT reach our vLLM.
#   They CAN reach this container by its docker-bridge IP, so we point the
#   agent at that IP instead of localhost.
#   Under --harbor-env modal the task container is in Modal's cloud, so NO
#   address of ours is reachable from it. That is fine for in-process agents
#   (Vanillux2Agent), whose LLM calls never leave this machine — they use plain
#   localhost. It is fatal for harbor's built-in agents, which are installed and
#   run INSIDE the task container; that combination is rejected below.
if [ "$SKIP_VLLM" = "1" ]; then
    AGENT_API_BASE=""
elif [ "$HARBOR_ENV" = "docker" ]; then
    HOST_IP="${HOST_IP:-$(hostname -i 2>/dev/null | awk '{print $1}')}"
    [ -n "$HOST_IP" ] || { echo "could not determine container IP (set HOST_IP)"; exit 1; }
    AGENT_API_BASE="http://$HOST_IP:$VLLM_PORT/v1"
    log "agent will reach vLLM at $AGENT_API_BASE (this container's bridge IP)"
else
    AGENT_API_BASE="http://localhost:$VLLM_PORT/v1"
    log "agent will reach vLLM at $AGENT_API_BASE (in-process agent, same machine)"
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
if [ -n "$AGENT_API_BASE" ]; then
    export OPENAI_API_BASE="$AGENT_API_BASE"
fi
# litellm reads OPENAI_BASE_URL; harbor's mini/swe agents forward it + the
# api-key var into the container. NOTE: do NOT set MSWEA_API_KEY — if it's set,
# harbor's mini-swe-agent forwards only that and skips OPENAI_API_KEY, and
# litellm's openai provider then fails with "Missing credentials".
unset MSWEA_API_KEY
if [ -n "$AGENT_API_BASE" ]; then
    export OPENAI_BASE_URL="$AGENT_API_BASE"
fi

HARBOR_CMD=( uv run harbor run
             --env "$HARBOR_ENV"
             --n-concurrent "$N_CONCURRENT"
             --job-name "$JOB_NAME"
             --yes
             -k "$N_ATTEMPTS" )
# Local dataset dir (harbor --path) overrides the registry --dataset ref.
if [ -n "$DATASET_PATH" ]; then
    HARBOR_CMD+=( --path "$DATASET_PATH" )
else
    HARBOR_CMD+=( --dataset "$DATASET" )
fi
# An agent value containing ":" is a module:Class import path (e.g.
# VanilluxAgent:VanilluxAgent, the Beaker default). Otherwise it's a harbor
# built-in agent name (e.g. mini-swe-agent, swe-agent, terminus).
#   * import-path SWE agents take an explicit api_base agent-kwarg and default to
#     the hosted_vllm/ litellm provider (Beaker parity).
#   * built-in agents resolve the endpoint from OPENAI_BASE_URL + --model and
#     want the openai/ provider (litellm has no "hosted_vllm" provider in the
#     installed harbor, so openai/<served-name> is the working spec).
# Override the prefix per agent with --model-provider (e.g. Vanillux2Agent is an
# import-path agent but uses its own litellm loop → needs openai/, not hosted_vllm).
if [[ "$AGENT_IMPORT_PATH" == *:* ]]; then
    MODEL_PROVIDER="${MODEL_PROVIDER:-hosted_vllm}"
    HARBOR_CMD+=( --model "${HARBOR_MODEL_NAME:-$MODEL_PROVIDER/$SERVED_MODEL_NAME}"
                  --agent-import-path "$AGENT_IMPORT_PATH" )
    if [ -n "$AGENT_API_BASE" ]; then
        HARBOR_CMD+=( --agent-kwarg "api_base=$AGENT_API_BASE" )
    fi
else
    if [ "$HARBOR_ENV" != "docker" ]; then
        echo "FATAL: agent '$AGENT_IMPORT_PATH' is a harbor built-in, which is installed"
        echo "       and run INSIDE the task container. Under --harbor-env $HARBOR_ENV that"
        echo "       container is a remote sandbox and cannot reach a model served here."
        echo "       Use an in-process agent (--agent Vanillux2Agent:Vanillux2Agent)."
        exit 1
    fi
    MODEL_PROVIDER="${MODEL_PROVIDER:-openai}"
    HARBOR_CMD+=( --model "${HARBOR_MODEL_NAME:-$MODEL_PROVIDER/$SERVED_MODEL_NAME}"
                  --agent "$AGENT_IMPORT_PATH" )
fi
if [ -n "$HARBOR_ENV_KWARGS" ]; then
    while IFS= read -r env_kwarg; do
        [ -n "$env_kwarg" ] || continue
        HARBOR_CMD+=( --environment-kwarg "$env_kwarg" )
    done <<< "$HARBOR_ENV_KWARGS"
fi
if [ -n "$SINGLE_TASK" ]; then
    HARBOR_CMD+=( --include-task-name "$SINGLE_TASK" )
else
    HARBOR_CMD+=( --n-tasks "$N_TASKS" )
fi

log "running harbor: ${HARBOR_CMD[*]}"
set +e
"${HARBOR_CMD[@]}"
HARBOR_RC=$?
set -e

# --- 4. Compute stats -------------------------------------------------------
JOB_DIR="jobs/$JOB_NAME"
if [ -d "$JOB_DIR" ]; then
    log "computing stats: scripts/compute_stats.py $JOB_DIR"
    set +e
    uv run python scripts/compute_stats.py "$JOB_DIR" \
        --json-output "$JOB_DIR/metrics.json" 2>&1 | tee "$JOB_DIR/stats.txt"
    set -e
fi

# --- 5. Optional copy to results dir ----------------------------------------
if [ -n "${RESULTS_DIR:-}" ] && [ -d "$JOB_DIR" ]; then
    log "copying $JOB_DIR -> $RESULTS_DIR/"
    mkdir -p "$RESULTS_DIR"; cp -r "$JOB_DIR" "$RESULTS_DIR/"
fi

log "harbor exit code: $HARBOR_RC — results in $JOB_DIR"
exit "$HARBOR_RC"
