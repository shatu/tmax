#!/usr/bin/env bash
#
# Launch a single beaker task that:
#   1. spins up a vLLM server on the local GPUs (8 by default)
#   2. configures podman + harbor (incl. the patches discovered while bringing
#      up harbor on the podman socket — see scripts/setup_podman_harbor.sh and
#      scripts/beaker/run_eval_in_job.sh)
#   3. runs `harbor run` on the chosen dataset against the local vLLM
#   4. copies the resulting jobs/<name>/ tree to a /weka path you can fetch.
#
# Usage:
#   ./beaker_configs/launch_eval.sh <model_path> [options]
#
# Example:
#   ./beaker_configs/launch_eval.sh allenai/open_instruct_dev \
#       --revision sft_qwen3_4b_tmax_4node \
#       --name sft-4b-tb2 \
#       --gpus 8 \
#       --dataset terminal-bench@2.0
#
# Outputs (set via --results-dir, default below) end up on weka:
#   /weka/oe-adapt-default/${USER}/tmax-eval/<job-name>/jobs/<job-name>/
#
# Uses Beaker Gantry to submit the current git HEAD. The SHA must be pushed to
# the remote; local dirty changes are not included in the remote job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- defaults ----------------------------------------------------------------
REVISION="main"
SERVED_MODEL_NAME=""
HARBOR_MODEL_NAME=""
GPU_COUNT=8
TP_SIZE=""
DP_SIZE=""
VLLM_PORT=8008
VLLM_VERSION="0.19.1"
VLLM_TOOL_CALL_PARSER="hermes"
VLLM_REASONING_PARSER=""
MIRROR_URL="${MIRROR_URL:-}"
MODEL_PROVIDER=""
VLLM_LANGUAGE_MODEL_ONLY=0
MAX_MODEL_LEN=""
DATASET="terminal-bench@2.0"
DATASET_PATH=""
HARBOR_ENV="docker"
AGENT_IMPORT_PATH="Vanillux2Agent:Vanillux2Agent"
N_CONCURRENT=8
N_ATTEMPTS=1
N_TASKS=""
JOB_NAME=""
RESULTS_DIR=""
CLUSTER="ai2/saturn"
BUDGET=""
PRIORITY="urgent"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/oe-agents}"
BEAKER_IMAGE="${BEAKER_IMAGE:-hamishivi/tmax-eval-interactive}"
BEAKER_DOCKER_IMAGE="${BEAKER_DOCKER_IMAGE:-}"
REPO_GIT_URL=""
REPO_GIT_REF=""
BEAKER_SCRIPTS_DATASET=""
EXTRA_UV_PIP_INSTALLS=""
EXTRA_AGENT_KWARGS=""
EXTRA_AGENT_ENVS=""
HOSTED_VLLM_MODEL_INFO=""
HARBOR_OVERRIDE_CPUS=""
HARBOR_OVERRIDE_MEMORY_MB=""
HARBOR_OVERRIDE_STORAGE_MB=""
HARBOR_OVERRIDE_GPUS=""
MIN_RUNTIME=""
HOSTNAME_CONSTRAINT=""              # optional: pin to a specific beaker node hostname (gantry --hostname)
HARBOR_TIMEOUT_MULTIPLIER=""
HARBOR_AGENT_TIMEOUT_MULTIPLIER=""
HARBOR_VERIFIER_TIMEOUT_MULTIPLIER=""
HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER=""
HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER=""
HARBOR_AGENT_TIMEOUT_SEC=""
VLLM_MODE="colocated"               # colocated | split | external
VLLM_BASE_URL=""                    # required for --vllm-base-url (external)
SYNC_START_TIMEOUT="30m"            # gantry --synchronized-start-timeout (split)
DRY_RUN=0
RDV_ROOT="${RDV_ROOT:-/weka/oe-adapt-default/${USER:-$(whoami)}/tmax-eval/rendezvous}"

usage() {
    cat <<EOF
Usage: $0 <model_path> [options]

Required:
  <model_path>           HF model path (e.g. allenai/Llama-3.1-Tulu-3-8B)
                         or a weka path the beaker image can read.

Options:
  --revision REV         HF revision/branch (default: main)
  --name NAME            served-model-name (default: basename of model_path)
  --harbor-model-name NAME
                        model name passed to harbor (default: hosted_vllm/<served-name>)
  --gpus N               GPUs (default: 8)
  --tp N                 tensor-parallel-size (default: GPU_COUNT)
  --dp N                 data-parallel-size (default: 1)
  --port PORT            vllm port (default: 8008)
  --vllm-version VER     vLLM package version for uvx (default: 0.19.1)
  --tool-call-parser P   vLLM tool call parser (default: hermes; use qwen3_xml
                         for Qwen3.5 with tool-calling agents like Vanillux2Agent)
  --reasoning-parser P   vLLM reasoning parser (default: none; use qwen3 for
                         Qwen3 so <think> blocks are split out of tool-calls)
  --mirror-url HOST:PORT docker.io pull-through mirror(s) for podman task-image
                         pulls (comma-sep; e.g. jupiter-cs-aus-137.reviz.ai2.in:5000).
                         Avoids Docker Hub rate limits under co-located jobs.
  --model-provider PROV  litellm provider prefix for --model (default: hosted_vllm
                         for import-path agents, openai otherwise). Use openai for
                         Vanillux2Agent.
  --language-model-only  pass --language_model_only to vLLM
  --max-model-len LEN    pass --max-model-len to vllm
  --dataset DS           harbor dataset (default: terminal-bench@2.0; also
                         valid: openthoughts-tblite@2.0)
  --harbor-env ENV       harbor environment backend (default: docker)
  --agent AGENT          harbor agent import path or named agent (default: Vanillux2Agent:Vanillux2Agent)
  --n-concurrent N       harbor --n-concurrent (default: 8)
  --n-attempts N         harbor -k (default: 1)
  --n-tasks N            harbor --n-tasks limit
  --job-name NAME        harbor --job-name (default: <served-name>-<dataset>)
  --results-dir DIR      where to copy the harbor jobs/ output
                         (default: /results; persisted by Gantry)
  --cluster CLUSTER      beaker cluster(s), comma-separated for multiple
                         (default: ai2/saturn; e.g. ai2/jupiter,ai2/saturn,ai2/ceres)
  --budget BUDGET        beaker budget (default: omitted; uses workspace default)
  --priority PRI         beaker priority (default: urgent)
  --workspace WS         beaker workspace (default: \$BEAKER_WORKSPACE or ai2/oe-agents)
  --image IMAGE          beaker image name or ID (clears default --docker-image)
  --docker-image IMAGE   public Docker image (default: $BEAKER_DOCKER_IMAGE)
  --beaker-scripts-dataset DS
                         existing Beaker dataset to mount at /uploaded-beaker-scripts
                         (default: upload local scripts/beaker)
  --repo-url URL         git URL of tmax (default: current 'origin' remote)
  --repo-ref REF         git SHA/branch of tmax (default: current HEAD SHA)
  --extra-uv-pip-install SPEC
                        extra package spec(s) to uv pip install in the job
  --agent-kwarg KV       extra harbor --agent-kwarg value (can be repeated)
  --agent-env KV         extra harbor --agent-env value (can be repeated)
  --hosted-vllm-model-info JSON
                        Harbor model_info JSON for hosted_vllm agents
  --override-cpus N      harbor per-task environment CPU override
  --override-memory-mb N harbor per-task environment memory override in MB
  --override-storage-mb N
                        harbor per-task environment storage override in MB
  --override-gpus N      harbor per-task environment GPU override
  --min-runtime DUR      gantry --min-runtime (e.g. 8h): guaranteed runtime before preemption
  --hostname HOST        pin the job to one beaker node (gantry --hostname; repeatable via comma list)
  --timeout-multiplier X harbor task timeout multiplier
  --agent-timeout-multiplier X
                        harbor agent timeout multiplier
  --verifier-timeout-multiplier X
                        harbor verifier timeout multiplier
  --agent-setup-timeout-multiplier X
                        harbor agent setup timeout multiplier
  --environment-build-timeout-multiplier X
                        harbor environment build timeout multiplier
  --agent-timeout-sec SEC
                        exact harbor agent timeout override in seconds

vLLM placement (see docs/running_evals.md, "Splitting vLLM off the agent node"):
  --split-vllm           run vLLM and harbor on SEPARATE nodes, as two replicas
                         of one Beaker task (replicas=2 + leaderSelection).
                         Rank 0 serves the model and nothing else; rank 1 runs
                         harbor's podman containers and never touches a GPU, so
                         trial CPU/disk load can no longer starve vLLM's API
                         server. Costs one extra node: Beaker replicas are
                         homogeneous, so the agent replica is allocated --gpus
                         GPUs it will not use, and the pair is scheduled
                         all-or-nothing.
  --vllm-base-url URL    do not start vLLM at all; point harbor at an existing
                         server, e.g. from beaker_configs/launch_vllm.sh. URL
                         must include /v1. Implies --gpus 0 unless overridden.
  --sync-start-timeout D gantry --synchronized-start-timeout for --split-vllm
                         (default: $SYNC_START_TIMEOUT)
  --rdv-root DIR         weka dir for split-mode rendezvous state
                         (default: $RDV_ROOT)

  --dry-run              build and validate the gantry command, but submit nothing
EOF
    exit 1
}

[ $# -lt 1 ] && usage
MODEL_PATH="$1"; shift

while [ $# -gt 0 ]; do
    case "$1" in
        --revision)        REVISION="$2"; shift 2 ;;
        --name)            SERVED_MODEL_NAME="$2"; shift 2 ;;
        --harbor-model-name) HARBOR_MODEL_NAME="$2"; shift 2 ;;
        --gpus)            GPU_COUNT="$2"; shift 2 ;;
        --tp)              TP_SIZE="$2"; shift 2 ;;
        --dp)              DP_SIZE="$2"; shift 2 ;;
        --port)            VLLM_PORT="$2"; shift 2 ;;
        --vllm-version)    VLLM_VERSION="$2"; shift 2 ;;
        --tool-call-parser) VLLM_TOOL_CALL_PARSER="$2"; shift 2 ;;
        --reasoning-parser) VLLM_REASONING_PARSER="$2"; shift 2 ;;
        --mirror-url)      MIRROR_URL="$2"; shift 2 ;;
        --model-provider)  MODEL_PROVIDER="$2"; shift 2 ;;
        --language-model-only|--language_model_only) VLLM_LANGUAGE_MODEL_ONLY=1; shift ;;
        --max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
        --dataset)         DATASET="$2"; shift 2 ;;
        --dataset-path)    DATASET_PATH="$2"; shift 2 ;;
        --harbor-env)      HARBOR_ENV="$2"; shift 2 ;;
        --agent)           AGENT_IMPORT_PATH="$2"; shift 2 ;;
        --n-concurrent)    N_CONCURRENT="$2"; shift 2 ;;
        --n-attempts)      N_ATTEMPTS="$2"; shift 2 ;;
        --n-tasks)         N_TASKS="$2"; shift 2 ;;
        --job-name)        JOB_NAME="$2"; shift 2 ;;
        --results-dir)     RESULTS_DIR="$2"; shift 2 ;;
        --cluster)         CLUSTER="$2"; shift 2 ;;
        --budget)          BUDGET="$2"; shift 2 ;;
        --priority)        PRIORITY="$2"; shift 2 ;;
        --workspace)       BEAKER_WORKSPACE="$2"; shift 2 ;;
        --image)           BEAKER_IMAGE="$2"; BEAKER_DOCKER_IMAGE=""; shift 2 ;;
        --docker-image)    BEAKER_DOCKER_IMAGE="$2"; BEAKER_IMAGE=""; shift 2 ;;
        --beaker-scripts-dataset) BEAKER_SCRIPTS_DATASET="$2"; shift 2 ;;
        --repo-url)        REPO_GIT_URL="$2"; shift 2 ;;
        --repo-ref)        REPO_GIT_REF="$2"; shift 2 ;;
        --extra-uv-pip-install) EXTRA_UV_PIP_INSTALLS="$2"; shift 2 ;;
        --agent-kwarg)     EXTRA_AGENT_KWARGS+="${EXTRA_AGENT_KWARGS:+$'\n'}$2"; shift 2 ;;
        --agent-env)       EXTRA_AGENT_ENVS+="${EXTRA_AGENT_ENVS:+$'\n'}$2"; shift 2 ;;
        --hosted-vllm-model-info) HOSTED_VLLM_MODEL_INFO="$2"; shift 2 ;;
        --override-cpus)   HARBOR_OVERRIDE_CPUS="$2"; shift 2 ;;
        --override-memory-mb) HARBOR_OVERRIDE_MEMORY_MB="$2"; shift 2 ;;
        --override-storage-mb) HARBOR_OVERRIDE_STORAGE_MB="$2"; shift 2 ;;
        --override-gpus)   HARBOR_OVERRIDE_GPUS="$2"; shift 2 ;;
        --min-runtime)     MIN_RUNTIME="$2"; shift 2 ;;
        --hostname)        HOSTNAME_CONSTRAINT="$2"; shift 2 ;;
        --timeout-multiplier) HARBOR_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --agent-timeout-multiplier) HARBOR_AGENT_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --verifier-timeout-multiplier) HARBOR_VERIFIER_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --agent-setup-timeout-multiplier) HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --environment-build-timeout-multiplier) HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        --agent-timeout-sec) HARBOR_AGENT_TIMEOUT_SEC="$2"; shift 2 ;;
        --split-vllm)      VLLM_MODE="split"; shift ;;
        --vllm-base-url)   VLLM_MODE="external"; VLLM_BASE_URL="$2"; shift 2 ;;
        --sync-start-timeout) SYNC_START_TIMEOUT="$2"; shift 2 ;;
        --rdv-root)        RDV_ROOT="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         usage ;;
        *) echo "unknown option: $1"; usage ;;
    esac
done

# --- derive defaults ---------------------------------------------------------
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
TP_SIZE="${TP_SIZE:-$GPU_COUNT}"
DP_SIZE="${DP_SIZE:-1}"

DATASET_SLUG="${DATASET//[^A-Za-z0-9]/-}"
JOB_NAME="${JOB_NAME:-${SERVED_MODEL_NAME}-${DATASET_SLUG}}"
RESULTS_DIR="${RESULTS_DIR:-/results}"

if [ -z "$REPO_GIT_REF" ]; then
    REPO_GIT_REF="$(git -C "$REPO_ROOT" rev-parse HEAD)"
fi

BEAKER_NAME="eval-${JOB_NAME}"

# --- vLLM placement validation ----------------------------------------------
case "$VLLM_MODE" in
    external)
        case "$VLLM_BASE_URL" in
            http://*|https://*) ;;
            *) echo "error: --vllm-base-url must be an http(s) URL including /v1" >&2; exit 1 ;;
        esac
        # The eval job runs no model, so it needs no GPUs. Keep an explicit
        # --gpus from the caller (some harbor tasks request GPUs of their own).
        if [ "$GPU_COUNT" = "8" ]; then
            GPU_COUNT=0
        fi
        ;;
    split)
        # A replica group is admitted all-or-nothing and placed on ONE cluster,
        # so a comma-separated list only widens where the PAIR may land, not
        # where each replica lands. Still worth saying out loud.
        if [ -z "$HOSTNAME_CONSTRAINT" ] && [ "${CLUSTER//[^,]/}" != "" ]; then
            echo "note: --split-vllm with multiple clusters — Beaker places the whole"
            echo "      replica group on one of them; it will not straddle clusters."
        fi
        if [ -n "$HOSTNAME_CONSTRAINT" ]; then
            echo "error: --split-vllm needs two nodes; --hostname pins the group to one." >&2
            exit 1
        fi
        ;;
esac

VLLM_PLACEMENT_DESC="colocated (vLLM + harbor on one node)"
case "$VLLM_MODE" in
    split)    VLLM_PLACEMENT_DESC="split (replicas=2: rank 0 serves vLLM, rank 1 runs harbor)" ;;
    external) VLLM_PLACEMENT_DESC="external (${VLLM_BASE_URL})" ;;
esac

cat <<EOF
=== Launching tmax eval on Beaker ===
  Model:        ${MODEL_PATH}@${REVISION}
  Served name:  ${SERVED_MODEL_NAME}
  Harbor model: ${HARBOR_MODEL_NAME:-${MODEL_PROVIDER:-hosted_vllm}/${SERVED_MODEL_NAME}}
  vLLM version: ${VLLM_VERSION}
  Tool parser:  ${VLLM_TOOL_CALL_PARSER}
  Reason parser: ${VLLM_REASONING_PARSER:-<none>}
  LM only:      ${VLLM_LANGUAGE_MODEL_ONLY}
  GPUs:         ${GPU_COUNT} (TP=${TP_SIZE}, DP=${DP_SIZE})
  vLLM placement: ${VLLM_PLACEMENT_DESC}
  Dataset:      ${DATASET}
  Harbor env:   ${HARBOR_ENV}
  Agent:        ${AGENT_IMPORT_PATH}
  Agent kwargs: ${EXTRA_AGENT_KWARGS:-<none>}
  Agent envs:   ${EXTRA_AGENT_ENVS:-<none>}
  Extra pip:    ${EXTRA_UV_PIP_INSTALLS:-<none>}
  N tasks:      ${N_TASKS:-<all>}
  Task resources: cpus=${HARBOR_OVERRIDE_CPUS:-<task default>} memory_mb=${HARBOR_OVERRIDE_MEMORY_MB:-<task default>} storage_mb=${HARBOR_OVERRIDE_STORAGE_MB:-<task default>} gpus=${HARBOR_OVERRIDE_GPUS:-<task default>}
  Timeouts:     agent_sec=${HARBOR_AGENT_TIMEOUT_SEC:-<task default>} timeout_mult=${HARBOR_TIMEOUT_MULTIPLIER:-<default>} agent_mult=${HARBOR_AGENT_TIMEOUT_MULTIPLIER:-<default>} verifier_mult=${HARBOR_VERIFIER_TIMEOUT_MULTIPLIER:-<default>}
  Model info:   ${HOSTED_VLLM_MODEL_INFO:-<auto>}
  Job name:     ${JOB_NAME}
  Results dir:  ${RESULTS_DIR}
  Image:        ${BEAKER_IMAGE:+beaker:${BEAKER_IMAGE}}${BEAKER_DOCKER_IMAGE:+docker:${BEAKER_DOCKER_IMAGE}}
  Repo ref:     ${REPO_GIT_REF}
  Gantry:       ${BEAKER_NAME}  cluster=${CLUSTER}  workspace=${BEAKER_WORKSPACE}
EOF

# Sanity check: the SHA must be reachable on the remote.
if ! git -C "$REPO_ROOT" branch -r --contains "$REPO_GIT_REF" 2>/dev/null | grep -q .; then
    echo
    echo "warning: $REPO_GIT_REF doesn't appear to be on any remote branch."
    echo "         the beaker job will fail to clone it. push first or pass --repo-ref."
fi

# --- launch via Gantry --------------------------------------------------------
GANTRY_CMD=(
    uvx --from beaker-gantry gantry --quiet run
    --yes
    --allow-dirty
    --workspace "$BEAKER_WORKSPACE"
    --name "$BEAKER_NAME"
    --description "Harbor eval (${DATASET}) of ${SERVED_MODEL_NAME} (${MODEL_PATH}@${REVISION}) via vLLM"
    --ref "$REPO_GIT_REF"
    --gpus "$GPU_COUNT"
    --priority "$PRIORITY"
    --weka "oe-adapt-default:/weka/oe-adapt-default"
    --env-secret HF_TOKEN
    --env-secret "DOCKER_PAT=${DOCKER_PAT_SECRET:-shashankg_DOCKER_PAT}"
    --env "MODEL_PATH=${MODEL_PATH}"
    --env "MODEL_REVISION=${REVISION}"
    --env "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
    --env "HARBOR_MODEL_NAME=${HARBOR_MODEL_NAME}"
    --env "VLLM_VERSION=${VLLM_VERSION}"
    --env "VLLM_TOOL_CALL_PARSER=${VLLM_TOOL_CALL_PARSER}"
    --env "VLLM_REASONING_PARSER=${VLLM_REASONING_PARSER}"
    --env "MIRROR_URL=${MIRROR_URL}"
    --env "MODEL_PROVIDER=${MODEL_PROVIDER}"
    --env "VLLM_LANGUAGE_MODEL_ONLY=${VLLM_LANGUAGE_MODEL_ONLY}"
    --env "VLLM_PORT=${VLLM_PORT}"
    --env "TP_SIZE=${TP_SIZE}"
    --env "DP_SIZE=${DP_SIZE}"
    --env "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
    --env "DATASET=${DATASET}"
    --env "DATASET_PATH=${DATASET_PATH}"
    --env "HARBOR_ENV=${HARBOR_ENV}"
    --env "AGENT_IMPORT_PATH=${AGENT_IMPORT_PATH}"
    --env "EXTRA_AGENT_KWARGS=${EXTRA_AGENT_KWARGS}"
    --env "EXTRA_AGENT_ENVS=${EXTRA_AGENT_ENVS}"
    --env "EXTRA_UV_PIP_INSTALLS=${EXTRA_UV_PIP_INSTALLS}"
    --env "HOSTED_VLLM_MODEL_INFO=${HOSTED_VLLM_MODEL_INFO}"
    --env "N_CONCURRENT=${N_CONCURRENT}"
    --env "N_ATTEMPTS=${N_ATTEMPTS}"
    --env "N_TASKS=${N_TASKS}"
    --env "HARBOR_OVERRIDE_CPUS=${HARBOR_OVERRIDE_CPUS}"
    --env "HARBOR_OVERRIDE_MEMORY_MB=${HARBOR_OVERRIDE_MEMORY_MB}"
    --env "HARBOR_OVERRIDE_STORAGE_MB=${HARBOR_OVERRIDE_STORAGE_MB}"
    --env "HARBOR_OVERRIDE_GPUS=${HARBOR_OVERRIDE_GPUS}"
    --env "HARBOR_TIMEOUT_MULTIPLIER=${HARBOR_TIMEOUT_MULTIPLIER}"
    --env "HARBOR_AGENT_TIMEOUT_MULTIPLIER=${HARBOR_AGENT_TIMEOUT_MULTIPLIER}"
    --env "HARBOR_VERIFIER_TIMEOUT_MULTIPLIER=${HARBOR_VERIFIER_TIMEOUT_MULTIPLIER}"
    --env "HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER=${HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER}"
    --env "HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER=${HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER}"
    --env "HARBOR_AGENT_TIMEOUT_SEC=${HARBOR_AGENT_TIMEOUT_SEC}"
    --env "JOB_NAME=${JOB_NAME}"
    --env "VLLM_MODE=${VLLM_MODE}"
    --env "VLLM_BASE_URL=${VLLM_BASE_URL}"
    --env "RDV_ROOT=${RDV_ROOT}"
    --env BEAKER_ALLOW_SUBCONTAINERS=1
    --env BEAKER_SKIP_DOCKER_SOCKET=1
    --host-networking
    --propagate-failure
    --no-python
)

# Split mode is one Beaker TASK with two REPLICAS, not two tasks: replicas are
# the only thing Beaker gives cross-node discovery for (leaderSelection +
# hostNetworking). Separate tasks in one experiment get no discovery env vars and
# no guarantee they land on different nodes.
#   --propagate-failure/-preemption: if the server replica dies, kill the agent
#     replica rather than let it error out every remaining trial.
#   --synchronized-start-timeout: don't start the agent replica against a server
#     replica that is still queued.
if [ "$VLLM_MODE" = "split" ]; then
    # Beaker expands `replicas: 2` server-side into <task-name>-replica-0 and
    # -replica-1; there is no per-replica naming hook, so the ONLY way to say
    # which is which is to encode the fixed rank->role mapping in the task name:
    #   vllm0-agent1-replica-0  = rank 0 = the vLLM server
    #   vllm0-agent1-replica-1  = rank 1 = harbor / the agent
    # run_eval_in_job.sh assigns roles from BEAKER_REPLICA_RANK on exactly this
    # convention, so the name stays true as long as that holds.
    GANTRY_CMD+=(
        --task-name "${SPLIT_TASK_NAME:-vllm0-agent1}"
        --replicas 2
        --leader-selection
        --propagate-preemption
        --synchronized-start-timeout "$SYNC_START_TIMEOUT"
    )
fi

if [ "$DRY_RUN" = "1" ]; then
    GANTRY_CMD+=(--dry-run)
fi

if [ -n "$MIN_RUNTIME" ]; then
    GANTRY_CMD+=(--min-runtime "$MIN_RUNTIME")
fi
if [ -n "$HOSTNAME_CONSTRAINT" ]; then
    for h in ${HOSTNAME_CONSTRAINT//,/ }; do GANTRY_CMD+=(--hostname "$h"); done
fi

# Gantry accepts repeated --cluster flags; CLUSTER may be comma-separated.
if [ -z "$HOSTNAME_CONSTRAINT" ]; then   # gantry forbids --cluster together with --hostname
    for cluster in ${CLUSTER//,/ }; do
        GANTRY_CMD+=(--cluster "$cluster")
    done
fi

# The daytona backend needs an API key; only register the secret then, so the
# default docker path doesn't require a DAYTONA_API_KEY secret in the workspace.
if [ "$HARBOR_ENV" = "daytona" ]; then
    GANTRY_CMD+=(--env-secret "DAYTONA_API_KEY=${DAYTONA_API_KEY_SECRET:-hamishivi_DAYTONA_API_KEY}")
fi

if [ -n "$BEAKER_IMAGE" ]; then
    GANTRY_CMD+=(--beaker-image "$BEAKER_IMAGE")
elif [ -n "$BEAKER_DOCKER_IMAGE" ]; then
    GANTRY_CMD+=(--docker-image "$BEAKER_DOCKER_IMAGE")
else
    echo "error: either --image or --docker-image must be set" >&2
    exit 1
fi

if [ -n "$BEAKER_SCRIPTS_DATASET" ]; then
    GANTRY_CMD+=(--dataset "${BEAKER_SCRIPTS_DATASET}:/uploaded-beaker-scripts")
else
    GANTRY_CMD+=(--upload "$REPO_ROOT/scripts/beaker:/uploaded-beaker-scripts")
fi

if [ -n "$BUDGET" ]; then
    GANTRY_CMD+=(--budget "$BUDGET")
fi

if [ "$RESULTS_DIR" = "/results" ]; then
    GANTRY_CMD+=(-- bash /uploaded-beaker-scripts/run_eval_in_job.sh)
else
    GANTRY_CMD+=(-- env "RESULTS_DIR=${RESULTS_DIR}" bash /uploaded-beaker-scripts/run_eval_in_job.sh)
fi

echo
printf 'Launching with:'
printf ' %q' "${GANTRY_CMD[@]}"
printf '\n\n'

"${GANTRY_CMD[@]}"
