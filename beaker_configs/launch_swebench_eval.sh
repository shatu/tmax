#!/usr/bin/env bash
#
# Launch a SWE-bench Verified eval (harbor dataset `swebench-verified@1.0`) on Beaker.
#
# Thin wrapper over beaker_configs/launch_eval.sh that pins the combination
# verified working on 2026-08-14 (tmax-4b smoke pass@1 0.60 on 5 tasks with
# 0 errored trials; tmax-9b k=5 full run ~1.4% error rate). It exists because
# the flag combo is easy to get subtly wrong and several of the failure modes
# are SILENT — they depress pass@k instead of erroring out.
#
# Usage:
#   ./beaker_configs/launch_swebench_eval.sh <model_path> [options]
#
# Examples:
#   # full 500-task run, k=5 (what produced the ~0.55 mean solve rate)
#   ./beaker_configs/launch_swebench_eval.sh allenai/tmax-9b --n-attempts 5
#
#   # quick smoke on 5 tasks
#   ./beaker_configs/launch_swebench_eval.sh allenai/tmax-4b --n-tasks 5 --gpus 1
#
#   # non-tmax base model (ships preprocessor_config.json -> no --language-model-only)
#   ./beaker_configs/launch_swebench_eval.sh Qwen/Qwen3.5-9B --no-language-model-only
#
# WHY EACH PINNED FLAG MATTERS (all of these fail quietly if wrong):
#   --agent Vanillux2Agent:Vanillux2Agent
#       The repo default and the only surviving custom agent.
#   --model-provider openai
#       Vanillux2Agent drives litellm directly; the installed harbor has no
#       usable hosted_vllm path. With the wrong provider the model is addressed
#       incorrectly and the run is wasted. (launch_eval.sh's banner used to print
#       "hosted_vllm/<name>" regardless of this flag — fixed, it now echoes the
#       real value; the job itself resolves it at run_eval_in_job.sh:548.)
#   --tool-call-parser qwen3_xml
#       Qwen3.5 emits <function=..><parameter=..> XML. With the default `hermes`
#       parser those tool calls are SILENTLY DROPPED and the agent loops on
#       "Format error" with ~0 useful steps -> a near-zero score that looks real.
#   --language-model-only
#       allenai/tmax-* are text-only fine-tunes of a multimodal-capable arch and
#       ship NO preprocessor_config.json, so vLLM dies with
#       "OSError: Can't load image processor" without this. Qwen/Qwen3.5-* and
#       shatu/*-Reasoning-Fix DO ship it -> pass --no-language-model-only.
#       (open-instruct RL/SFT Qwen3.5 checkpoints need CG CONVERSION instead;
#        this flag does not help them. See scripts/beaker/README.md.)
#   --max-model-len 65536
#       SWE-bench problem statements + repo exploration are long; 32k truncates.
#
# SCALE / STORAGE NOTES (swebench-verified is unlike terminal-bench):
#   * 500 tasks -> 500 UNIQUE prebuilt images, all docker.io/swebench/sweb.eval.x86_64.*
#     (1:1, no layer sharing). ~618 GiB compressed, ~1.5 TB on disk if retained.
#     Nothing is built locally.
#   * Therefore harbor's stock `compose down --rmi all` MUST stay on (the default
#     since 682989c8) or the node fills and wedges. Do NOT set
#     HARBOR_KEEP_TASK_IMAGES=1 here — that is for small sets like tb2's 89 images.
#   * Because every trial re-pulls, a LIVE mirror matters much more than on tb2.
#   * --n-concurrent: 12 is validated. 32 caused ~45% of trials to error via
#     per-node podman/disk contention (see the setup-timeout note in
#     Vanillux2Agent/agent.py). To go faster, SHARD ACROSS JOBS rather than
#     raising concurrency on one node.
#
# AFTER THE RUN — never quote a score without this check:
#   result.json -> stats.n_errored_trials.  5-25% = normal (model's own timeouts).
#   50-90% = poisoned (dead mirror / bad node) and pass@k comes out silently LOW.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- pinned defaults ---------------------------------------------------------
DATASET="swebench-verified@1.0"
AGENT="Vanillux2Agent:Vanillux2Agent"
MODEL_PROVIDER="openai"
TOOL_CALL_PARSER="qwen3_xml"
MAX_MODEL_LEN="65536"
LANGUAGE_MODEL_ONLY=1          # correct for allenai/tmax-*; see header
N_CONCURRENT=12                # validated; 32 induces per-node contention
N_ATTEMPTS=5
N_TASKS=""                     # empty = all 500
GPUS=4
TP_SIZE=1
DP_SIZE=""                     # default: equal to GPUS (data-parallel replicas)
CLUSTER="ai2/saturn"
PRIORITY="urgent"
WORKSPACE="${BEAKER_WORKSPACE:-ai2/oe-agents}"
JOB_NAME=""
MIRROR_URL="${MIRROR_URL:-}"   # REQUIRED (or "auto" to derive); see below
SKIP_MIRROR_CHECK=0

# Beaker workload whose CURRENT node hosts the docker.io pull-through mirror.
MIRROR_WORKLOAD="01KTYHEWPNFFNZ7BPMPTBJ1XQ5"
# A real swebench task image used to prove the mirror serves THIS namespace
# (a /v2/ 200 is NOT sufficient — a mirror can answer 200 and serve corrupt
# manifests; that has happened, see the -102 incident).
MIRROR_PROBE_IMAGE="swebench/sweb.eval.x86_64.sympy_1776_sympy-11618"

usage() {
    sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --mirror-url HOST:PORT   docker.io pull-through mirror, or 'auto' to derive it
                           from Beaker workload $MIRROR_WORKLOAD. REQUIRED.
  --skip-mirror-check      don't preflight the mirror (not recommended)
  --n-tasks N              limit to N tasks (default: all 500)
  --n-attempts N           harbor -k (default: $N_ATTEMPTS)
  --n-concurrent N         concurrent trials (default: $N_CONCURRENT)
  --gpus N                 GPUs for vLLM (default: $GPUS)
  --tp N / --dp N          tensor / data parallel (default: tp=$TP_SIZE, dp=gpus)
  --max-model-len N        vLLM context (default: $MAX_MODEL_LEN)
  --tool-call-parser P     override parser (default: $TOOL_CALL_PARSER)
  --no-language-model-only for models that DO ship preprocessor_config.json
  --cluster C              beaker cluster (default: $CLUSTER)
  --priority P             beaker priority (default: $PRIORITY). NOTE: on a
                           congested Jupiter, 'high' often schedules SOONER than
                           'urgent' — strict priority with lower-priority
                           backfill means a small job slots into gaps that big
                           urgent jobs cannot use.
  --workspace W            beaker workspace (default: $WORKSPACE)
  --job-name NAME          harbor job name (default: derived from model)
  --dry-run                print the launch_eval.sh command and exit
EOF
    exit 1
}

[ $# -lt 1 ] && usage
MODEL_PATH="$1"; shift

DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --mirror-url)             MIRROR_URL="$2"; shift 2 ;;
        --skip-mirror-check)      SKIP_MIRROR_CHECK=1; shift ;;
        --n-tasks)                N_TASKS="$2"; shift 2 ;;
        --n-attempts|-k)          N_ATTEMPTS="$2"; shift 2 ;;
        --n-concurrent)           N_CONCURRENT="$2"; shift 2 ;;
        --gpus)                   GPUS="$2"; shift 2 ;;
        --tp)                     TP_SIZE="$2"; shift 2 ;;
        --dp)                     DP_SIZE="$2"; shift 2 ;;
        --max-model-len)          MAX_MODEL_LEN="$2"; shift 2 ;;
        --tool-call-parser)       TOOL_CALL_PARSER="$2"; shift 2 ;;
        --no-language-model-only) LANGUAGE_MODEL_ONLY=0; shift ;;
        --cluster)                CLUSTER="$2"; shift 2 ;;
        --priority)               PRIORITY="$2"; shift 2 ;;
        --workspace)              WORKSPACE="$2"; shift 2 ;;
        --job-name)               JOB_NAME="$2"; shift 2 ;;
        --dry-run)                DRY_RUN=1; shift ;;
        -h|--help)                usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done

log() { printf '=== %s ===\n' "$*"; }

# --- mirror: derive if asked ------------------------------------------------
if [ "$MIRROR_URL" = "auto" ]; then
    log "deriving live mirror from beaker workload $MIRROR_WORKLOAD"
    node_id="$(beaker experiment get "$MIRROR_WORKLOAD" --format json \
        | python3 -c 'import sys,json; d=json.load(sys.stdin); e=d[0] if isinstance(d,list) else d; print((e.get("jobs") or [{}])[-1].get("node") or "")')"
    [ -n "$node_id" ] || { echo "FATAL: could not read mirror node from $MIRROR_WORKLOAD" >&2; exit 1; }
    host="$(beaker node get "$node_id" --format json \
        | python3 -c 'import sys,json; d=json.load(sys.stdin); e=d[0] if isinstance(d,list) else d; print(e.get("hostname") or "")')"
    [ -n "$host" ] || { echo "FATAL: could not resolve hostname for node $node_id" >&2; exit 1; }
    MIRROR_URL="${host}:5000"
    log "derived MIRROR_URL=$MIRROR_URL"
fi

if [ -z "$MIRROR_URL" ]; then
    cat >&2 <<EOF
FATAL: --mirror-url is required.

swebench-verified re-pulls a ~1.2 GiB image for EVERY trial (500 unique images,
harbor deletes each after use). Without a live mirror those pulls all go to
Docker Hub; podman's fallback is SILENT, so you get a slow run and rate-limit
failures that surface as a quietly LOW pass@k rather than an error.

Pass --mirror-url auto to derive the current host, or give HOST:PORT explicitly.
Mirrors move constantly — never reuse a hardcoded host from an old script.
EOF
    exit 1
fi

# --- mirror: preflight ------------------------------------------------------
# Two checks, because a mirror can answer /v2/ with 200 while serving corrupt
# manifests for the namespace you actually need.
if [ "$SKIP_MIRROR_CHECK" = "0" ]; then
    log "verifying mirror $MIRROR_URL"
    code="$(curl -s -o /dev/null -m 10 -w '%{http_code}' "http://${MIRROR_URL}/v2/" || true)"
    [ "$code" = "200" ] || { echo "FATAL: mirror $MIRROR_URL /v2/ returned '$code' (expected 200). Re-derive with --mirror-url auto." >&2; exit 1; }
    code="$(curl -s -o /dev/null -m 30 \
        -H 'Accept: application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.index.v1+json' \
        -w '%{http_code}' "http://${MIRROR_URL}/v2/${MIRROR_PROBE_IMAGE}/manifests/latest" || true)"
    [ "$code" = "200" ] || { echo "FATAL: mirror $MIRROR_URL cannot serve $MIRROR_PROBE_IMAGE (HTTP '$code'). It is up but not usable for swebench." >&2; exit 1; }
    log "mirror OK (/v2/ + real swebench manifest both 200)"
fi

# --- assemble ---------------------------------------------------------------
[ -n "$DP_SIZE" ] || DP_SIZE="$GPUS"
[ -n "$JOB_NAME" ] || JOB_NAME="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')-swebench-verified-k${N_ATTEMPTS}"

CMD=( "$REPO_ROOT/beaker_configs/launch_eval.sh" "$MODEL_PATH"
      --dataset "$DATASET"
      --mirror-url "$MIRROR_URL"
      --agent "$AGENT"
      --model-provider "$MODEL_PROVIDER"
      --tool-call-parser "$TOOL_CALL_PARSER"
      --max-model-len "$MAX_MODEL_LEN"
      --gpus "$GPUS" --tp "$TP_SIZE" --dp "$DP_SIZE"
      --n-attempts "$N_ATTEMPTS"
      --n-concurrent "$N_CONCURRENT"
      --cluster "$CLUSTER"
      --priority "$PRIORITY"
      --workspace "$WORKSPACE"
      --job-name "$JOB_NAME" )
[ "$LANGUAGE_MODEL_ONLY" = "1" ] && CMD+=( --language-model-only )
[ -n "$N_TASKS" ] && CMD+=( --n-tasks "$N_TASKS" )

echo
log "swebench-verified launch"
printf '  model:        %s\n' "$MODEL_PATH"
printf '  tasks:        %s x k=%s\n' "${N_TASKS:-all 500}" "$N_ATTEMPTS"
printf '  concurrency:  %s\n' "$N_CONCURRENT"
printf '  mirror:       %s\n' "$MIRROR_URL"
printf '  cluster:      %s (priority %s)\n' "$CLUSTER" "$PRIORITY"
echo

if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

exec "${CMD[@]}"
