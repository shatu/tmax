#!/usr/bin/env bash
# Auto-restart watchdog for the swerl-tmax15k VMVM DPPO training jobs.
#
# Monitors one training job (by id) and, when it leaves the queue WITHOUT having
# printed "finished training", decides whether to resubmit:
#   - Resubmits (resuming from checkpoint_state_dir via a pinned RUN_ID) only when
#     the failure looks like a transient/random infra error (NCCL segfault, node
#     failure, Ray actor death, CUDA/GPU faults, connection resets, OOM-killer).
#   - Refuses to restart deterministic failures (config/import/ValueError/etc.)
#     and fast-failing loops (a run that dies in < MIN_GOOD_RUNTIME_S), so a real
#     bug does not burn the queue.
# Runs as its own tiny cpu sbatch job so it survives Cursor/SSH logout.
#
# Submit, e.g.:
#   sbatch --job-name=autorestart-16x32 \
#     --export=ALL,TARGET_JOB_ID=9525486,\
# RUN_ID=swerl_qwen35_9b_dppo_vmvm_16x32__seed42__20260715_003251,\
# EXP_NAME=swerl_qwen35_9b_dppo_vmvm_16x32,\
# NUM_UNIQUE_PROMPTS_ROLLOUT=16,NUM_SAMPLES_PER_PROMPT_ROLLOUT=32 \
#     scripts/train/debug/envs/autorestart_watchdog.sh
#
#SBATCH --job-name=autorestart-watchdog
#SBATCH --account=comem
#SBATCH --qos=cpu_lowest
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --output=/checkpoint/comem/rulin/logs/swerl_tmax15k_vmvm_dppo/autorestart-%x-%j.out
#SBATCH --error=/checkpoint/comem/rulin/logs/swerl_tmax15k_vmvm_dppo/autorestart-%x-%j.err

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/checkpoint/comem/rulin/open-instruct}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/scripts/train/debug/envs/swerl_vanillux_tmax15k_vmvm_dppo_slurm.sh}"
# Overridable for the same reason as the launcher's mkdir: the watchdog greps
# ${LOGDIR}/${JOB_NAME}-${jid}.{out,err} to classify failures, so a hardcoded path a
# collaborator cannot read makes it silently mis-classify every failure — the same
# shape as the stale-JOB_NAME veto bug. Default unchanged.
LOGDIR="${TMAX_LOGDIR:-/checkpoint/comem/rulin/logs/swerl_tmax15k_vmvm_dppo}"

: "${TARGET_JOB_ID:?set TARGET_JOB_ID to the training job to watch}"
: "${RUN_ID:?set RUN_ID (pins checkpoint_state_dir so restarts resume)}"
: "${EXP_NAME:?set EXP_NAME}"
: "${NUM_UNIQUE_PROMPTS_ROLLOUT:?set NUM_UNIQUE_PROMPTS_ROLLOUT}"
: "${NUM_SAMPLES_PER_PROMPT_ROLLOUT:?set NUM_SAMPLES_PER_PROMPT_ROLLOUT}"
# tmax-private#1: the launcher's historical default backend (vmvm) is DEAD at
# tmax@61b5a85d, so a resubmit that lost this pin used to silently revert the arm to a
# boot-fatal backend and burn every retry to MAX_RESTARTS. Refuse to even start without
# an explicit pin — the watchdog cannot lose on resubmit what it cannot start without.
: "${SWERL_SANDBOX_BACKEND:?set SWERL_SANDBOX_BACKEND explicitly (vmvm default is dead at tmax@61b5a85d)}"

# Slurm --job-name of the training job. Log files are ${LOGDIR}/${JOB_NAME}-${jid}.{out,err}
# (Slurm's %x-%j pattern). Must match the --job-name the training job was submitted
# with, else the watchdog greps the wrong (missing) log and never classifies failures.
# Defaults to the launcher's #SBATCH job-name; the watchdog re-uses it on resubmit so
# the log path stays consistent across restarts.
# Our launches use --job-name=jb-${EXP_NAME}; a wrong JOB_NAME makes every log
# grep silently miss (2>/dev/null) and the fast-fail guard then vetoes restarts.
JOB_NAME="${JOB_NAME:-jb-${EXP_NAME}}"

MAX_RESTARTS="${MAX_RESTARTS:-8}"
POLL_S="${POLL_S:-120}"
MIN_GOOD_RUNTIME_S="${MIN_GOOD_RUNTIME_S:-600}"   # runs dying faster than this are treated as deterministic
TAIL_LINES="${TAIL_LINES:-4000}"

# Transient / random infra failure signatures -> safe to resume-and-retry.
TRANSIENT_RE='Segfault encountered|ncclSocketsCompare|\bNCCL\b|nccl (error|timeout|unhandled)|DUE TO NODE FAILURE|ActorDiedError|RayActorError|ActorUnavailableError|keepalive watchdog timeout|keepalive|RpcError|rpc_code|PREEMPT|SYSTEM_ERROR|connection error code|Connection reset by peer|CUDA error|an illegal memory access|uncorrectable|Xid|ECC|GPU is lost|NVML|Watchdog caught|Timed out initializing|torch.distributed.*[Tt]imeout|Socket Timeout|out of memory|oom-kill|Killed|Out of range float values are not JSON compliant|not JSON compliant: nan|BadRequestError|vLLM request task .* failed|vLLM engine initialization failed|Engine core initialization failed|EADDRINUSE|address already in use|All containers are busy|vacli lease failed|vacli exited early|VMVM vacli|Reset failed after|retryable'

# Startup-phase transient signatures. These frequently fail FAST (< MIN_GOOD_RUNTIME_S)
# during vLLM/Ray/NCCL init on a flaky node or a stale port, but are NOT deterministic
# config bugs, so we allow a restart even inside the fast-fail window (bounded by
# MAX_RESTARTS). Keep this narrow so genuine config/import errors are still not retried.
STARTUP_TRANSIENT_RE='vLLM engine initialization failed|Engine core initialization failed|EADDRINUSE|address already in use|\bNCCL\b|nccl (error|timeout|unhandled)|CUDA error|GPU is lost|Xid|DUE TO NODE FAILURE|ActorDiedError|RayActorError|Timed out initializing'

# Deterministic (code/config) failure signatures -> NEVER restart, regardless of
# any transient NCCL/RuntimeError noise emitted during the crash teardown. These are
# bugs a restart cannot fix (a fresh run dies identically), so retrying just burns the
# queue. Checked BEFORE the transient regex. (e.g. the ref-policy ZeRO-3 all-gather
# shape bug that spun this run 14x before it was fixed upstream.)
# IMPORTANT: keep these narrow and trainer-INTERNAL only. Do NOT include generic Python
# errors (ImportError/ModuleNotFoundError/SyntaxError/argparse) — those appear constantly in
# ROLLOUT tool output (the agent runs Python in the sandbox that throws them), so they would
# match on every run and wrongly block restarts of genuinely transient failures.
DETERMINISTIC_RE='output tensor size must be equal to world_size|Policy and reference policy parameter (names|shapes) do not match|ZeRO-3 parameter .* is missing its local partition|You are using an untested ZeRO Optimizer'

restarts=0
cur="${TARGET_JOB_ID}"

log() { echo "[$(date -Is)] $*"; }

classify_and_maybe_resubmit() {
    local jid="$1"
    local errlog="${LOGDIR}/${JOB_NAME}-${jid}.err"
    local outlog="${LOGDIR}/${JOB_NAME}-${jid}.out"

    # Success?
    if grep -qa "finished training" "$errlog" "$outlog" 2>/dev/null; then
        log "job ${jid}: 'finished training' found -> SUCCESS. Watchdog exiting."
        return 2
    fi

    # State + elapsed from sacct.
    local state elapsed_s
    state="$(sacct -j "${jid}" --format=State -n -P 2>/dev/null | head -1)"
    elapsed_s="$(sacct -j "${jid}.batch" --format=ElapsedRaw -n -P 2>/dev/null | head -1)"
    [ -z "${elapsed_s}" ] && elapsed_s="$(sacct -j "${jid}" --format=ElapsedRaw -n -P 2>/dev/null | head -1)"
    elapsed_s="${elapsed_s:-0}"
    log "job ${jid}: state=${state} elapsed_s=${elapsed_s}"

    # Fast-fail guard: a run dying quickly is almost certainly a deterministic
    # startup error (bad config/import), NOT a random infra blip.
    if [ "${elapsed_s}" -lt "${MIN_GOOD_RUNTIME_S}" ]; then
        if grep -qaiE "${STARTUP_TRANSIENT_RE}" "$errlog" "$outlog" 2>/dev/null; then
            log "job ${jid}: died after ${elapsed_s}s but matches a startup-transient signature (vLLM/Ray/NCCL/GPU init); allowing restart despite the fast-fail guard."
        else
            log "job ${jid}: died after ${elapsed_s}s (< ${MIN_GOOD_RUNTIME_S}s) with no startup-transient signature. Treating as deterministic; NOT auto-restarting."
            return 1
        fi
    fi

    # Deterministic failure? Refuse to restart even if transient noise is also present.
    if grep -qaE "${DETERMINISTIC_RE}" "$errlog" "$outlog" 2>/dev/null; then
        local dhit
        dhit="$(grep -haoE "${DETERMINISTIC_RE}" "$errlog" "$outlog" 2>/dev/null | sort | uniq -c | sort -rn | head -3 | tr '\n' ';')"
        log "job ${jid}: DETERMINISTIC failure signature(s): ${dhit} -> NOT auto-restarting (needs a code/config fix)."
        return 1
    fi

    # Look for a transient signature. Grep the WHOLE err+out logs (not just a
    # tail): on an 8-node crash the fatal root cause can sit thousands of lines
    # before the end amid shutdown noise, and async log flushing can reorder
    # things, so a fixed tail window intermittently misses it.
    if grep -qaiE "${TRANSIENT_RE}" "$errlog" "$outlog" 2>/dev/null; then
        local hit
        hit="$(grep -haoiE "${TRANSIENT_RE}" "$errlog" "$outlog" 2>/dev/null | sort | uniq -c | sort -rn | head -3 | tr '\n' ';')"
        log "job ${jid}: transient failure signature(s): ${hit}"
        if [ "${restarts}" -ge "${MAX_RESTARTS}" ]; then
            log "job ${jid}: reached MAX_RESTARTS=${MAX_RESTARTS}. NOT restarting."
            return 1
        fi
        restarts=$((restarts + 1))
        log "Resubmitting (restart ${restarts}/${MAX_RESTARTS}) resuming RUN_ID=${RUN_ID} ..."
        local newid
        # cgroup-v2 fix: do NOT pass `--export` (any explicit --export triggers Slurm user-env
        # retrieval, which fails and holds the job). Instead export the vars into THIS watchdog's
        # environment and submit with no --export, so sbatch propagates the env by default.
        export WITH_X2P=1 WANDB_MODE=online NCCL_RAS_ENABLE=0
        export RUN_ID EXP_NAME NUM_UNIQUE_PROMPTS_ROLLOUT NUM_SAMPLES_PER_PROMPT_ROLLOUT
        # Sandbox backend selection MUST be forwarded. The launcher defaults to
        # SWERL_SANDBOX_BACKEND=vmvm + the vmvm dataset; vmvm-registry is dead, so a
        # restart that drops these silently reverts an apptainer run to vmvm and then
        # wedges forever ("podman pull failed" -> no rollouts -> DataPreparationActor
        # waits at current_prepared_step=-1, with the job still shown RUNNING).
        [ -n "${DATASET_JSONL:-}" ] && export DATASET_JSONL
        [ -n "${SWERL_SANDBOX_BACKEND:-}" ] && export SWERL_SANDBOX_BACKEND
        [ -n "${SWERL_APPTAINER_SIF_DIR:-}" ] && export SWERL_APPTAINER_SIF_DIR
        # Sandfleet backend knobs. Same failure mode as the vmvm reversion above:
        # a restart that drops SANDFLEET_* silently reverts the arm to the
        # launcher-default backend, and the run stops being what its name says.
        [ -n "${SANDFLEET_CLIENT_TOKEN:-}" ] && export SANDFLEET_CLIENT_TOKEN
        [ -n "${SANDFLEET_POOL:-}" ] && export SANDFLEET_POOL
        [ -n "${SWERL_SANDBOX_MEM_LIMIT:-}" ] && export SWERL_SANDBOX_MEM_LIMIT
        [ -n "${SANDFLEET_EMBED_CONTROLLER:-}" ] && export SANDFLEET_EMBED_CONTROLLER
        [ -n "${SANDFLEET_SRC:-}" ] && export SANDFLEET_SRC
        # Oscar's report (tmax-collaboration#1): without these, a resubmit reverts to
        # launcher defaults — WANDB_ENTITY=rulin silently reattributes a collaborator's
        # run; the timeouts revert to defaults that may not match the submitted run.
        [ -n "${WANDB_ENTITY:-}" ] && export WANDB_ENTITY
        [ -n "${SWERL_SANDBOX_TIMEOUT:-}" ] && export SWERL_SANDBOX_TIMEOUT
        [ -n "${SWERL_SANDBOX_TEST_TIMEOUT:-}" ] && export SWERL_SANDBOX_TEST_TIMEOUT
        # Worktree/topology overrides (a sandfleet arm runs from the geomean worktree
        # with a non-default topology; dropping these reverts code AND shape).
        [ -n "${REPO_ROOT:-}" ] && export REPO_ROOT
        [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] && export UV_PROJECT_ENVIRONMENT
        [ -n "${PYTHONPATH:-}" ] && export PYTHONPATH
        [ -n "${GRPO_NUM_LEARNERS_PER_NODE:-}" ] && export GRPO_NUM_LEARNERS_PER_NODE
        [ -n "${VLLM_NUM_ENGINES:-}" ] && export VLLM_NUM_ENGINES
        [ -n "${RAY_NUM_CPUS:-}" ] && export RAY_NUM_CPUS
        [ -n "${NUM_UNIQUE_PROMPTS_ROLLOUT:-}" ] && export NUM_UNIQUE_PROMPTS_ROLLOUT
        [ -n "${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-}" ] && export NUM_SAMPLES_PER_PROMPT_ROLLOUT
        [ -n "${SEQUENCE_TIS_MASK_LOWER:-}" ] && export SEQUENCE_TIS_MASK_LOWER
        [ -n "${SEQUENCE_TIS_MASK_UPPER:-}" ] && export SEQUENCE_TIS_MASK_UPPER
        [ -n "${TOTAL_EPISODES:-}" ] && export TOTAL_EPISODES
        [ -n "${MAX_STEPS:-}" ] && export MAX_STEPS
        # ASYNC_STEPS may legitimately be 0 (synchronous), so forward it whenever set (incl. 0).
        [ -n "${ASYNC_STEPS+x}" ] && export ASYNC_STEPS
        [ -n "${LEARNING_RATE:-}" ] && export LEARNING_RATE
        [ -n "${OPTIMIZER_TYPE:-}" ] && export OPTIMIZER_TYPE
        [ -n "${SGD_MOMENTUM:-}" ] && export SGD_MOMENTUM
        [ -n "${MAX_GRAD_NORM:-}" ] && export MAX_GRAD_NORM
        [ -n "${LOSS_FN:-}" ] && export LOSS_FN
        [ -n "${BETA:-}" ] && export BETA
        [ -n "${ALPHA:-}" ] && export ALPHA
        [ -n "${REF_POLICY_UPDATE_FREQ:-}" ] && export REF_POLICY_UPDATE_FREQ
        [ -n "${CLIP_LOWER:-}" ] && export CLIP_LOWER
        [ -n "${CLIP_HIGHER:-}" ] && export CLIP_HIGHER
        [ -n "${TIS_MASK_LOWER:-}" ] && export TIS_MASK_LOWER
        [ -n "${TIS_MASK_UPPER:-}" ] && export TIS_MASK_UPPER
        [ -n "${USE_VLLM_LOGPROBS:-}" ] && export USE_VLLM_LOGPROBS
        [ -n "${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP:-}" ] && export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP
        [ -n "${SEQUENCE_TIS_MASK_LOWER:-}" ] && export SEQUENCE_TIS_MASK_LOWER
        [ -n "${SEQUENCE_TIS_MASK_UPPER:-}" ] && export SEQUENCE_TIS_MASK_UPPER
        [ -n "${CHECKPOINT_STATE_FREQ:-}" ] && export CHECKPOINT_STATE_FREQ
        [ -n "${KEEP_LAST_N_CHECKPOINTS:-}" ] && export KEEP_LAST_N_CHECKPOINTS
        [ -n "${LOG_MOMENTUM_DIAGNOSTICS:-}" ] && export LOG_MOMENTUM_DIAGNOSTICS
        # Resubmit to a specific QOS if requested (e.g. h100_comm_shared); default keeps the launcher's #SBATCH qos.
        QOS_ARG=()
        [ -n "${QOS:-}" ] && QOS_ARG=(--qos="${QOS}")
        # Sick-node pin (Oscar's catch, collab#1): manual launches carried --exclude but
        # resubmits did not, so exactly the restarts that follow a crash could land back on
        # the nodes that caused it. Rides the watchdog env like QOS does.
        EXCLUDE_ARG=()
        [ -n "${EXCLUDE_NODES:-}" ] && EXCLUDE_ARG=(--exclude="${EXCLUDE_NODES}")
        # tmax-private#1 preflight: run the tool-config emission + dataclass assertion HERE,
        # on this cpu node, before the resubmit can consume 64 GPUs on a schema-drift boot.
        # A preflight failure is config drift, not a transient — needs a human, exit 1.
        # Mirror the launcher defaults for vars the watchdog env may not carry, so the
        # preflight evaluates the same config the resubmitted launcher would emit.
        if ! PYTHONPATH="${REPO_ROOT}" \
                TASK_DATA_HF_REPO="${TASK_DATA_HF_REPO:-hamishivi/swerl-tmax-15k}" \
                SWERL_SANDBOX_TEST_TIMEOUT="${SWERL_SANDBOX_TEST_TIMEOUT:-120}" \
                SWERL_SANDBOX_TIMEOUT="${SWERL_SANDBOX_TIMEOUT:-120}" \
                "${UV_PROJECT_ENVIRONMENT:-/checkpoint/comem/rulin/open-instruct/.venv}/bin/python" \
                "${REPO_ROOT}/scripts/train/debug/envs/emit_tool_configs.py" > /dev/null; then
            log "ERROR: emit_tool_configs preflight FAILED — refusing to resubmit onto a boot-fatal config. Watchdog exiting for a human."
            return 1
        fi
        newid="$(cd "${REPO_ROOT}" && sbatch --parsable --job-name="${JOB_NAME}" "${QOS_ARG[@]}" "${EXCLUDE_ARG[@]}" "${LAUNCHER}" 2>/dev/null)"
        if [ -z "${newid}" ]; then
            log "ERROR: resubmit failed (empty job id). Watchdog exiting."
            return 1
        fi
        log "Resubmitted as job ${newid} (resumes from checkpoint_state_dir of RUN_ID=${RUN_ID})."
        cur="${newid}"
        return 0
    fi

    log "job ${jid}: no transient signature found in last ${TAIL_LINES} lines. NOT auto-restarting (needs a human)."
    return 1
}

# Return 0 if the job should be considered still-active, 1 only if it has
# genuinely reached a terminal state. Robust against transient squeue/slurmctld
# hiccups: an empty/failed `squeue` is NOT treated as "ended" unless `sacct`
# independently confirms a terminal state. This prevents the watchdog from
# mis-firing (and exiting) when squeue momentarily returns nothing for a job
# that is actually still RUNNING.
job_is_active() {
    local jid="$1" out rc state
    out="$(squeue -j "${jid}" -h -o '%T' 2>/dev/null)"; rc=$?
    if [ -n "${out}" ]; then return 0; fi                   # present in queue -> active
    # squeue failed OR returned empty. A failure is usually a slurmctld hiccup, but it is
    # ALSO what "Invalid job id specified" looks like once Slurm has purged an old job from
    # its in-memory queue -- and blindly assuming "active" there wedges the watchdog forever
    # on a job that died long ago (observed: adamw sat 23h on a purged id, never restarted).
    # Either way sacct is the authority, so fall through to it instead of guessing.
    state="$(sacct -j "${jid}" --format=State -n -P 2>/dev/null | head -1)"
    case "${state}" in
        ""|RUNNING*|PENDING*|REQUEUED*|COMPLETING*|CONFIGURING*|SUSPENDED*|RESIZING*)
            return 0 ;;                                      # unknown/non-terminal -> assume active
        *)
            return 1 ;;                                      # terminal (COMPLETED/FAILED/CANCELLED/TIMEOUT/NODE_FAIL/...)
    esac
}

log "Watchdog start: watching job ${cur} RUN_ID=${RUN_ID} exp=${EXP_NAME} (${NUM_UNIQUE_PROMPTS_ROLLOUT}x${NUM_SAMPLES_PER_PROMPT_ROLLOUT}), max_restarts=${MAX_RESTARTS}"

while true; do
    # Wait for the current job to genuinely leave the queue. Require TWO
    # consecutive terminal reads (POLL_S apart) so a single transient blip in
    # squeue/sacct can never be mistaken for job termination.
    while true; do
        if ! job_is_active "${cur}"; then
            sleep "${POLL_S}"
            job_is_active "${cur}" || break
        fi
        sleep "${POLL_S}"
    done
    log "job ${cur} confirmed terminal (two consecutive reads); evaluating outcome."
    # Give Slurm/sacct a moment to finalize the record + fully flush the 8-node
    # crash logs (async actor output can lag the job's terminal state).
    sleep 90
    classify_and_maybe_resubmit "${cur}"
    rc=$?
    if [ "${rc}" -eq 2 ]; then exit 0; fi          # success
    # Deliberate decline (deterministic failure / max restarts / unknown signature) exits 0:
    # the watchdog WORKED — it decided not to restart. A non-zero exit here makes sacct show
    # the watchdog itself as FAILED, so "watchdog FAILED" stops meaning "watchdog broke"
    # (Oscar's report, tmax-collaboration#2). The decision is in the log either way.
    if [ "${rc}" -eq 1 ]; then log "Watchdog exiting after deliberate no-restart decision (see above)."; exit 0; fi
    # rc==0 -> resubmitted, loop to watch the new job
done
