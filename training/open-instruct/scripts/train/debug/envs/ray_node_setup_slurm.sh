#!/usr/bin/env bash
# Start a Ray cluster inside a Slurm allocation.
#
# Source this script from every Slurm task. Rank 0 starts the Ray head and
# returns to the caller; non-zero ranks join as workers and block while
# monitoring the head node.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/checkpoint_env.sh"

if [ -z "${RAY_NODE_PORT:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        RAY_NODE_PORT="$((20000 + SLURM_JOB_ID % 20000))"
    else
        RAY_NODE_PORT="28888"
    fi
fi
NODE_RANK="${SLURM_NODEID:-${SLURM_PROCID:-0}}"

if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')"
else
    HEAD_NODE="$(hostname)"
fi

HEAD_IP="$(getent hosts "${HEAD_NODE}" | awk '{print $1; exit}')"
if [ -z "${HEAD_IP}" ]; then
    echo "Could not resolve Ray head node ${HEAD_NODE}" >&2
    exit 1
fi

export RAY_ADDRESS="${HEAD_IP}:${RAY_NODE_PORT}"
RAY_CPU_ARGS=()
if [ -n "${RAY_NUM_CPUS:-}" ]; then
    RAY_CPU_ARGS=(--num-cpus="${RAY_NUM_CPUS}")
fi

echo "SLURM_NODEID=${NODE_RANK} HEAD_NODE=${HEAD_NODE} RAY_ADDRESS=${RAY_ADDRESS}"
if [ -n "${RAY_NUM_CPUS:-}" ]; then
    echo "RAY_NUM_CPUS=${RAY_NUM_CPUS}"
fi
ray stop --force >/dev/null 2>&1 || true

archive_ray_logs() {
    if [ -z "${RAY_LOG_ARCHIVE_DIR:-}" ] || [ -z "${RAY_TMPDIR:-}" ] || [ ! -d "${RAY_TMPDIR}" ]; then
        return
    fi
    mkdir -p "${RAY_LOG_ARCHIVE_DIR}"
    archive="${RAY_LOG_ARCHIVE_DIR}/ray-${SLURM_JOB_ID:-nojob}-rank${NODE_RANK}-$(hostname).tgz"
    tar -C "${RAY_TMPDIR}" -czf "${archive}" . >/dev/null 2>&1 || true
    echo "[ray_node_setup_slurm] archived Ray logs to ${archive}"
}

if [ "${NODE_RANK}" = "0" ]; then
    trap 'archive_ray_logs' EXIT
    echo "Starting Ray head on ${HEAD_IP}:${RAY_NODE_PORT}"
    ray start \
        --head \
        --node-ip-address="${HEAD_IP}" \
        --port="${RAY_NODE_PORT}" \
        --dashboard-host=0.0.0.0 \
        --temp-dir="${RAY_TMPDIR}" \
        "${RAY_CPU_ARGS[@]}"
else
    echo "Starting Ray worker ${NODE_RANK}, connecting to ${RAY_ADDRESS}"
    ray start \
        --address="${RAY_ADDRESS}" \
        --dashboard-host=0.0.0.0 \
        "${RAY_CPU_ARGS[@]}"

    cleanup() {
        exit_code="${1:-1}"
        echo "[ray_node_setup_slurm] stopping worker ${NODE_RANK}; exit=${exit_code}"
        archive_ray_logs
        ray stop --force >/dev/null 2>&1 || true
        trap - TERM INT HUP EXIT
        exit "${exit_code}"
    }

    trap 'cleanup 143' TERM
    trap 'cleanup 130' INT
    trap 'cleanup 129' HUP
    trap 'cleanup 1' EXIT

    while true; do
        if ! ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
            echo "[ray_node_setup_slurm] Ray head is unreachable; worker ${NODE_RANK} exiting."
            cleanup 0
        fi
        sleep 5
    done
fi
