#!/usr/bin/env bash
#
# Build + push the task images a Dockerfile-only harbor dataset needs on the
# OpenSandbox backend, as a CPU-only Beaker job (podman, linux/amd64, fast
# registry uplink). Building 100 TBLite images under qemu on a laptop is slow;
# this does it where the podman eval path already builds them today.
#
# Usage:
#   ./scripts/opensandbox/launch_build_task_images.sh <dataset> --repo <registry repo> [options]
#
# Example:
#   ./scripts/opensandbox/launch_build_task_images.sh openthoughts-tblite@2.0 \
#       --repo docker.io/<user>/tmax-harbor-tasks \
#       --docker-username <user> --docker-pat-secret <user>_DOCKER_PAT_RW
#
# The registry credential must have WRITE scope (the read-only pull PATs used
# by the eval jobs are not enough). Afterwards, run evals with
#   ./beaker_configs/launch_eval.sh ... --harbor-env opensandbox --task-image-repo <repo>
#
# Options:
#   --repo REPO              registry repo to push to (required)
#   --docker-username USER   registry username (default: $DOCKERHUB_USERNAME)
#   --docker-pat-secret S    Beaker secret holding the registry password/PAT
#                            (default: ${USER}_DOCKER_PAT)
#   --dataset-path DIR       build from a local task dir (weka) instead of a
#                            registry dataset name
#   --jobs N                 parallel builds (default: 4)
#   --cluster C  --workspace W  --priority P  --budget B  (Beaker placement)
#   --repo-ref REF           tmax git ref to run (default: current HEAD)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET=""
DATASET_PATH=""
IMAGE_REPO=""
DOCKER_USERNAME="${DOCKERHUB_USERNAME:-}"
DOCKER_PAT_SECRET="${DOCKER_PAT_SECRET:-${USER}_DOCKER_PAT}"
JOBS=4
CLUSTER="${CLUSTER:-ai2/jupiter}"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/oe-agents}"
PRIORITY="normal"
BUDGET=""
REPO_GIT_REF=""
BEAKER_DOCKER_IMAGE="${BEAKER_DOCKER_IMAGE:-hamishivi/tmax-eval-interactive}"

if [ $# -gt 0 ] && [[ "$1" != --* ]]; then DATASET="$1"; shift; fi
while [ $# -gt 0 ]; do
    case "$1" in
        --repo)             IMAGE_REPO="$2"; shift 2 ;;
        --docker-username)  DOCKER_USERNAME="$2"; shift 2 ;;
        --docker-pat-secret) DOCKER_PAT_SECRET="$2"; shift 2 ;;
        --dataset-path)     DATASET_PATH="$2"; shift 2 ;;
        --jobs)             JOBS="$2"; shift 2 ;;
        --cluster)          CLUSTER="$2"; shift 2 ;;
        --workspace)        BEAKER_WORKSPACE="$2"; shift 2 ;;
        --priority)         PRIORITY="$2"; shift 2 ;;
        --budget)           BUDGET="$2"; shift 2 ;;
        --repo-ref)         REPO_GIT_REF="$2"; shift 2 ;;
        -h|--help)          grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

[ -n "$IMAGE_REPO" ] || { echo "error: --repo is required" >&2; exit 1; }
[ -n "$DATASET$DATASET_PATH" ] || { echo "error: a dataset name or --dataset-path is required" >&2; exit 1; }
[ -n "$DOCKER_USERNAME" ] || { echo "error: --docker-username (or DOCKERHUB_USERNAME) is required" >&2; exit 1; }
REPO_GIT_REF="${REPO_GIT_REF:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
SLUG="${DATASET:-$(basename "$DATASET_PATH")}"
SLUG="${SLUG//[^A-Za-z0-9]/-}"

cat <<EOF
=== Building task images on Beaker ===
  Dataset:   ${DATASET:-$DATASET_PATH}
  Repo:      ${IMAGE_REPO}
  Registry:  ${DOCKER_USERNAME} / secret ${DOCKER_PAT_SECRET}
  Cluster:   ${CLUSTER}  workspace=${BEAKER_WORKSPACE}
  tmax ref:  ${REPO_GIT_REF}
EOF

# In-job script: podman is already in the eval image; harbor downloads the
# dataset; the build script probes the registry and builds+pushes the rest.
read -r -d '' INNER <<'INNER' || true
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
if [ -n "${DATASET_PATH:-}" ]; then
    TASKS_DIR="$DATASET_PATH"
else
    uv run harbor datasets download "$DATASET" -o /tmp/tasks --export
    TASKS_DIR="/tmp/tasks/${DATASET%%@*}"
fi
printf '%s' "$DOCKER_PAT" | podman login -u "$DOCKER_USERNAME" --password-stdin "${IMAGE_REPO%%/*}"
uv run python scripts/opensandbox/build_task_images.py "$TASKS_DIR" \
    --repo "$IMAGE_REPO" --builder podman --push --jobs "$JOBS" \
    --log-dir /results/build-logs --manifest /results/task_images.json
INNER

GANTRY_CMD=(
    uvx --from beaker-gantry gantry --quiet run
    --yes --allow-dirty
    --workspace "$BEAKER_WORKSPACE"
    --name "build-task-images-${SLUG}"
    --description "Build+push OpenSandbox task images for ${DATASET:-$DATASET_PATH} -> ${IMAGE_REPO}"
    --ref "$REPO_GIT_REF"
    --gpus 0
    --priority "$PRIORITY"
    --weka "oe-adapt-default:/weka/oe-adapt-default"
    --docker-image "$BEAKER_DOCKER_IMAGE"
    --env-secret "DOCKER_PAT=${DOCKER_PAT_SECRET}"
    --env "DOCKER_USERNAME=${DOCKER_USERNAME}"
    --env "DATASET=${DATASET}"
    --env "DATASET_PATH=${DATASET_PATH}"
    --env "IMAGE_REPO=${IMAGE_REPO}"
    --env "JOBS=${JOBS}"
    --env BEAKER_ALLOW_SUBCONTAINERS=1
    --env BEAKER_SKIP_DOCKER_SOCKET=1
    --host-networking
    --propagate-failure
    --no-python
)
for cluster in ${CLUSTER//,/ }; do GANTRY_CMD+=(--cluster "$cluster"); done
[ -n "$BUDGET" ] && GANTRY_CMD+=(--budget "$BUDGET")
GANTRY_CMD+=(-- bash -c "$INNER")

printf 'Launching with:'; printf ' %q' "${GANTRY_CMD[@]}"; printf '\n\n'
"${GANTRY_CMD[@]}"
