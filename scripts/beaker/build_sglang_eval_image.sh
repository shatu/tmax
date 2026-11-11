#!/usr/bin/env bash
# Build scripts/beaker/Dockerfile.sglang_eval at the current commit and push it to
# Beaker twice, with the same conventions as build_vllm_eval_image.sh:
#   <user>/tmax-sglang-eval_<commit>   permanent record of this build
#   <user>/tmax-sglang-eval_latest     the image `launch_eval.sh --engine sglang` should use
# Both get "tmax scripts/beaker/Dockerfile.sglang_eval @ <full sha> (base <name> = <id>)"
# as their description: the base is the Miles training image, so it is an input
# of the build just like the Dockerfile.
#
# The previous _latest loses its name (its _<commit> twin keeps one) only after
# the new one is uploaded, so _latest never goes missing.
#
# Usage (from anywhere in the tmax repo, on a machine with podman + beaker CLI;
# the base image is ~52 GB and each upload ~51 GB):
#   scripts/beaker/build_sglang_eval_image.sh [--base shashankg/tmax-miles-train-20261029] \
#       [--workspace ai2/oe-agents] [--allow-dirty]
set -euo pipefail

BASE_BEAKER_IMAGE="shashankg/tmax-miles-train-20261029"
WORKSPACE="ai2/oe-agents"
ALLOW_DIRTY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --base)        BASE_BEAKER_IMAGE="$2"; shift 2 ;;
        --workspace)   WORKSPACE="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        -h|--help)     sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd "$(git rev-parse --show-toplevel)"
DOCKERFILE=scripts/beaker/Dockerfile.sglang_eval
OI=training/open-instruct
OI_INPUTS=(docker/podman scripts/docker/setup_podman.sh)

# The commit in the image's name and description must describe what is in it.
dirty=0
git diff --quiet HEAD -- "$DOCKERFILE" "$OI" || dirty=1
git -C "$OI" diff --quiet HEAD -- "${OI_INPUTS[@]}" || dirty=1
if [ "$dirty" = 1 ] && [ "$ALLOW_DIRTY" != 1 ]; then
    echo "error: $DOCKERFILE or its open-instruct inputs differ from HEAD; commit first (or --allow-dirty)" >&2
    exit 1
fi

SHA="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short=12 HEAD)"
BEAKER_USER="$(beaker account whoami --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["name"])')"
BASE_ID="$(beaker image get "$BASE_BEAKER_IMAGE" --format json \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print(d["id"])')"
NAME_COMMIT="tmax-sglang-eval_${SHORT}"
NAME_LATEST="tmax-sglang-eval_latest"
DESC="tmax scripts/beaker/Dockerfile.sglang_eval @ ${SHA} (base ${BASE_BEAKER_IMAGE} = ${BASE_ID})"
[ "$dirty" = 1 ] && DESC="$DESC (dirty tree)"
BASE_LOCAL_TAG="localhost/tmax-sglang-eval-base:${BASE_ID,,}"
LOCAL_TAG="localhost/tmax-sglang-eval:${SHORT}"

if beaker image get "$BEAKER_USER/$NAME_COMMIT" >/dev/null 2>&1; then
    echo "error: $BEAKER_USER/$NAME_COMMIT already exists; nothing to do for this commit" >&2
    exit 1
fi

# beaker image pull/create talk the Docker API; serve podman's on a short socket
# path (unix socket paths are limited to ~108 bytes).
SOCK="/tmp/tmax-sglang-eval-build-$$.sock"
podman system service --time=0 "unix://$SOCK" >/dev/null 2>&1 &
SERVICE_PID=$!
trap 'kill "$SERVICE_PID" 2>/dev/null || true; rm -f "$SOCK"' EXIT
for _ in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.2; done
[ -S "$SOCK" ] || { echo "error: podman API socket did not come up" >&2; exit 1; }
export DOCKER_HOST="unix://$SOCK"

if ! podman image exists "$BASE_LOCAL_TAG"; then
    echo "== pulling base $BASE_BEAKER_IMAGE ($BASE_ID) as $BASE_LOCAL_TAG"
    beaker image pull "$BASE_ID" "$BASE_LOCAL_TAG"
fi

echo "== building $LOCAL_TAG"
podman build -f "$DOCKERFILE" --build-arg BASE_IMAGE="$BASE_LOCAL_TAG" -t "$LOCAL_TAG" .

echo "== uploading $BEAKER_USER/$NAME_COMMIT"
beaker image create "$LOCAL_TAG" -n "$NAME_COMMIT" -w "$WORKSPACE" --description "$DESC"

# Upload the new _latest under a temporary name, then swap names.
TMP_NAME="${NAME_LATEST}-new-${SHORT}"
echo "== uploading $BEAKER_USER/$TMP_NAME"
beaker image create "$LOCAL_TAG" -n "$TMP_NAME" -w "$WORKSPACE" --description "$DESC"
if beaker image get "$BEAKER_USER/$NAME_LATEST" >/dev/null 2>&1; then
    echo "== un-naming the previous $NAME_LATEST"
    beaker image rename "$BEAKER_USER/$NAME_LATEST" ""
fi
beaker image rename "$BEAKER_USER/$TMP_NAME" "$NAME_LATEST"

echo "== done: $BEAKER_USER/$NAME_LATEST and $BEAKER_USER/$NAME_COMMIT -> $DESC"
echo "   Point launch_eval.sh's SGLANG_BEAKER_IMAGE default at $BEAKER_USER/$NAME_LATEST if it isn't already."
