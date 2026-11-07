#!/usr/bin/env bash
# Build scripts/beaker/Dockerfile.vllm_eval at the current commit and push it to
# Beaker twice:
#   <user>/tmax-vllm-eval_<commit>   permanent record of this build
#   <user>/tmax-vllm-eval_latest     what launch_eval.sh uses by default
# Both get "tmax scripts/beaker/Dockerfile.vllm_eval @ <full sha>" as their
# description. launch_eval.sh resolves _latest to an image ID at launch and
# eval_meta.json records it, so an eval stays traceable after _latest moves.
#
# The previous _latest loses its name (its _<commit> twin keeps one) only after
# the new one is uploaded, so _latest never goes missing.
#
# Usage (from anywhere in the tmax repo, on a machine with podman + beaker CLI):
#   scripts/beaker/build_vllm_eval_image.sh [--workspace ai2/oe-agents] [--allow-dirty]
set -euo pipefail

WORKSPACE="ai2/oe-agents"
ALLOW_DIRTY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --workspace)   WORKSPACE="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        -h|--help)     sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd "$(git rev-parse --show-toplevel)"
DOCKERFILE=scripts/beaker/Dockerfile.vllm_eval
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
NAME_COMMIT="tmax-vllm-eval_${SHORT}"
NAME_LATEST="tmax-vllm-eval_latest"
DESC="tmax scripts/beaker/Dockerfile.vllm_eval @ ${SHA}"
[ "$dirty" = 1 ] && DESC="$DESC (dirty tree)"
LOCAL_TAG="localhost/tmax-vllm-eval:${SHORT}"

if beaker image get "$BEAKER_USER/$NAME_COMMIT" >/dev/null 2>&1; then
    echo "error: $BEAKER_USER/$NAME_COMMIT already exists; nothing to do for this commit" >&2
    exit 1
fi

echo "== building $LOCAL_TAG"
podman build -f "$DOCKERFILE" -t "$LOCAL_TAG" .

# beaker image create talks the Docker API; serve podman's on a short socket path
# (unix socket paths are limited to ~108 bytes).
SOCK="/tmp/tmax-vllm-eval-build-$$.sock"
podman system service --time=0 "unix://$SOCK" >/dev/null 2>&1 &
SERVICE_PID=$!
trap 'kill "$SERVICE_PID" 2>/dev/null || true; rm -f "$SOCK"' EXIT
for _ in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.2; done
[ -S "$SOCK" ] || { echo "error: podman API socket did not come up" >&2; exit 1; }
export DOCKER_HOST="unix://$SOCK"

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
