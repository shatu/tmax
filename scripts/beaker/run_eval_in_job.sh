#!/usr/bin/env bash
#
# Inner script invoked inside a beaker task. Sets up podman + harbor, spins up
# a vLLM server on the local GPUs, then runs harbor against it.
#
# Driven entirely by env vars (set by beaker_configs/launch_eval.sh):
#   MODEL_PATH               HF model path or weka path (required)
#   MODEL_REVISION           HF revision/branch (default: main)
#   SERVED_MODEL_NAME        --served-model-name for vLLM (required)
#   HARBOR_MODEL_NAME        optional model name passed to harbor
#   VLLM_VERSION             vLLM package version for uvx (default: 0.19.1)
#   VLLM_TOOL_CALL_PARSER    vLLM tool parser (default: hermes)
#   VLLM_LANGUAGE_MODEL_ONLY pass --language_model_only to vLLM when set to 1
#   VLLM_PORT                port for vLLM (default: 8008)
#   TP_SIZE                  --tensor-parallel-size (required)
#   DP_SIZE                  --data-parallel-size (default: 1)
#   MAX_MODEL_LEN            optional --max-model-len
#   DATASET                  harbor dataset, e.g. terminal-bench@2.0
#   HARBOR_ENV               harbor environment backend (default: docker)
#   AGENT_IMPORT_PATH        e.g. Vanillux2Agent:Vanillux2Agent
#   EXTRA_UV_PIP_INSTALLS    optional space-separated packages to uv pip install
#   EXTRA_AGENT_KWARGS       optional newline-separated harbor --agent-kwarg values
#   EXTRA_AGENT_ENVS         optional newline-separated harbor --agent-env values
#   HOSTED_VLLM_MODEL_INFO   optional JSON model_info for Harbor agents
#   N_CONCURRENT             default 8
#   N_ATTEMPTS               default 1
#   N_TASKS                  optional harbor --n-tasks limit
#   INCLUDE_TASK_NAMES       optional newline-separated --include-task-name globs
#   EXCLUDE_TASK_NAMES       optional newline-separated --exclude-task-name globs
#   HARBOR_OVERRIDE_CPUS     optional per-task environment CPU override
#   HARBOR_OVERRIDE_MEMORY_MB
#                            optional per-task environment memory override
#   HARBOR_OVERRIDE_STORAGE_MB
#                            optional per-task environment storage override
#   HARBOR_OVERRIDE_GPUS     optional per-task environment GPU override
#   HARBOR_*_TIMEOUT_MULTIPLIER
#                            optional Harbor timeout multiplier flags
#   HARBOR_AGENT_TIMEOUT_SEC optional exact per-task agent timeout override
#   JOB_NAME                 harbor job name
#   RESULTS_DIR              /weka path to copy jobs/$JOB_NAME into
#   REPO_GIT_URL, REPO_GIT_REF
#                            optional — if set, this script self-clones into a
#                            workdir; otherwise it assumes pwd is the repo.

set -euo pipefail

log() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$*"; }

# --- 0. Workdir: clone repo if URL given, else use cwd ----------------------
if [ -n "${REPO_GIT_URL:-}" ]; then
    WORKDIR="${WORKDIR:-/workspace/tmax}"
    if [ ! -d "$WORKDIR/.git" ]; then
        log "cloning $REPO_GIT_URL @ ${REPO_GIT_REF:-HEAD} -> $WORKDIR"
        mkdir -p "$(dirname "$WORKDIR")"
        git clone "$REPO_GIT_URL" "$WORKDIR"
        if [ -n "${REPO_GIT_REF:-}" ]; then
            git -C "$WORKDIR" checkout "$REPO_GIT_REF"
        fi
    fi
    cd "$WORKDIR"
fi

# --- 1. Install podman + deps -----------------------------------------------
if ! command -v podman >/dev/null 2>&1; then
    log "installing podman + helpers"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq podman crun uidmap fuse-overlayfs slirp4netns \
        curl git ca-certificates
fi

# Some images (e.g. AI2's cuda gantry images) ship a real Docker CLI without
# the compose plugin. Harbor shells out to `docker compose ...`, so without
# the plugin every up/down errors with "unknown shorthand flag: 'p' in -p"
# (Docker CLI rejects `compose` as a subcommand and then mis-parses `-p`).
# Drop in the official compose v2 static binary as a user-level CLI plugin —
# it talks the Docker API, which podman serves on /tmp/podman.sock.
if ! docker compose version >/dev/null 2>&1; then
    log "installing docker compose v2 plugin"
    DOCKER_COMPOSE_VERSION="${DOCKER_COMPOSE_VERSION:-v2.39.4}"
    DOCKER_COMPOSE_ARCH="$(uname -m)"
    mkdir -p /root/.docker/cli-plugins
    curl -fsSL \
        "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-${DOCKER_COMPOSE_ARCH}" \
        -o /root/.docker/cli-plugins/docker-compose
    chmod +x /root/.docker/cli-plugins/docker-compose
fi

# --- 2. Write containers.conf -----------------------------------------------
log "writing /etc/containers/containers.conf"
mkdir -p /etc/containers
# NOTE (TB3 / harbor 0.21): bridge/compose networking is IMPOSSIBLE in a
# beaker job container — proven empirically (smokes 3-9 + a diagnostics job):
#   * the job has only the plain docker default caps (no NET_ADMIN/SYS_ADMIN),
#   * /proc/sys is a LOCKED read-only mount (netavark's sysctl writes fail,
#     and no user/mount-namespace trick can undo a locked mount),
#   * /proc has tmpfs-masked paths, so fresh proc mounts inside a userns are
#     kernel-blocked ("fully visible" rule),
#   * no-new-privileges blocks the setuid newuidmap/newgidmap helpers, so
#     multi-uid ROOTLESS podman can't start either.
# So every container runs on the HOST network namespace (netns="host", the
# TB2-era proven mechanism), and harbor is patched (step 3) to force
# network_mode: host on every compose service with extra_hosts aliases
# (service-name -> 127.0.0.1) standing in for compose DNS.
# userns="auto" was dropped (2026-08-15): harbor 0.21 moves files with
# `podman cp`-style API uploads, which create files owned by uids OUTSIDE an
# auto userns's mapping — verifier scripts that chmod their own /logs/verifier
# (ks-solver-cpp, ontology-kg-querying) then die with "Operation not
# permitted" before writing reward.txt. Without a userns, container root is
# the job container's root: uploads match, chmod works, and setpriv-to-nobody
# verifiers still work under real CAP_SETUID.
cat > /etc/containers/containers.conf <<'CONF'
[containers]
netns="host"
ipcns="host"
utsns="host"
cgroupns="host"
cgroups="disabled"
keyring=false
log_driver = "k8s-file"
volumes = [
        "/proc:/proc",
]
default_sysctls = []
[engine]
cgroup_manager = "cgroupfs"
events_logger="file"
runtime="crun"
compose_warning_logs=false
CONF


# --- 2b. Matching netavark/aardvark-dns --------------------------------------
# The beaker image ships a NEW podman (5.x) next to Ubuntu noble's ANCIENT
# netavark/aardvark-dns 1.4 debs. podman 5.x changed the netavark IPAM
# contract (>= 1.6 required), so every bridge/compose network dies with
# "IPAM error: failed to get ips ..." + "failed to parse ipam options: no
# static ips provided" + a misleading aardvark-dns-directory IO error. Every
# harbor 0.21 compose project creates a network, so this errors 100% of
# trials. Install matching static release binaries when the found netavark is
# too old, and point podman at them.
NETAVARK_MIN_VER="1.6"
NETAVARK_PIN="v2.1.0"
netavark_bin="$(ls /usr/local/lib/podman/netavark /usr/lib/podman/netavark /usr/libexec/podman/netavark 2>/dev/null | head -1 || true)"
netavark_ver="$("${netavark_bin:-false}" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
if [ -z "$netavark_ver" ] || [ "$(printf '%s\n' "$netavark_ver" "$NETAVARK_MIN_VER" | sort -V | head -1)" != "$NETAVARK_MIN_VER" ]; then
    log "netavark ${netavark_ver:-<none>} too old for podman $(podman --version 2>/dev/null || echo '?') — installing static $NETAVARK_PIN"
    mkdir -p /usr/local/lib/podman
    curl -fsSL "https://github.com/containers/netavark/releases/download/$NETAVARK_PIN/netavark.gz" \
        | gunzip > /usr/local/lib/podman/netavark
    curl -fsSL "https://github.com/containers/aardvark-dns/releases/download/$NETAVARK_PIN/aardvark-dns.gz" \
        | gunzip > /usr/local/lib/podman/aardvark-dns
    chmod +x /usr/local/lib/podman/netavark /usr/local/lib/podman/aardvark-dns
    /usr/local/lib/podman/netavark --version
    # containers.conf: force podman onto the new binaries (searched before the
    # distro paths). Insert into the existing [engine] table (TOML forbids a
    # duplicate [engine]) and add a [network] table.
    sed -i '/^\[engine\]/a network_cmd_path = "/usr/local/lib/podman/netavark"\nhelper_binaries_dir = ["/usr/local/lib/podman", "/usr/local/libexec/podman", "/usr/lib/podman", "/usr/libexec/podman"]' /etc/containers/containers.conf
    printf '\n[network]\nnetwork_backend = "netavark"\n' >> /etc/containers/containers.conf
fi

log "running uv sync"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync
if [ "${HARBOR_ENV:-docker}" = "daytona" ]; then
    log "installing harbor daytona extra"
    uv pip install 'harbor[daytona]'
fi
if [ -n "${EXTRA_UV_PIP_INSTALLS:-}" ]; then
    log "installing extra packages: ${EXTRA_UV_PIP_INSTALLS}"
    # shellcheck disable=SC2086
    uv pip install ${EXTRA_UV_PIP_INSTALLS}
fi

log "patching harbor for podman compat (harbor 0.21.x)"
uv run python - <<'PY'
import os, pathlib, harbor
hdir = pathlib.Path(harbor.__file__).parent

# NOTE (harbor 0.21 upgrade, TB3): the old 0.6.6-era patches are gone on
# purpose:
#   * verifier/oracle/paths chmod + compose ":U" bind-mount patches — obsolete.
#     0.21's docker environment has NO bind mounts; agent/verifier logs and
#     artifacts move via upload/download over the API, and TrialPaths already
#     chmods its dirs (models/trial/paths.py:chmod_dir).
#   * mini-swe-agent step_limit + openhands install patches — dropped in the
#     upgrade (anchors changed upstream). Re-derive from git history if a
#     mini-swe/openhands run is ever needed on this branch.

# (1) OPTIONAL network_mode: host on the generated "main" service.
# Vanillux2Agent runs HOST-side (only bash execs enter the container), so task
# containers never need to reach vLLM and stock compose networking is correct —
# and REQUIRED for TB3's multi-service tasks (a host-netns "main" cannot
# resolve sibling services like "db" by compose DNS). Set
# HARBOR_NETWORK_MODE_HOST=1 only for agents that run INSIDE the task
# container and call vLLM at localhost (mini-swe-agent, swe-agent).
if os.environ.get("HARBOR_NETWORK_MODE_HOST", "0") == "1":
    for name in ("docker-compose-build.yaml", "docker-compose-prebuilt.yaml"):
        compose = hdir / "environments/docker" / name
        text = compose.read_text()
        if "network_mode: host" not in text:
            assert "  main:\n" in text, f"anchor missing in {name}"
            text = text.replace("  main:\n", "  main:\n    network_mode: host\n", 1)
            compose.write_text(text)
            print(f"patched {name}: main runs with network_mode: host")
else:
    print("compose templates: stock networking (HARBOR_NETWORK_MODE_HOST!=1)")

# (2) Image retention after each trial. harbor 0.21 stock is
# `compose down --rmi local` (removes the locally-built task image after every
# trial; prebuilt/tagged images survive).
#
# DEFAULT (HARBOR_KEEP_TASK_IMAGES unset/0): keep harbor's stock behaviour.
#   Required for large per-task-image datasets (swebench-verified ~1.5 TB
#   retained would wedge the node).
#
# HARBOR_KEEP_TASK_IMAGES=1: drop `--rmi local` so built images persist in
#   podman storage. Sensible for small sets (tb2 = 89 images / ~9 GB); for
#   TB3's 74 locally-BUILT images it also skips a full rebuild on every
#   attempt (relevant at k>1 — at k=1 each task builds once either way).
docker_py = hdir / "environments/docker/docker.py"
text = docker_py.read_text()
_rmi = '["down", "--rmi", "local", "--volumes", "--remove-orphans"]'
if os.environ.get("HARBOR_KEEP_TASK_IMAGES", "0") == "1":
    if _rmi in text:
        text = text.replace(_rmi, '["down", "--volumes", "--remove-orphans"]')
        docker_py.write_text(text)
        print("patched docker.py: dropped --rmi local (HARBOR_KEEP_TASK_IMAGES=1)")
    else:
        assert '"--rmi"' not in text, "docker.py --rmi anchor changed; fix the patch"
        print("docker.py: already patched")
else:
    print("docker.py: keeping harbor stock --rmi local (built images deleted per trial)")

# (3) Exact per-task agent timeout override. Harbor's CLI exposes timeout
# MULTIPLIERS but not AgentConfig.override_timeout_sec; let launch_eval set
# HARBOR_AGENT_TIMEOUT_SEC for evals needing a uniform wall-clock cap.
jobs_py = hdir / "cli/jobs.py"
text = jobs_py.read_text()
if "HARBOR_AGENT_TIMEOUT_SEC" not in text:
    anchor = "    if n_concurrent_agents is not None:\n"
    assert anchor in text, "cli/jobs.py anchor changed; fix the HARBOR_AGENT_TIMEOUT_SEC patch"
    text = text.replace(
        anchor,
        "    harbor_agent_timeout_sec = __import__(\"os\").environ.get(\"HARBOR_AGENT_TIMEOUT_SEC\")\n"
        "    if harbor_agent_timeout_sec:\n"
        "        for agent in config.agents:\n"
        "            agent.override_timeout_sec = float(harbor_agent_timeout_sec)\n\n"
        + anchor,
        1,
    )
    jobs_py.write_text(text)
    print("patched jobs.py: added HARBOR_AGENT_TIMEOUT_SEC override")

# (4) Host-network compose overlay. Bridge/compose networking is IMPOSSIBLE
# in the beaker job container (locked ro /proc/sys, masked /proc, docker
# default caps only — see the containers.conf note in this script). Append a
# generated overlay as the LAST compose file that forces network_mode: host
# on EVERY service (main + task-authored) and clears all networks (compose
# `!reset`), with extra_hosts aliases (service-name -> 127.0.0.1) standing in
# for compose DNS — TB3 multi-service tasks address each other as
# http://<service>:<port> on distinct ports. HARBOR_HOST_NETWORK_OVERLAY=0
# disables (e.g. for environments with working bridges).
if os.environ.get("HARBOR_HOST_NETWORK_OVERLAY", "1") == "1":
    docker_py = hdir / "environments/docker/docker.py"
    text = docker_py.read_text()
    if "_hostnet_overlay_path" not in text:
        method = '''
    def _hostnet_overlay_path(self, paths):
        """Overlay forcing every service onto the host netns (see run_eval_in_job.sh)."""
        import json as _json
        import shlex as _shlex
        import tempfile

        services = {"main"}
        healthchecks = {}
        for p in paths:
            try:
                doc = yaml.safe_load(Path(p).read_text())
            except Exception:
                continue
            if isinstance(doc, dict) and isinstance(doc.get("services"), dict):
                for name, svc in doc["services"].items():
                    services.add(name)
                    test = ((svc or {}).get("healthcheck") or {}).get("test")
                    # podman's docker-compat API WORD-SPLITS exec-form
                    # healthcheck argv (["CMD","python3","-c","a b"] is stored
                    # as [...,"a","b"]), so any check whose argument contains
                    # spaces always fails. CMD-SHELL's single string survives
                    # the join/split round-trip (single-space scripts), so
                    # rewrite exec-form checks to an equivalent CMD-SHELL.
                    if (
                        isinstance(test, list)
                        and test
                        and test[0] == "CMD"
                        and any(" " in str(a) for a in test[1:])
                    ):
                        healthchecks[name] = _shlex.join(str(a) for a in test[1:])
        aliases = "".join(
            f'      - "{s}:127.0.0.1"\\n' for s in sorted(services)
        )
        blocks = ""
        for s in sorted(services):
            blocks += (
                f"  {s}:\\n"
                f"    network_mode: host\\n"
                f"    networks: !reset null\\n"
                f"    extra_hosts:\\n{aliases}"
            )
            if s in healthchecks:
                blocks += (
                    "    healthcheck:\\n"
                    f"      test: [\\"CMD-SHELL\\", {_json.dumps(healthchecks[s])}]\\n"
                )
        content = "networks: !reset {}\\nservices:\\n" + blocks
        f = tempfile.NamedTemporaryFile(
            "w", suffix="-hostnet-overlay.yaml", delete=False
        )
        f.write(content)
        f.close()
        return Path(f.name)

'''
        anchor = "    def _egress_controlled_service_names(self"
        assert anchor in text, "docker.py anchor changed; fix the hostnet overlay patch"
        text = text.replace(anchor, method + anchor, 1)
        ret_anchor = (
            "        if self._enable_egress_control:\n"
            "            paths.append(self._DOCKER_COMPOSE_EGRESS_CONTROL_PATH)\n"
            "            if self._egress_control_services_compose_path:\n"
            "                paths.append(self._egress_control_services_compose_path)\n"
            "\n"
            "        return paths"
        )
        assert ret_anchor in text, "docker.py return anchor changed; fix the hostnet overlay patch"
        text = text.replace(
            ret_anchor,
            ret_anchor[: -len("        return paths")]
            + "        paths.append(self._hostnet_overlay_path(paths))\n"
            + "        return paths",
            1,
        )
        docker_py.write_text(text)
        print("patched docker.py: host-network compose overlay on every service")
    else:
        print("docker.py: hostnet overlay already patched")
else:
    print("docker.py: hostnet overlay disabled (HARBOR_HOST_NETWORK_OVERLAY=0)")

# (5) Docker platform detection fallback. podman's docker-compat shim has no
# {{.Server.Arch}} template field, so harbor's default_docker_platform()
# raises "Failed to detect Docker platform" — which errors every trial whose
# task ships its own [verifier.environment] (batched-eval-parity,
# lake-temp-glm, ...). Fall back to deriving the platform locally.
utils_py = hdir / "environments/docker/utils.py"
text = utils_py.read_text()
if "podman-compat platform fallback" not in text:
    old = (
        "    stdout, stderr = await process.communicate()\n"
        "    if process.returncode != 0:\n"
        "        raise RuntimeError(\n"
        "            f\"Failed to detect Docker platform: {stderr.decode(errors='replace')}\"\n"
        "        )\n"
    )
    assert old in text, "utils.py anchor changed; fix the platform fallback patch"
    new = (
        "    stdout, stderr = await process.communicate()\n"
        "    if process.returncode != 0:\n"
        "        # podman-compat platform fallback: podman's docker shim has no\n"
        "        # {{.Server.Arch}} template field; derive the platform locally.\n"
        "        import platform as _plat\n"
        "        _arch = {\"x86_64\": \"amd64\", \"aarch64\": \"arm64\"}.get(\n"
        "            _plat.machine(), _plat.machine()\n"
        "        )\n"
        "        return f\"{_plat.system().lower()}/{_arch}\"\n"
    )
    text = text.replace(old, new, 1)
    utils_py.write_text(text)
    print("patched utils.py: podman-compat platform fallback")
PY

# --- 4. Bring podman service up (uses scripts/setup_podman_harbor.sh) -------
# ROOTFUL, host-netns podman — the only container networking that works in a
# beaker job container (see the containers.conf note above; rootless podman is
# blocked by no-new-privileges killing the setuid newuidmap helper).
log "starting podman service"
# shellcheck disable=SC1091
source scripts/setup_podman_harbor.sh
export DOCKER_HOST="${DOCKER_HOST:-unix:///tmp/podman.sock}"
pdm() { podman "$@"; }

# --- 4b. Podman healthcheck driver -------------------------------------------
# Podman schedules container healthchecks via systemd transient timers; there
# is no systemd inside the beaker job container, so containers with a
# healthcheck stay "starting" FOREVER and `docker compose up --wait` /
# `depends_on: condition: service_healthy` hang until harbor's environment
# build timeout kills the trial (all 12 TB3 multi-service tasks do this).
# Drive the checks ourselves: run `podman healthcheck run` on every running
# container whose health status isn't "healthy" yet, forever.
log "starting podman healthcheck driver (no systemd => timers never fire)"
(
    while true; do
        for cid in $(pdm ps -q 2>/dev/null); do
            line="$(pdm inspect "$cid" --format \
                '{{if .Config.Healthcheck}}{{.State.Healthcheck.Status}}|{{.State.StartedAt}}|{{.Config.Healthcheck.StartPeriod}}{{end}}' \
                2>/dev/null || true)"
            [ -n "$line" ] || continue
            status="${line%%|*}"; rest="${line#*|}"
            started_at="${rest%%|*}"; sp_raw="${rest#*|}"
            case "$status" in starting|unhealthy) ;; *) continue ;; esac
            # RESPECT start_period: a manually-driven probe that fails while
            # the service is still booting increments podman's failing streak
            # and flips the container to "unhealthy" immediately — compose
            # `up --wait` then bails on whichever service was slowest to bind
            # (observed on heat-pump-warranty/freight-dispatch even when
            # trials ran serially). Only drive checks once start_period has
            # elapsed since the container started. StartPeriod renders as a
            # Go duration ("30s", "1m30s") or nanoseconds depending on
            # version; over-waiting is safe (containers just stay "starting").
            case "$sp_raw" in
                ''|0|0s)      sp_s=0 ;;
                *[mh]*)       sp_s=180 ;;
                *s)           sp_s="${sp_raw%s}"; sp_s="${sp_s%%.*}" ;;
                *[!0-9]*)     sp_s=60 ;;
                *)            sp_s=$(( sp_raw / 1000000000 )) ;;
            esac
            started_s="$(date -d "$started_at" +%s 2>/dev/null || echo 0)"
            if [ "$started_s" -gt 0 ] && [ $(( $(date +%s) - started_s )) -lt "${sp_s:-0}" ]; then
                continue
            fi
            pdm healthcheck run "$cid" >/dev/null 2>&1 || true
        done
        sleep 5
    done
) &
HEALTHCHECK_DRIVER_PID=$!

# --- 4a. Docker Hub auth + mirror -------------------------------------------
# tb2 task images live on Docker Hub; on a 267-trial run, harbor's
# `compose down --rmi all` deletes each image after a trial, so the next
# trial re-pulls and we blow past Docker Hub's 100 pulls/6hr unauthenticated
# cap somewhere around trial 100 (whole 2nd half of the run fails with
# "toomanyrequests: You have reached your unauthenticated pull rate limit").
# Defense in depth:
#   - if the image ships an internal Docker Hub mirror, use it
#   - if DOCKER_PAT is set (beaker secret), write the auth config (200/6hr
#     authenticated cap, or unlimited on a Docker Hub paid account)
#   - image retention is controlled in step 3 by HARBOR_KEEP_TASK_IMAGES
#     (default 0 = keep harbor's stock --rmi all; set 1 to persist images,
#     only sane for small image sets like tb2's 89)
if [ -x /usr/local/bin/setup_dockerio_mirror ]; then
    /usr/local/bin/setup_dockerio_mirror || log "setup_dockerio_mirror failed (continuing)"
fi
if [ -n "${DOCKER_PAT:-}" ]; then
    # Authenticate to Docker Hub so task-image pulls don't hit the
    # unauthenticated rate cap. We `docker login` to VERIFY the credentials and
    # HARD-ABORT on failure — no anonymous fallback — so a wrong username/PAT
    # fails fast and unambiguously here, rather than silently rate-limiting or
    # erroring on every image pull mid-run. DOCKERHUB_USERNAME must be the
    # Docker Hub account that owns the DOCKER_PAT secret.
    DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-shashankg209}"
    log "docker login as '$DOCKERHUB_USERNAME'"
    if printf '%s' "$DOCKER_PAT" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin docker.io >/dev/null 2>&1; then
        log "Docker Hub login OK ($DOCKERHUB_USERNAME)"
    else
        log "FATAL: Docker Hub login failed for '$DOCKERHUB_USERNAME'. Check DOCKERHUB_USERNAME and the DOCKER_PAT secret. Aborting."
        exit 1
    fi
    # Harbor pulls task images via the podman socket (DOCKER_HOST=.../podman.sock);
    # podman reads registry creds from containers/auth.json, NOT ~/.docker/config.json,
    # so a plain `docker login` leaves the podman service pulling ANONYMOUSLY (which
    # then hits the shared-IP unauthenticated rate cap under --host-networking, even
    # with a paid account). Authenticate the ROOTLESS podman user's own store too.
    if printf '%s' "$DOCKER_PAT" | pdm login -u "$DOCKERHUB_USERNAME" --password-stdin docker.io >/dev/null 2>&1; then
        log "podman Docker Hub login OK ($DOCKERHUB_USERNAME, rootless)"
    else
        log "FATAL: podman Docker Hub login failed for '$DOCKERHUB_USERNAME'. Aborting."
        exit 1
    fi
else
    log "FATAL: DOCKER_PAT not set; refusing to fall back to anonymous pulls. Provide the DOCKER_PAT secret. Aborting."
    exit 1
fi

# --- 4c. Container preflight --------------------------------------------------
# Every service (main + task-authored) runs with network_mode: host via the
# harbor overlay patch above — bridge networking is impossible in this job
# container (locked ro /proc/sys, docker default caps; smokes 3-9). Probe that
# a host-netns container runs AND has internet egress before anything heavy
# starts, failing fast with diagnostics instead of erroring all trials.
log "host-network container preflight"
if pdm run --rm --network host docker.io/library/busybox:latest \
        sh -c 'wget -q -T 15 -O /dev/null http://archive.ubuntu.com/ubuntu/'; then
    log "host-network preflight OK (container ran, egress works)"
else
    log "FATAL: cannot run a host-network container with egress — every trial would error."
    log "podman info follows for diagnosis:"
    pdm info 2>&1 | sed -n '1,120p' || true
    pdm run --rm --network host docker.io/library/busybox:latest sh -c 'wget -q -T 15 -O /dev/null http://archive.ubuntu.com/ubuntu/' || true
    exit 1
fi

# --- 5. Start vLLM in the background ----------------------------------------
: "${VLLM_VERSION:=0.19.1}"
: "${VLLM_TOOL_CALL_PARSER:=hermes}"
: "${VLLM_REASONING_PARSER:=}"
: "${VLLM_PORT:=8008}"
: "${DP_SIZE:=1}"
# Under gantry --host-networking, co-located jobs share the host netns, so any
# FIXED port collides across jobs. Two consequences, both fixed by randomizing:
#   1. vLLM derives its INTERNAL TP-rendezvous ports from the VLLM_PORT env var
#      (VLLM_PORT+1, etc.) — a fixed value makes co-located TP>1 jobs collide and
#      die at startup ("DistNetworkError ... EADDRINUSE"). UNSET it so vLLM picks
#      random free internal ports.
#   2. The OpenAI API server port: a fixed 8008 makes a job's harbor reach a
#      *neighbor's* vLLM (a different served model), so every trial fails with
#      "model does not exist". Bind the API server to a per-job free high port.
unset VLLM_PORT
API_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); p=s.getsockname()[1]; s.close(); print(p)')"
VLLM_LOG=/tmp/vllm.log
VLLM_LOG_TAIL_LINES="${VLLM_LOG_TAIL_LINES:-300}"
# Pin fastapi < 0.137: fastapi 0.137 changed the router internals and breaks
# prometheus-fastapi-instrumentator (which vLLM mounts on every route), so the
# API server 500s on every request including /v1/models — the readiness probe
# below then never passes and the job is killed after 30 min.
# See vllm-project/vllm#45596 and #45597.
VLLM_CMD=( uvx --with "fastapi<0.137"
           "vllm==${VLLM_VERSION}" serve "$MODEL_PATH"
           --revision "$MODEL_REVISION"
           --tokenizer-revision "$MODEL_REVISION"
           --served-model-name "$SERVED_MODEL_NAME"
           --enable-auto-tool-choice
           --enable-prefix-caching
           --tool-call-parser "$VLLM_TOOL_CALL_PARSER"
           --port "$API_PORT"
           --gpu-memory-utilization 0.85
           --tensor-parallel-size "$TP_SIZE"
           --data-parallel-size "$DP_SIZE" )
if [ -n "${MAX_MODEL_LEN:-}" ]; then
    VLLM_CMD+=( --max-model-len "$MAX_MODEL_LEN" )
fi
# Reasoning models (e.g. Qwen3) emit <think>...</think>; --reasoning-parser
# splits that into reasoning_content so tool-calls/content parse cleanly. Leave
# empty for non-reasoning models (e.g. Qwen3.5).
if [ -n "${VLLM_REASONING_PARSER:-}" ]; then
    VLLM_CMD+=( --reasoning-parser "$VLLM_REASONING_PARSER" )
fi
if [ "${VLLM_LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    VLLM_CMD+=( --language_model_only )
fi

log "launching vllm: ${VLLM_CMD[*]}"
"${VLLM_CMD[@]}" >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    log "cleanup: killing vllm pid $VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    if [ -n "${HEALTHCHECK_DRIVER_PID:-}" ]; then
        kill "$HEALTHCHECK_DRIVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# A 200 on /v1/models can precede the engine actually being able to GENERATE
# (CUDA-graph capture etc.) — the first completion then fails with "model does
# not exist", fatal for small runs and lost trials for large ones (notably the
# 9B). So gate readiness on a real /v1/chat/completions succeeding.
vllm_can_generate() {
    curl -sf -X POST "http://localhost:$API_PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
        >/dev/null 2>&1
}
# Readiness cap: 5s * VLLM_READY_MAX_ITERS. Default 720 = 60 min (large models
# like the 27B on TP>1 need >30 min for weight-load + torch.compile before the
# API server binds). Override with VLLM_READY_MAX_ITERS.
VLLM_READY_MAX_ITERS="${VLLM_READY_MAX_ITERS:-720}"
log "waiting for vllm to serve completions on :$API_PORT (up to $((VLLM_READY_MAX_ITERS*5/60)) min)"
VLLM_READY=0
for _ in $(seq 1 "$VLLM_READY_MAX_ITERS"); do
    if vllm_can_generate; then
        log "vllm ready (completion probe ok)"
        VLLM_READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "vllm process died — tail of $VLLM_LOG:"
        tail -"$VLLM_LOG_TAIL_LINES" "$VLLM_LOG" || true
        exit 1
    fi
    sleep 5
done

if [ "$VLLM_READY" -ne 1 ]; then
    log "vllm did not become ready in 30 min — tail of $VLLM_LOG:"
    tail -"$VLLM_LOG_TAIL_LINES" "$VLLM_LOG" || true
    exit 1
fi

# --- 6. Run harbor ----------------------------------------------------------
: "${N_CONCURRENT:=8}"
: "${N_ATTEMPTS:=1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_API_BASE="http://localhost:$API_PORT/v1"
# Harbor's SWE-agent adapter copies OPENAI_BASE_URL (litellm convention) —
# not OPENAI_API_BASE — into the container, and only then does it pass
# --agent.model.api_base=... to sweagent. Without this, litellm in the
# container falls back to https://api.openai.com and every trial exits
# with NotFoundError: Hosted_vllmException on step 1.
export OPENAI_BASE_URL="http://localhost:$API_PORT/v1"

if [ -z "${HOSTED_VLLM_MODEL_INFO:-}" ]; then
    MODEL_INFO_MAX_INPUT_TOKENS="${MAX_MODEL_LEN:-40960}"
    HOSTED_VLLM_MODEL_INFO="$(python3 - <<PY
import json

print(json.dumps({
    "max_input_tokens": int("$MODEL_INFO_MAX_INPUT_TOKENS"),
    "max_output_tokens": 8192,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}, separators=(",", ":")))
PY
)"
fi
# AGENT_IMPORT_PATH with a ":" is a module:Class import path; otherwise it's a
# harbor built-in agent name (e.g. mini-swe-agent, swe-agent, terminus).
#
# Litellm provider prefix (MODEL_PROVIDER, overridable via launch_eval.sh
# --model-provider). Defaults:
#   * import-path SWE agents (VanilluxAgent): hosted_vllm/ + an api_base kwarg.
#   * everything else (built-in agents, and the custom litellm BaseAgent
#     Vanillux2Agent): openai/ — the installed harbor's litellm has no usable
#     "hosted_vllm" path, and openai/<served-name> + OPENAI_BASE_URL works.
#   NOTE Vanillux2Agent: launch with --agent Vanillux2Agent:Vanillux2Agent
#        --model-provider openai --tool-call-parser qwen3_xml (Qwen3.5 emits
#        <function=..><parameter=..> XML that the hermes parser drops, which
#        otherwise loops the agent on format errors).
# Do NOT set MSWEA_API_KEY for built-in agents: harbor's mini-swe-agent forwards
# only that when present and skips OPENAI_API_KEY, which litellm then reports as
# "Missing credentials".
if [[ "$AGENT_IMPORT_PATH" == *:* ]]; then
    MODEL_PROVIDER="${MODEL_PROVIDER:-hosted_vllm}"
else
    MODEL_PROVIDER="${MODEL_PROVIDER:-openai}"
    unset MSWEA_API_KEY
fi
# An explicit HARBOR_MODEL_NAME wins; otherwise derive it from the provider.
: "${HARBOR_MODEL_NAME:=$MODEL_PROVIDER/$SERVED_MODEL_NAME}"

HARBOR_CMD=( uv run harbor run
             --model "$HARBOR_MODEL_NAME"
             --env "${HARBOR_ENV:-docker}"
             --n-concurrent "$N_CONCURRENT"
             --job-name "$JOB_NAME"
             -k "$N_ATTEMPTS" )
# DATASET_PATH (a local dir on a mounted weka fs, harbor --path) overrides the
# registry --dataset ref. Used for datasets not in harbor 0.6.6's registry
# (e.g. terminal-bench-2-1, downloaded via a newer harbor). The dir must be
# under a weka mount the job has (launch_eval mounts oe-adapt-default).
if [ -n "${DATASET_PATH:-}" ]; then
    HARBOR_CMD+=( --path "$DATASET_PATH" )
else
    HARBOR_CMD+=( --dataset "$DATASET" )
fi
if [ -n "${N_TASKS:-}" ]; then
    HARBOR_CMD+=( --n-tasks "$N_TASKS" )
fi
# Newline-separated globs from launch_eval --include-task-name.
if [ -n "${INCLUDE_TASK_NAMES:-}" ]; then
    while IFS= read -r task_glob; do
        [ -n "$task_glob" ] || continue
        HARBOR_CMD+=( --include-task-name "$task_glob" )
    done <<< "$INCLUDE_TASK_NAMES"
fi
# Retry exception-errored trials (NOT reward-0 ones) — absorbs transient
# flakes: anonymous github API 403s during builds, mirror hiccups, and
# host-netns port collisions (the colliding neighbour is usually gone by the
# retry).
if [ -n "${HARBOR_MAX_RETRIES:-}" ]; then
    HARBOR_CMD+=( --max-retries "$HARBOR_MAX_RETRIES" )
fi
# Newline-separated globs from launch_eval --exclude-task-name. Needed for
# TB3's 4 GPU tasks: harbor 0.21 hard-errors ("Task requires N GPU(s) but
# EnvironmentType.DOCKER does not support GPU allocation") and ABORTS THE
# WHOLE JOB when a GPU task is scheduled on the docker env.
if [ -n "${EXCLUDE_TASK_NAMES:-}" ]; then
    while IFS= read -r task_glob; do
        [ -n "$task_glob" ] || continue
        HARBOR_CMD+=( --exclude-task-name "$task_glob" )
    done <<< "$EXCLUDE_TASK_NAMES"
fi
if [ -n "${HARBOR_OVERRIDE_CPUS:-}" ]; then
    HARBOR_CMD+=( --override-cpus "$HARBOR_OVERRIDE_CPUS" )
fi
if [ -n "${HARBOR_OVERRIDE_MEMORY_MB:-}" ]; then
    HARBOR_CMD+=( --override-memory-mb "$HARBOR_OVERRIDE_MEMORY_MB" )
fi
if [ -n "${HARBOR_OVERRIDE_STORAGE_MB:-}" ]; then
    HARBOR_CMD+=( --override-storage-mb "$HARBOR_OVERRIDE_STORAGE_MB" )
fi
if [ -n "${HARBOR_OVERRIDE_GPUS:-}" ]; then
    HARBOR_CMD+=( --override-gpus "$HARBOR_OVERRIDE_GPUS" )
fi
if [ -n "${HARBOR_TIMEOUT_MULTIPLIER:-}" ]; then
    HARBOR_CMD+=( --timeout-multiplier "$HARBOR_TIMEOUT_MULTIPLIER" )
fi
if [ -n "${HARBOR_AGENT_TIMEOUT_MULTIPLIER:-}" ]; then
    HARBOR_CMD+=( --agent-timeout-multiplier "$HARBOR_AGENT_TIMEOUT_MULTIPLIER" )
fi
if [ -n "${HARBOR_VERIFIER_TIMEOUT_MULTIPLIER:-}" ]; then
    HARBOR_CMD+=( --verifier-timeout-multiplier "$HARBOR_VERIFIER_TIMEOUT_MULTIPLIER" )
fi
if [ -n "${HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER:-}" ]; then
    HARBOR_CMD+=( --agent-setup-timeout-multiplier "$HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER" )
fi
if [ -n "${HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER:-}" ]; then
    HARBOR_CMD+=( --environment-build-timeout-multiplier "$HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER" )
fi
if [[ "$HARBOR_MODEL_NAME" == hosted_vllm/* ]]; then
    HARBOR_CMD+=( --agent-kwarg "model_info=$HOSTED_VLLM_MODEL_INFO" )
fi
if [ -n "${EXTRA_AGENT_KWARGS:-}" ]; then
    while IFS= read -r agent_kwarg; do
        [ -n "$agent_kwarg" ] || continue
        HARBOR_CMD+=( --agent-kwarg "$agent_kwarg" )
    done <<< "$EXTRA_AGENT_KWARGS"
fi
if [ -n "${EXTRA_AGENT_ENVS:-}" ]; then
    while IFS= read -r agent_env; do
        [ -n "$agent_env" ] || continue
        if [[ "$agent_env" == *=* ]]; then
            export "$agent_env"
        fi
        HARBOR_CMD+=( --agent-env "$agent_env" )
    done <<< "$EXTRA_AGENT_ENVS"
fi
if [[ "$AGENT_IMPORT_PATH" == *:* ]]; then
    HARBOR_CMD+=( --agent-import-path "$AGENT_IMPORT_PATH"
                  --agent-kwarg "api_base=http://localhost:$API_PORT/v1" )
else
    HARBOR_CMD+=( --agent "$AGENT_IMPORT_PATH" )
fi
log "running harbor: ${HARBOR_CMD[*]}"

# Background progress reporter — harbor's built-in progress bar uses ANSI
# escapes that gantry logs flatten into noise, so we tail result.json
# ourselves and emit one human-readable line per interval.
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
RESULT_JSON="jobs/$JOB_NAME/result.json"
(
    while true; do
        sleep "$PROGRESS_INTERVAL"
        [ -f "$RESULT_JSON" ] || continue
        python3 - "$RESULT_JSON" <<'PY' || true
import json, sys, datetime
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
total = d.get("n_total_trials", "?")
stats = d.get("stats", {}) or {}
done = stats.get("n_trials", stats.get("n_completed_trials", 0))
errs = stats.get("n_errors", stats.get("n_errored_trials", 0))
evals = stats.get("evals", {}) or {}
mean = 0.0
for v in evals.values():
    done = max(done, v.get("n_trials", 0))
    errs = max(errs, v.get("n_errors", 0))
    m = (v.get("metrics") or [{}])[0].get("mean")
    if m is not None:
        mean = m
        break
ts = datetime.datetime.utcnow().strftime("%H:%M:%S")
print(f"=== [{ts}] progress: {done}/{total} trials  errors={errs}  mean={mean:.3f} ===",
      flush=True)
PY
    done
) &
PROGRESS_PID=$!

set +e
"${HARBOR_CMD[@]}"
HARBOR_RC=$?
set -e

kill "$PROGRESS_PID" 2>/dev/null || true
wait "$PROGRESS_PID" 2>/dev/null || true

# --- 7. Compute aggregate stats ---------------------------------------------
JOB_DIR="jobs/$JOB_NAME"
if [ -f "$VLLM_LOG" ]; then
    log "preserving vllm log in $JOB_DIR"
    mkdir -p "$JOB_DIR"
    cp "$VLLM_LOG" "$JOB_DIR/vllm.log" || true
    tail -"$VLLM_LOG_TAIL_LINES" "$VLLM_LOG" >"$JOB_DIR/vllm.tail.txt" 2>/dev/null || true
fi
if [ -d "$JOB_DIR" ]; then
    STATS_LOG="$JOB_DIR/stats.txt"
    METRICS_JSON="$JOB_DIR/metrics.json"
    log "computing stats: scripts/compute_stats.py $JOB_DIR"
    set +e
    uv run python scripts/compute_stats.py "$JOB_DIR" --json-output "$METRICS_JSON" 2>&1 | tee "$STATS_LOG"
    STATS_RC=${PIPESTATUS[0]}
    set -e
    if [ "$STATS_RC" -ne 0 ]; then
        log "compute_stats.py exited with $STATS_RC; preserving harbor exit code $HARBOR_RC"
    else
        log "stats written to $STATS_LOG"
        log "metrics written to $METRICS_JSON"
    fi
else
    log "job directory $JOB_DIR not found; skipping compute_stats.py"
fi

# --- 8. Persist results ------------------------------------------------------
# Always drop metrics.json into /results so Beaker surfaces it in the UI,
# even when RESULTS_DIR redirects the full copy elsewhere (e.g. weka).
if [ -f "$JOB_DIR/metrics.json" ] && [ -d /results ]; then
    cp "$JOB_DIR/metrics.json" /results/metrics.json || true
    log "metrics also available at /results/metrics.json (Beaker UI)"
fi
if [ -n "${RESULTS_DIR:-}" ]; then
    if [ -d "$JOB_DIR" ]; then
        log "copying $JOB_DIR -> $RESULTS_DIR/"
        mkdir -p "$RESULTS_DIR"
        cp -r "$JOB_DIR" "$RESULTS_DIR/"
        log "results available at $RESULTS_DIR/$JOB_NAME"
        if [ -f "$JOB_DIR/metrics.json" ]; then
            cp "$JOB_DIR/metrics.json" "$RESULTS_DIR/metrics.json"
            log "metrics available at $RESULTS_DIR/metrics.json"
        fi
    else
        log "job directory $JOB_DIR not found; skipping result copy"
    fi
fi

exit "$HARBOR_RC"
