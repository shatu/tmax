"""harbor environment backed by a self-hosted OpenSandbox service.

Each harbor trial gets its own sandbox pod on the OpenSandbox cluster; the
harbor process (and therefore host-side agents such as Vanillux2Agent, plus the
verifier driver) stays wherever ``harbor run`` executes and only shell
commands / file transfers cross the network. Nothing container-related runs on
the eval node, so the podman-in-Beaker patch stack is not needed.

Select it with::

    harbor run --environment-import-path tmax_envs.opensandbox:OpenSandboxEnvironment \\
               --environment-kwarg task_image_repo=docker.io/<user>/tmax-harbor-tasks ...

Configuration (constructor kwargs via ``--environment-kwarg``; env var fallback):

    domain              TMAX_OPENSANDBOX_DOMAIN     service host (default: the
                        AI2 sandbox-standard endpoint)
    protocol            TMAX_OPENSANDBOX_PROTOCOL   http|https (default https)
    api_key             OPEN_SANDBOX_API_KEY        service API key (required)
    task_image_repo     TMAX_TASK_IMAGE_REPO        registry repo holding prebuilt
                        images for tasks WITHOUT [environment].docker_image
                        (built by scripts/opensandbox/build_task_images.py)
    image_prefix        TMAX_OPENSANDBOX_IMAGE_PREFIX
                        pull-through mirror prefix for bare Docker Hub refs
    sandbox_lifetime_sec TMAX_OPENSANDBOX_LIFETIME_S hard pod lifetime (default 7200)
    ready_timeout_sec   TMAX_OPENSANDBOX_READY_TIMEOUT_S (default 600)
    start_concurrency   TMAX_OPENSANDBOX_START_CONCURRENCY concurrent creates
                        per harbor process (default 16)
    app_name            TMAX_OPENSANDBOX_APP_NAME   metadata tag for the janitor
    registry_username / registry_password
                        DOCKERHUB_USERNAME / DOCKER_PAT  registry auth for
                        direct (non-mirrored) pulls
    max_cpus            TMAX_OPENSANDBOX_MAX_CPUS   cap on the pod CPU request
                        (default 2). sandbox-standard's node pool is 4-vCPU
                        machines, so a task.toml asking for cpus = 4 (TB 2.1:
                        caffe-cifar-10, mcmc-sampling-stan, rstan-to-pystan)
                        never schedules and the control plane fails the create
                        after 180 s. CPU requests are not enforced as limits
                        there anyway, so capping costs nothing.
    use_server_proxy    TMAX_OPENSANDBOX_USE_SERVER_PROXY  route exec/file
                        traffic through the control plane instead of the
                        ingress gateway (default true: the sandbox-standard
                        deployment's gateway hostname resolves to the training
                        cluster, so direct execd access never becomes healthy)

Image resolution: ``[environment].docker_image`` when the task declares one
(Terminal-Bench 2.x); otherwise ``<task_image_repo>:<task>-<envdir hash>``
(see :mod:`tmax_envs.task_images`). OpenSandbox never builds Dockerfiles.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import io
import logging
import os
import shlex
import tarfile
import time
import uuid
from datetime import timedelta
from pathlib import Path, PurePosixPath

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

from tmax_envs.task_images import DockerfileFacts, parse_dockerfile, task_image_ref

try:
    from opensandbox import Sandbox, SandboxManager
    from opensandbox.config import ConnectionConfig
    from opensandbox.exceptions import SandboxException
    from opensandbox.models.execd import RunCommandOpts
    from opensandbox.models.filesystem import WriteEntry
    from opensandbox.models.sandboxes import (
        NetworkPolicy,
        SandboxFilter,
        SandboxImageAuth,
        SandboxImageSpec,
        SandboxState,
    )

    _HAS_OPENSANDBOX = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _HAS_OPENSANDBOX = False

DEFAULT_DOMAIN = "sandbox-standard.oe-rl-sandbox.apps.allenai.org"
DEFAULT_PROTOCOL = "https"
DEFAULT_APP_NAME = "tmax-harbor-eval"
ENVIRONMENT_TYPE = "opensandbox"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _as_int(value, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_bool(value, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def rewrite_image_for_mirror(image: str, prefix: str) -> str:
    """Route a bare Docker Hub reference through a pull-through mirror.

    References already qualified with a registry host (first path component
    contains ``.`` or ``:``, or is ``localhost``) pass through unchanged.
    Official images (no namespace) gain Docker Hub's implicit ``library/``.
    """
    if not prefix:
        return image
    first = image.split("/", 1)[0]
    if "/" in image and ("." in first or ":" in first or first == "localhost"):
        return image
    if "/" not in image:
        return f"{prefix}/library/{image}"
    return f"{prefix}/{image}"


def _tag_value(value: str, limit: int = 63) -> str:
    """Metadata values travel as pod labels on Kubernetes; keep them label-safe."""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value)
    return safe.strip("-_.")[:limit] or "x"


# Sandboxes this process created and has not yet killed; the atexit reaper
# kills whatever is left so an interrupted `harbor run` leaks as little as
# possible (a SIGKILLed process still leaks — see cleanup_sandboxes.py).
_LIVE_SANDBOX_IDS: dict[str, "ConnectionConfig"] = {}


def _reap_live_sandboxes() -> None:  # pragma: no cover - process teardown
    if not _LIVE_SANDBOX_IDS or not _HAS_OPENSANDBOX:
        return
    from opensandbox.config import ConnectionConfigSync
    from opensandbox.sync import SandboxManagerSync

    by_config: dict[tuple, list[str]] = {}
    for sandbox_id, config in list(_LIVE_SANDBOX_IDS.items()):
        key = (config.domain, config.protocol, config.api_key)
        by_config.setdefault(key, []).append(sandbox_id)
    for (domain, protocol, api_key), ids in by_config.items():
        try:
            manager = SandboxManagerSync.create(
                ConnectionConfigSync(domain=domain, protocol=protocol, api_key=api_key)
            )
        except Exception:
            continue
        for sandbox_id in ids:
            with contextlib.suppress(Exception):
                manager.kill_sandbox(sandbox_id)
        with contextlib.suppress(Exception):
            manager.close()


atexit.register(_reap_live_sandboxes)

# The SDK logs a full traceback at ERROR when a file read 404s. Exec output
# files can legitimately be gone (a command that wipes /tmp); _collect_output
# handles that and logs it at DEBUG, so silence the adapter's duplicate.
logging.getLogger("opensandbox.adapters.filesystem_adapter").setLevel(logging.CRITICAL)


class OpenSandboxEnvironment(BaseEnvironment):
    # Extra time the server-side command kill and the SSE read get beyond the
    # in-sandbox `timeout` wrapper, so exit 124 (our timeout) always fires first.
    _EXEC_TIMEOUT_MARGIN_S = 30
    # Per-request HTTP timeout for control-plane calls (create/kill/list).
    _CONTROL_REQUEST_TIMEOUT_S = 300
    # The exec HTTP stream must outlive the longest command harbor will run
    # (verifier timeouts are ~15 min on Terminal-Bench).
    _EXEC_REQUEST_TIMEOUT_S = 4 * 3600
    _ADOPT_POLL_INTERVAL_S = 5.0
    _MAX_OUTPUT_CHARS = 1_000_000
    # Where exec stdout/stderr are spooled inside the sandbox (see _exec_raw).
    # A dot-directory under /tmp: /tmp is world-writable in every image we run
    # (so non-root exec users can create it), and an agent's `rm -rf /tmp/*`
    # skips dotfiles. It is (re)created by every exec, so it also survives an
    # agent that deletes it outright.
    _EXEC_OUTPUT_DIR = "/tmp/.tmax-exec"

    _start_semaphore: asyncio.Semaphore | None = None
    _start_semaphore_size: int | None = None

    @classmethod
    def preflight(cls) -> None:
        if not _HAS_OPENSANDBOX:
            raise SystemExit(
                "The OpenSandbox environment requires the 'opensandbox' package "
                "(uv sync should install it)."
            )
        if not os.environ.get("OPEN_SANDBOX_API_KEY"):
            raise SystemExit(
                "OpenSandbox requires OPEN_SANDBOX_API_KEY to be set (Beaker secret "
                "pradeepd_OPEN_SANDBOX_API_KEY). Export it and try again."
            )

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args,
        domain: str | None = None,
        protocol: str | None = None,
        api_key: str | None = None,
        task_image_repo: str | None = None,
        image_prefix: str | None = None,
        sandbox_lifetime_sec=None,
        ready_timeout_sec=None,
        start_concurrency=None,
        app_name: str | None = None,
        registry_username: str | None = None,
        registry_password: str | None = None,
        enforce_network_policy=None,
        use_server_proxy=None,
        max_cpus=None,
        **kwargs,
    ):
        if not _HAS_OPENSANDBOX:
            raise RuntimeError(
                "The 'opensandbox' package is required for OpenSandboxEnvironment."
            )
        # `capabilities` is read during BaseEnvironment.__init__ validators, so
        # nothing below may be needed by it; it is static for this backend.
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self._domain = (domain or os.getenv("TMAX_OPENSANDBOX_DOMAIN") or DEFAULT_DOMAIN).strip()
        self._protocol = (protocol or os.getenv("TMAX_OPENSANDBOX_PROTOCOL") or DEFAULT_PROTOCOL).strip()
        if "://" in self._domain:
            self._protocol, self._domain = self._domain.split("://", 1)
        self._api_key = api_key or os.getenv("OPEN_SANDBOX_API_KEY")
        self._task_image_repo = (task_image_repo or os.getenv("TMAX_TASK_IMAGE_REPO") or "").strip()
        self._image_prefix = (image_prefix or os.getenv("TMAX_OPENSANDBOX_IMAGE_PREFIX") or "").rstrip("/")
        self._sandbox_lifetime = _as_int(sandbox_lifetime_sec, _env_int("TMAX_OPENSANDBOX_LIFETIME_S", 7200))
        self._ready_timeout = _as_int(ready_timeout_sec, _env_int("TMAX_OPENSANDBOX_READY_TIMEOUT_S", 600))
        self._start_concurrency = _as_int(start_concurrency, _env_int("TMAX_OPENSANDBOX_START_CONCURRENCY", 16))
        self._app_name = app_name or os.getenv("TMAX_OPENSANDBOX_APP_NAME") or DEFAULT_APP_NAME
        self._registry_username = registry_username or os.getenv("DOCKERHUB_USERNAME")
        self._registry_password = registry_password or os.getenv("DOCKER_PAT")
        self._enforce_network_policy = _as_bool(
            enforce_network_policy, os.getenv("TMAX_OPENSANDBOX_ENFORCE_NETWORK_POLICY", "1") == "1"
        )
        self._use_server_proxy = _as_bool(
            use_server_proxy, os.getenv("TMAX_OPENSANDBOX_USE_SERVER_PROXY", "1") == "1"
        )
        self._max_cpus = _as_int(max_cpus, _env_int("TMAX_OPENSANDBOX_MAX_CPUS", 2))

        self._dockerfile: DockerfileFacts = (
            parse_dockerfile(self._dockerfile_path) if self._dockerfile_path.is_file() else DockerfileFacts()
        )
        self._workdir: str | None = self.task_env_config.workdir or self._dockerfile.workdir
        # The image's USER is what `docker compose exec` runs as when harbor
        # passes no user; execd does not apply it, so we do.
        self._image_user: str | None = self._dockerfile.user

        self._sandbox: Sandbox | None = None
        self._uid_cache: dict[str, tuple[int, int, dict[str, str]]] = {}

    # ── harbor plumbing ──────────────────────────────────────────────────

    @staticmethod
    def type() -> str:
        return ENVIRONMENT_TYPE

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    def _validate_definition(self):
        if self.task_env_config.docker_image:
            return
        if not self._dockerfile_path.exists():
            raise FileNotFoundError(
                f"{self._dockerfile_path} not found and the task declares no "
                "[environment].docker_image; nothing to run."
            )

    # ── image / connection ───────────────────────────────────────────────

    def resolve_image(self) -> str:
        """The image the sandbox is created from, before mirror rewriting."""
        if self.task_env_config.docker_image:
            return self.task_env_config.docker_image
        if not self._task_image_repo:
            raise RuntimeError(
                f"Task '{self.environment_name}' has no [environment].docker_image, and "
                "OpenSandbox cannot build its Dockerfile. Build and push the task images "
                "once with scripts/opensandbox/build_task_images.py and pass "
                "--environment-kwarg task_image_repo=<registry repo> (or set "
                "TMAX_TASK_IMAGE_REPO)."
            )
        return task_image_ref(self._task_image_repo, self.environment_dir, self.environment_name)

    def _image_spec(self) -> "SandboxImageSpec | str":
        image = self.resolve_image()
        effective = rewrite_image_for_mirror(image, self._image_prefix)
        if effective == image and self._registry_username and self._registry_password:
            return SandboxImageSpec(
                image=image,
                auth=SandboxImageAuth(username=self._registry_username, password=self._registry_password),
            )
        return effective

    def _connection_config(self, request_timeout_s: int) -> "ConnectionConfig":
        return ConnectionConfig(
            domain=self._domain,
            protocol=self._protocol,
            api_key=self._api_key,
            request_timeout=timedelta(seconds=request_timeout_s),
            use_server_proxy=self._use_server_proxy,
        )

    def _resource(self) -> dict[str, str]:
        cpus = int(self.task_env_config.cpus)
        if self._max_cpus and cpus > self._max_cpus:
            self.logger.info(
                f"Capping task cpus={cpus} to {self._max_cpus} (pods requesting more never schedule "
                "on this cluster; set max_cpus=0 to disable)"
            )
            cpus = self._max_cpus
        return {
            "cpu": str(cpus),
            "memory": f"{int(self.task_env_config.memory_mb)}Mi",
        }

    def _metadata(self, create_id: str) -> dict[str, str]:
        return {
            "tmax_app": _tag_value(self._app_name),
            "tmax_task": _tag_value(self.environment_name),
            "tmax_trial": _tag_value(self.session_id),
            "tmax_create_id": create_id,
        }

    def _network_policy(self) -> "NetworkPolicy | None":
        if self.task_env_config.allow_internet or not self._enforce_network_policy:
            return None
        return NetworkPolicy(default_action="deny", egress=[])

    @classmethod
    def _semaphore(cls, size: int) -> asyncio.Semaphore:
        if cls._start_semaphore is None or cls._start_semaphore_size != size:
            cls._start_semaphore = asyncio.Semaphore(size)
            cls._start_semaphore_size = size
        return cls._start_semaphore

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self, force_build: bool) -> None:
        if self._sandbox is not None:
            await self.stop(delete=True)
        if force_build:
            self.logger.info(
                "force_build has no effect on OpenSandbox (images are pulled, not built); "
                "rebuild with scripts/opensandbox/build_task_images.py instead."
            )

        image_spec = self._image_spec()
        image_ref = image_spec if isinstance(image_spec, str) else image_spec.image
        create_id = uuid.uuid4().hex
        started = time.perf_counter()
        self.logger.info(
            f"Creating OpenSandbox sandbox for {self.session_id}: image={image_ref} "
            f"resource={self._resource()} lifetime={self._sandbox_lifetime}s domain={self._domain}"
        )

        async with self._semaphore(self._start_concurrency):
            try:
                self._sandbox = await Sandbox.create(
                    image_spec,
                    timeout=timedelta(seconds=self._sandbox_lifetime),
                    ready_timeout=timedelta(seconds=self._ready_timeout),
                    env=dict(self._persistent_env) or None,
                    metadata=self._metadata(create_id),
                    resource=self._resource(),
                    network_policy=self._network_policy(),
                    connection_config=self._connection_config(self._EXEC_REQUEST_TIMEOUT_S),
                )
            except SandboxException as e:
                if not self._is_gateway_timeout(e):
                    raise
                # The ingress cut the create response but the server usually
                # still brings the pod up; adopt it instead of leaking it.
                self.logger.warning(
                    f"OpenSandbox create hit a gateway timeout (create_id={create_id}); "
                    "polling to adopt the sandbox it may still have spawned."
                )
                self._sandbox = await self._adopt_after_gateway_timeout(create_id, e)

        _LIVE_SANDBOX_IDS[self._sandbox.id] = self._sandbox.connection_config
        self.logger.info(
            f"OpenSandbox sandbox {self._sandbox.id} ready in {time.perf_counter() - started:.1f}s"
        )
        await self._prepare_filesystem()

    async def _prepare_filesystem(self) -> None:
        """Create harbor's fixed in-container directories.

        The docker backend gets these as bind mounts; here they are plain dirs
        that upload/download go through. World-writable so a non-root agent or
        verifier user can write its logs.
        """
        paths = self.env_paths
        dirs = [paths.agent_dir, paths.verifier_dir, paths.artifacts_dir, paths.tests_dir, paths.solution_dir]
        quoted = " ".join(shlex.quote(str(p)) for p in dirs)
        log_dirs = " ".join(
            shlex.quote(str(p)) for p in (paths.agent_dir, paths.verifier_dir, paths.artifacts_dir)
        )
        result = await self._exec_raw(
            f"mkdir -p {quoted} && chmod 777 {log_dirs}", timeout_sec=60, uid=None, gid=None, cwd="/"
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to prepare harbor directories in sandbox {self._sandbox_id()}: "
                f"exit={result.return_code} {result.stderr or result.stdout}"
            )

    @staticmethod
    def _is_gateway_timeout(error: Exception) -> bool:
        if getattr(error, "status_code", None) == 504:
            return True
        return "HTTP 504" in str(error) or "504 Gateway" in str(error)

    async def _adopt_after_gateway_timeout(self, create_id: str, original: Exception) -> "Sandbox":
        deadline = time.monotonic() + self._ready_timeout
        adopted_id: str | None = None
        manager = await SandboxManager.create(self._connection_config(self._CONTROL_REQUEST_TIMEOUT_S))
        try:
            while time.monotonic() < deadline:
                page = await manager.list_sandbox_infos(
                    SandboxFilter(metadata={"tmax_create_id": create_id}, page=1, page_size=10)
                )
                infos = page.sandbox_infos
                running = [i for i in infos if i.status.state == SandboxState.RUNNING]
                if running:
                    adopted_id = running[0].id
                    for info in infos:
                        if info.id != adopted_id:
                            self.logger.warning(f"Killing duplicate sandbox {info.id} from timed-out create")
                            with contextlib.suppress(Exception):
                                await manager.kill_sandbox(info.id)
                    break
                await asyncio.sleep(self._ADOPT_POLL_INTERVAL_S)
        finally:
            await manager.close()
        if adopted_id is None:
            self.logger.error(
                f"No sandbox appeared for create_id={create_id} within {self._ready_timeout}s; "
                "if one shows up later, scripts/opensandbox/cleanup_sandboxes.py reclaims it by app tag."
            )
            raise original
        sandbox = await Sandbox.connect(
            adopted_id,
            connection_config=self._connection_config(self._EXEC_REQUEST_TIMEOUT_S),
            connect_timeout=timedelta(seconds=self._ready_timeout),
        )
        self.logger.info(f"Adopted sandbox {adopted_id} after gateway timeout")
        return sandbox

    async def stop(self, delete: bool) -> None:
        if self._sandbox is None:
            return
        if not delete:
            self.logger.info("OpenSandbox sandboxes are ephemeral; killing despite delete=False.")
        sandbox, self._sandbox = self._sandbox, None
        self._uid_cache.clear()
        for attempt in (1, 2):
            try:
                await sandbox.kill()
                break
            except Exception as e:
                if attempt == 1:
                    self.logger.warning(f"kill failed for sandbox {sandbox.id}: {e}; retrying once")
                    await asyncio.sleep(2)
                else:
                    self.logger.error(
                        f"kill failed twice for sandbox {sandbox.id}: {e}. It will live until its "
                        f"{self._sandbox_lifetime}s lifetime cap; cleanup_sandboxes.py can reap it."
                    )
        with contextlib.suppress(Exception):
            await sandbox.close()
        _LIVE_SANDBOX_IDS.pop(sandbox.id, None)

    # ── exec ─────────────────────────────────────────────────────────────

    def _require_sandbox(self) -> "Sandbox":
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        return self._sandbox

    def _sandbox_id(self) -> str:
        return self._sandbox.id if self._sandbox else "<none>"

    async def _resolve_user_identity(self, user: str | int | None) -> tuple[int | None, int | None, dict[str, str]]:
        """Map harbor's username/UID to execd's numeric uid/gid plus the env vars
        ``docker exec -u`` would set (HOME/USER/LOGNAME from /etc/passwd)."""
        if user is None:
            return None, None, {}
        if isinstance(user, int) or str(user).isdigit():
            return int(user), None, {}
        name = str(user)
        if ":" in name:  # "user:group" Dockerfile form
            name = name.split(":", 1)[0]
        if name not in self._uid_cache:
            result = await self._exec_raw(
                f"getent passwd {shlex.quote(name)}", timeout_sec=30, uid=None, gid=None, cwd="/"
            )
            fields = (result.stdout or "").strip().split(":")
            if result.return_code != 0 or len(fields) < 6:
                raise RuntimeError(
                    f"Cannot resolve user {name!r} in sandbox {self._sandbox_id()}: "
                    f"{result.stderr or result.stdout}"
                )
            uid, gid, home = int(fields[2]), int(fields[3]), fields[5] or "/"
            self._uid_cache[name] = (uid, gid, {"HOME": home, "USER": name, "LOGNAME": name})
        uid, gid, env = self._uid_cache[name]
        return uid, gid, dict(env)

    async def _exec_raw(
        self,
        command: str,
        *,
        timeout_sec: int | None,
        uid: int | None,
        gid: int | None,
        cwd: str | None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        sandbox = self._require_sandbox()
        shell = ["bash", "-c", command]
        if timeout_sec:
            # Mirror harbor's docker backend: the command is killed at
            # timeout_sec. GNU timeout returns 124, which we surface as the
            # same RuntimeError the docker backend raises.
            shell = ["timeout", "--signal=TERM", "--kill-after=10", str(int(timeout_sec))] + shell
        # Output goes to files inside the sandbox and comes back over the
        # filesystem API, NOT over execd's stdout stream. execd emits one SSE
        # event per output line and the proxied stream drains at ~5k lines/s:
        # a 150k-line `seq` took 36s to arrive, and a chatty compile or pip
        # install blocks on the full pipe until GNU timeout kills it at
        # timeout_sec. Redirected, the same output is a single ~0.1s read.
        token = uuid.uuid4().hex
        out_path = f"{self._EXEC_OUTPUT_DIR}/{token}.out"
        err_path = f"{self._EXEC_OUTPUT_DIR}/{token}.err"
        outer = (
            f"mkdir -p {shlex.quote(self._EXEC_OUTPUT_DIR)} 2>/dev/null; "
            f"chmod 1777 {shlex.quote(self._EXEC_OUTPUT_DIR)} 2>/dev/null; "
            f"{shlex.join(shell)} >{shlex.quote(out_path)} 2>{shlex.quote(err_path)}"
        )
        opts = RunCommandOpts(
            working_directory=cwd,
            envs=env or None,
            uid=uid,
            gid=gid if uid is not None else None,
            timeout=timedelta(seconds=timeout_sec + self._EXEC_TIMEOUT_MARGIN_S) if timeout_sec else None,
        )
        try:
            execution = await sandbox.commands.run(shlex.join(["bash", "-c", outer]), opts=opts)
        except SandboxException as e:
            if not await self._sandbox_is_alive():
                raise RuntimeError(
                    f"Sandbox {sandbox.id} died during exec (expired, evicted, or crashed): {e}"
                ) from e
            raise
        stdout, stderr = await self._collect_output(sandbox, out_path, err_path)
        # Anything execd itself streamed (e.g. a shell that failed before the
        # redirect took effect) is still worth surfacing.
        streamed_err = "".join(m.text for m in execution.logs.stderr)
        if streamed_err and not stderr:
            stderr = streamed_err[: self._MAX_OUTPUT_CHARS]
        return_code = execution.exit_code
        if return_code is None:
            if execution.error is not None:
                stderr = f"{stderr}\n[{execution.error.name}] {execution.error.value}".strip()
            return_code = -1
        return ExecResult(stdout=stdout or None, stderr=stderr or None, return_code=return_code)

    async def _collect_output(self, sandbox: "Sandbox", out_path: str, err_path: str) -> tuple[str, str]:
        """Read and remove the redirected stdout/stderr files (capped at _MAX_OUTPUT_CHARS)."""
        byte_range = f"bytes=0-{self._MAX_OUTPUT_CHARS - 1}"

        async def read(path: str) -> str:
            try:
                data = await sandbox.files.read_bytes(path, range_header=byte_range)
            except Exception as e:  # noqa: BLE001 - missing file if the redirect itself failed
                self.logger.debug(f"could not read exec output {path}: {e}")
                return ""
            return data.decode("utf-8", errors="replace")

        stdout, stderr = await asyncio.gather(read(out_path), read(err_path))
        with contextlib.suppress(Exception):
            await sandbox.files.delete_files([out_path, err_path])
        return stdout, stderr

    async def _sandbox_is_alive(self) -> bool:
        if self._sandbox is None:
            return False
        try:
            info = await self._sandbox.get_info()
            return info.status.state == SandboxState.RUNNING
        except Exception:
            return False

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        user = self._resolve_user(user)
        if user is None:
            user = self._image_user
        # Task env is also set on the pod at creation; re-merging it per call is
        # harmless and matches the docker backend's `-e` flags exactly.
        env = self._merge_env(env)
        uid, gid, user_env = await self._resolve_user_identity(user)
        if user_env:
            env = {**user_env, **(env or {})}
        result = await self._exec_raw(
            command, timeout_sec=timeout_sec, uid=uid, gid=gid, cwd=cwd or self._workdir, env=env
        )
        if timeout_sec and result.return_code == 124:
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        return result

    # ── file transfer ────────────────────────────────────────────────────

    @staticmethod
    def _mode_int(path: Path) -> int:
        """WriteEntry.mode is the octal digits as an int (0o755 -> 755)."""
        return int(oct(path.stat().st_mode & 0o777)[2:])

    async def _write_bytes(self, target_path: str, data: bytes, mode: int) -> None:
        sandbox = self._require_sandbox()
        await sandbox.files.write_files([WriteEntry(path=target_path, data=data, mode=mode)])

    async def _sh_root(self, command: str, timeout_sec: int, what: str) -> None:
        result = await self._exec_raw(command, timeout_sec=timeout_sec, uid=None, gid=None, cwd="/")
        if result.return_code != 0:
            raise RuntimeError(f"{what} failed (exit={result.return_code}): {result.stderr or result.stdout}")

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        parent = str(PurePosixPath(target_path).parent) or "/"
        await self._sh_root(f"mkdir -p {shlex.quote(parent)}", 60, f"mkdir for {target_path}")
        await self._write_bytes(target_path, source.read_bytes(), self._mode_int(source))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Copy the *contents* of source_dir into target_dir (docker cp src/. semantics)."""
        source = Path(source_dir)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            tar.add(source, arcname=".")
        remote_tmp = f"/tmp/.tmax_upload_{uuid.uuid4().hex}.tgz"
        await self._write_bytes(remote_tmp, buffer.getvalue(), 644)
        q_target, q_tmp = shlex.quote(target_dir), shlex.quote(remote_tmp)
        await self._sh_root(
            f"mkdir -p {q_target} && tar -xzf {q_tmp} -C {q_target} && rm -f {q_tmp}",
            300,
            f"upload_dir {source} -> {target_dir}",
        )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        sandbox = self._require_sandbox()
        data = await sandbox.files.read_bytes(source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Copy the contents of source_dir into target_dir, overwriting existing files."""
        sandbox = self._require_sandbox()
        remote_tmp = f"/tmp/.tmax_download_{uuid.uuid4().hex}.tgz"
        q_src, q_tmp = shlex.quote(source_dir), shlex.quote(remote_tmp)
        await self._sh_root(f"tar -czf {q_tmp} -C {q_src} .", 300, f"download_dir tar of {source_dir}")
        try:
            data = await sandbox.files.read_bytes(remote_tmp)
        finally:
            with contextlib.suppress(Exception):
                await self._exec_raw(f"rm -f {q_tmp}", timeout_sec=30, uid=None, gid=None, cwd="/")
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tar.extractall(path=target, filter="data")
