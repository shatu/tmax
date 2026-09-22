"""Unit tests for the OpenSandbox harbor environment (no network)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from tmax_envs import task_images
from tmax_envs.opensandbox import OpenSandboxEnvironment, rewrite_image_for_mirror


# ── task_images ───────────────────────────────────────────────────────────


def _make_env_dir(tmp_path: Path, dockerfile: str, extra: dict[str, bytes] | None = None) -> Path:
    env_dir = tmp_path / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text(dockerfile)
    for rel, data in (extra or {}).items():
        p = env_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return env_dir


def test_environment_dir_hash_changes_with_content_and_ignores_pycache(tmp_path):
    env_dir = _make_env_dir(tmp_path, "FROM ubuntu:24.04\n", {"data/a.txt": b"1"})
    h1 = task_images.environment_dir_hash(env_dir)
    (env_dir / "__pycache__").mkdir()
    (env_dir / "__pycache__" / "x.pyc").write_bytes(b"junk")
    (env_dir / ".DS_Store").write_bytes(b"junk")
    assert task_images.environment_dir_hash(env_dir) == h1
    (env_dir / "data" / "a.txt").write_bytes(b"2")
    assert task_images.environment_dir_hash(env_dir) != h1


def test_task_image_ref_shape(tmp_path):
    env_dir = _make_env_dir(tmp_path, "FROM ubuntu:24.04\n")
    ref = task_images.task_image_ref("docker.io/u/repo/", env_dir, "my task/name")
    repo, tag = ref.rsplit(":", 1)
    assert repo == "docker.io/u/repo"
    name, digest = tag.rsplit("-", 1)
    assert name == "my-task-name"
    assert len(digest) == 12 and int(digest, 16) >= 0


def test_parse_dockerfile_last_stage_workdir_user_and_continuations(tmp_path):
    dockerfile = """
# comment
FROM python:3.12-slim AS builder
WORKDIR /build
USER nobody
FROM ubuntu:22.04
RUN apt-get update && \\
    apt-get install -y curl
ENV FOO=bar
WORKDIR "/app"
USER alice:research
"""
    facts = task_images.parse_dockerfile(_make_env_dir(tmp_path, dockerfile) / "Dockerfile")
    assert facts.from_images == ["python:3.12-slim", "ubuntu:22.04"]
    assert facts.workdir == "/app"
    assert facts.user == "alice:research"


def test_parse_dockerfile_without_user_or_workdir(tmp_path):
    facts = task_images.parse_dockerfile(_make_env_dir(tmp_path, "FROM --platform=linux/amd64 ubuntu:24.04\nRUN true\n") / "Dockerfile")
    assert facts.from_images == ["ubuntu:24.04"]
    assert facts.workdir is None and facts.user is None


# ── mirror rewriting ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "image,expected",
    [
        ("ubuntu:24.04", "m.example/hub/library/ubuntu:24.04"),
        ("alexgshaw/foo:1", "m.example/hub/alexgshaw/foo:1"),
        ("ghcr.io/org/img:tag", "ghcr.io/org/img:tag"),
        ("localhost:5000/img", "localhost:5000/img"),
        ("docker.io/ubuntu:22.04", "docker.io/ubuntu:22.04"),
    ],
)
def test_rewrite_image_for_mirror(image, expected):
    assert rewrite_image_for_mirror(image, "m.example/hub") == expected
    assert rewrite_image_for_mirror(image, "") == image


# ── environment construction (no network) ────────────────────────────────


def _make_env(tmp_path: Path, dockerfile: str, docker_image: str | None = None, **kwargs) -> OpenSandboxEnvironment:
    env_dir = _make_env_dir(tmp_path, dockerfile)
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    return OpenSandboxEnvironment(
        environment_dir=env_dir,
        environment_name="demo-task",
        session_id="demo-task__abc123",
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(docker_image=docker_image, cpus=2, memory_mb=4096, allow_internet=True),
        logger=logging.getLogger("test"),
        api_key="k",
        **kwargs,
    )


def test_prebuilt_image_wins_and_resources_map(tmp_path):
    env = _make_env(tmp_path, "FROM ubuntu:24.04\nWORKDIR /app\nUSER alice\n", docker_image="alexgshaw/x:1")
    assert env.resolve_image() == "alexgshaw/x:1"
    assert env._resource() == {"cpu": "2", "memory": "4096Mi"}
    assert env._workdir == "/app"
    assert env._image_user == "alice"
    assert env.type() == "opensandbox"
    assert env.capabilities.disable_internet is True and env.capabilities.mounted is False


def test_dockerfile_only_task_requires_task_image_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("TMAX_TASK_IMAGE_REPO", raising=False)
    env = _make_env(tmp_path, "FROM ubuntu:24.04\n")
    with pytest.raises(RuntimeError, match="task_image_repo"):
        env.resolve_image()
    env2 = _make_env(tmp_path / "b", "FROM ubuntu:24.04\n", task_image_repo="docker.io/u/tasks")
    ref = env2.resolve_image()
    assert ref.startswith("docker.io/u/tasks:demo-task-")
    assert ref == task_images.task_image_ref("docker.io/u/tasks", env2.environment_dir, "demo-task")


def test_kwargs_are_coerced_from_cli_strings(tmp_path):
    env = _make_env(
        tmp_path,
        "FROM ubuntu:24.04\n",
        docker_image="x:1",
        sandbox_lifetime_sec="123",
        start_concurrency="3",
        domain="https://sb.example.org",
        enforce_network_policy="false",
    )
    assert env._sandbox_lifetime == 123
    assert env._start_concurrency == 3
    assert env._domain == "sb.example.org" and env._protocol == "https"
    assert env._network_policy() is None


def test_network_policy_denies_when_task_disallows_internet(tmp_path):
    env_dir = _make_env_dir(tmp_path, "FROM ubuntu:24.04\n")
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    env = OpenSandboxEnvironment(
        environment_dir=env_dir,
        environment_name="t",
        session_id="t__1",
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(docker_image="x:1", allow_internet=False),
        api_key="k",
    )
    policy = env._network_policy()
    assert policy is not None and policy.default_action == "deny"


def test_metadata_is_label_safe(tmp_path):
    env = _make_env(tmp_path, "FROM ubuntu:24.04\n", docker_image="x:1", app_name="my app!")
    meta = env._metadata("abc")
    assert meta["tmax_app"] == "my-app"
    assert meta["tmax_task"] == "demo-task"
    assert meta["tmax_create_id"] == "abc"


class _FakeExecution:
    def __init__(self, stdout: str, exit_code: int):
        class _Msg:
            def __init__(self, text):
                self.text = text

        class _Logs:
            pass

        self.logs = _Logs()
        self.logs.stdout = [_Msg(stdout)]
        self.logs.stderr = []
        self.exit_code = exit_code
        self.error = None


class _FakeCommands:
    """Records exec calls; stdout is delivered through _FakeFiles like the real backend."""

    def __init__(self, files):
        self.calls: list[tuple[str, object]] = []
        self._files = files

    async def run(self, command, *, opts=None, handlers=None):
        self.calls.append((command, opts))
        # The backend redirects "... >/tmp/<tok>.out 2>/tmp/<tok>.err"; emulate the files.
        # the redirect is the first ">" that is not part of "2>/dev/null"
        out_path = None
        for part in command.replace("2>/dev/null", "").split(">")[1:2]:
            out_path = part.split()[0].strip("'\"")
        if "getent passwd" in command:
            name = command.split("getent passwd ")[1].split("'")[0].split('"')[0].split()[0]
            self._files.contents[out_path] = f"{name}:x:1001:1002::/home/{name}:/bin/bash\n".encode()
            return _FakeExecution("", 0)
        if "exit 124" in command:
            return _FakeExecution("", 124)
        self._files.contents[out_path] = b"ok\n"
        return _FakeExecution("", 0)


class _FakeFiles:
    def __init__(self):
        self.contents: dict[str, bytes] = {}
        self.deleted: list[str] = []

    async def read_bytes(self, path, *, range_header=None):
        if path not in self.contents:
            raise FileNotFoundError(path)
        return self.contents[path]

    async def delete_files(self, paths):
        self.deleted.extend(paths)


class _FakeSandbox:
    id = "sb-1"

    def __init__(self):
        self.files = _FakeFiles()
        self.commands = _FakeCommands(self.files)


def test_exec_resolves_named_user_and_applies_workdir_and_timeout(tmp_path):
    env = _make_env(tmp_path, "FROM ubuntu:24.04\nWORKDIR /app\nUSER alice\n", docker_image="x:1")
    env._sandbox = _FakeSandbox()

    result = asyncio.run(env.exec("echo hi", timeout_sec=20, env={"A": "1"}))
    assert result.return_code == 0 and result.stdout == "ok\n"
    calls = env._sandbox.commands.calls
    # first call resolves alice -> uid/gid/home as root, second runs the command
    assert "getent passwd alice" in calls[0][0] and calls[0][1].uid is None
    cmd, opts = calls[1]
    # outer `bash -c` wraps the timeout'd command and redirects both streams to files
    assert cmd.startswith("bash -c ")
    assert "timeout --signal=TERM --kill-after=10 20 bash -c " in cmd
    assert ">/tmp/.tmax-exec/" in cmd and ".out 2>/tmp/.tmax-exec/" in cmd
    assert cmd.index("mkdir -p /tmp/.tmax-exec") < cmd.index("timeout ")
    assert len(env._sandbox.files.deleted) == 4  # both files for both execs so far
    assert opts.uid == 1001 and opts.gid == 1002
    assert opts.working_directory == "/app"
    assert opts.envs == {"HOME": "/home/alice", "USER": "alice", "LOGNAME": "alice", "A": "1"}
    assert opts.timeout.total_seconds() == 20 + env._EXEC_TIMEOUT_MARGIN_S

    # cached: no second id lookup, explicit root user bypasses the image USER
    asyncio.run(env.exec("true", user="root"))
    assert "getent passwd root" in env._sandbox.commands.calls[-2][0]
    asyncio.run(env.exec("true", user=0, cwd="/tmp"))
    assert env._sandbox.commands.calls[-1][1].uid == 0
    assert env._sandbox.commands.calls[-1][1].working_directory == "/tmp"


def test_exec_timeout_raises_like_docker_backend(tmp_path):
    env = _make_env(tmp_path, "FROM ubuntu:24.04\n", docker_image="x:1")
    env._sandbox = _FakeSandbox()
    with pytest.raises(RuntimeError, match="timed out after 5 seconds"):
        asyncio.run(env.exec("exit 124", timeout_sec=5))
