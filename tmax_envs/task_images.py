"""Deterministic image names for harbor tasks that ship a Dockerfile only.

OpenSandbox pulls OCI images; it cannot build a task's ``environment/Dockerfile``.
Tasks that declare ``[environment].docker_image`` (Terminal-Bench 2.x) need
nothing else. Tasks that do not (OpenThoughts TBLite) are built once by
``scripts/opensandbox/build_task_images.py`` and pushed to a registry repo under
a tag derived here, so the environment can name the image without a manifest:

    <repo>:<task-name>-<12 hex chars of the environment dir's content hash>

The hash covers every file under ``environment/`` (path + bytes), so editing a
Dockerfile or a COPY'd file changes the tag and the stale image is never reused.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

_IGNORED_DIR_NAMES = {"__pycache__", ".git"}
_IGNORED_FILE_NAMES = {".DS_Store"}
_TAG_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_HASH_LEN = 12
# Docker tags are capped at 128 chars; leave room for "-<hash>".
_MAX_NAME_LEN = 128 - _HASH_LEN - 1


def environment_dir_hash(environment_dir: Path | str) -> str:
    """sha256 over the sorted (relative path, contents) of every file in the dir."""
    root = Path(environment_dir)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if any(part in _IGNORED_DIR_NAMES for part in rel.parts[:-1]):
            continue
        if rel.name in _IGNORED_FILE_NAMES:
            continue
        digest.update(rel.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def sanitize_tag_component(name: str) -> str:
    """Make *name* a valid Docker tag fragment (``[A-Za-z0-9_.-]``, no leading ``.``/``-``)."""
    safe = _TAG_SAFE_RE.sub("-", name).strip(".-") or "task"
    return safe[:_MAX_NAME_LEN]


def task_image_tag(environment_dir: Path | str, task_name: str) -> str:
    return f"{sanitize_tag_component(task_name)}-{environment_dir_hash(environment_dir)[:_HASH_LEN]}"


def task_image_ref(repo: str, environment_dir: Path | str, task_name: str) -> str:
    """Full image reference ``<repo>:<tag>`` for a Dockerfile-only task."""
    return f"{repo.rstrip('/')}:{task_image_tag(environment_dir, task_name)}"


@dataclass
class DockerfileFacts:
    """The few Dockerfile instructions an exec-only backend must honour itself.

    OpenSandbox starts the image with its own entrypoint (``tail -f /dev/null``)
    and runs commands through ``execd``, so the image's ``WORKDIR`` and ``USER``
    are not applied to exec'd commands the way ``docker compose exec`` applies
    them. We read them from the Dockerfile and pass them explicitly.
    """

    from_images: list[str] = field(default_factory=list)
    workdir: str | None = None
    user: str | None = None


def _logical_lines(text: str):
    """Yield Dockerfile instructions with comments stripped and continuations joined.

    Heredocs (``RUN <<EOF ... EOF``) are passed through as opaque lines; we only
    care about WORKDIR/USER/FROM, which are never heredocs.
    """
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not buf and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        yield buf.strip()
        buf = ""
    if buf.strip():
        yield buf.strip()


def parse_dockerfile(path: Path | str) -> DockerfileFacts:
    facts = DockerfileFacts()
    text = Path(path).read_text(errors="replace")
    for line in _logical_lines(text):
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        instruction, rest = parts[0].upper(), parts[1].strip()
        if instruction == "FROM":
            # "FROM image[:tag] [AS name]" — build-stage names are irrelevant here.
            tokens = rest.split()
            if tokens and tokens[0].startswith("--"):
                tokens = tokens[1:]  # e.g. --platform=...
            if tokens:
                facts.from_images.append(tokens[0])
            # A new stage resets WORKDIR/USER; only the final stage matters.
            facts.workdir = None
            facts.user = None
        elif instruction == "WORKDIR":
            facts.workdir = _unquote(rest)
        elif instruction == "USER":
            facts.user = _unquote(rest)
    return facts


def _unquote(value: str) -> str:
    try:
        tokens = shlex.split(value)
    except ValueError:
        return value.strip()
    return tokens[0] if tokens else value.strip()
