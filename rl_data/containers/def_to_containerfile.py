"""Convert Apptainer base ``.def`` files into podman/docker build contexts.

Every base def in this directory has the same uniform shape::

    Bootstrap: docker
    From: <image>

    %post
        <arbitrary shell>

    %environment
        export KEY=VALUE
        ...

which maps mechanically onto a build context::

    docker/<base_name>/
    ├── Containerfile   # FROM <image> + ENV lines + RUN bash -e /tmp/post.sh
    └── post.sh         # the %post body, verbatim

The %post body is shipped as a script (not inlined into RUN) so arbitrary
shell — line continuations, loops, heredocs — survives translation
untouched. ``bash -e`` mirrors apptainer's fail-on-error %post semantics.

Usage (regenerates every docker/<base>/ context from the sibling defs)::

    uv run python -m rl_data.containers.def_to_containerfile

The generated contexts are committed; rerun this after editing any .def.
Build them with::

    podman build -t localhost/tmax/<base_name> rl_data/containers/docker/<base_name>
"""
from __future__ import annotations

import re
from pathlib import Path

CONTAINERS_DIR = Path(__file__).parent
DOCKER_DIR = CONTAINERS_DIR / "docker"

_SECTION_RE = re.compile(r"^%(\w+)", re.MULTILINE)


def parse_def(def_text: str) -> tuple[str, str, list[str]]:
    """Return (from_image, post_body, environment_exports) for a uniform def.

    Raises ValueError on defs that use features this converter doesn't
    support (%files, non-docker bootstrap, ...) so drift is loud, not silent.
    """
    bootstrap_m = re.search(r"^Bootstrap:\s*(\S+)", def_text, re.MULTILINE)
    from_m = re.search(r"^From:\s*(\S+)", def_text, re.MULTILINE)
    if not bootstrap_m or bootstrap_m.group(1) != "docker" or not from_m:
        raise ValueError("expected 'Bootstrap: docker' + 'From: <image>'")

    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(def_text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(def_text)
        sections[m.group(1)] = def_text[m.end():end]

    unsupported = set(sections) - {"post", "environment"}
    if unsupported:
        raise ValueError(f"unsupported def sections: {sorted(unsupported)}")
    if "post" not in sections:
        raise ValueError("def has no %post section")

    # %post body verbatim, minus the uniform 4-space def indentation.
    post_lines = []
    for line in sections["post"].splitlines():
        post_lines.append(line[4:] if line.startswith("    ") else line)
    post_body = "\n".join(post_lines).strip("\n") + "\n"

    env_exports: list[str] = []
    for line in sections.get("environment", "").splitlines():
        line = line.strip()
        if line.startswith("export "):
            env_exports.append(line[len("export "):])
        elif line:
            raise ValueError(f"unsupported %environment line: {line!r}")

    return from_m.group(1), post_body, env_exports


def convert(def_path: Path) -> Path:
    """Write the build context for one def; return the context dir."""
    from_image, post_body, env_exports = parse_def(def_path.read_text())
    name = def_path.stem
    ctx = DOCKER_DIR / name
    ctx.mkdir(parents=True, exist_ok=True)

    (ctx / "post.sh").write_text(post_body, encoding="utf-8")

    lines = [
        f"# Generated from ../{def_path.name} by def_to_containerfile.py — do not edit by hand.",
        f"FROM {from_image}",
    ]
    for kv in env_exports:
        lines.append(f"ENV {kv}")
    lines += [
        "COPY post.sh /tmp/post.sh",
        "RUN bash -e /tmp/post.sh && rm -f /tmp/post.sh",
    ]
    (ctx / "Containerfile").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ctx


def main() -> None:
    defs = sorted(CONTAINERS_DIR.glob("base_*.def"))
    if not defs:
        raise SystemExit(f"no base_*.def found in {CONTAINERS_DIR}")
    for def_path in defs:
        ctx = convert(def_path)
        print(f"{def_path.name} -> {ctx.relative_to(CONTAINERS_DIR)}/")
    print(f"\n{len(defs)} contexts written under {DOCKER_DIR}")


if __name__ == "__main__":
    main()
