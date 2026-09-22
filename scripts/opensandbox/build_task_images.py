#!/usr/bin/env python3
"""Build and push images for harbor tasks that ship only a Dockerfile.

OpenSandbox pulls images; it cannot build. Terminal-Bench 2.x tasks declare a
prebuilt ``[environment].docker_image`` and need nothing here. OpenThoughts
TBLite tasks (and any other Dockerfile-only dataset) must be built once and
pushed to a registry repo the sandbox cluster can pull from. The tag is derived
from the task name plus a hash of ``environment/`` (see
``tmax_envs/task_images.py``), so the OpenSandbox environment resolves the same
reference at run time without a manifest, and any edit to a task's environment
yields a new tag.

Typical use (Docker Hub repo; the AI2 sandbox cluster's mirror caches it):

    # 1. get the tasks locally (harbor's cache mode keeps them under ~/.cache/harbor)
    uv run harbor datasets download openthoughts-tblite@2.0 -o /tmp/tblite --export

    # 2. see what is missing in the registry
    uv run python scripts/opensandbox/build_task_images.py /tmp/tblite/openthoughts-tblite \\
        --repo docker.io/<user>/tmax-harbor-tasks --check

    # 3. build + push what is missing (needs `docker login` for the repo)
    uv run python scripts/opensandbox/build_task_images.py /tmp/tblite/openthoughts-tblite \\
        --repo docker.io/<user>/tmax-harbor-tasks --push

Images are built for linux/amd64 (the sandbox nodes) via ``docker buildx``;
``--builder podman`` uses ``podman build`` instead for Beaker jobs without
Docker. Tasks that already declare docker_image are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tmax_envs.task_images import task_image_ref  # noqa: E402


def iter_task_dirs(dataset_dir: Path):
    for task_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        if (task_dir / "task.toml").is_file() and (task_dir / "environment").is_dir():
            yield task_dir


def task_docker_image(task_dir: Path) -> str | None:
    config = tomllib.loads((task_dir / "task.toml").read_text())
    return (config.get("environment") or {}).get("docker_image")


def image_exists(ref: str, tool: str) -> bool:
    cmd = ["docker", "manifest", "inspect", ref] if tool == "docker" else ["podman", "manifest", "inspect", ref]
    if tool == "podman":
        # `podman manifest inspect` only handles manifest lists; skopeo is the
        # reliable registry probe when it is present.
        if shutil.which("skopeo"):
            cmd = ["skopeo", "inspect", "--raw", f"docker://{ref}"]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def build_one(task_dir: Path, ref: str, *, tool: str, push: bool, platform: str, log_dir: Path | None) -> tuple[str, bool, str]:
    env_dir = task_dir / "environment"
    if tool == "docker":
        cmd = ["docker", "buildx", "build", "--platform", platform, "-t", ref, "--provenance=false", "--sbom=false"]
        cmd += ["--push"] if push else ["--load"]
        cmd += [str(env_dir)]
    else:
        cmd = ["podman", "build", "--platform", platform, "-t", ref, str(env_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    ok = proc.returncode == 0
    if ok and push and tool == "podman":
        push_proc = subprocess.run(["podman", "push", ref], capture_output=True, text=True)
        output += push_proc.stdout + push_proc.stderr
        ok = push_proc.returncode == 0
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{task_dir.name}.log").write_text(output)
    return task_dir.name, ok, output[-2000:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="directory whose subdirs are harbor tasks (task.toml + environment/)")
    parser.add_argument("--repo", required=True, help="registry repo, e.g. docker.io/<user>/tmax-harbor-tasks")
    parser.add_argument("--check", action="store_true", help="only report which images exist in the registry")
    parser.add_argument("--push", action="store_true", help="push after building (otherwise images stay local)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the tag already exists in the registry")
    parser.add_argument("--builder", choices=["docker", "podman"], default="docker")
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--jobs", type=int, default=2, help="parallel builds")
    parser.add_argument("--task", action="append", default=[], help="only these task names (repeatable)")
    parser.add_argument("--log-dir", type=Path, default=None, help="write per-task build logs here")
    parser.add_argument("--manifest", type=Path, default=None, help="write {task: image_ref} JSON here")
    args = parser.parse_args()

    tasks = list(iter_task_dirs(args.dataset_dir))
    if args.task:
        wanted = set(args.task)
        tasks = [t for t in tasks if t.name in wanted]
        missing = wanted - {t.name for t in tasks}
        if missing:
            print(f"warning: tasks not found: {sorted(missing)}", file=sys.stderr)
    if not tasks:
        print(f"no tasks found under {args.dataset_dir}", file=sys.stderr)
        return 2

    plan: dict[str, str] = {}
    skipped_prebuilt: list[str] = []
    for task_dir in tasks:
        prebuilt = task_docker_image(task_dir)
        if prebuilt:
            skipped_prebuilt.append(f"{task_dir.name} ({prebuilt})")
            continue
        plan[task_dir.name] = task_image_ref(args.repo, task_dir / "environment", task_dir.name)

    if skipped_prebuilt:
        print(f"{len(skipped_prebuilt)} task(s) declare docker_image; nothing to build for them.")
    if args.manifest:
        args.manifest.write_text(json.dumps(plan, indent=2, sort_keys=True))
        print(f"manifest written to {args.manifest}")

    existing: set[str] = set()
    if not args.force:
        print(f"probing registry for {len(plan)} image(s)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            probes = {pool.submit(image_exists, ref, args.builder): name for name, ref in plan.items()}
            for fut in concurrent.futures.as_completed(probes):
                if fut.result():
                    existing.add(probes[fut])
    to_build = {name: ref for name, ref in plan.items() if name not in existing}
    print(f"{len(existing)} present, {len(to_build)} to build")
    for name in sorted(to_build):
        print(f"  MISSING  {name} -> {to_build[name]}")
    if args.check or not to_build:
        return 0 if (not to_build or args.check) else 1

    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(
                build_one,
                args.dataset_dir / name,
                ref,
                tool=args.builder,
                push=args.push,
                platform=args.platform,
                log_dir=args.log_dir,
            )
            for name, ref in sorted(to_build.items())
        ]
        for fut in concurrent.futures.as_completed(futures):
            name, ok, tail = fut.result()
            print(f"  {'OK  ' if ok else 'FAIL'} {name}", flush=True)
            if not ok:
                failures.append((name, tail))
    if failures:
        print(f"\n{len(failures)} build(s) failed:")
        for name, tail in failures:
            print(f"--- {name} ---\n{tail}\n")
        return 1
    print(f"all {len(to_build)} image(s) built{' and pushed' if args.push else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
