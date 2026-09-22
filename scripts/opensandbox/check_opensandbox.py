#!/usr/bin/env python3
"""Preflight for the OpenSandbox eval backend: reach the service, run one sandbox.

Run it where `harbor run` will run (laptop, or inside the Beaker job before the
eval starts). It creates a sandbox from a small image, execs a command as root
and as a named user, round-trips a file, and kills the sandbox.

    export OPEN_SANDBOX_API_KEY=...
    uv run python scripts/opensandbox/check_opensandbox.py
    uv run python scripts/opensandbox/check_opensandbox.py --image alexgshaw/adaptive-rejection-sampler:20251031
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opensandbox import Sandbox  # noqa: E402
from opensandbox.config import ConnectionConfig  # noqa: E402
from opensandbox.models.execd import RunCommandOpts  # noqa: E402
from opensandbox.models.filesystem import WriteEntry  # noqa: E402

from tmax_envs.opensandbox import DEFAULT_DOMAIN, DEFAULT_PROTOCOL, rewrite_image_for_mirror  # noqa: E402


async def main_async(args) -> int:
    domain = args.domain or os.getenv("TMAX_OPENSANDBOX_DOMAIN") or DEFAULT_DOMAIN
    protocol = args.protocol or os.getenv("TMAX_OPENSANDBOX_PROTOCOL") or DEFAULT_PROTOCOL
    if "://" in domain:
        protocol, domain = domain.split("://", 1)
    prefix = args.image_prefix if args.image_prefix is not None else os.getenv("TMAX_OPENSANDBOX_IMAGE_PREFIX", "")
    image = rewrite_image_for_mirror(args.image, prefix)
    config = ConnectionConfig(domain=domain, protocol=protocol, api_key=os.getenv("OPEN_SANDBOX_API_KEY"),
                              request_timeout=timedelta(seconds=600), use_server_proxy=not args.direct)
    print(f"endpoint: {protocol}://{domain}\nimage:    {image}")

    t0 = time.perf_counter()
    sandbox = await Sandbox.create(
        image,
        timeout=timedelta(minutes=10),
        ready_timeout=timedelta(seconds=args.ready_timeout),
        resource={"cpu": "1", "memory": "1024Mi"},
        metadata={"tmax_app": "tmax-opensandbox-check"},
        connection_config=config,
    )
    print(f"[PASS] sandbox {sandbox.id} ready in {time.perf_counter() - t0:.1f}s")
    ok = True
    try:
        async def run(cmd: str, **opts):
            t = time.perf_counter()
            ex = await sandbox.commands.run(cmd, opts=RunCommandOpts(timeout=timedelta(seconds=60), **opts))
            out = "".join(m.text for m in ex.logs.stdout).strip()
            err = "".join(m.text for m in ex.logs.stderr).strip()
            return ex.exit_code, out, err, time.perf_counter() - t

        code, out, err, dt = await run("id -u; id -un; pwd; echo $PATH; bash --version | head -1")
        print(f"[{'PASS' if code == 0 else 'FAIL'}] exec as default user ({dt:.2f}s):\n         " + out.replace("\n", "\n         ") + (f"\n         stderr: {err}" if err else ""))
        ok &= code == 0

        code, out, err, dt = await run("useradd -m tmaxcheck 2>/dev/null || true; id -u tmaxcheck")
        if code == 0 and out.isdigit():
            uid = int(out)
            code, out, err, dt = await run("id -un; echo $HOME", uid=uid)
            print(f"[{'PASS' if code == 0 else 'FAIL'}] exec as uid {uid} ({dt:.2f}s): {out!r} {err!r}")
            ok &= code == 0
        else:
            print(f"[WARN] could not create a test user (exit {code}: {err}); skipping uid check")

        code, out, err, dt = await run("timeout --signal=TERM 2 sleep 10; echo exit=$?")
        print(f"[{'PASS' if 'exit=124' in out else 'FAIL'}] GNU timeout wrapper ({dt:.2f}s): {out!r}")
        ok &= "exit=124" in out

        payload = os.urandom(4096)
        await sandbox.files.write_files([WriteEntry(path="/tmp/tmax_check.bin", data=payload, mode=600)])
        back = await sandbox.files.read_bytes("/tmp/tmax_check.bin")
        print(f"[{'PASS' if back == payload else 'FAIL'}] binary file round trip (4 KiB)")
        ok &= back == payload

        code, out, err, dt = await run("env | grep -E '^(PYTHON_VERSION|LANG|HOME)=' | sort | tr '\\n' ' '")
        print(f"[INFO] image ENV visible to exec: {out!r}")

        code, out, err, dt = await run("curl -sS -m 5 -o /dev/null -w '%{http_code}' https://pypi.org/simple/ || echo curl-exit=$?")
        print(f"[INFO] internet egress from sandbox: {out!r} {err!r}")
    finally:
        await sandbox.kill()
        await sandbox.close()
        print(f"[PASS] sandbox {sandbox.id} killed")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default="ubuntu:24.04")
    parser.add_argument("--image-prefix", default=None, help="mirror prefix (default: $TMAX_OPENSANDBOX_IMAGE_PREFIX)")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--ready-timeout", type=int, default=600)
    parser.add_argument("--direct", action="store_true", help="talk to execd via the ingress gateway instead of the server proxy")
    args = parser.parse_args()
    if not os.getenv("OPEN_SANDBOX_API_KEY"):
        print("RESULT: FAIL — OPEN_SANDBOX_API_KEY is not set", file=sys.stderr)
        return 2
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
