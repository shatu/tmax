#!/usr/bin/env python3
"""Kill leftover OpenSandbox sandboxes from tmax harbor evals.

A `harbor run` that is SIGKILLed (Beaker job cancelled, node preempted) never
reaches its atexit reaper, so its sandboxes live until their lifetime cap
(default 2h) while still occupying cluster capacity. This lists sandboxes by
the metadata tag the environment sets and kills them.

    export OPEN_SANDBOX_API_KEY=...
    uv run python scripts/opensandbox/cleanup_sandboxes.py                 # dry run, app=tmax-harbor-eval
    uv run python scripts/opensandbox/cleanup_sandboxes.py --kill
    uv run python scripts/opensandbox/cleanup_sandboxes.py --app my-run --kill
    uv run python scripts/opensandbox/cleanup_sandboxes.py --all-apps --older-than 3600 --kill
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opensandbox import SandboxManager  # noqa: E402
from opensandbox.config import ConnectionConfig  # noqa: E402
from opensandbox.models.sandboxes import SandboxFilter, SandboxState  # noqa: E402

from tmax_envs.opensandbox import DEFAULT_APP_NAME, DEFAULT_DOMAIN, DEFAULT_PROTOCOL  # noqa: E402


async def run(args) -> int:
    domain = args.domain or os.getenv("TMAX_OPENSANDBOX_DOMAIN") or DEFAULT_DOMAIN
    protocol = args.protocol or os.getenv("TMAX_OPENSANDBOX_PROTOCOL") or DEFAULT_PROTOCOL
    if "://" in domain:
        protocol, domain = domain.split("://", 1)
    config = ConnectionConfig(domain=domain, protocol=protocol, api_key=os.getenv("OPEN_SANDBOX_API_KEY"),
                              request_timeout=timedelta(seconds=120))
    metadata = None if args.all_apps else {"tmax_app": args.app}
    if not args.all_apps:
        print(f"listing sandboxes with tmax_app={args.app} on {protocol}://{domain}")
    else:
        print(f"listing ALL sandboxes on {protocol}://{domain} (--all-apps)")

    cutoff = None
    if args.older_than is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=args.older_than)

    manager = await SandboxManager.create(config)
    victims = []
    try:
        page = 1
        while True:
            result = await manager.list_sandbox_infos(SandboxFilter(metadata=metadata, page=page, page_size=100))
            infos = result.sandbox_infos
            if not infos:
                break
            for info in infos:
                state = info.status.state
                if state not in (SandboxState.RUNNING, SandboxState.PENDING) and not args.include_finished:
                    continue
                created = getattr(info, "created_at", None)
                if cutoff is not None and created is not None and created > cutoff:
                    continue
                meta = getattr(info, "metadata", None) or {}
                if args.all_apps and "tmax_app" not in meta and not args.include_untagged:
                    continue
                victims.append(info)
                print(f"  {info.id}  state={state}  created={created}  task={meta.get('tmax_task')}  trial={meta.get('tmax_trial')}")
            if len(infos) < 100:
                break
            page += 1
        print(f"{len(victims)} sandbox(es) matched")
        if not args.kill:
            print("dry run; pass --kill to terminate them")
            return 0
        killed = 0
        for info in victims:
            try:
                await manager.kill_sandbox(info.id)
                killed += 1
            except Exception as e:  # noqa: BLE001
                print(f"  failed to kill {info.id}: {e}")
        print(f"killed {killed}/{len(victims)}")
        return 0 if killed == len(victims) else 1
    finally:
        await manager.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", default=os.getenv("TMAX_OPENSANDBOX_APP_NAME") or DEFAULT_APP_NAME,
                        help="tmax_app metadata tag to match (default: %(default)s)")
    parser.add_argument("--all-apps", action="store_true", help="match every tmax-tagged sandbox regardless of app")
    parser.add_argument("--include-untagged", action="store_true", help="with --all-apps, also match sandboxes without a tmax_app tag")
    parser.add_argument("--include-finished", action="store_true", help="also list non-running sandboxes")
    parser.add_argument("--older-than", type=int, default=None, help="only sandboxes created more than N seconds ago")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--kill", action="store_true", help="actually kill (default: dry run)")
    args = parser.parse_args()
    if not os.getenv("OPEN_SANDBOX_API_KEY"):
        print("OPEN_SANDBOX_API_KEY is not set", file=sys.stderr)
        return 2
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
