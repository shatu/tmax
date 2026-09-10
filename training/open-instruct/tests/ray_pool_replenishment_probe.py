"""Bounded real-Ray probe of the production pool's discard/replacement path."""

import ast
import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ray


def load_pool():
    path = Path(__file__).parents[1] / "open_instruct/environments/pool.py"
    tree = ast.parse(path.read_text())
    # Keep the real class, constructor, helpers and Ray decorators. Only the
    # unrelated application logger and Docker imports require lightweight stand-ins.
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("open_instruct"))
    ]
    namespace = {
        "__name__": "pool_replenishment_probe",
        "logger_utils": SimpleNamespace(setup_logger=lambda name: Mock()),
        "is_docker_host_connectivity_error": lambda error: False,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["EnvironmentPool"]


class ProbeEnvironment:
    def __init__(self):
        self.ready = False

    async def setup(self):
        self.ready = True

    async def step(self, fail=False):
        assert self.ready, "replacement did not execute setup"
        if fail:
            raise RuntimeError("injected step failure")
        return "ready"


async def probe():
    pool = load_pool().remote(pool_size=1, actor_class=ProbeEnvironment, acquire_timeout=10)
    generations = []
    try:
        for _ in range(2):
            actor = await asyncio.wait_for(pool.acquire.remote(), timeout=30)
            generations.append(actor._actor_id.hex())
            assert await actor.step.remote() == "ready"
            try:
                await actor.step.remote(fail=True)
            except ray.exceptions.RayTaskError:
                pass
            else:
                raise AssertionError("injected failure did not reach the caller")
            # Exercise duplicate requests while replacement setup is in flight.
            await asyncio.wait_for(
                asyncio.gather(pool.discard.remote(actor, "failed step"), pool.discard.remote(actor, "duplicate")),
                timeout=30,
            )
            assert await pool.size.remote() == 1
        actor = await asyncio.wait_for(pool.acquire.remote(), timeout=30)
        generations.append(actor._actor_id.hex())
        assert await actor.step.remote() == "ready"
        assert len(set(generations)) == 3
        assert await pool.size.remote() == 1
        await pool.release.remote(actor)
        print(json.dumps({"ray_version": ray.__version__, "generations": generations, "pool_size": 1}), flush=True)
    finally:
        ray.kill(pool, no_restart=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ray-pool-", dir="/tmp") as temporary:
        ray.init(
            address="local",
            num_cpus=2,
            include_dashboard=False,
            object_store_memory=80 * 1024 * 1024,
            _temp_dir=temporary,
        )
        try:
            asyncio.run(asyncio.wait_for(probe(), timeout=120))
        finally:
            ray.shutdown()
