"""Bounded CPU-only integration probe for Ray ObjectRef timeout handling."""

import asyncio
import json
import tempfile
import time
from types import SimpleNamespace

import ray
from test_training_step_drain import StepDrainTest


@ray.remote(num_cpus=1)
class BlockingEnvironment:
    async def ready(self):
        return True

    async def step(self, outcome):
        time.sleep(1.0 if outcome == "overrun" else 0.1)
        if outcome == "error":
            raise RuntimeError("injected worker execution failure")
        metadata = {"sandbox_lost": True} if outcome == "lost" else {}
        return SimpleNamespace(metadata=metadata, reward=0.5)


async def probe():
    results = []
    for outcome in ("success", "lost", "error", "overrun"):
        actor = BlockingEnvironment.remote()
        try:
            await actor.ready.remote()
            fixture = StepDrainTest()
            fixture.setUp()
            namespace = fixture.dispatch.__globals__

            async def wait(awaitable, timeout):
                return await asyncio.wait_for(awaitable, timeout=0.5 if timeout == 60 else timeout)

            namespace["step_timeout"] = 0.05
            namespace["asyncio"].wait_for = wait
            namespace["target"] = actor
            namespace["pool_setup"] = SimpleNamespace(acquired={"sandbox": (None, actor)})
            started = time.monotonic()
            await fixture.dispatch(actor.step.remote(outcome))
            elapsed = time.monotonic() - started
            assert bool(fixture.unsafe) == (outcome in {"error", "overrun"}), (outcome, fixture.unsafe)
            assert bool(fixture.rollout.info.get("infrastructure_failure")) == (outcome != "success")
            assert elapsed < 2, elapsed
            results.append(
                {
                    "outcome": outcome,
                    "elapsed_seconds": elapsed,
                    "quarantined": bool(fixture.unsafe),
                    "info": fixture.rollout.info,
                }
            )
        finally:
            ray.kill(actor, no_restart=True)
    print(json.dumps({"ray_version": ray.__version__, "results": results}), flush=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ray-drain-") as temporary:
        ray.init(
            address="local",
            num_cpus=2,
            include_dashboard=False,
            object_store_memory=80 * 1024 * 1024,
            _temp_dir=temporary,
        )
        try:
            asyncio.run(probe())
        finally:
            ray.shutdown()
