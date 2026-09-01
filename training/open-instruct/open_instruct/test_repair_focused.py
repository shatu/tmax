"""Focused per-failure-mode tests for the tmax-private#1 repair.

One test per restored guard, complementing the behavioral oracle in
test_rollout_termination.py (oseyosey). Each test exercises exactly one
restoration so a future regression names its failure mode.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import os
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import ray

from open_instruct.data_types import EnvConfig, EnvConfigEntry
from open_instruct.environments.base import EnvCall, StepResult
from open_instruct.environments.swerl_vanillux_sandbox import SWERLVanilluxSandboxEnv
from open_instruct.vllm_utils import LLMRayActor, PoolSetup, SamplingConfig, process_request

TOOL = "swerl_vanillux_sandbox"
TOOL_CALL_TIMEOUT_S = 0.05
MAX_STEPS = 12
INSTANCE_NOT_STARTED = "Instance not started. Call start() first."


# ---------------------------------------------------------------- fakes (per test_rollout_termination.py)


class _Logprobs:
    token_logprobs = [0.0]


class _Choice:
    text = f'<tool_call>{{"name": "{TOOL}", "arguments": {{"command": "x"}}}}</tool_call>'
    token_ids = [7]
    logprobs = _Logprobs()
    finish_reason = "stop"


class _ApiResponse:
    choices = [_Choice()]


class _Completions:
    async def create(self, **_kwargs):
        return _ApiResponse()


class _Client:
    completions = _Completions()


class _ToolCall:
    def __init__(self, name):
        self.id = ""
        self.name = name
        self.args = {"command": "x"}


class _ParseResult:
    def __init__(self):
        self.tool_calls = [_ToolCall(TOOL)]
        self.had_tool_call = True


class _Parser:
    def parse_tool_calls(self, _text):
        return _ParseResult()


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text, **_kw):
        return [11] * max(1, min(len(text), 4))


class _ModelConfig:
    max_model_len = 4096


class _Engine:
    tokenizer = _Tokenizer()
    model_config = _ModelConfig()


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **kw):
        return self._fn(*a, **kw)


class _Pool:
    def __init__(self):
        self.released = []
        self.discarded = []
        self.release = _Remote(self._release)
        self.discard = _Remote(self._discard)

    def _release(self, actor):
        self.released.append(actor)

    def _discard(self, actor, reason=""):
        self.discarded.append((actor, reason))


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _Actor:
    def __init__(self, pool):
        self.client = _Client()
        self.tool_parser = _Parser()
        self.llm_engine = _Engine()
        self.model_name = "fake"
        self.server_port = 0
        self.tool_call_timeout = TOOL_CALL_TIMEOUT_S
        self.per_turn_max_tokens = None
        self.mask_tool_use = False
        self.tool_call_format_error_feedback = False
        self.pools = {TOOL: pool}
        self.active_tasks = {}
        self.completion_queue = _Queue()
        self.request_metadata = {
            "train_0": {
                "prompt_token_ids": [1, 2, 3],
                "active_tools": [TOOL],
                "env_config": EnvConfig(
                    max_steps=MAX_STEPS, env_configs={TOOL: EnvConfigEntry(env_name=TOOL, is_text_env=False)}
                ),
                "original_sampling_params": SamplingConfig(n=1),
            }
        }


def _pool_setup(pool, env):
    return PoolSetup(
        acquired={TOOL: (pool, env)},
        actor_map={TOOL: env},
        allowed_tools={TOOL},
        tool_response_roles={TOOL: "tool"},
        tool_call_format_error_messages={},
        active_env_names=[TOOL],
        text_env_names=[],
    )


async def _no_health(_port):
    return None


def _drive(pool, env, cancel_recorder=None):
    actor = _Actor(pool)
    setup = _pool_setup(pool, env)

    async def _pools(*_a, **_kw):
        return setup

    cancel = cancel_recorder if cancel_recorder is not None else (lambda *a, **k: None)
    with (
        patch("open_instruct.vllm_utils._check_health", _no_health),
        patch("open_instruct.vllm_utils._acquire_and_reset_pools", _pools),
        patch("ray.cancel", cancel),
        patch("open_instruct.vllm_utils.process_tool_tokens", lambda *a, **kw: ([9], [0.0], [0], 0)),
    ):
        asyncio.run(process_request(actor, "train_0_1", SamplingConfig(n=1, max_tokens=64)))
    return actor


# ---------------------------------------------------------------- vllm_utils-side tests


class _HangingEnv:
    def __init__(self):
        self.step = _Remote(self._step)
        self.get_metrics = _Remote(self._get_metrics)

    async def _step(self, _call: EnvCall) -> StepResult:
        await asyncio.sleep(TOOL_CALL_TIMEOUT_S * 20)
        return StepResult(result="never reached")

    async def _get_metrics(self):
        return {}


class TestProductionCancellation(unittest.TestCase):
    """The timeout path must ray.cancel(step_ref, force=True) — a controlled
    recorder substitute proves invocation without needing a Ray cluster."""

    def test_timeout_invokes_force_cancel(self):
        calls = []

        def recorder(ref, *args, **kwargs):
            calls.append((ref, kwargs.get("force", False)))

        pool, env = _Pool(), _HangingEnv()
        _drive(pool, env, cancel_recorder=recorder)

        self.assertEqual(len(calls), 1, f"ray.cancel invocations: {calls}")
        ref, force = calls[0]
        self.assertIsNotNone(ref)
        self.assertTrue(force, "ray.cancel must be called with force=True")


class _DeadMetricsEnv:
    """Steps fine once (done), then the actor is dead by metrics time."""

    def __init__(self):
        self.step = _Remote(self._step)
        self.get_metrics = _Remote(self._get_metrics)

    async def _step(self, _call: EnvCall) -> StepResult:
        return StepResult(result="ok", reward=0.0, done=True)

    async def _get_metrics(self):
        raise ray.exceptions.RayActorError()


class TestDeadHandleDisposition(unittest.TestCase):
    """An actor dying during metrics must not abort finalization (the completion
    still lands) and must be discarded, never released back to the pool."""

    def test_discard_not_release_and_completion_enqueued(self):
        pool, env = _Pool(), _DeadMetricsEnv()
        actor = _drive(pool, env)

        self.assertEqual(len(actor.completion_queue.items), 1, "completion must still be enqueued")
        self.assertEqual(pool.released, [], "dead handle must not be released")
        self.assertEqual(len(pool.discarded), 1, "dead handle must be discarded")


# ---------------------------------------------------------------- env-side tests


class _CmdResult(types.SimpleNamespace):
    pass


def _bare_env(backend):
    env = SWERLVanilluxSandboxEnv.__new__(SWERLVanilluxSandboxEnv)
    env._backend = backend
    env._task_id = "t0"
    env._step_count = 0
    env._max_steps = 10
    env._last_step_warning = False
    env._append_turns_remaining = False
    return env


class TestResetFailFast(unittest.TestCase):
    """_prepare_vanillux_runtime must raise when init or chmod exits nonzero
    instead of nominally succeeding over a torn sandbox."""

    def test_init_failure_raises(self):
        backend = types.SimpleNamespace(
            run_command=lambda _cmd: _CmdResult(exit_code=1, stdout="", stderr="torn"), write_file=lambda *_a: None
        )
        env = _bare_env(backend)
        with self.assertRaisesRegex(RuntimeError, "initialize Vanillux runtime"):
            env._prepare_vanillux_runtime()

    def test_chmod_failure_raises(self):
        results = iter(
            [_CmdResult(exit_code=0, stdout="", stderr=""), _CmdResult(exit_code=1, stdout="", stderr="chmod failed")]
        )
        backend = types.SimpleNamespace(run_command=lambda _cmd: next(results), write_file=lambda *_a: None)
        env = _bare_env(backend)
        with self.assertRaisesRegex(RuntimeError, "bash wrapper"):
            env._prepare_vanillux_runtime()


class TestInstanceNotStartedTerminal(unittest.TestCase):
    """A backend torn down mid-episode ends the episode cleanly (reward 0,
    done, backend_unavailable) instead of propagating out of step()."""

    def test_step_returns_clean_terminal(self):
        def _raise(_cmd):
            raise RuntimeError(INSTANCE_NOT_STARTED)

        env = _bare_env(types.SimpleNamespace(run_command=_raise))
        result = asyncio.run(env.step(EnvCall(id="1", name="bash", args={"command": "x"})))
        self.assertTrue(result.done)
        self.assertEqual(result.reward, 0.0)
        self.assertTrue(result.metadata.get("backend_unavailable"))
        self.assertTrue(result.metadata.get("timeout"))

    def test_other_runtime_errors_still_raise(self):
        def _raise(_cmd):
            raise RuntimeError("something else entirely")

        env = _bare_env(types.SimpleNamespace(run_command=_raise))
        with self.assertRaisesRegex(RuntimeError, "something else"):
            asyncio.run(env.step(EnvCall(id="1", name="bash", args={"command": "x"})))


class TestExitCode124Terminal(unittest.TestCase):
    """A sandbox-side command timeout (exit 124) is a distinct terminal timeout
    observation, not ordinary output."""

    def test_124_ends_episode(self):
        backend = types.SimpleNamespace(
            run_command=lambda _cmd: _CmdResult(exit_code=124, stdout="partial", stderr="")
        )
        env = _bare_env(backend)
        result = asyncio.run(env.step(EnvCall(id="1", name="bash", args={"command": "sleep 999"})))
        self.assertTrue(result.done)
        self.assertEqual(result.reward, 0.0)
        self.assertTrue(result.metadata.get("timeout"))
        self.assertEqual(result.metadata.get("exit_code"), 124)


class TestTarOwnership(unittest.TestCase):
    """_upload_directory must force uid/gid 0 and root names on every member."""

    def test_all_members_root(self):
        captured = {}
        backend = types.SimpleNamespace(put_archive=lambda _path, data: captured.update(data=data))
        env = _bare_env(backend)

        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "sub"))
            with open(os.path.join(d, "seed.txt"), "w") as f:
                f.write("x")
            with open(os.path.join(d, "sub", "verify.sh"), "w") as f:
                f.write("#!/bin/bash\n")
            env._upload_directory(d, "/tests")

        with tarfile.open(fileobj=io.BytesIO(captured["data"])) as tar:
            members = tar.getmembers()
        self.assertGreaterEqual(len(members), 3)
        for m in members:
            self.assertEqual((m.uid, m.gid, m.uname, m.gname), (0, 0, "root", "root"), m.name)


class _FaultingEnv:
    """Raises a generic (non-timeout) exception on the FIRST call — the
    worker-disappearance shape: no preceding tool-call timeout, so the generic
    except gate is the first thing that touches the failure."""

    def __init__(self):
        self.step = _Remote(self._step)
        self.get_metrics = _Remote(self._get_metrics)

    async def _step(self, _call: EnvCall) -> StepResult:
        raise RuntimeError("Lost connection to sandbox step")

    async def _get_metrics(self):
        return {}


class TestNonTimeoutFaultTerminal(unittest.TestCase):
    """Repair r2 regression (11082494): a mid-step fault with NO preceding
    timeout must terminate the rollout via the classification gate — and must
    not die on a missing actor attribute."""

    def test_first_call_generic_fault_terminates(self):
        pool, env = _Pool(), _FaultingEnv()
        actor = _drive(pool, env)
        self.assertEqual(len(actor.completion_queue.items), 1, "rollout must still complete")


class TestFormatFeedbackProducerRestored(unittest.TestCase):
    """The gate consumes actor.tool_call_format_error_feedback; repair r1
    restored the consumer without the producer and the real LLMRayActor lacked
    the attribute (AttributeError on first mid-step fault, trainer down).
    Fakes can mask that class of bug, so this checks the REAL class."""

    def test_ctor_param_and_assignment_exist(self):
        sig = inspect.signature(LLMRayActor.__init__)
        self.assertIn("tool_call_format_error_feedback", sig.parameters)
        self.assertIs(sig.parameters["tool_call_format_error_feedback"].default, False)
        self.assertIn(
            "self.tool_call_format_error_feedback = tool_call_format_error_feedback",
            inspect.getsource(LLMRayActor._init_config),
        )


if __name__ == "__main__":
    unittest.main()
