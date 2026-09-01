"""Emit TOOL_CONFIGS JSON for the swerl_vanillux_sandbox tool — the single choke point.

Every launcher (production, 4-node, smokes) builds its tool config through this
script instead of a per-file heredoc, so the schema lives in exactly one place and
is asserted against the real dataclass before any training process starts.
The watchdog also runs this as a preflight before resubmitting, so schema drift
fails on the cpu watchdog node instead of on 64 allocated GPUs.

Reads env:
  SWERL_SANDBOX_BACKEND        required ("vmvm" hard-fails: backend removed at 61b5a85d)
  TASK_DATA_HF_REPO            required
  SWERL_SANDBOX_TEST_TIMEOUT   required (int)
  SWERL_SANDBOX_TIMEOUT        required (int)
  SWERL_SANDBOX_MEM_LIMIT      optional -> mem_limit
  TOOL_CONFIG_IMAGE            optional -> image
  TASK_DATA_DIR                optional -> task_data_dir
  SWERL_TOOL_PENALTY           optional float -> penalty
  SWERL_TOOL_CALL_FORMAT_ERROR_FEEDBACK / SWERL_TOOL_LAST_STEP_WARNING /
  SWERL_TOOL_APPEND_TURNS_REMAINING   optional bools ("1/true/yes/on")

Exit codes: 0 ok; 1 missing/invalid input; 2 unsupported (removed) backend; 3 schema assertion failure.
"""

import dataclasses
import json
import os
import sys

from open_instruct.environments.swerl_vanillux_sandbox import SWERLVanilluxSandboxEnvConfig


def _bool_env(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    backend = os.environ.get("SWERL_SANDBOX_BACKEND", "")
    if not backend:
        sys.exit(
            "emit_tool_configs: SWERL_SANDBOX_BACKEND must be set explicitly (no default; the old vmvm default is dead at tmax@61b5a85d)"
        )
    if backend == "vmvm":
        sys.stderr.write(
            "emit_tool_configs: FATAL: backend 'vmvm' no longer exists at tmax@61b5a85d "
            "(factory supports docker/apptainer/sandfleet; pwd/network/tenant/ttl fields removed). "
            "Set SWERL_SANDBOX_BACKEND explicitly.\n"
        )
        sys.exit(2)

    cfg: dict = {
        "backend": backend,
        "task_data_hf_repo": os.environ["TASK_DATA_HF_REPO"],
        "test_timeout": int(os.environ["SWERL_SANDBOX_TEST_TIMEOUT"]),
        "timeout": int(os.environ["SWERL_SANDBOX_TIMEOUT"]),
    }
    if os.environ.get("SWERL_SANDBOX_MEM_LIMIT"):
        cfg["mem_limit"] = os.environ["SWERL_SANDBOX_MEM_LIMIT"]
    if os.environ.get("TOOL_CONFIG_IMAGE"):
        cfg["image"] = os.environ["TOOL_CONFIG_IMAGE"]
    if os.environ.get("TASK_DATA_DIR"):
        cfg["task_data_dir"] = os.environ["TASK_DATA_DIR"]
    if os.environ.get("SWERL_TOOL_PENALTY"):
        cfg["penalty"] = float(os.environ["SWERL_TOOL_PENALTY"])
    for env_name, field in (
        ("SWERL_TOOL_CALL_FORMAT_ERROR_FEEDBACK", "tool_call_format_error_feedback"),
        ("SWERL_TOOL_LAST_STEP_WARNING", "last_step_warning"),
        ("SWERL_TOOL_APPEND_TURNS_REMAINING", "append_turns_remaining"),
    ):
        val = _bool_env(env_name)
        if val is not None:
            cfg[field] = val

    # The assertion hamishivi asked for: emitted keys must be real dataclass fields
    # of the pinned config, checked against the class itself, so future schema drift
    # dies here — before a single sandbox or training actor exists.
    allowed = {f.name for f in dataclasses.fields(SWERLVanilluxSandboxEnvConfig)}
    extra = set(cfg) - allowed
    if extra:
        sys.stderr.write(
            f"emit_tool_configs: FATAL: emitted keys not in SWERLVanilluxSandboxEnvConfig: {sorted(extra)}; "
            f"allowed: {sorted(allowed)}\n"
        )
        sys.exit(3)

    print(json.dumps(cfg))


if __name__ == "__main__":
    main()
