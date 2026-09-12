# Optional dataset setup

The Vanillux training environment accepts an optional string `setup_command` in
per-row `env_config` (passed to `reset`). Missing, null, empty, or whitespace-only
commands do nothing. No image name or dataset name triggers implicit setup.

Setup runs after instruction/seed loading and shell initialization, before the
first action. Seeds are available under `/workspace`; verifier tests stay hidden
until submission. Setup uses the normal command timeout (`timeout`, default 120
seconds). A nonzero exit or timeout fails reset and follows its existing retry
policy; it is not a task reward. Failures report task, exit status, elapsed time,
and bounded diagnostic output.

Use Bash commands with explicit error propagation (`&&`, or `set -e`). Exported
variables and `cd` persist to agent actions. The verifier starts with a snapshot
of the successful setup environment and cwd, independent of later agent shell
changes. Do not use `exit`, `exec`, or background initialization on a successful
setup path: the wrapper must finish saving its shell state. Each reset reruns the
current row's command in the fresh sandbox. These commands execute trusted dataset
code, just like the dataset's verifier; there is no task-id interpolation by the
harness.

For example, a SWE-smith dataset producer can populate each row as follows:

```python
import shlex

ref = f"refs/remotes/origin/{task_id}^{{commit}}"
env_config["setup_command"] = (
    "set -e\n"
    "cd /testbed\n"
    f"commit=$(git rev-parse --verify --end-of-options {shlex.quote(ref)})\n"
    'git checkout --detach --force "$commit"\n'
    "test -x /opt/miniconda3/envs/testbed/bin/python\n"
    'export PATH="/opt/miniconda3/envs/testbed/bin:$PATH"\n'
    'printf "Task commit: %s\\n" "$commit"\n'
)
```

The dataset producer owns repository paths, interpreter selection, and ref
availability. Missing refs must be fixed in the dataset/images before launching;
this hook does not substitute refs or filter tasks. Existing SWE-smith rows need
this metadata added in a separately pinned dataset revision before using the hook.
