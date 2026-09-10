"""Bounded Docker regression checks for a prepared ETL task (not model scores)."""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def check(task, image, case):
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--cpus",
        "1",
        "--memory",
        "2g",
        "--entrypoint",
        "bash",
    ]
    for source, target in (
        ("environment/start_postgres.sh", "/workspace/start_postgres.sh"),
        ("tests/test.sh", "/tests/test.sh"),
        ("tests/test_outputs.py", "/workspace/tests/test_outputs.py"),
    ):
        command += ["-v", f"{task / source}:{target}:ro"]
    script = "set -e\n"
    if case == "warm":
        script += "/workspace/start_postgres.sh true\n"
        # Old marker files must not bypass initialization or cause failure.
        script += "touch /tmp/.etl_bootstrap_ready /tmp/.etl_bootstrap_failed\n"
    if case == "concurrent":
        script += """/workspace/start_postgres.sh bash -c 'touch /tmp/lock_held; sleep 2; test ! -e /tmp/second_init; echo LOCK_SERIALIZATION_PASS' &
first=$!
until [ -f /tmp/lock_held ]; do sleep 0.1; done
/workspace/start_postgres.sh touch /tmp/second_init &
second=$!
wait "$first"
wait "$second"
test -f /tmp/second_init
"""
    elif case == "failure":
        script += "printf '%s\\n' 'raise RuntimeError(\"ETL_TEST_INIT_FAILURE\")' > /workspace/scripts/setup_databases.py\n"
        script += "if bash /tests/test.sh; then exit 99; fi\n"
        script += "test ! -e /logs/verifier/reward.txt\n"
    else:
        # Diagnostic reference fix, applied only inside the disposable container.
        script += (
            'sed -i "s/if False:/if True:/" /workspace/app/migration_orchestrator.py\n'
        )
        script += "bash /tests/test.sh\n"
        script += 'test "$(cat /logs/verifier/reward.txt)" = 1\n'
    result = subprocess.run(
        command + [image, "-c", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    print(f"=== {case} exit={result.returncode} ===\n{result.stdout}", flush=True)
    if result.returncode:
        raise RuntimeError(f"{case} regression failed")
    if case == "concurrent":
        assert "LOCK_SERIALIZATION_PASS" in result.stdout
    elif case == "failure":
        assert "ETL_TEST_INIT_FAILURE" in result.stdout
        assert "test session starts" not in result.stdout
    else:
        assert "5 passed" in result.stdout
        assert result.stdout.index("Test data populated") < result.stdout.index(
            "test session starts"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--image", required=True, help="Existing complete ETL image")
    args = parser.parse_args()
    # Dockerfile also sets this permission when building the packaged task.
    (args.task / "environment/start_postgres.sh").chmod(0o755)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(
            executor.map(
                lambda case: check(args.task.resolve(), args.image, case),
                ("cold", "warm", "failure", "concurrent"),
            )
        )
