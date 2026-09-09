"""Run with TBLITE_SOURCE set to the pinned, complete upstream task corpus."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class PreparationTest(unittest.TestCase):
    def test_preserves_tasks_and_records_final_hashes(self):
        source = Path(os.environ["TBLITE_SOURCE"])
        scripts = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("prepare.py", "prepare_maven.py", "extract_dependency_pom.py"):
                shutil.copyfile(scripts / name, root / name)
                subprocess.run([sys.executable, str(root / name)], check=True)
            for name in (
                "maven-slf4j-conflict",
                "breast-cancer-mlflow",
                "okhttp-trailers-crash",
            ):
                task = root / "tasks" / name
                hashes = json.loads((root / f"{name}-files.json").read_text())
                actual = {
                    str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in task.rglob("*")
                    if p.is_file()
                }
                self.assertEqual(hashes, actual)
                allowed = (
                    {"tests/test.sh"} if name != "okhttp-trailers-crash" else set()
                )
                if name == "maven-slf4j-conflict":
                    allowed.add("solution/solve.sh")
                for path in (source / name).rglob("*"):
                    relative = path.relative_to(source / name)
                    if path.is_file() and str(relative) not in allowed:
                        self.assertEqual(
                            path.read_bytes(),
                            (task / relative).read_bytes(),
                            str(relative),
                        )
                subprocess.run(["bash", "-n", str(task / "tests/test.sh")], check=True)
                subprocess.run(
                    ["bash", "-n", str(task / "solution/solve.sh")], check=True
                )
            original = (source / "maven-slf4j-conflict/solution/solve.sh").read_text()
            prepared = (
                root / "tasks/maven-slf4j-conflict/solution/solve.sh"
            ).read_text()
            self.assertEqual(
                prepared,
                original.replace(
                    "apt-get update && apt-get install -y maven",
                    "export MAVEN_ARGS=--offline\ncommand -v mvn",
                ),
            )
            self.assertIn("<project", (root / "dependency-pom.xml").read_text())
            repeat = subprocess.run(
                [sys.executable, str(root / "prepare.py")], capture_output=True
            )
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn(b"FileExistsError", repeat.stderr)


if __name__ == "__main__":
    unittest.main()
