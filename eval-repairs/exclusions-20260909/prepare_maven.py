import os
from pathlib import Path
root = Path(__file__).resolve().parent
task = root / 'tasks/maven-slf4j-conflict'
source = Path(os.environ['TBLITE_SOURCE']) / 'maven-slf4j-conflict'
test = (source / 'tests/test.sh').read_text()
start = test.index('apt-get update && apt-get install -y maven')
end = test.index('uv venv', start)
test = test[:start] + 'set -e\nexport MAVEN_ARGS=--offline UV_OFFLINE=1 UV_CACHE_DIR=/opt/uv-cache UV_LINK_MODE=copy\ncommand -v mvn\n\n' + test[end:]
test = test.replace('pytest "$TEST_DIR/test_outputs.py" -rA -v', 'set +e\npytest "$TEST_DIR/test_outputs.py" -rA -v', 1)
(task / 'tests/test.sh').write_text(test)
solve = (source / 'solution/solve.sh').read_text()
old = 'apt-get update && apt-get install -y maven'
assert solve.count(old) == 1
solve = solve.replace(old, 'export MAVEN_ARGS=--offline\ncommand -v mvn')
(task / 'solution/solve.sh').write_text(solve)
assert (source / 'tests/test_outputs.py').read_bytes() == (task / 'tests/test_outputs.py').read_bytes()
