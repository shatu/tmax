"""Create explicitly labelled task variants; never modify upstream task inputs."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ['TBLITE_SOURCE'])

for name in ('maven-slf4j-conflict', 'okhttp-trailers-crash', 'breast-cancer-mlflow'):
    source = UPSTREAM / name
    target = ROOT / 'tasks' / name
    shutil.copytree(source, target)
    original = (source / 'tests/test.sh').read_text()
    if name == 'breast-cancer-mlflow':
        start = original.index('# Install curl')
        end = original.index("# Check if we're in a valid working directory")
        modified = original[:start] + 'set -e\nexport UV_CACHE_DIR=/opt/uv-cache UV_OFFLINE=1 UV_LINK_MODE=copy\n\n' + original[end:]
        modified = modified.replace('uv run pytest /tests/test_outputs.py -rA', 'set +e\nuv run pytest /tests/test_outputs.py -rA', 1)
        (target / 'tests/test.sh').write_text(modified)
    assert (target / 'tests/test.sh').is_file()
    # Verifier assertions and solution are byte-identical; only bootstrap may differ.
    for path in source.rglob('*'):
        if path.is_file() and path.relative_to(source).as_posix() != 'tests/test.sh':
            assert path.read_bytes() == (target / path.relative_to(source)).read_bytes()
    receipt = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in target.rglob('*') if p.is_file()}
    (ROOT / f'{name}-files.json').write_text(json.dumps(receipt, indent=2))
