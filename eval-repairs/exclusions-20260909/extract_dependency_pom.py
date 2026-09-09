"""Build-stage dependency manifest only; never copy the repaired POM to runtime."""
import os
from pathlib import Path
root = Path(__file__).resolve().parent
solve = (root / 'tasks/maven-slf4j-conflict/solution/solve.sh').read_text()
pom = solve.split("cat > pom.xml << 'EOF'\n", 1)[1].split('\nEOF', 1)[0]
(root / 'dependency-pom.xml').write_text(pom)
