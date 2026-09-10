# Current site handoff

## Maven and OkHttp: use the already-accepted path

Use the full tasks prepared by `../exclusions-20260909/prepare.py`,
`prepare_maven.py`, and `extract_dependency_pom.py`, and the exact Docker build
commands in that directory's README. These are the accepted recipes, not a new
composed-context candidate. Start with pristine upstream `f075e463472c7790b85793b392dff1fff20cc0e3`.
The corrected wrapper preserves verifier exit status and does not classify
model-controlled error text. Do not add the superseded phrase-based guard.

On our Hyak site the validated SIFs still exist at:

- `/gscratch/scrubbed/hamishiv/tblite-exclusion-repair.GVkFIZoB/maven-v1.sif`,
  SHA256 `c9f25275317d3aeb9ac117cb295526137a5d904afd5de7d81dae5562d2b89fa1`.
- `/gscratch/scrubbed/hamishiv/tblite-exclusion-repair.GVkFIZoB/okhttp-v1.sif`,
  SHA256 `966e5f988ca03a2b1117016a2dabe25c2d01d7e7680712f01410b59f9a43e426`.

Other sites should build/export/import as documented, record their resulting SIF
hashes and OCI ENV/WORKDIR, and run a fresh oracle smoke before model scoring.
No transfer from Oscar is required. Use merged Sandfleet `ad99c92b` or a later
reviewed pin with fresh workers. Keep original task resources: Maven 2 CPU,
4096 MiB RAM, 600s agent/300s verifier; OkHttp 4 CPU, 8192 MiB RAM,
1200s agent/300s verifier. Disk waivers remain explicitly labelled.

## React: capture exact manifests, then replay

Use runner-side `capture_react_manifests.py`, not task-side edits, as described
in REACT_REPLAY.md. Historical reconstructions from agent commands are useful
diagnostics but are not guaranteed to equal the final on-disk manifests.
Our eight reconstructed manifests exposed seven additional missing exact
packages. After filling those entries, all eight installed successfully with
network disabled and lifecycle scripts disabled (previously two of eight).
The input hashes and outcomes are in evidence/react/reconstructed-cache-replay.json.
The replay used the original complete image with a separately expanded cache
mounted at /opt/npm-cache, not a newly built combined SIF. The eleven total
extra versions (including four earlier additions) are in react-cache-packages.txt.
Build that complete candidate context and smoke it on the destination before
scoring. This is not complete scored coverage or a new accepted image identity.
