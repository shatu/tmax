# Offline exclusion repairs — 2026-09-09

These are labelled evaluation variants, not changes to task assertions or a
retroactive reclassification of model scores. Start with the complete upstream
TB-Lite task directories at `f075e463` (including environment, solution, tests,
instruction.md, and task.toml). Do not stage a repaired application in an image.

Original source: [open-thoughts/OpenThoughts-TBLite at
f075e463472c7790b85793b392dff1fff20cc0e3](https://github.com/open-thoughts/OpenThoughts-TBLite/tree/f075e463472c7790b85793b392dff1fff20cc0e3).
Check out that exact commit in a fresh clone and use its repository root as
`TBLITE_SOURCE`; do not use already-modified campaign task directories.

## Reproduce the images and bootstrap variants

From this directory, set `TBLITE_SOURCE` to the pinned upstream task root:

```sh
export TBLITE_SOURCE=/absolute/path/to/pinned/tasks
python test_prepare.py
python prepare.py
python prepare_maven.py
python extract_dependency_pom.py
docker build --platform linux/amd64 -t tblite-repair-maven:20260909 "$TBLITE_SOURCE/maven-slf4j-conflict/environment"
docker build --platform linux/amd64 -t tblite-repair-mlflow:20260909 "$TBLITE_SOURCE/breast-cancer-mlflow/environment"
docker build --platform linux/amd64 -f Maven.Dockerfile -t tblite-repair-maven-offline:20260909 .
docker build --platform linux/amd64 -f MLflow.Dockerfile -t tblite-repair-mlflow-offline:20260909 .
docker build --platform linux/amd64 -t tblite-repair-okhttp:20260909 "$TBLITE_SOURCE/okhttp-trailers-crash/environment"
docker build --platform linux/amd64 -f OkHttp.Dockerfile -t tblite-repair-okhttp-offline:20260909 .
```

### Offline failures versus model failures

Maven and OkHttp now capture verifier output and abort with `SETUP-FAILED`
(exit 90, without writing a reward) for explicit Maven/Gradle offline-cache
diagnostics. Ordinary compilation errors and failed assertions preserve the
original verifier status and reward logic. This is a narrow diagnostic guard,
not a claim that every possible provisioning failure is classified. MLflow's
dependency installs already run under `set -e` before grading.

The guard has positive/negative shell regression tests. The earlier full-oracle
receipts below validate the images and original grading paths; they predate this
guard and are not a live acceptance test of its new failure path. Historical
scores are unchanged.

Preparation refuses to overwrite existing task directories. The file manifests
record the prepared files; `prepare_maven.py` refreshes Maven's manifest after
its bootstrap changes. Record these hashes alongside every image/run.

The Maven build stage resolves dependencies for both POMs and explicitly caches
Surefire's JUnit4 provider. Only `/root/.m2/repository` crosses into the final
image: the repaired POM and build-stage filesystem do not. Maven's reference
solution still makes the application repair during evaluation. Its initial apt
bootstrap is replaced by a check for the preinstalled executable and offline mode.

MLflow's verifier installs pytest first, then its scientific/MLflow dependencies.
That sequence needs both packaging 26.3 and 24.2 cached: copying only the final
environment's packages misses the first-stage version. The image build preserves
both stages. Tests install into their own fresh venv with networking disabled.
The unchanged MLflow solution expects the original Docker entrypoint to start the
tracking server. For Apptainer instance mode, preserve that initialization using:

```text
%startscript
    exec /app/start.sh sleep infinity
```

Do not confuse a Docker-only pass with Sandfleet/Harbor acceptance. Use the full
generated task directory, exact SIF hash/OCI environment and WORKDIR, Sandfleet
PR10 at least c0bdf83, and Harbor 0.6.6. Keep task CPU/RAM/time limits; report
nominal disk and actual available space separately. Never turn bootstrap or
cleanup exceptions into model reward zero.

## Verified results and remaining work

### Export to an Apptainer-only site

Build the images above on a Docker-capable machine, then export each with
`docker save -o maven-offline.docker.tar tblite-repair-maven-offline:20260909`
(substitute `mlflow` or `okhttp` for the other images). Transfer the uncompressed
Docker archives and prepared task directories through an authorized private
route. Record and compare SHA256 before and after transfer. No Docker daemon
is needed on the destination:

```sh
export APPTAINER_TMPDIR=$(mktemp -d /tmp/tblite-build.XXXXXXXX)
export APPTAINER_CACHEDIR="$APPTAINER_TMPDIR/cache"
apptainer build --fakeroot maven.sif docker-archive:///absolute/path/maven-offline.docker.tar
apptainer build --fakeroot okhttp.sif docker-archive:///absolute/path/okhttp-offline.docker.tar
```

For MLflow, create a definition file with the following content and run
`apptainer build --fakeroot mlflow.sif mlflow.def`:

```text
Bootstrap: docker-archive
From: /absolute/path/mlflow-offline.docker.tar

%startscript
    exec /app/start.sh sleep infinity
```

Use new output paths, check build exit status, then hash each SIF. Run conversion
in an allocation allowed by site policy, with node-local temporary space. Keep
OCI `Config.Env`, `WorkingDir` and `User` from `docker image inspect` alongside
the archive for the Harbor image manifest. Record the base/final image IDs,
`mvn -version`, installed package versions (`dpkg-query -W`), and Gradle JDK
`release` files alongside run evidence to diagnose build-version differences.
The full original corpus is required; do not run preparation over another site's
already-modified task copies. OkHttp warms only its verifier target's external
runtime artifacts; local project outputs must still be rebuilt during evaluation.

### Acceptance receipts

- **Maven:** local network-none Docker oracle 10/10; full Hyak Sandfleet/Harbor
  oracle 10/10 in 29.32s, reward 1.0, no exceptions. Driver 39897571, worker
  39897658 terminal, pool deleted and temporary credentials removed. SIF
  `c9f25275317d3aeb9ac117cb295526137a5d904afd5de7d81dae5562d2b89fa1`.
  Original 2 CPUs/4096 MiB RAM/8192 MiB nominal disk, agent 600s/verifier 300s.
  Final image POM SHA256 matches upstream:
  `e95f996e4f1f23c65f2f29f9a151fb67821c4b41f95f54b0a59989285b4a4742`.
  `/app/service/target` and `/tmp/dependency-resolution` are absent in the image.
- **MLflow:** local network-none oracle 34/34; Hyak full oracle also 34/34 in
  27.78s, reward 1.0, but teardown reports an instance still listed after stop.
  Driver 39897329 subsequently deleted
  the pool; worker 39897655 is terminal and credentials were removed. SIF
  `5761a86257fbaf21c0e331e401b231848256a8b0668839478766d075b276b500`.
  Original 2 CPUs/4096 MiB RAM/10240 MiB nominal disk, 900s agent and verifier.
  **Clean rerun:** Sandfleet `773b12e`, driver 39898255, same SIF and limits:
  34 assertions pass in 28.47s, reward 1.0, `exception_info=null`. Worker
  39898498 is terminal CANCELLED, driver COMPLETED, pool deleted, queue empty
  and three temporary role-token files removed. The cleanup change waits
  boundedly for the instance listing after successful stop, retaining disk
  and raising if the instance remains listed.
- **OkHttp:** Python is absent in the original Docker image, so the reference
  solution exits 127. Adding Python gets past that failure, then offline Gradle
  fails on uncached compile and test-runtime JARs. The dependency-only cache
  repair now passes the unchanged reference solution and verifier in a fresh
  network-none Docker sandbox (2m52s Gradle, reward 1, exit 0), at the actual
  4 CPUs/8192 MiB, 1200s agent/300s verifier limits. Image SHA256:
  `345b706740ba2cfbdec868131bc24524b4561e3c8ee783974e6577192be64f9d`.
  Only Gradle caches/JDKs cross from the build stage, not application build
  outputs or repaired source. **Full Hyak Sandfleet/Harbor acceptance passed**:
  source `773b12e`, driver 39900827, worker 39901175, reward 1.0,
  `exception_info=null`. Verifier wall time 172.39s (Gradle 2m48s), inside the
  original 300s limit. Effective CPU set `32-33,35,37`, RAM 8589934592 bytes;
  nominal disk 10240 MiB, total 10568986624 bytes, initially available
  9941266432 bytes. SIF SHA256:
  `966e5f988ca03a2b1117016a2dabe25c2d01d7e7680712f01410b59f9a43e426`.
  Pool deleted, worker terminal CANCELLED, driver COMPLETED, queue empty and
  all three temporary role-token files removed.
- **ACL:** explicitly deferred and excluded from this repair/backfill scope.

New image builds can resolve unpinned transitive versions differently. Retain
image digests, dependency inventories and original task hashes for every run;
these recipes are not a claim of bit-for-bit reproducible external repositories.
