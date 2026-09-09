# Offline exclusion repairs — 2026-09-09

These are labelled evaluation variants, not changes to task assertions or a
retroactive reclassification of model scores. Start with the complete upstream
TB-Lite task directories at `f075e463` (including environment, solution, tests,
instruction.md, and task.toml). Do not stage a repaired application in an image.

## Reproduce the images and bootstrap variants

From this directory, set `TBLITE_SOURCE` to the pinned upstream task root:

```sh
export TBLITE_SOURCE=/absolute/path/to/pinned/tasks
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

Preparation refuses to overwrite existing task directories. The file manifests
written by `prepare.py` describe the initial copy; the subsequent Maven bootstrap
changes are explicit in `prepare_maven.py`. Record final hashes when publishing
an image/run, rather than treating those initial manifests as final.

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
  outputs or repaired source. Full Sandfleet/Harbor acceptance remains pending.
- **ACL:** remains with Rulin's backend/permissions investigation.

New image builds can resolve unpinned transitive versions differently. Retain
image digests, dependency inventories and original task hashes for every run;
these recipes are not a claim of bit-for-bit reproducible external repositories.
