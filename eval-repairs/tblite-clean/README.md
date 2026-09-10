# tblite-clean (Harbor dataset release tooling)

Build a portable snapshot from the complete pinned upstream corpus and reviewed
repair directories. The package contains full task directories and a manifest;
it does not modify the source corpus or publish anything externally.

```sh
python package.py --source /path/to/pinned/upstream \
  --repairs /path/to/validated/tasks --plan /path/to/release-plan.json \
  --output /path/to/new/tblite-clean
```

The release plan enumerates **every** source task under `tasks`, records its
`source_files` SHA256 map, and assigns one status:

- `unchanged`: copied byte-for-byte; not a claim of independent validation.
- `validated-repair`: requires `evidence`, `image_sha256`, and an exact `changes`
  mapping from relative path to `{before: SHA256-or-null, after: SHA256-or-null}`.
- `excluded`: requires a reason. ACL remains explicitly excluded.

The top-level `source_revision` identifies the upstream commit. Pending repairs
are rejected, as are source drift, unlisted changes, symlinks, existing output
directories, and changes to instructions, task resources or Python assertions.
Task TOML may add only the four fixed offline verifier environment entries
(`UV_OFFLINE`, `PIP_NO_INDEX`, `PIP_FIND_LINKS`, `UV_LINK_MODE`) used by the
received bootstrap bundle; existing environment values, time limits and all
other configuration must remain unchanged. These changes still require exact
before/after hashes in the reviewed repair manifest.
Reviewed bootstrap scripts and image recipes may change, but must be listed in
the repair manifest. Receipt references must be portable before public release;
do not include credentials or private cluster metadata. A release plan is a
reviewed input, not an automatically generated assertion of validity.

Keep image recipes in each task's environment directory, reference immutable
image digests, and retain the validated OCI environment/working-directory data.
Do not include a repaired application or oracle outputs in the runtime image.
Different package/image manifests are different evaluation variants; historical
scores are not rewritten or silently combined with repaired variants.

Release gate: clean fixed-dependency provisioning failures must be distinguished
from grader/model failures without classifying arbitrary grader text. Both the
positive oracle and negative/error-path tests must pass on the target runtime.
The narrow Maven/OkHttp exit-status correction is merged in TMAX PR #8 and
passed fresh full-Harbor oracle runs. The composed image contexts still require
build validation. Broad guard work is deferred and is not a release dependency.
This directory is packaging tooling, not the completed dataset release.

`prepare_etl.py` creates a full candidate ETL task with TCP endpoints and
synchronous initialization. Startup accepts an optional command: it completes
database/schema/seed initialization and runs that command under the same lock.
The verifier uses this to run the unchanged pytest/reward sequence without a
background initializer racing it. With no command, startup retains its original
keepalive behavior and releases the initialization lock before waiting forever.
The database server does not inherit the lock. Existing servers/databases are
accepted; original schema and seed scripts remain unchanged.

There are no readiness markers, source rewrites during grading, or additional
time budgets. Startup errors retain their full output and exit without producing
a misleading model reward. Instructions, application source, assertions, oracle,
Dockerfile and task resources are copied unchanged. This supersedes the
endpoint-only variant, whose original verifier had an initialization race.

Run `python validate_etl.py --task /path/to/prepared/etl --image <complete-etl-image>`
for offline Docker cold/warm reference-fix and bootstrap-failure checks. These
diagnostics are not model evaluation scores or target-cluster acceptance.

Run local packaging tests with `python -m unittest discover -s .`.

## Candidate validation (not a dataset release)

- Synchronous ETL candidate: offline Docker checks at 1 CPU/2 GiB passed
  cold and fully initialized cases (all five unchanged assertions, 18.54s and
  18.41s), with the reference conditional fix applied only in the disposable
  test container. Deliberate initialization failure retained its traceback,
  ran no pytest and wrote no reward. Concurrent initializers were serialized
  through the verification command. Base image
  `00abffa0bce5ace8a80dbf37b4ca6acd0abae4b79286d72a77d9d7eb4c3ec7cd`,
  with the candidate startup/verifier bind-mounted; this is not a newly built
  image identity. Oscar subsequently reported target-runtime real-agent smoke
  5/5 with no exceptions (issue #1 comment 5622966465). The full series has
  incomplete coverage; see REMAINING.md rather than treating smoke as scores.
- Historical ETL candidate (with the now-removed verifier gate), Docker image
  `00abffa0bce5ace8a80dbf37b4ca6acd0abae4b79286d72a77d9d7eb4c3ec7cd`:
  unchanged oracle passed all five assertions in 18.21 seconds locally.
  Hyak Harbor job 39936627 **did not pass**: PostgreSQL reported
  `pg_ctl: cannot be run as root` after `su postgres`; no reward was produced.
  Image identity and 1 CPU/2048 MiB limits were recorded correctly. Worker
  39936641 is terminal, pool deleted, and the three temporary tokens removed.
  Root-mapped fakeroot user switching needs resolution; do not mark this task
  validated on Hyak based on the Docker test. These receipts do not validate
  the current gate-free candidate. It needs a fresh real-agent smoke on a
  supported runtime, including already-running PostgreSQL and agent-edited
  startup files; check upstream startup/test timing rather than assuming it safe.
- `prepare_hydra.py` appends a self-contained mirror/wheelhouse image recipe.
  The oracle's git URL is unchanged and redirects to the pinned upstream
  commit; the verifier installs the built wheel for that same commit.
  Full Docker image `8aa2e1b1f0f7403935450adbe1083680c0fd5617d351e81a45073248f9386661`
  passed the unchanged oracle and all seven assertions with `--network none`
  (pytest 1.38 seconds). Hyak full Harbor job 39937179 also passed: reward 1.0,
  no exception, seven tests in 2.25 seconds. SIF SHA256
  `bccef31fa30eb5c16b8eb788ebccdbce3d63c6d586dc477fa4358210ef045080`,
  context `f9a610f577f1f01a9da98e2832d5a905b5c174b47645f67f4182d30f7eff8e8a`.
  Recorded 1 CPU/2048 MiB, 10240 MiB nominal overlay (10568986624 total bytes,
  9941266432 initially available). Worker 39937232 terminal, pool deleted,
  temporary tokens removed.
  This candidate is separate from Oscar's earlier Hydra image/receipts.
- `prepare_react.py` builds the original app image plus a dependency-only npm
  cache stage and staged verifier tools. Only that stage's cache reaches the
  runtime, not its reference dependency manifest or application modifications.
  There is no model-manifest preflight or arbitrary-output classifier. Node
  20.19.6/npm 10.9.2 are verified at build time; original grader is retained.
  Image `46a48dc57c8c8872888a4c3bdc5b18bc62bc7e67dd1578a5d861728e939805d4`
  passed the unchanged oracle and all 17 assertions with Docker networking
  disabled (34.45 seconds, exploratory 1 CPU/2 GiB). Full Harbor validation at
  the task's 2 CPU/4096 MiB limits passed in job 39937635: reward 1.0, no
  exception, 17 assertions in 302.53 seconds, inside the original 900s budget.
  See `evidence/react/` for the SIF/resource/result/cleanup receipts.
  That oracle-only cache is not sufficient for every model-selected dependency.
  `prepare_react.py --cache-package package@exact-version` can now add dependency
  closures in separate build-only directories. Repeat the flag for each extra
  version; only npm's cache is copied to the runtime, never those directories,
  their manifests, or installed application files. Installation scripts are
  disabled during warming. The original application and scored assertions stay
  unchanged. Preserve the final image digest because transitive resolution can
  change between builds.

  Observed missing tarballs from the failed-model artifact snapshot `95249aa`
  can be warmed with these additional arguments:

  ```sh
  --cache-package @types/react@18.3.10 \
  --cache-package @types/react-dom@18.3.0 \
  --cache-package @types/react-dom@18.3.5 \
  --cache-package identity-obj-proxy@3.0.0
  ```

  Independent reproduction: the earlier image fails an offline install of
  `@types/react@18.3.10` with the same ENOTCACHED tarball error. After warming
  these four dependency closures, all four installs succeed with Docker
  networking disabled. The unchanged oracle and all 17 assertions also pass
  (26.14 seconds, 2 CPUs/4 GiB) with only the extended cache mounted over the
  earlier image. This is a cache diagnostic, not a rebuilt-image acceptance or
  model score. Other published failures request package metadata already present
  in our cache; exact failed model manifests/lockfiles still need replay. No
  finite cache guarantees arbitrary npm choices, and no failed-model manifest
  should be rewritten to make it fit the cache.
- The composed MLflow context builds successfully as image
  `3de102f7e0d417385adc05ebc93bf91f5b91ff7c9468204699a162d346c513ec`.
  This build alone does not substitute for its earlier validated SIF or certify
  new model scores.
  Its unchanged oracle and 34 assertions also passed with Docker networking
  disabled (pytest 9.35s, 2 CPUs/4 GiB). The oracle's curl-based health polling
  logged missing curl; subsequent Python API checks passed. The recipe now stages
  curl. Rebuilt image
  `2bcbd592af623063eb4ba0e2fc659c1aec06163af9b9e629853ca4cb6f0be67a`
  again passed the unchanged oracle and all 34 assertions offline (pytest 7.36s,
  reward 1.0). Earlier SIFs are retained with their original identities.
- The composed Maven context builds as
  `b86f57d3d459241577528e3095c694be643f01f8b89f4ef517ea2734d4f7c839`
  and passed the unchanged oracle plus ten verifier assertions offline in Docker
  (37.34 seconds, 2 CPUs/4 GiB). This is independent evidence for the composed
  recipe, not an identity claim with the earlier SIF.
- The composed OkHttp context builds as
  `37808077013275204070e39bd6f17c7d7be6561d31994981387df7117f62806c`.
  Its dependency-cache compilation and runtime-cache warmup succeeded. Its
  earlier SIF acceptance does not by itself validate this newly built image.
  This composed image independently passed the unchanged oracle and verifier
  offline in Docker at 4 CPUs/8 GiB: Gradle BUILD SUCCESSFUL in 2m40s,
  55 build tasks, reward 1.0.
