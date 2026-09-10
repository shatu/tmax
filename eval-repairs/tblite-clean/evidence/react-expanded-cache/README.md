# Expanded React cache: Hyak acceptance

The combined image passed the unchanged oracle and all 17 original assertions:
reward 1.0, `exception_info: null`, verifier tests 309.86 seconds. This is an
oracle acceptance check, not a model checkpoint score or proof that all possible
agent dependency edits work.

- Build job: 39974823; Harbor driver: 39974835; worker: 39975210.
- Sandfleet: merged `ad99c92` (0.7.3); no resource or timeout overrides.
- Task: 2 CPUs, 4096 MiB RAM, 10240 MiB nominal writable disk, 900-second
  agent and verifier limits. Nominal disk is not an available-free-space claim.
- SIF: `/gscratch/scrubbed/hamishiv/react-final.eG3nLQkn/react-v1.sif`
- SHA256: `e96be2bc5d4536f91e35f36ea0a4ee65efbe30e9ad97f8a60e4398c81185dae5`
- Base SIF SHA256: `41a773c300942740d21e460393c8dd6a84402e39b05fa4eb1bc5bc98d1bc99ac`
- Cache archive SHA256: `42b4ce9e8efe1818d09277a795cd6740ba1fe082a7d0508d6e9c30ebd9d667c5`

The build layers only expanded `cache/_cacache` into `/opt/npm-cache/` on the
accepted base SIF. It does not copy reconstructed agent source into the image.
The eight reconstructed dependency manifests also install offline successfully;
that separate diagnostic is recorded in `../react/reconstructed-cache-replay.json`.

Cleanup deleted the pool and cancelled worker 39975210. `sacct` confirmed the
driver COMPLETED 0:0, worker CANCELLED, batch/extern COMPLETED; `squeue` was empty
for both jobs. The three temporary role-token files were removed after this
terminal check. Raw trial result, original verifier output and cleanup response
are retained beside this receipt.
