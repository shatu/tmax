# Repeatable React cache validation

These CPU-only checks precede a new, separately labelled scored task series.
They do not run the model or grade its solution.

Retention belongs in the runner, not the task or its grader. Call
`await capture_react_manifests.capture(environment, output_directory, workdir)`
after agent execution and before verification, while the original sandbox is
alive. Use the actual pinned task WORKDIR. This records exact files and hashes;
missing optional files are explicit. Report capture errors separately without
rewriting the trial reward. Do not put a copy step into model-visible test code
or wait until after sandbox cleanup. A Harbor hook needs a closure over the
live environment; `TrialHookEvent` alone does not contain it.

1. Save each trial's final `package.json` and, when present, `package-lock.json`
   or `npm-shrinkwrap.json` under `manifests/<trial-id>/`. Retain exact bytes;
   do not substitute the reference manifest or alter model choices.
2. Replay against the candidate cache and pinned complete image:

   ```sh
   python replay_react_cache.py --image sha256:<image-id> \
     --cache /path/to/npm-cache --trials /path/to/manifests \
     --output /path/to/new-replay.jsonl
   ```

   Each install has no network, skips lifecycle scripts, uses a disposable copy
   of the manifests, and has a 120-second diagnostic limit. Input hashes, resolved
   image ID, exit code and complete npm output are retained per trial. This is
   not the task's verifier budget and does not change scored runs. Inspect failed
   installs: dependency conflicts or malformed model manifests are not cache
   misses, and successful installation is not proof the solution is correct.
3. Add missing *exact* registry versions through `prepare_react.py`'s repeatable
   `--cache-package` option, build the image, and rerun the same saved manifests.
   Only dependency caches reach the image; never copy a solved app into it.
   Preserve the final image/cache digest, since transitive dependency versions
   can change when rebuilding. Missing metadata can require additional package
   versions; a small list of first errors is not proof of complete coverage.

   The observed extra versions are in `react-cache-packages.txt`. To prepare a
   fresh complete context with those cache entries (Bash):

   ```bash
   cache_args=()
   while IFS= read -r package; do cache_args+=(--cache-package "$package"); done < react-cache-packages.txt
   python prepare_react.py --source "$PINNED_REACT_TASK" --output "$NEW_REACT_TASK" "${cache_args[@]}"
   docker build --platform linux/amd64 -t tblite-react-cache-candidate "$NEW_REACT_TASK/environment"
   ```

   Record the resulting image hash and validate at the destination; external
   resolution is not guaranteed to recreate our populated cache byte-for-byte.
4. Once the candidate passes a full task smoke on the target runtime, use one
   pinned task/image manifest and the existing campaign launcher for all 31
   models, five attempts each. Save its exact config/command alongside results.
   This is a full replacement task series, not selective retries of model zeros.
   Keep prior variants separate and preserve genuine failures and timeouts.

Hyak cache-only acceptance: four observed extra package versions installed
successfully with Apptainer networking disabled. That receipt and the local
17-test oracle pass do not yet establish coverage of all failed model manifests.

September 10 local follow-up: eight reconstructed manifests went from 2/8 to
8/8 successful network-disabled installs after seven further exact-version
cache additions. See BUILD_HANDOFF.md and the hash-indexed replay summary.
These reconstructed inputs still need checking against captured final files;
installation success is not model correctness.

The combined expanded-cache SIF subsequently passed the full Hyak Harbor
oracle: 17/17 assertions, reward 1.0, no exception, original resource limits,
and verified worker cleanup. See `evidence/react-expanded-cache/README.md` for
the exact image digest and raw results. Model reruns and retained final-manifest
coverage remain separate from this successful destination acceptance.
