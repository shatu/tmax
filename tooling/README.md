# Pre-allocation image gate and audit tooling

Consolidated delta for `guard/fail-closed-sif`. Two things live here: the gate
that must pass before a 64-GPU allocation is requested, and the audit tooling
whose artifacts are under `run-evidence/`.

## Why this exists

Trainer `11096823` ended at step 78. 58 of the 1,170 task images it touched were
absent from the local SIF pool, and `prefer_local_sif()` answered a pool miss by
returning the bare image reference. The sandfleet worker then attempted a
per-lease `docker://` pull plus OCI→SIF conversion — 600–1200 s under restricted
egress, ending `exit=255`. Because the symptom was latency rather than a missing
file, it presented as a fleet-wide restart cliff (3.0 s → 639 s median within
five minutes) and cost a 6.5-hour 64-GPU run.

The fix is two-sided, and both sides are required:

| | mechanism | covers |
|---|---|---|
| **Before allocation** | `preflight/preflight.sh` | every image we can enumerate |
| **At runtime** | fail-closed guard in `apptainer_images.py` | every image we cannot |

The second row is not belt-and-braces, it is load-bearing. Replay-based
enumeration provably cannot bound the image set: on `11096823`, active sampling
filtered 46% of draws (`batch/filtered_prompts`=529 against 616 accepted, with
`no_resampled_prompts`=0), so the trainer reaches prompts the replay never saw.
A manifest audit alone would have given a green light and hit the same cliff.

## The gate

`preflight/preflight.sh`, wired into the launch wrapper ahead of any `sbatch`
(see `preflight/launch.sh.oscar-wrapper`). Non-zero exit means no allocation is
requested at all.

0. `SWERL_ALLOW_REMOTE_IMAGE_FALLBACK` is not set in the launching environment.
1. **Coverage** — all 14,488 manifest images resolve via the tree's *own*
   `prefer_local_sif()`, not a reimplementation, so the audit cannot drift from
   runtime behaviour.
2. **Non-empty** — no zero-byte SIFs.
3. **Size floor** — nothing under 10 MB. Added after a `du` reading of "2.0K"
   turned out to be a 131 MB file whose blocks had not flushed on Lustre. The
   real hole that scare exposed: non-empty + `inspect` would both accept a
   *truncated* image. Observed distribution is min 49.5 MB / p50 112.4 MB /
   max 3,449.8 MB, so 10 MB sits well below the floor of real images.
4. **`apptainer inspect` valid** — over all 14,488, not a sample. A sample is
   what would have missed the original defect: 58 bad out of 1,170 is a rate at
   which a 200-image sample usually reads clean.
5. **Attestation freshness** — the inspect reports must be newer than every file
   in the pool. An attestation that predates a pool mutation is worthless.
6. **Guard liveness** — the tree being launched is imported and a miss is
   actually provoked against an empty pool. Grep is insufficient: a partially
   applied patch can define `MissingLocalSifError` and never raise it.

Gates 1–4 run from `pool_audit.py`; gate 4 consumes the shard reports written by
`inspect_all.sh` (a 32-way array; `apptainer` is not on the login node and
14,488 inspects is a multi-minute parallel job, not something to run inline).

### Verified in both directions

A gate that has never been observed to fail is not known to be a gate.

```
unguarded launch tree  repos/cand-r3                  -> BLOCK, exit 1
    FAIL tree returns 'hamishi740/swerl-tmax-v3:0000preflightprobe'
         on a pool miss instead of raising
guarded tree           guard/fail-closed-sif          -> CLEAR, exit 0
    PASS fail-closed guard live: miss raises MissingLocalSifError
```

Both trees resolve all 14,488 identically, so the guard does not change the
happy path at scale — it only changes what a miss does.

**Open item this surfaced:** `cand-r3`, the tree `tmax_env.sh` currently points
`OPEN_INSTRUCT_DIR` at, does *not* carry the guard and is blocked by gate 6.
Relaunch must run from the guarded tree (i.e. after `guard/fail-closed-sif`
fast-forwards into `geomean_mask`), not from `cand-r3` as configured today.

## Current result

`run-evidence/pool-audit/` — 14,488/14,488 on every gate, 2.10 TB, `inspect`
`TOTAL ok=14488 bad=0` (job 11137452). Full per-image results in
`inspect-all-14488.jsonl.gz`; the audit reports the count it consumed so full
coverage is proven rather than inferred from an absence of failures.

## Audit tooling

- `audit/test_fail_closed_sif.py` — 6 regression tests for the guard. Red on
  canonical `da146695`, green with the guard.
- `audit/tool_audit.py` — per-rollout classifier for `11096823`.
- `audit/reward_csv.py` — the steps 1–100 reward CSV.

### Correction applied to the audit artifacts

The previous artifact carried classes derived from
`rollout_state.info.oom_killed` and `.sandbox_lost`. A key census settles it:
`request_info.rollout_state` is present in 19,712/19,712 records and its `info`
sub-dict likewise — but `info` carries exactly two keys, `env_name` and
`step_count`. Those two fields occur in **zero** records at any nesting level,
so both detectors were dead code, and a reported "0" from a detector that cannot
fire reads as *checked and clean* when it means *never checked*. They are
removed rather than left at zero. Terminal-state claims here are **timeout-only**
(`request_info.timeouts`, 2,419 rollouts; `rollout_state.timeout` agrees
exactly).

Class totals are **bit-identical** to the superseded artifact — the correction is
to what is claimed, not to what was counted. Rulin's 30/30 reproduction and the
observation that 15 apparent pass signals were model echo banners are unaffected
and preserved.

Two further additions, both prompted by the above:

- `schema_census` in the JSON records the keys present at *every* nesting level.
  A first attempt at this census looked only at the top level, concluded
  `rollout_state` did not exist at all, and so manufactured exactly the
  false-absence it existed to prevent. It now descends.
- `class_provenance` marks each class `TEXT_MATCH` or `STRUCTURED`. Without it a
  reader sees `INFRA:sandbox_oom 234` next to "no OOM claim is made" and cannot
  tell that the 234 is a regex hit on transcript text rather than a
  platform-reported event.
