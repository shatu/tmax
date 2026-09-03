# Script inventory

Everything Oscar's side ran against this collaboration, so a reviewer can tell
which scripts produced posted evidence and which were throwaway. Requested by
Rulin as part of the consolidated delta.

Paths are relative to `/checkpoint/memorization/oscaryinn/tmax` on fair-sc.
**Vendored** = copied into this branch under `tooling/`. **Local** = stayed on
the cluster; named here so nothing is invisible.

## Load-bearing — produced evidence that was posted or gates the launch

| script | vendored | what it does |
|---|---|---|
| `scripts/preflight.sh` | ✔ `tooling/preflight/` | the hard pre-allocation gate (7 checks) |
| `probes/pool_audit.py` | ✔ `tooling/preflight/` | 14,488-image resolver audit via the tree's own `prefer_local_sif()` |
| `probes/inspect_all.sh` | ✔ `tooling/preflight/` | 32-way `apptainer inspect` over the whole pool |
| `scripts/launch.sh` | ✔ as `launch.sh.oscar-wrapper` | launch wrapper; shows where the gate is wired |
| `probes/patches/test_fail_closed_sif.py` | ✔ `tooling/audit/` | 6 regression tests for the guard |
| `tooling/audit/test_required_launch_files.sh` | ✔ (in-tree) | 7 neg/pos tests for preflight gate 7; regression for trainer 11141229 |
| `probes/dep_gate.py` | local — **deliberately NOT in the production path** | recursive launcher-dependency walker built during 11141229 triage. Correct in both directions, but it must reason about guarded sources and mentions-vs-executions, so a parser bug could block a good launch or pass a bad one. Gate 7 uses a fixed two-file list instead. Parked for the hardening pass (hamishivi ruling) |
| `probes/tool_audit.py` | ✔ `tooling/audit/` | per-rollout classifier; two `TESTS_FAILED` defects fixed (see `tooling/audit/tests/`) |
| `probes/exit0_adjudicate.py` | ✔ `tooling/audit/` | full-population apportionment of `exit0_with_failure_in_same_call`; asserts reconciliation against the published class count |
| `probes/calibrate_failed.py` | ✔ `tooling/audit/` | calibrates the test-verdict regex against the real corpus and prints the behavioural diff both ways |
| `probes/lostconn_incidents.py` | ✔ `tooling/audit/` | counts each `lost connection` phrasing separately and groups rows into incidents |
| `probes/patches/test_tests_failed_regex.py` | ✔ `tooling/audit/tests/` | 25 regression cases for the verdict detector, all drawn from real corpus text |
| `probes/scrub_bundle.py` | local | structural redaction of internal identifiers before publication |
| `probes/reward_csv.py` | ✔ `tooling/audit/` | steps 1–100 reward CSV, both step keys |
| `probes/curve_bundle.py` | ✔ `run-evidence/11096823/` | W&B history → PNG panels + tidy CSV/JSON |
| `probes/reward_curves.py` | ✔ `run-evidence/11096823/` | reward curves from the CSV |
| `probes/full_warm.sh` | local | 4-shard SIF build; deterministic disjoint shards, atomic rename |
| `probes/retry_warm.sh` | local | retried the 39 build failures → 39 resolved, 0 hard |
| `probes/last_one.sh` | local | built the final missing image `ffb66f421c8c` |
| `probes/pr6_reset_check.py` | local | reproduced the reaper `ConnectionResetError` with real `SO_LINGER` RSTs |
| `probes/fleet_state.py`, `probes/fleet_sampler.sh` | local | lease/capacity sampling; produced the 49-`lost` lease evidence |
| `probes/deleted_guards.py` | local | deleted-guard detector for schema-drift review |
| `probes/orphan_attrs.py` | local | class-scoped orphaned-attribute detector |
| `probes/vram_sampler.sh` | local | VRAM sampling during rollout |
| `probes/wandb_pull.sh` | local | pulled run history from the internal instance (the authoritative source) |

## Infrastructure — no evidence claims

Repo/branch fetches (`fetch_canon.sh`, `fetch_sf_054.sh`, `fetch_sf_pr6.sh`,
`upstream_fetch.sh`, `sf_fetch.sh`, `prefetch.sh`), venv builds (`build_venv*.sh`,
`build_sandfleet_venv.sh`, `build_sf054_venv.sh`, `build_sf6_venv.sh`), the PR
watch loop (`pr_fetch.sh`, `pr_post.sh`, `gh_from_node.sh`, `gh_auth_probe.sh`),
babysitting (`health.sh`, `monitor.sh`, `tmax_env.sh`), and one-shot capability
probes (`access_check.sh`, `perm_check.sh`, `cuda_probe.sh`, `model_probe.sh`,
`tok_probe.sh`, `mks_probe.sh`, `export_test.sh`, `overlap_test.sh`,
`probe_fairsc.sh`, `repo_scan.sh`, `branch_scan.sh`, `files_probe.sh`,
`tp_probe.sh`, `wandb_reach.sh`, `wandb_discover.sh`, `wandb_verify.sh`,
`wb_metrics.sh`, `wb_final.sh`, `guard_test.sh`, `push_guard.sh`,
`review_candidate.sh`, `pr6_live_check.sh`).

## Superseded — listed so their outputs are not mistaken for current

| script | superseded by | why |
|---|---|---|
| `probes/warm_sifs.sh`, `smoke_warm.sh`, `bulk_warm.sh`, `warm_one.sh`, `warm_daemon.sh` | `full_warm.sh` | earlier non-sharded / subset builds |
| `probes/enum.sh`, `enum8.sh` | the launch manifest | replay enumeration; **cannot bound coverage** — see `README.md` on the 46% filter |
| `probes/full_sweep.sh` | `inspect_all.sh` | predates the size floor and the inspect gate |
| `probes/patches/test_rollout_termination.py` | — | exploratory, never posted as evidence |

## Known-wrong outputs, retracted on-thread

Kept in the record because the retraction is part of the evidence:

- An early `tool_audit.py` matched failure text transcript-wide and flagged 81.5%
  of rollouts (→ 8.96% once segmented per tool call), and produced 254 + 225
  reward mismatches that were entirely false (→ 0 once compared against terminal
  state only).
- The first `deleted_guards.py` used a sev-3-only threshold that hid the whole
  `vllm_utils` cluster, and reported narrowed conditions as deleted ones.
- The first `orphan_attrs.py` was name-based and tree-wide, and returned 0 on a
  known positive; it is now class-scoped.
- `pr_fetch.sh` originally issued a single unpaged `per_page=100` request and
  silently missed 32 comments once threads passed 100. It now pages and
  **self-checks fetched vs API-reported counts**, printing `SELF_CHECK_PASS`.
- The first census inside `tool_audit.py` looked only at top-level record keys,
  concluded `rollout_state` was absent, and would have justified dropping a
  column (`done`, genuinely 18,043 True / 1,669 False) as fake. It now descends
  through `request_info`.
- `TESTS_FAILED` has been wrong twice, the same way both times — it matched text
  that is not a test verdict, and because `_verdict_positions()` is line-wise and
  failure-dominant, one bad line set a whole rollout's terminal verdict.
  (a) `0 failed` scored as a failure, so a clean pass read as terminal failure.
  (b) any digits before "failed" scored as a count: 19,727 of 31,169 DPPO matches
  (63%) were ports (`bind() to 0.0.0.0:8080 failed`, 11,385 alone), indices
  (`Drive 3 failed`), patch hunks, and the `8` inside `UTF-8 failed`. Both fixed;
  the class fell 5,907 → 5,003 across the two corrections.
- The same detector was then wrong a third time, on the *other* side: I fixed
  `0 failed` and never checked `0 passed`, and left the PASS token bare while
  tightening FAILED. A strict FAILED against a loose PASSED biased every rollout
  toward `final_passed`. Corrected to require a strictly positive, non-fractional
  count and to reject digitless model-echo banners; `zero_reward_though_final_
  tests_passed` fell 120 → 96 (DPPO) and 158 → 104 (SGD). **The asymmetry was the
  bug — fixing one side of a symmetric pattern is not fixing it.**
- The first attempt at that fix claimed to be "purely subtractive by
  construction" and was not: a `[\d/.]` lookbehind instead of `[\w/.-]` matched
  the `1` in `test_..._defaults_to_1 PASSED` and **added** 198 lines.
  `subtractive_check.py` caught it against the real corpus. Subtractiveness is
  now measured, never argued from the regex shape.
- I reported **2** `lost connection to sandbox` rollouts for the steps 1–200
  window. The correct figures are **16** (DPPO) / **20** (SGD), matching Rulin's
  independent scan; the 2 came from the 11096823 corpus and was carried across
  without recounting.
