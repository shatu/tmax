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
| `probes/tool_audit.py` | ✔ `tooling/audit/` | per-rollout classifier for 11096823 |
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
