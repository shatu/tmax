# `lost connection to sandbox` — rows reconciled into incidents

Requested by hamishivi after two counts of the same phenomenon disagreed: I had
reported **2**, Rulin's independent scan **16** (DPPO) / **20** (SGD).

## The disagreement, settled

Every candidate phrasing is counted **separately** over the same bounded window
rather than picking one and calling it the answer:

| pattern | DPPO | SGD |
|---|---:|---:|
| `lost_connection_to_sandbox_step` | 16 | 20 |
| `lost_connection_to_sandbox_any` | 16 | 20 |
| `sandbox_worker_lost` | 0 | 0 |
| `could_not_reach_endpoint` | 0 | 0 |
| `sandbox_lost_error_cls` | 0 | 0 |

**Rulin's 16 / 20 is right and my 2 was not, for this window.** The loose
phrasing and the strict one return the *same* 16 / 20 here, so the gap was not a
pattern-width difference — my earlier figure came from a different corpus (the
earlier 11096823 audit) and I carried it across to this run without recounting.
It does not transfer. The figure for steps 1–200 of these two arms is 16 and 20.

`Could not reach Sandfleet endpoint` is kept in its own column as instructed and
is **0 on both arms** — no rollout failed to reach the Sandfleet control plane.

Note `sandbox_worker_lost` = 0 here is **not** in tension with the 2,518 / 3,380
`sandbox worker lost` occurrences I reported earlier: those are in the **trainer
logs**, a different source. Zero of them surface in the rollout records.

## Rows are not incidents

| arm | rollout rows | distinct incidents |
|---|---:|---:|
| production (DPPO) 11149108 | 16 | 12 |
| control (SGD) 11151208 | 20 | 16 |

**Grouping basis, stated because it limits the claim:** no record in either arm
carries a sandbox worker URL, lease id or sandbox id. The only URLs present are
the task's own loopback services, which are explicitly excluded from identity —
an earlier pass grouped one row under `127.0.0.1:5000` and would have published a
task-internal address as if it identified a worker.

So every incident here is grouped by **step alone**, which *merges* rows that may
well be separate events. The incident counts are a **lower bound** on
distinctness, not an established count.

### Correction: the per-step task check was vacuous when first published

An earlier revision of this file said "every multi-row step is a single task".
That was read off a `task` column populated by `r["task_id"]` -- a field this
schema does not have. It returned `""` for every record without erroring, so
"distinct tasks per step" was computing `len({""}) == 1` for every step. The
statement was not evidence; it was an artifact of an empty column.
The task id actually lives in `ground_truth[0]`. With the field read correctly,
the claim does hold -- but it holds on evidence that did not exist when it was
first made:

- **production (DPPO) 11149108**:
  - step 137: 4 rows, 1 distinct task — `task_001838_5ad95d97`
  - step 173: 2 rows, 1 distinct task — `task_003391_f8b99429`
- **control (SGD) 11151208**:
  - step 62: 2 rows, 1 distinct task — `task_004568_aad28eb7`
  - step 83: 2 rows, 1 distinct task — `task_006433_70340fff`
  - step 151: 3 rows, 1 distinct task — `task_000820_650d9059`

Multiple samples of one prompt hitting the same failure at the same step is
consistent with one sandbox loss per step, which is what the step-only grouping
assumes. It does not prove it: without worker identity in the record, two
distinct workers failing on the same step at the same time remain
indistinguishable from one.

Per-row detail with task, reward, `num_calls` and timeout flag is in
`lostconn-<arm>.csv`; the incident table is in `lostconn-<arm>.json`.

## Scale

16 and 20 rows against 51,200 rollouts per arm is **0.031%** and
**0.039%**. All carry reward 0.0. None is a control-plane
reachability failure.

