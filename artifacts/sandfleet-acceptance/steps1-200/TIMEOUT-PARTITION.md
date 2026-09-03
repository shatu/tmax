# Timeout-cause partition, steps 1–200

Included in this bundle at hamishivi's direction. **This analysis is Rulin's**,
not mine: produced independently by `verify_1200.py` in
`exchange/rulin_verify_steps1-200_v2/`, reproduced here so the bundle is
self-contained. Source file SHA256:

```
765e3e12d3a2f1dd1fc3946a51a3587ef580f66dba063baa89e921ab70a736a1  verify_1200_results.json
```

Counts are **unique rollouts**, not regex occurrences. The six phases are
mutually exclusive with explicit precedence, and the overlap-flag matrix came
back **empty on both arms** — the partition is clean, so no row is double-counted.

| phase | DPPO 11149108 | SGD 11151208 |
|---|---:|---:|
| 1. total deadline exhausted before any useful work (`step_count=0`) | 588 (10.4%) | 1,184 (21.2%) |
| 2. total deadline during the FIRST generation | 10 (0.2%) | 4 (0.1%) |
| 3. total deadline during a LATER generation | 2,021 (35.9%) | 1,434 (25.7%) |
| 4. total deadline during a tool step | 250 (4.4%) | 203 (3.6%) |
| 5. per-tool timeout | 2,381 (42.3%) | 2,508 (44.9%) |
| 6. metadata-only, none of the above text (`unknown`) | 379 (6.7%) | 256 (4.6%) |
| **total `marked_timeout`** | **5,629** | **5,589** |

## The per-tool ceiling was measured, not assumed

hamishivi asked for the configured duration rather than an assumption of 120 s.
Across every phase-5 record the configured value was **120.0 s in 100% of cases**:

- dppo: `{"120.0": 2381}`
- sgd: `{"120.0": 2508}`

## What this bounds about 600 → 900 s

Phases 1–3 are the total-deadline cases a longer deadline could plausibly help:
**46.5% of DPPO timeouts** and **46.9% of SGD**.
Phase 5 (42–45%) is the 120 s per-tool limit, which a longer *total* deadline
does not touch at all; phase 4 largely likewise.

**Censoring, stated plainly:** a rollout killed at 600 s cannot demonstrate it
would have finished by 900 s. This partition bounds the possible upside — at
most ~47% of timeouts, roughly 5% of all rollouts — and cannot establish it.
Only a controlled canary that varies the total deadline separately from the
per-tool deadline answers the causal question. Neither live job's timeout was
changed to produce this analysis.

