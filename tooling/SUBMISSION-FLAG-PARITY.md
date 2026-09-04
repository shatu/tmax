# launch.sh vs watchdog resubmit — scheduler flag parity

Requested by hamishivi after defect 4, on the grounds that *"the recurring
defect family is duplicated submission logic"*. This is the checklist that
turns that family from a series of outages into a table.

Every defect below was found the same way: `launch.sh` passes something on its
`sbatch` line that the watchdog's resubmit did not, and nothing compares them.

## The two submission sites

```
launch.sh (trainer)                       autorestart_watchdog.sh (resubmit)
  sbatch --parsable                         sbatch --parsable
    --job-name="jb-$EXP_NAME"                 --job-name="${JOB_NAME}"
    --account=memorization                    "${SCHED_ARGS[@]}"   <- inherited
    --qos="$QOS"                              "${QOS_ARG[@]}"
    "${TIME_ARG[@]}"                          "${EXCLUDE_ARG[@]}"
    "${EXCL_ARG[@]}"                          "${IO_ARGS[@]}"
    --output=$OSCAR_ROOT/logs/slurm/%x-%j.out "${LAUNCHER}"
    --error=$OSCAR_ROOT/logs/slurm/%x-%j.err
    "$LAUNCHER"
```

## Parity table

| flag | launch.sh | resubmit, before | resubmit, now | consequence when absent |
|---|---|---|---|---|
| `--parsable` | yes | yes | yes | — |
| `--job-name` | yes | yes | yes | — |
| `--account` | `memorization` | **MISSING** | inherited from replaced job | **defect 1**: every resubmit ever attempted was rejected `Invalid account or account/partition combination`; supervision never worked |
| `--qos` | `$QOS` | via `QOS_ARG` if env set | inherited + `QOS_ARG` | would fall to the launcher's `#SBATCH --qos=h100_comem_high`, another account's QOS |
| `--time` | `$WALLTIME` | **MISSING** | inherited (`TimelimitRaw`) | **defect 5**: falls back to launcher's `#SBATCH --time=7-00:00:00`. Invisible on production/control (also 7d); **cap500 runs 3d and would silently get 7** |
| `--exclude` | `$EXCLUDE_NODES` | via `EXCLUDE_ARG` | same | restarts could land back on the sick nodes that caused the crash |
| `--output` | `logs/slurm/%x-%j.out` | **MISSING** | `${LOGDIR}/${JOB_NAME}-%j.out` | **defect 4**: falls back to launcher's `#SBATCH --output=/checkpoint/comem/rulin/...`, unwritable → Slurm kills the job at ~3s with `ExitCode 0:53` and no log |
| `--error` | `logs/slurm/%x-%j.err` | **MISSING** | `${LOGDIR}/${JOB_NAME}-%j.err` | as above, **and** the classifier reads this exact path — a surviving restart would be judged against a file that never existed |
| `--partition` | not passed | not passed | not passed | correct: fair-sc infers it from the QOS prefix and warns if given |

## Why inheritance rather than copying the list

Three of the four defects were "launch.sh passes X, the resubmit doesn't".
Copying launch.sh's list into the watchdog would have produced a fourth copy to
drift. `SCHED_ARGS` is instead read from the job being replaced via `sacct`, so
whatever launched the original governs the replacement and the two cannot
diverge. `--output`/`--error` cannot be inherited that way — they must point at
a path *this* account can write — so they are set from `LOGDIR`, the same
expression the classifier reads, which makes writer and reader impossible to
separate.

## Residual gaps, stated rather than left implicit

* **Anything the launcher bakes into `#SBATCH` that neither site overrides** is
  still silently in force. The three that bit us were `--output`, `--time` and
  `--account`/`--qos`; the launcher also hardcodes `--nodes`, `--gpus-per-node`,
  `--cpus-per-task`, `--mem` and `--exclusive`, which happen to be correct for
  every arm we run. That is luck, not design.
* **`LOGDIR` has no writability probe.** Its baked-in default is the same
  unwritable path that caused defect 4, so a deployment that forgets
  `TMAX_LOGDIR` reproduces it exactly. RulinShao's suggested one-line
  touch-and-remove check at watchdog startup would close it.
* **The real fix is one submission path, not two.** Proposed post-incident, not
  during: a single `submit_trainer()` used by both `launch.sh` and the resubmit,
  so parity is structural rather than a document someone has to re-read.
