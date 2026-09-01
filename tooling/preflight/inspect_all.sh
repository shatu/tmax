#!/bin/bash
#SBATCH --job-name=sifinspect
#SBATCH --output=/checkpoint/memorization/oscaryinn/tmax/probes/inspect-%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --qos=cpu_lowest
#SBATCH --array=0-31
# Full-pool `apptainer inspect` validity check — every image, not a sample.
#
# hamishivi's gate wording is "apptainer inspect valid" over the whole pool. A
# sampled pass would leave the exact hole this run already fell into once (the
# step-78 miss was 58 images out of 1,170 — a sample of 200 would very likely
# have shown clean). 32 shards keep the wall clock at minutes.
#
# Shard rule is the same NR%n as full_warm.sh: deterministic and disjoint, so a
# rerun of one shard reproduces exactly that slice.
set -uo pipefail

ROOT=/checkpoint/memorization/oscaryinn/tmax
MANIFEST=$ROOT/probes/launch_image_manifest.tsv
POOL=$ROOT/sifs
N=${SLURM_ARRAY_TASK_COUNT:-32}
S=${SLURM_ARRAY_TASK_ID:-0}
OUT=$ROOT/probes/logs/inspect-${SLURM_ARRAY_JOB_ID:-local}_${S}.jsonl
mkdir -p "$ROOT/probes/logs"
: > "$OUT"

command -v apptainer >/dev/null || { echo "FATAL: no apptainer on $(hostname)"; exit 2; }

n_ok=0; n_bad=0
while IFS=$'\t' read -r image sif; do
  [ -z "${image:-}" ] && continue
  path="$POOL/$sif"
  if [ ! -s "$path" ]; then
    printf '{"image":"%s","rc":90,"err":"missing or empty"}\n' "$image" >> "$OUT"
    n_bad=$((n_bad+1)); continue
  fi
  # inspect reads the SIF header + the metadata descriptor; a truncated or
  # half-written image fails here even though it is non-empty on disk.
  err=$(apptainer inspect "$path" 2>&1 >/dev/null); rc=$?
  if [ $rc -eq 0 ]; then
    n_ok=$((n_ok+1))
  else
    n_bad=$((n_bad+1))
    esc=$(printf '%s' "$err" | tr '\n' ' ' | sed 's/"/\\"/g' | cut -c1-200)
    printf '{"image":"%s","rc":%d,"err":"%s"}\n' "$image" "$rc" "$esc" >> "$OUT"
  fi
  # record every success too, so the audit can prove FULL coverage rather than
  # inferring it from the absence of failures
  [ $rc -eq 0 ] && printf '{"image":"%s","rc":0}\n' "$image" >> "$OUT"
done < <(awk -v n="$N" -v s="$S" 'NR % n == s' "$MANIFEST")

echo "SHARD $S/$N  ok=$n_ok  bad=$n_bad"
echo "INSPECT_SHARD_DONE"
