#!/bin/bash
# Hard pre-allocation gate. Runs BEFORE any sbatch in launch.sh; a non-zero exit
# here means no 64-GPU allocation is requested at all.
#
# Why this exists: trainer 11096823 died at step 78 because 58 of the 1,170 task
# images it touched were absent from the SIF pool, and `prefer_local_sif()`
# answered a miss with the bare `docker://` ref instead of an error. The cost was
# a 6.5-hour 64-GPU run. Every check below is one of the things that, had it run
# before the allocation, would have stopped that launch.
#
# What this gate can and cannot bound (stated plainly, because the distinction
# cost us a wrong claim once already):
#   CAN  — that every image in the enumerated manifest resolves to a real,
#          plausibly-sized, structurally valid local SIF.
#   CANNOT — that the manifest is the complete set the run will touch. Replay
#          enumeration provably cannot bound that: on 11096823, active sampling
#          filtered 46% of draws (batch/filtered_prompts=529 against 616
#          accepted, no_resampled_prompts=0), so the trainer draws prompts the
#          replay never saw.
# Gate 5 is the answer to the second bullet: the fail-closed guard must be
# present in the tree being launched, so an image outside the manifest fails
# loudly in seconds instead of silently entering a 600-1200 s docker:// pull.
set -uo pipefail

ROOT=/checkpoint/memorization/oscaryinn/tmax
MANIFEST=$ROOT/probes/launch_image_manifest.tsv
POOL=${SWERL_APPTAINER_SIF_DIR:-$ROOT/sifs}
TREE=${OPEN_INSTRUCT_DIR:?source tmax_env.sh first}
PY=$ROOT/venvs/tmax/bin/python
ATTEST=$ROOT/probes/logs/inspect-attestation.json
REPORTS='probes/logs/inspect-*.jsonl'

fail=0
note() { printf '  %-4s %s\n' "$1" "$2"; }
echo "=== pre-allocation gate ==="
echo "tree     : $TREE"
echo "pool     : $POOL"
echo "manifest : $MANIFEST"
echo

# ---- Gate 0: the operator has not disabled the guard out from under us -------
if [ -n "${SWERL_ALLOW_REMOTE_IMAGE_FALLBACK:-}" ]; then
  note FAIL "SWERL_ALLOW_REMOTE_IMAGE_FALLBACK is set ('$SWERL_ALLOW_REMOTE_IMAGE_FALLBACK')"
  echo "       That re-enables the exact silent docker:// fallback that ended 11096823."
  fail=1
else
  note PASS "remote-image fallback not enabled in this environment"
fi

# ---- Gates 1-4: coverage / non-empty / size floor / inspect ------------------
# Coverage runs live against the tree's own prefer_local_sif(). The inspect gate
# consumes the array's shard reports rather than re-running apptainer inline:
# apptainer is not on the login node, and 14,488 inspects is a multi-minute
# parallel job, not something to do inside a launch wrapper.
cd "$ROOT" || exit 2
AUDIT_JSON=probes/logs/pool-audit-preflight.json
"$PY" probes/pool_audit.py --tree "$TREE" --manifest "$MANIFEST" --pool "$POOL" \
      --inspect-reports "$REPORTS" --json "$AUDIT_JSON" || fail=1
echo

# ---- Gate 5: the inspect attestation must be FRESHER than the pool ----------
# An attestation that predates a pool mutation is worthless: any SIF added,
# rebuilt, or truncated since the array ran is unverified. Fingerprint the pool
# and refuse if it has moved.
"$PY" - "$POOL" "$ATTEST" "$AUDIT_JSON" <<'PY'
import json, os, sys, glob
pool, attest, audit = sys.argv[1], sys.argv[2], sys.argv[3]
files = glob.glob(os.path.join(pool, "*.sif"))
st = [os.stat(f) for f in files]
fp = {"count": len(files),
      "total_bytes": sum(s.st_size for s in st),
      "newest_mtime": max((s.st_mtime for s in st), default=0)}
newest_report = max((os.path.getmtime(f) for f in glob.glob("probes/logs/inspect-*.jsonl")),
                    default=0)
ok = True
if fp["newest_mtime"] > newest_report:
    print(f"  FAIL pool changed after the inspect run "
          f"(newest SIF {fp['newest_mtime']:.0f} > newest report {newest_report:.0f})")
    print("       re-run: sbatch probes/inspect_all.sh")
    ok = False
else:
    print("  PASS inspect attestation is newer than every file in the pool")
if os.path.exists(attest):
    prev = json.load(open(attest))
    if prev.get("count") != fp["count"] or prev.get("total_bytes") != fp["total_bytes"]:
        print(f"  NOTE pool fingerprint moved since last gate: "
              f"{prev.get('count')}/{prev.get('total_bytes')} -> "
              f"{fp['count']}/{fp['total_bytes']} (re-verified above)")
fp["audit_verdict"] = json.load(open(audit))["VERDICT"]
json.dump(fp, open(attest, "w"), indent=1)
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || fail=1

# ---- Gate 6: the launched tree actually contains the fail-closed guard ------
# Grep is not enough — a partially applied patch can define the error class and
# never raise it. Import the module and exercise a miss against an empty pool.
"$PY" - "$TREE" <<'PY'
import os, sys, tempfile
sys.path.insert(0, sys.argv[1])
os.environ.pop("SWERL_ALLOW_REMOTE_IMAGE_FALLBACK", None)
with tempfile.TemporaryDirectory() as tmp:
    os.environ["SWERL_APPTAINER_SIF_DIR"] = tmp
    from open_instruct.environments.apptainer_images import prefer_local_sif
    probe = "hamishi740/swerl-tmax-v3:0000preflightprobe"
    try:
        got = prefer_local_sif(probe)
    except Exception as e:
        print(f"  PASS fail-closed guard live: miss raises {type(e).__name__}")
        sys.exit(0)
print(f"  FAIL tree returns {got!r} on a pool miss instead of raising")
print("       this is the unguarded behaviour that ended trainer 11096823")
sys.exit(1)
PY
[ $? -eq 0 ] || fail=1

echo
if [ $fail -ne 0 ]; then
  echo "PREFLIGHT: BLOCK — not requesting an allocation."
  exit 1
fi
echo "PREFLIGHT: CLEAR"
exit 0
