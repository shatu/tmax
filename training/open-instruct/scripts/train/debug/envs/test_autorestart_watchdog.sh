#!/bin/bash
# Focused tests for the autorestart watchdog's resubmission path.
#
# Written after supervision failed silently on two arms in one night:
#   cap500     trainer 11212260 died 21:18Z, watchdog 11212261 self-exited
#   production trainer 11149108 died 01:26Z, watchdog 11149109 self-exited
#              -> production sat dead and unsupervised for ~3 hours
#
# Both logged only "ERROR: resubmit failed (empty job id)". The cause was that
# the watchdog never passed --account while launch.sh does, so every resubmit
# it had ever attempted was rejected by the scheduler -- and `2>/dev/null` on
# the sbatch line threw the reason away.
#
# These tests therefore exercise a REJECTED submission, not just a happy path:
# a happy-path test would have passed against the broken watchdog for months.
#
# Run:  bash test_autorestart_watchdog.sh
set -uo pipefail

WD="$(dirname "$0")/autorestart_watchdog.sh"
pass=0; fail=0
ok()  { printf '  PASS  %s\n' "$1"; pass=$((pass+1)); }
bad() { printf '  FAIL  %s\n' "$1"; fail=$((fail+1)); }

# Pull the regexes and the helper out of the watchdog without running its main
# loop, so the tests exercise the real definitions rather than copies of them.
DETERMINISTIC_RE="$(grep -m1 '^DETERMINISTIC_RE=' "$WD" | cut -d"'" -f2)"
TRANSIENT_RE="$(grep -m1 '^TRANSIENT_RE=' "$WD" | cut -d"'" -f2)"
eval "$(sed -n '/^sched_args_from_job() {/,/^}/p' "$WD")"

# --- 1. scheduler args are recovered from a real job ------------------------
# The regression that matters: --account must be present. A job id is passed in
# so this runs against the accounting DB rather than a mock.
REF_JOB="${REF_JOB:-11149108}"
if sacct -j "$REF_JOB" -X -n -o Account -P >/dev/null 2>&1 && \
   [ -n "$(sacct -j "$REF_JOB" -X -n -o Account -P 2>/dev/null | head -1)" ]; then
    if sched_args_from_job "$REF_JOB"; then
        if printf '%s\n' "${SCHED_ARGS[@]}" | grep -q -- '--account='; then
            ok "1 --account recovered from job $REF_JOB ($SCHED_RECOVERED)"
        else
            bad "1 no --account in recovered args -- this IS the production defect"
        fi
        if printf '%s\n' "${SCHED_ARGS[@]}" | grep -q -- '--partition='; then
            bad "1b --partition passed; fair-sc infers it from the QOS prefix and warns"
        else
            ok "1b --partition correctly omitted"
        fi
    else
        bad "1 sched_args_from_job returned non-zero for a real job"
    fi
else
    printf '  SKIP  1 job %s not in the accounting DB on this cluster\n' "$REF_JOB"
fi

# --- 2. a REJECTED submission is loud, and preserves the reason -------------
# --test-only so nothing is ever queued. A deliberately invalid account
# reproduces the exact rejection that killed supervision twice.
# A REAL batch script, so the rejection is the account rejection and not
# "this does not look like a batch script" -- otherwise the test would pass on
# the wrong error and prove nothing about the defect it exists for.
tmpjob="$(mktemp -t wd-testjob.XXXXXX.sh)"
printf '#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --time=00:01:00\nexit 0\n' > "$tmpjob"
err="$(mktemp)"; out="$(sbatch --test-only --parsable --account=definitely-not-an-account \
        --qos=h100_memorization_high "$tmpjob" 2>"$err")"; rc=$?
newid="$(printf '%s' "$out" | tr -d '[:space:]')"
if [ "$rc" -ne 0 ] || ! printf '%s' "$newid" | grep -qE '^[0-9]+$'; then
    ok "2 bad-account submission is detected as failure (rc=$rc, id='${newid:-<empty>}')"
else
    bad "2 bad-account submission was accepted as success -- validation is not working"
fi
if [ -s "$err" ]; then
    ok "2b scheduler stderr is non-empty and available to log: $(head -c 90 "$err" | tr '\n' ' ')"
else
    bad "2b scheduler stderr was empty -- the failure reason would be unrecoverable"
fi
if grep -qi 'account' "$err"; then
    ok "2c the preserved stderr names the ACCOUNT problem -- the real defect is reproduced"
else
    bad "2c stderr did not mention the account; test is not exercising the real rejection: $(head -c 100 "$err")"
fi
rm -f "$err" "$tmpjob"

# --- 3. job-id validation accepts only ^[0-9]+$ -----------------------------
for probe in "12345:accept" "12345\n:accept" " 12345 :accept" \
             ":reject" "Submitted batch job 12345:reject" "abc:reject" \
             "12345abc:reject" "sbatch: error: foo:reject"; do
    raw="${probe%:*}"; want="${probe##*:}"
    got="$(printf '%b' "$raw" | tr -d '[:space:]')"
    if printf '%s' "$got" | grep -qE '^[0-9]+$'; then verdict=accept; else verdict=reject; fi
    if [ "$verdict" = "$want" ]; then ok "3 id '${raw}' -> $verdict"
    else bad "3 id '${raw}' -> $verdict, wanted $want"; fi
done

# --- 4. permanent failures are non-retryable, keyed on EXHAUSTION -----------
# The distinction that matters: healthy production and control each carry 27-34
# transient bearer-token lines and must still be restartable. Only the terminal
# form is permanent.
perm_yes=(
  "RuntimeError: Reset failed after 3 attempts: Sandfleet PermissionError: Invalid or missing client bearer token"
  "ValueError: Some specified arguments are not used by the HfArgumentParser: ['--foo']"
  "MissingLocalSifError: no local SIF for image sha256:abc"
)
# NOTE the flags mirror the watchdog exactly: it greps DETERMINISTIC_RE with
# -qaE (case SENSITIVE) and TRANSIENT_RE with -qaiE. Testing both with -i would
# pass signatures that the real classifier would miss on case alone.
for t in "${perm_yes[@]}"; do
    if printf '%s' "$t" | grep -qaE "$DETERMINISTIC_RE"; then ok "4 permanent: ${t:0:58}..."
    else bad "4 NOT classified permanent: ${t:0:58}..."; fi
done

perm_no=(
  "WARNING - SWERLVanilluxSandboxEnv.reset attempt 2 failed: Sandfleet PermissionError: Invalid or missing client bearer token. Retrying in 2.69s..."
  "RuntimeError: Reset failed after 3 attempts: Could not reach Sandfleet endpoint 'http://10.137.78.52:42949': timed out"
)
for t in "${perm_no[@]}"; do
    if printf '%s' "$t" | grep -qaE "$DETERMINISTIC_RE"; then
        bad "4b WRONGLY permanent (must stay retryable): ${t:0:58}..."
    else ok "4b stays retryable: ${t:0:58}..."; fi
done

# --- 5. genuinely transient classes are still retryable ---------------------
# Explicitly including the ECC case from the production failure: 318 ecc hits
# were the dominant transient signature and must not be swallowed.
trans=("Uncorrectable ECC error encountered" "NVML: Xid 63" "nccl unhandled error"
       "keepalive watchdog timeout" "address already in use")
for t in "${trans[@]}"; do
    if printf '%s' "$t" | grep -qaE "$DETERMINISTIC_RE"; then
        bad "5 transient signature wrongly classified permanent: $t"
    elif printf '%s' "$t" | grep -qaiE "$TRANSIENT_RE"; then ok "5 transient: $t"
    else bad "5 matched neither classifier: $t"; fi
done

# --- 6. ordering: deterministic is checked BEFORE transient -----------------
# A credential exhaustion also contains "Reset failed after", which TRANSIENT_RE
# matches. Permanence must win, or the fix is inert.
mixed="RuntimeError: Reset failed after 3 attempts: Sandfleet PermissionError: Invalid or missing client bearer token"
if printf '%s' "$mixed" | grep -qaiE "$TRANSIENT_RE" && \
   printf '%s' "$mixed" | grep -qaE "$DETERMINISTIC_RE"; then
    det_line="$(grep -n 'grep -qaE "\${DETERMINISTIC_RE}"' "$WD" | head -1 | cut -d: -f1)"
    tr_line="$(grep -n 'grep -qaiE "\${TRANSIENT_RE}"' "$WD" | head -1 | cut -d: -f1)"
    if [ -n "$det_line" ] && [ -n "$tr_line" ] && [ "$det_line" -lt "$tr_line" ]; then
        ok "6 both regexes match; DETERMINISTIC evaluated first (line $det_line < $tr_line)"
    else
        bad "6 credential exhaustion matches both, but ordering is wrong (det=${det_line:-?} transient=${tr_line:-?})"
    fi
else
    bad "6 expected the credential-exhaustion form to match BOTH classifiers; it did not"
fi

# --- 6b. the resubmit must override the launcher's unwritable --output ------
# The launcher hardcodes #SBATCH --output into another account's directory.
# Without a command-line override Slurm kills the job at ~3s (ExitCode 0:53)
# with no log, and the watchdog's own classifier then reads a file that was
# never written. Both replacements died this way on 2026-09-04.
if grep -q 'IO_ARGS=(--output=' "$WD" && grep -q '"${IO_ARGS\[@\]}"' "$WD"; then
    ok "6b resubmit passes an explicit --output/--error override"
else
    bad "6b resubmit does not override the launcher's hardcoded --output"
fi
if grep -q 'IO_ARGS=(--output="\${LOGDIR}' "$WD"; then
    ok "6c override targets LOGDIR, the same path the classifier reads"
else
    bad "6c override does not target LOGDIR; classifier would read a missing file"
fi

# --- 7. no sbatch invocation may discard stderr -----------------------------
if grep -n 'sbatch' "$WD" | grep -q '2>/dev/null'; then
    bad "7 an sbatch call still redirects stderr to /dev/null"
else
    ok "7 no sbatch call discards stderr"
fi

echo
echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
echo "WATCHDOG_RESUBMIT_TESTS_OK"
