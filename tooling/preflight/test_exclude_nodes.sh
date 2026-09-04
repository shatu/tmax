#!/bin/bash
# Tests for the EXCLUDE_NODES set.
#
# h100-139-020-027 was added because it is the only H100 node reporting
# "prolog pre-hook failed (sunk:verify-undrain)". It was NOT implicated in the
# 2026-09-04 08:00Z deaths -- it appeared in neither dead trainer's allocation
# -- so the entry is prophylactic. These tests exist so it cannot be silently
# dropped the next time the list is edited, and so the list stays well-formed.
#
# HERMETIC-FIRST, after RulinShao ran the first version from a clean checkout
# and got 6/9 where I got 9/9. That version sourced ./scripts/tmax_env.sh -- the
# DEPLOYMENT path, which exists on my box and nowhere else -- so a third party
# or CI saw three false FAILs. Worse, it did
#
#     ( source ./scripts/tmax_env.sh "$MODE" >/dev/null 2>&1; echo "$EXCLUDE_NODES" )
#
# which swallowed the missing-file error, so "file not found" was reported as
# "does NOT exclude the prolog-failed node". A misleading verdict from a
# discarded stderr: exactly the defect-2 failure mode, reproduced by me inside
# the test written to guard against that family.
#
# So: the REPO copy is the hermetic subject and always runs. The DEPLOYMENT copy
# is checked additionally when present, and its absence is an explicit SKIP with
# a reason -- never an exclusion failure.
#
# Run:  bash test_exclude_nodes.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ENV="$HERE/tmax_env.sh.oscar"                      # vendored, in-tree
DEPLOY_ENV="${OSCAR_ROOT:-/checkpoint/memorization/oscaryinn/tmax}/scripts/tmax_env.sh"
REQUIRED_NODE="h100-139-020-027"
MODES=(accept8x32 sgd8x32 cap500)

pass=0; fail=0; skip=0
ok()   { printf '  PASS  %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  FAIL  %s\n' "$1"; fail=$((fail+1)); }
skipf(){ printf '  SKIP  %s\n' "$1"; skip=$((skip+1)); }

# Read EXCLUDE_NODES for one mode from one env file.
# Prints the value on stdout; returns 1 and prints the sourcing error on stderr
# if the file cannot be sourced. Errors are NOT discarded -- that is the point.
read_excludes() {
    local envfile="$1" mode="$2" err out rc
    err="$(mktemp)"
    out="$( set +u; source "$envfile" "$mode" >/dev/null 2>"$err"; printf '%s' "${EXCLUDE_NODES:-}" )"
    rc=$?
    if [ -s "$err" ] && [ -z "$out" ]; then
        printf 'sourcing %s failed: %s\n' "$envfile" "$(head -2 "$err" | tr '\n' ' ')" >&2
        rm -f "$err"; return 1
    fi
    rm -f "$err"
    printf '%s' "$out"
    [ -n "$out" ] || return 1
    return 0
}

check_file() {  # $1=label  $2=path  $3=required(1) or optional(0)
    local label="$1" envfile="$2" required="$3"
    if [ ! -r "$envfile" ]; then
        if [ "$required" = "1" ]; then
            bad "$label: $envfile is not readable -- the vendored copy must exist in-tree"
        else
            skipf "$label: $envfile not present (deployment-only check; not an exclusion failure)"
        fi
        return
    fi
    local mode ex
    for mode in "${MODES[@]}"; do
        if ! ex="$(read_excludes "$envfile" "$mode" 2>/tmp/rx.$$)"; then
            bad "$label/$mode: could not read EXCLUDE_NODES -- $(cat /tmp/rx.$$ 2>/dev/null)"
            rm -f /tmp/rx.$$; continue
        fi
        rm -f /tmp/rx.$$
        case "$ex" in
            *"$REQUIRED_NODE"*) ok "$label/$mode excludes $REQUIRED_NODE" ;;
            *) bad "$label/$mode does NOT exclude $REQUIRED_NODE (list: ${ex:0:60}...)" ;;
        esac
        local n u
        n=$(tr ',' '\n' <<<"$ex" | grep -c .)
        u=$(tr ',' '\n' <<<"$ex" | sort -u | grep -c .)
        [ "$n" = "$u" ] && ok "$label/$mode no duplicates ($n nodes)" \
                        || bad "$label/$mode has duplicates ($n vs $u unique)"
        # A space anywhere makes sbatch --exclude misparse the WHOLE set, which
        # silently voids every exclusion at once rather than just one entry.
        case "$ex" in
            *" "*) bad "$label/$mode contains a space -- sbatch --exclude would void the entire set" ;;
            *)     ok "$label/$mode is space-free" ;;
        esac
    done
}

echo "=== repo copy (hermetic, always runs) ==="
check_file "repo" "$REPO_ENV" 1

echo "=== deployment copy (additional, when present) ==="
check_file "deploy" "$DEPLOY_ENV" 0

echo
echo "  $pass passed, $fail failed, $skip skipped"
[ "$fail" -eq 0 ] || exit 1
echo "EXCLUDE_NODES_TESTS_OK"
