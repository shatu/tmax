#!/bin/bash
# Focused test for the EXCLUDE_NODES set (hamishivi: staged + tested, next launch only).
#
# h100-139-020-027 was added because it is the only H100 node reporting
# "prolog pre-hook failed (sunk:verify-undrain)". It was NOT implicated in the
# 2026-09-04 08:00Z deaths -- it appeared in neither dead trainer's allocation --
# so this is prophylactic. The test exists so the entry cannot be silently lost
# the next time the list is edited, and so the list stays well-formed.
set -uo pipefail
cd "$(dirname "$0")/../.."
pass=0; fail=0
ok(){ printf '  PASS  %s\n' "$1"; pass=$((pass+1)); }
bad(){ printf '  FAIL  %s\n' "$1"; fail=$((fail+1)); }

for MODE in accept8x32 sgd8x32 cap500; do
  ( set +u; source ./scripts/tmax_env.sh "$MODE" >/dev/null 2>&1
    echo "$EXCLUDE_NODES" ) > /tmp/excl.$$ 2>/dev/null
  EX=$(cat /tmp/excl.$$); rm -f /tmp/excl.$$ 2>/dev/null
  case "$EX" in *h100-139-020-027*) ok "$MODE excludes the prolog-failed node";;
                *) bad "$MODE does NOT exclude h100-139-020-027";; esac
  # well-formedness: no empty entries, no spaces, no duplicates
  n=$(tr ',' '\n' <<<"$EX" | grep -c .)
  u=$(tr ',' '\n' <<<"$EX" | sort -u | grep -c .)
  [ "$n" = "$u" ] && ok "$MODE list has no duplicates ($n nodes)" || bad "$MODE has duplicates ($n vs $u unique)"
  case "$EX" in *" "*) bad "$MODE list contains a space -- sbatch --exclude would misparse";;
                *) ok "$MODE list is space-free";; esac
done
echo; echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
echo "EXCLUDE_NODES_TESTS_OK"
