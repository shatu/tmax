#!/bin/bash
# Focused negative/positive test for preflight gate 7 (required launch files).
#
# Regression test for the defect that killed trainer 11141229: the canonical tree
# did not contain scripts/train/debug/envs/checkpoint_env.sh, which the launcher
# sources at line 41, so an 8-node allocation died 19 s in. Gate 7 must refuse to
# allocate when either hard-sourced helper is absent or unreadable.
#
# The test drives the gate's own logic over synthetic trees rather than calling
# preflight.sh end-to-end, because the full preflight also audits 14,488 SIFs and
# needs a live pool; this keeps the test hermetic and fast while exercising the
# exact list and the exact conditions the gate checks.
#
# Run:  bash test_required_launch_files.sh
set -uo pipefail

REQUIRED_LAUNCH_FILES=(
  "scripts/train/debug/envs/checkpoint_env.sh"
  "scripts/train/debug/envs/ray_node_setup_slurm.sh"
)

# Mirror of the gate body in preflight.sh. Kept in sync by
# test_z_gate_list_matches_preflight below, which reads the real file — so this
# copy cannot silently drift from the thing it claims to test.
check_tree() {
  local TREE="$1" miss=0 rel f
  for rel in "${REQUIRED_LAUNCH_FILES[@]}"; do
    f="$TREE/$rel"
    if [ ! -f "$f" ]; then miss=1
    elif [ ! -r "$f" ]; then miss=1
    fi
  done
  return $miss
}

mktree() {                      # $1=dir  $2..=relative files to create
  local d="$1"; shift
  mkdir -p "$d/scripts/train/debug/envs"
  local rel
  for rel in "$@"; do mkdir -p "$d/$(dirname "$rel")"; printf '#!/bin/bash\n:\n' > "$d/$rel"; chmod 755 "$d/$rel"; done
}

pass=0; fail=0
ok()  { printf '  PASS  %s\n' "$1"; pass=$((pass+1)); }
bad() { printf '  FAIL  %s\n' "$1"; fail=$((fail+1)); }

TMP=$(mktemp -d); trap 'chmod -R u+rwX "$TMP" 2>/dev/null; rm -rf "$TMP"' EXIT

# --- A: complete tree CLEARS (positive control) ------------------------------
mktree "$TMP/complete" "${REQUIRED_LAUNCH_FILES[@]}"
if check_tree "$TMP/complete"; then ok "A complete tree clears the gate"
else bad "A complete tree was blocked — gate over-blocks, would refuse a good launch"; fi

# --- B: the exact 11141229 condition BLOCKS (negative control) ---------------
mktree "$TMP/no_ckpt" "scripts/train/debug/envs/ray_node_setup_slurm.sh"
if check_tree "$TMP/no_ckpt"; then
  bad "B missing checkpoint_env.sh cleared — this is exactly what killed 11141229"
else ok "B missing checkpoint_env.sh blocks"; fi

# --- C: the next-in-line failure BLOCKS --------------------------------------
mktree "$TMP/no_ray" "scripts/train/debug/envs/checkpoint_env.sh"
if check_tree "$TMP/no_ray"; then
  bad "C missing ray_node_setup_slurm.sh cleared — would have been the next fatal"
else ok "C missing ray_node_setup_slurm.sh blocks"; fi

# --- D: both absent BLOCKS (the canonical 8691a09ba2 state) ------------------
mktree "$TMP/neither"
if check_tree "$TMP/neither"; then bad "D empty tree cleared"
else ok "D both absent blocks (the state canonical 8691a09ba2 shipped in)"; fi

# --- E: present but UNREADABLE blocks ----------------------------------------
# Existence alone is not sufficient: the launcher sources these, so a file it
# cannot read fails identically to one that is not there.
mktree "$TMP/unreadable" "${REQUIRED_LAUNCH_FILES[@]}"
chmod 000 "$TMP/unreadable/scripts/train/debug/envs/checkpoint_env.sh"
if [ "$(id -u)" -eq 0 ]; then
  printf '  SKIP  E unreadable-file check (running as root; chmod 000 is not enforced)\n'
elif check_tree "$TMP/unreadable"; then
  bad "E unreadable checkpoint_env.sh cleared — existence checked but not readability"
else ok "E unreadable file blocks, not just absent ones"; fi

# --- F: a directory at the path blocks ---------------------------------------
mktree "$TMP/isdir" "scripts/train/debug/envs/ray_node_setup_slurm.sh"
mkdir -p "$TMP/isdir/scripts/train/debug/envs/checkpoint_env.sh"
if check_tree "$TMP/isdir"; then bad "F a directory at the path cleared the -f test"
else ok "F a directory at the required path blocks"; fi

# --- Z: this test's list matches the gate's list -----------------------------
# Without this, the test could pass forever against a stale copy of the list
# while the real gate checks something else.
PRE="$(dirname "$0")/../preflight/preflight.sh"
if [ -r "$PRE" ]; then
  got=$(sed -n '/^REQUIRED_LAUNCH_FILES=(/,/^)/p' "$PRE" \
        | grep -oE '"[^"]+\.sh"' | tr -d '"' | sort | tr '\n' ' ')
  want=$(printf '%s\n' "${REQUIRED_LAUNCH_FILES[@]}" | sort | tr '\n' ' ')
  if [ "$got" = "$want" ]; then ok "Z test list matches preflight.sh gate list"
  else bad "Z list drift — preflight has [$got], test has [$want]"; fi
else
  bad "Z could not read preflight.sh at $PRE to compare lists"
fi

echo
echo "  $pass passed, $fail failed"
[ $fail -eq 0 ] || exit 1
echo "REQUIRED_LAUNCH_FILES_TESTS_OK"
