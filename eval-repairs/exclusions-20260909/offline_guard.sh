# Preserve verifier output and status, except for explicit offline-cache failures.
# Compilation errors and failed assertions are still model failures.
run_offline_verifier() {
    local output status
    output=$(mktemp) || {
        echo 'SETUP-FAILED: cannot create verifier output file' >&2
        exit 90
    }
    "$@" >"$output" 2>&1
    status=$?
    cat "$output"
    if [ "$status" -ne 0 ] && grep -Eq \
        'Cannot access .* in offline mode|has not been downloaded from it before|No cached version .* available for offline mode|No cached resource available for offline mode' "$output"; then
        rm -f "$output"
        echo 'SETUP-FAILED: verifier dependency unavailable offline; not a model reward' >&2
        exit 90
    fi
    rm -f "$output"
    return "$status"
}
