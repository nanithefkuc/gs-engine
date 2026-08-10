#!/bin/sh
# Aggregate the cross-library comparison with provenance.
#
# Builds the controller once, validates the frozen corpus against the gs-engine
# reference, runs each adapter over the corpus, and records machine/toolchain
# and pinned-revision provenance alongside the classification table. Adapters
# that are not built are skipped with a visible gap, never silently.
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
controller=$crate_dir/external-bench/controller
fixtures=$crate_dir/external-bench/fixtures
record_dir=${GS_BENCH_RECORD_DIR:-"$crate_dir/external-bench/results/$(date -u +%Y%m%dT%H%M%SZ)"}
metadata=$record_dir/metadata.txt
table=$record_dir/aggregate.txt
mkdir -p "$record_dir"

# Provenance: hardware, toolchain, pinned external revisions.
{
    printf '%s\n' "aggregate-record-version=1"
    printf '%s\n' "timestamp-utc=$(date -u +%Y%m%dT%H%M%SZ)"
    uname -a
    rustc -Vv
    cargo -V
    printf '%s\n' "ecosystem-revisions:"
    for repository in "$crate_dir" "$crate_dir/../fgf" "$crate_dir/../butterfly-fft" "$crate_dir/../gfm" "$crate_dir/../simdispatch"; do
        if git -C "$repository" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            printf '%s\n' "  library=$(basename "$repository");revision=$(git -C "$repository" rev-parse HEAD)"
        fi
    done
    printf '%s\n' "external-revisions:"
    printf '%s\n' "  percyxx=$(awk -F= '/^revision=/{print $2; exit}' "$crate_dir/external-bench/revisions.lock")"
    printf '%s\n' "  decoding=$(awk -F= '/^sha256=/{print $2; exit}' "$crate_dir/external-bench/revisions.lock")"
    printf '%s\n' "  lambdaworks=$(awk -F= '/^\[LAMBDAWORKS\]/{f=1} f && /^revision=/{print $2; exit}' "$crate_dir/external-bench/revisions.lock")"
} >"$metadata"

# Build the controller and the adapters.
cargo build --release --manifest-path "$controller/Cargo.toml" >/dev/null 2>&1

{
    printf '%s\n' "=== reference self-validation (gs-engine) ==="
    cargo run --release --manifest-path "$controller/Cargo.toml" -- validate "$fixtures" 2>&1 || true
    echo

    for adapter in decoding/decoding-gs percyxx/percyxx-gs lambdaworks/lambdaworks-gs; do
        exe=$crate_dir/external-bench/$adapter
        printf '%s\n' "=== adapter: $adapter ==="
        if [ -x "$exe" ]; then
            cargo run --release --manifest-path "$controller/Cargo.toml" -- run "$exe" "$fixtures" 2>&1 || true
        else
            printf '%s\n' "  (not built: $exe missing)"
        fi
        echo
    done
} >"$table"

cat "$metadata"
echo "---"
cat "$table"
printf '%s\n' "aggregate_record=$record_dir"
