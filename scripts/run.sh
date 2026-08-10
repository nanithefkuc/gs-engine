#!/bin/sh
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
record_dir=${GS_BENCH_RECORD_DIR:-"$crate_dir/target/benchmark-record/$timestamp"}
mkdir -p "$record_dir"
metadata=$record_dir/metadata.txt
allocations=$record_dir/allocations.csv
results=$record_dir/criterion.log

{
    printf '%s\n' "benchmark-record-version=1"
    printf '%s\n' "timestamp-utc=$timestamp"
    printf '%s\n' "command=cargo bench --all-features --bench decoder --bench interpolation --bench products --bench root_extraction --bench scoring -- $*"
    printf '%s\n' "RUSTFLAGS=${RUSTFLAGS-}"
    printf '%s\n' "CARGO_PROFILE_BENCH_LTO=${CARGO_PROFILE_BENCH_LTO-}"
    uname -a
    rustc -Vv
    cargo -V
    lscpu
    printf '%s\n' "Cargo.lock-sha256=$(sha256sum "$crate_dir/Cargo.lock" | cut -d ' ' -f 1)"
    printf '%s\n' "external-revisions-sha256=$(sha256sum "$crate_dir/external-bench/revisions.lock" | cut -d ' ' -f 1)"
    for repository in "$crate_dir" "$crate_dir/../fgf" "$crate_dir/../butterfly-fft" "$crate_dir/../gfm" "$crate_dir/../simdispatch"; do
        if git -C "$repository" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            revision=$(git -C "$repository" rev-parse HEAD)
            dirty=clean
            if ! git -C "$repository" diff --quiet || ! git -C "$repository" diff --cached --quiet; then
                dirty=dirty
            fi
            printf '%s\n' "library=$(basename "$repository");revision=$revision;state=$dirty"
        fi
    done
    cargo run --quiet --release --all-features --example benchmark_metadata
} >"$metadata"

if ! GS_BENCH_ALLOC_CSV=$allocations cargo bench --all-features \
    --bench decoder \
    --bench interpolation \
    --bench products \
    --bench root_extraction \
    --bench scoring \
    -- "$@" >"$results" 2>&1
then
    cat "$results"
    exit 1
fi
cat "$results"
rm -rf "$record_dir/criterion"
cp -R "$crate_dir/target/criterion" "$record_dir/criterion"
printf '%s\n' "benchmark_record=$record_dir"
