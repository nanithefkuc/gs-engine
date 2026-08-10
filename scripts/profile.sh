#!/bin/sh
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
profile_dir=${GS_PERF_RECORD_DIR:-"$crate_dir/scripts/profiles/2026-08-10-intel-core-ultra-7-258v"}
seconds=${GS_PERF_SECONDS:-1}
mkdir -p "$profile_dir"
backend=$(cargo run --quiet --release --all-features --example benchmark_metadata \
    | sed -n 's/^selected-backend=//p')

{
    printf '%s\n' "perf-record-version=1"
    printf '%s\n' "profile-seconds=$seconds"
    printf '%s\n' "selected-backend=$backend"
    uname -a
    rustc -Vv
    lscpu
    git -C "$crate_dir" rev-parse HEAD
    printf '%s\n' "cargo bench --all-features --bench decoder -- <filter> --profile-time $seconds"
} >"$profile_dir/metadata.txt"

profile_case() {
    tier=$1
    filter=$2
    data=$profile_dir/$tier.data
    stat=$profile_dir/$tier.stat.txt
    report=$profile_dir/$tier.report.txt
    perf stat --output "$stat" -- \
        cargo bench --all-features --bench decoder -- "$filter" --profile-time "$seconds"
    sed -i 's/[[:space:]]*$//' "$stat"
    perf record --quiet --freq 999 --call-graph dwarf --output "$data" -- \
        cargo bench --all-features --bench decoder -- "$filter" --profile-time "$seconds"
    perf report --stdio --percent-limit 1 --input "$data" \
        | sed 's/[[:space:]]*$//' >"$report"
    rm -f "$data"
    printf '%s\n' "tier=$tier;filter=$filter" >>"$profile_dir/metadata.txt"
}

profile_case small "warmed-changed-word/gf8/$backend/additive/n4/k1"
profile_case medium "warmed-changed-word/gf16/$backend/additive/n16/k4"
profile_case large "warmed-changed-word/gf16/$backend/additive/n64/k16"
