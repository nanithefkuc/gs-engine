#!/bin/sh
set -eu

revision=3c8d8f65546cde6e847dd29b2ef6aefc38c0895a
url=https://github.com/lambdaclass/lambdaworks.git
root=${EXTERNAL_BENCH_ROOT:-"$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"}
source_dir=$root/sources/lambdaworks

if [ ! -d "$source_dir/.git" ]; then
    mkdir -p "$root/sources"
    git clone --filter=blob:none --no-checkout "$url" "$source_dir"
fi
git -C "$source_dir" fetch --depth 1 origin "$revision"
git -C "$source_dir" checkout --detach "$revision"

rustflags=${RUSTFLAGS:--C target-cpu=native}
# The reed-solomon-codes example is its own workspace and ships no lockfile at
# the pinned revision, so the lock is generated here; the revision stays pinned.
RUSTFLAGS=$rustflags cargo build \
    --release \
    --manifest-path "$source_dir/examples/reed-solomon-codes/Cargo.toml"

# Build the standalone native-prime comparison adapter. It is its own workspace
# and links nothing from gs-engine, the controller, or lambdaworks; it only
# implements the .gso protocol. The controller run gate invokes the executable
# at lambdaworks/lambdaworks-gs, so publish it there.
adapter_dir=$root/lambdaworks/adapter
RUSTFLAGS=$rustflags cargo build \
    --release \
    --manifest-path "$adapter_dir/Cargo.toml"
cp -f "$adapter_dir/target/release/lambdaworks-gs" "$root/lambdaworks/lambdaworks-gs"

printf '%s\n' "lambdaworks_revision=$revision"
printf '%s\n' "lambdaworks_rustflags=$rustflags"
printf '%s\n' "lambdaworks_manifest=$source_dir/examples/reed-solomon-codes/Cargo.toml"
printf '%s\n' "lambdaworks_adapter=$root/lambdaworks/lambdaworks-gs"
