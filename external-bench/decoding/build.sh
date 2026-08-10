#!/bin/sh
set -eu

release=0.4
archive=decoding-$release.tar.bz2
url=http://www.coincoin169.org/research/decoding/$archive
sha256=0746943594f0a5b2c63d0ec20ad42ceb8b20df211a50c095e4bbf1ff365bea62
root=${EXTERNAL_BENCH_ROOT:-"$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"}
download=$root/sources/$archive
source_dir=$root/sources/decoding-$release
backend=${DECODING_BACKEND:-gf2n_word}

mkdir -p "$root/sources"
if [ ! -f "$download" ]; then
    curl --fail --location "$url" --output "$download"
fi
printf '%s  %s\n' "$sha256" "$download" | sha256sum --check --status
if [ ! -d "$source_dir" ]; then
    tar -xjf "$download" -C "$root/sources"
fi

custom_flags=${CFLAGS:--O3 -DNDEBUG -std=c89}
case "$backend" in
    gf2n_word)
        make -C "$source_dir" CUSTOM_FLAGS="$custom_flags" MPFQ_GF2N_CFLAGS= MPFQ_GF2N_LDFLAGS=
        ;;
    mpfq)
        : "${MPFQ_INCLUDE:?set MPFQ_INCLUDE for the MPFQ headers}"
        : "${MPFQ_LIB:?set MPFQ_LIB for the MPFQ library directory}"
        make -C "$source_dir" \
            CUSTOM_FLAGS="$custom_flags" \
            MPFQ_GF2N_CFLAGS="-I$MPFQ_INCLUDE" \
            MPFQ_GF2N_LDFLAGS="-L$MPFQ_LIB -lmpfq_gf2n"
        ;;
    *)
        printf '%s\n' "unsupported DECODING_BACKEND=$backend" >&2
        exit 2
        ;;
esac

printf '%s\n' "decoding_release=$release"
printf '%s\n' "decoding_sha256=$sha256"
printf '%s\n' "decoding_backend=$backend"
printf '%s\n' "decoding_cflags=$custom_flags"
printf '%s\n' "decoding_source=$source_dir"

############################################################
#
# Build the standalone decoding-gs adapter, linking the
# (GPL) libdecoding.a into this separate executable only.
#
############################################################

adapter_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
adapter_cflags=${ADAPTER_CFLAGS:--O2 -std=gnu11 -Wall -Wextra}

cc $adapter_cflags -I"$source_dir/include" \
    -o "$adapter_dir/decoding-gs" \
    "$adapter_dir/adapter.c" \
    "$source_dir/libdecoding.a" \
    -lgmp -lm

printf '%s\n' "decoding_adapter=$adapter_dir/decoding-gs"
printf '%s\n' "decoding_adapter_cflags=$adapter_cflags"
