#!/bin/sh
set -eu

# WP5 Percy++ GS adapter build: compile only the RS-decoder support TUs
# (never the PIR client/server) and link percyxx-gs against NTL + GMP.

revision=b0cbb083b76ee9d55747954cbdb3b878e1dc24c7
url=https://github.com/gfanti/P2P-PIR-Cpp.git
adapter_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=${EXTERNAL_BENCH_ROOT:-"$(CDPATH= cd -- "$adapter_dir/.." && pwd)"}
source_dir=$root/sources/percyxx

# Fetch the pinned upstream revision (gitignored under sources/).
if [ ! -d "$source_dir/.git" ]; then
    mkdir -p "$root/sources"
    git clone --no-checkout "$url" "$source_dir"
fi
git -C "$source_dir" fetch --depth 1 origin "$revision"
git -C "$source_dir" checkout --detach "$revision"

ntl_include=${NTL_INCLUDE:-/usr/include}
# -fpermissive is required: the 2013-era source has minor conformance issues
# the probe downgraded to warnings. -I$source_dir finds rsdecoder.h etc.
cxxflags="-O2 -std=c++11 -fpermissive -I$source_dir -I$ntl_include/NTL"

# REQUIRED one-line source patch: a std::set/std::map comparator's operator()
# is declared non-const (rsdecoder_impl.h:2294); modern libstdc++ rejects this
# with a static_assert. Add `const` to the call operator. Applied to a freshly
# checked-out tree, so it is always reapplied cleanly.
sed -i 's/const FX& b) {/const FX\& b) const {/' "$source_dir/rsdecoder_impl.h"
# Runtime fix (REQUIRED): interpolate_kotter does `delete[] g;` then
# `return g[minindex].first;` — a use-after-free that corrupts the returned
# bivariate polynomial (segfault on gf8, "out of memory" on gf16 when the
# freed FXY is copied). Save the result before deleting. This does not change
# the algorithm, only the order of the delete relative to the copy-out.
perl -0pi -e 's/    delete\[\] g;\n\n    return g\[minindex\]\.first;/    FXY _gs_kotter_result = g[minindex].first;\n    delete[] g;\n    return _gs_kotter_result;/' "$source_dir/rsdecoder_impl.h"

# Selective decoder TUs only. Do NOT compile rsdecoder.cc/recover.cc (they hold
# the ZZ_p explicit specialization and template bodies we do not need for the
# GF2E path). The adapter TU includes both rsdecoder.h and rsdecoder_impl.h, so
# the GF2E specialization is instantiated from the adapter.
support="FXY gf2e pulse percyio percyparams portfolio subset subset_iter"
objs=""
for tu in $support; do
    g++ $cxxflags -c -o "$adapter_dir/$tu.o" "$source_dir/$tu.cc"
    objs="$objs $adapter_dir/$tu.o"
done

g++ $cxxflags -c -o "$adapter_dir/adapter.o" "$adapter_dir/adapter.cc"

# Link: NTL + GMP + m. No PIR client/server objects, no GPL in the controller.
g++ $cxxflags -o "$adapter_dir/percyxx-gs" "$adapter_dir/adapter.o" $objs -lntl -lgmp -lm

# Provenance.
printf '%s\n' "percyxx_revision=$revision"
printf '%s\n' "percyxx_cxxflags=$cxxflags"
printf '%s\n' "percyxx_selective_tus=$support"
printf '%s\n' "percyxx_kotter_uaf_patch=rsdecoder_impl.h: save result before delete[] g in interpolate_kotter"
printf '%s\n' "percyxx_artifact=$adapter_dir/percyxx-gs"
printf '%s\n' "percyxx_link=-lntl -lgmp -lm"

# gf16 isomorphism fingerprint: sha256 of the forward table (65536 uint16 LE).
fingerprint=$("$adapter_dir/percyxx-gs" --fingerprint | sha256sum | cut -d' ' -f1)
printf '%s\n' "percyxx_gf16_fwd_sha256=$fingerprint"
