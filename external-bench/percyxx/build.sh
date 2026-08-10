#!/bin/sh
set -eu

revision=b0cbb083b76ee9d55747954cbdb3b878e1dc24c7
url=https://github.com/gfanti/P2P-PIR-Cpp.git
root=${EXTERNAL_BENCH_ROOT:-"$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"}
source_dir=$root/sources/percyxx

if [ ! -d "$source_dir/.git" ]; then
    mkdir -p "$root/sources"
    git clone --no-checkout "$url" "$source_dir"
fi
git -C "$source_dir" fetch --depth 1 origin "$revision"
git -C "$source_dir" checkout --detach "$revision"

ntl_include=${NTL_INCLUDE:-/usr/include}
cxxflags=${CXXFLAGS:--O3 -DNDEBUG -Wall -Wextra -std=c++11}
make -C "$source_dir" \
    CXXFLAGS="$cxxflags -I$ntl_include" \
    libpercyclient.a

printf '%s\n' "percyxx_revision=$revision"
printf '%s\n' "percyxx_cxxflags=$cxxflags -I$ntl_include"
printf '%s\n' "percyxx_artifact=$source_dir/libpercyclient.a"
