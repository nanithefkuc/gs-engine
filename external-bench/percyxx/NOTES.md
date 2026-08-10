# Percy++ GS adapter (percyxx-gs) — notes

Standalone C++ executable wrapping Percy++'s `RSDecoder_GF2E` Guruswami-Sudan
decoder (Kotter interpolation + Roth-Ruckenstein root finding) for the WP5
`.gsf` / `.gso` protocol. Links only NTL + GMP + the Percy++ RS-decoder support
objects; **never** the PIR client/server (`percyclient.cc`, `percyserver.cc`,
`pirclient.cc`, `pirserver.cc`, `distserver.cc`, `threadedserver.cc`). No GPL
code is linked into the MIT `gs-engine` crate or the Rust controller; the
adapter is a separate process.

## Pinned upstream

- repo: https://github.com/gfanti/P2P-PIR-Cpp.git
- revision: `b0cbb083b76ee9d55747954cbdb3b878e1dc24c7`
- fetched under `external-bench/sources/percyxx` (gitignored)

## Build

    cd external-bench/percyxx && sh build.sh

`build.sh` fetches/checks out the pinned revision, applies the two source
patches below, compiles the 8 selective decoder TUs directly into the adapter
(it does **not** build `libpercyclient.a`), compiles `adapter.cc`, and links
`percyxx-gs` against `-lntl -lgmp -lm`. It also prints provenance and the gf16
isomorphism fingerprint.

### cxxflags

`-O2 -std=c++11 -fpermissive -I$source_dir -I/usr/include/NTL`

`-fpermissive` is required: the 2013-era source has minor conformance issues
the probe downgraded to warnings.

### Selective TUs (the only Percy++ objects linked)

    FXY gf2e pulse percyio percyparams portfolio subset subset_iter

`rsdecoder.cc` and `recover.cc` are **not** compiled (they hold the
`RSDecoder_ZZ_p` explicit specialization and template bodies we do not need
for the GF2E path). The adapter TU includes both `rsdecoder.h` and
`rsdecoder_impl.h`, so the `RSDecoder_GF2E` specialization is instantiated from
the adapter TU. The adapter also defines the two `extern` bookkeeping globals
(`hasseop`, `kotter_usec`) that `rsdecoder_impl.h` references and that
`rsdecoder.cc` would otherwise have defined.

## Source patches (both required; neither changes the algorithm)

1. **const comparator (rsdecoder_impl.h:2294).** A `std::set`/`std::map`
   comparator's `operator()` is declared non-`const`; modern libstdc++ rejects
   this with a `static_assert`. Patch:
   `sed -i 's/const FX& b) {/const FX\& b) const {/' "$src/rsdecoder_impl.h"`

2. **interpolate_kotter use-after-free (rsdecoder_impl.h ~292-294).** Upstream
   does `delete[] g;` then `return g[minindex].first;` — a use-after-free that
   corrupts the returned bivariate polynomial (segfault on gf8, "out of memory"
   on gf16 when the freed `FXY` is copied). The verified probe only exercised
   compile/link, not this runtime path. Patch (save the result before
   deleting):
   `perl -0pi -e 's/    delete\[\] g;\n\n    return g\[minindex\]\.first;/    FXY _gs_kotter_result = g[minindex].first;\n    delete[] g;\n    return _gs_kotter_result;/' "$src/rsdecoder_impl.h"`

Both patches are applied to a freshly checked-out tree each build, so they are
always reapplied cleanly.

## Exposing the GS entry

`findpolys_gs` is `private` in `rsdecoder.h`, exposed under
`#if defined(TEST_FINDPOLYS)`. The adapter defines `TEST_FINDPOLYS 1` before
including `rsdecoder.h`/`rsdecoder_impl.h`.

Signature:

    vector<RecoveryPoly<GF2EX>> findpolys_gs(
        unsigned int k,        // max_degree (v = k-1)
        unsigned int t,        // min agreement = n - target_radius
        const vector<unsigned short>& goodservers,  // 0..n-1
        const vec_GF2E& indices, const vec_GF2E& shares);

`interpolate_kotter` computes multiplicity `m` and list size `L` internally
from `(v, n, t)`, so the decode is `radius`-matched (same code/radius,
decoder-chosen internal params); the adapter emits `status=radius`.

The `v == 0` case (`k == 1`, max_degree 0) divides by zero in Kotter's
`L = (m*t - 1)/v`, so the adapter handles degree-0 directly (all constants
agreeing with >= `t` received points) and still emits `status=radius`.

## Field setup

### gf8 — identity map

NTL `GF2E` initialized with the AES modulus `x^8+x^4+x^3+x+1` (0x11B):
`SetCoeff(P,8,4,3,1,0)`. This is byte-for-byte the canonical gf8 field; NTL
`rep(e)` holds the AES byte. The map is the identity, verified **exhaustively**
(all 256 elements: preserve `+`, `*`, `0`, `1`, inverse).

- canonical modulus: **0x11B**
- NTL modulus:     **0x11B** (same)

### gf16 — explicit GF(2)-linear isomorphism

Percy++'s native GF(2^16) modulus is `x^16+x^5+x^3+x^2+1` (**0x1002D**) — from
the commented `#if 0 // GF(2^16)` block in `rsdecoder.cc` (bits 16,5,3,2,0).
This **differs** from the canonical tower `gf8[u]/(u^2+u+0x20)`
(gf8 = AES 0x11B; element = two LE bytes, low = degree-0 gf8 component).

The adapter implements the canonical tower arithmetic clean-room in C++
(gf8 add = XOR, gf8 mul = AES via Percy++'s verified `GF28_mult_table`; gf16 add
= XOR of both bytes; gf16 mul = tower multiply reducing `u^2 = u + 0x20`), then
builds the isomorphism:

1. find a primitive element `g_canon` of the canonical tower (order 65535, via
   the prime-factor test on 65535 = 3*5*17*257);
2. compute its minimal polynomial over GF(2) (Frobenius product
   `prod_{k=0}^{15}(x - g^(2^k))`);
3. lift that minimal polynomial to a `GF2EX` over the NTL field and scan NTL
   elements by **ascending 16-bit rep**, picking the **first** that is a root
   of the canonical minimal polynomial. This is deterministic — NTL's
   `FindRoots` uses randomized equal-degree factoring (a different Frobenius
   conjugate each run), which would make the fingerprint unstable. Any root of
   a primitive polynomial is itself primitive, so `g_percy` has order 65535
   (confirmed with the prime-factor test);
4. align powers: `fwd[g_canon^i] = g_percy^i` (and `fwd[0]=0`), `inv` inverse.
- Percy++ native modulus:    **0x1002D** (`x^16+x^5+x^3+x^2+1`)

### Isomorphism verification (gf16)

- `fwd`/`inv` round-trip for all 65536 elements (exhaustive bijection);
- `0 -> 0`, `1 -> 1`;
- `+` and `*` homomorphism on a large deterministic sample
  (all 65536 `a` x a 29-element sample of `b`, including 0,1,small primes,
  structural values `0x0100`/`0x0020`/`0x8000`, and field extremes);
- `inverse` homomorphism on the nonzero sample (canonical Fermat inverse
  `a^65534` vs NTL `inv`).

Result: **verified**. If verification ever fails, gf16 fixtures emit
`status=unsupported:<reason>` rather than a wrong candidate set.

### Isomorphism fingerprint

sha256 of the forward table (65536 `uint16` little-endian = 131072 bytes,
written by `percyxx-gs --fingerprint`):

    percyxx_gf16_fwd_sha256 = d98841e7dc2d3719e2a3b6d6f0e1a079b49f0a6cded53f8a4dc63e09d8cd4aa2

The adapter parses the fixture line-oriented (keys in order; `expected-*` and
the GS-geometry hints `multiplicity`/`y-degree`/`weighted-degree` are ignored
by the decoder). It maps support+received canonical bytes into NTL `GF2E`
(identity for gf8, `fwd[]` for gf16), calls
`findpolys_gs(max_degree, n - target_radius, goods, indices, shares)`, maps
each returned `.phi`'s coefficients back to canonical bytes (`inv[]` for gf16),
and prints `status=radius` then one `candidate=<hex>,<hex>...` line per
polynomial (constant term first, no trailing zero coefficients, zero poly =
single zero element). All parsing/mapping/startup is outside any timed region.
Malformed input -> `status=error:<message>`.

## Correctness gate (self-validated)

    cd external-bench/controller
    cargo run --release -- run $(pwd)/../percyxx/percyxx-gs ../fixtures

Result (all four fixtures `RadiusMatched`; no `Discrepancy`, no `Error`):

    fixture                          domain     class            status
    gf16-arbitrary-15-5-radius-4     arbitrary  RadiusMatched    Radius
    gf16-arbitrary-15-5-radius-6     arbitrary  RadiusMatched    Radius
    gf8-additive-4-1-radius-2       additive   RadiusMatched    Radius
    gf8-arbitrary-7-3-radius-1       arbitrary  RadiusMatched    Radius

## Exact commands

Build:

    cd external-bench/percyxx && sh build.sh

Run the controller gate:

    cd external-bench/controller && cargo run --release -- run "$(pwd)/../percyxx/percyxx-gs" ../fixtures

Run the adapter directly on one fixture:

    external-bench/percyxx/percyxx-gs external-bench/fixtures/<name>.gsf

Dump the gf16 forward table for hashing:

    external-bench/percyxx/percyxx-gs --fingerprint | sha256sum
