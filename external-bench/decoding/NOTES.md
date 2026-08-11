# decoding-gs adapter notes

Standalone `.gso` comparison adapter wrapping Guillaume Quintin's **DECODING
0.4** (GPL). The GPL library is linked **only** into this separate executable
(`decoding-gs`); it is never a dependency of the MIT `gs-engine` crate or the
Rust controller.

## Files
- `adapter.c`   — the adapter (clean-room canonical arithmetic + iso + GS glue).
- `build.sh`    — fetches/builds `libdecoding.a`, then builds `decoding-gs`.
- `decoding-gs` — the built executable (invoked as `decoding-gs <fixture.gsf>`).

## DECODING ring / irreducible polynomials
Backend: `gf2n_word` (`include/decoding/rings/gf2n_word.c`,
`gf2n_word_irr.h`). Fixed irreducible polynomials for the two degrees used:

| field | ext. degree | DECODING irreducible poly            | mask     |
|-------|-------------|--------------------------------------|----------|
| gf8   | m = 8       | x^8 + x^4 + x^3 + x + 1              | `0x11B`  |
| gf16  | m = 16      | x^16 + x^5 + x^3 + x + 1             | `0x1002B`|

The gf8 polynomial (`0x11B`) is **identical** to the canonical AES field; the
gf16 polynomial (`0x1002B`) is a monolithic GF(2^16) basis, unrelated to the
canonical tower — an explicit isomorphism is required for gf16.

## Canonical field definitions (fixture element encoding)
Implemented clean-room in `adapter.c` from the contract / `FORMAT.md`:
- **gf8**  : `gf2[x]/(x^8+x^4+x^3+x+1)` (AES `0x11B`), little-endian polynomial
  basis, one byte per element.
- **gf16** : `gf8[u]/(u^2+u+0x20)`, gf8 = AES `0x11B`; two little-endian
  components (low byte = degree-0 gf8 component, high byte = degree-1). Product
  rule used: `(a0+a1 u)(b0+b1 u) = (a0 b0 + 0x20 a1 b1) + (a0 b1 + a1 b0 + a1 b1) u`.

## GF(2)-linear isomorphism (canonical <-> DECODING)
Built by matching a primitive element through its minimal polynomial, then
mapping generator powers (`build_iso`): find a canonical primitive `alpha`,
compute its GF(2) minimal polynomial `p`, find a DECODING root `beta` of `p`,
set `phi(alpha^i) = beta^i`, `phi(0) = 0`. Support+received are mapped IN with
the forward table; codewords are mapped OUT with the inverse table. All mapping
happens at startup, outside any timed region.

Verification (`verify_iso`) before any decode — preserves `+`, `*`, `0`, `1`,
inverse:
- **gf8**  : exhaustive over all 256 elements (all 65536 pairs). PASS.
- **gf16** : deterministic 4,000,000-pair sample (fixed LCG seed `0xC0FFEE`).
  PASS. (On failure the adapter emits `status=unsupported:...` for gf16.)

Isomorphism parameters (reproduce with `decoding-gs --iso 8|16`):

| field | canonical primitive alpha | minpoly(alpha) bits | DECODING image beta |
|-------|---------------------------|---------------------|---------------------|
| gf8   | `0x3`                     | `0x11D`             | `0x3`               |
| gf16  | `0x108`                   | `0x17EED`           | `0x5354`            |

Forward-table fingerprints (`decoding-gs --dump 8|16 | sha256sum`; raw
little-endian table, 1 byte/entry for gf8, 2 bytes/entry for gf16):
- gf8  forward table sha256 = `40aff2e9d2d8922e47afd4648e6967497158785fbd1da870e7110266bf944880`
- gf16 forward table sha256 = `a174342e9ee48b13114b36370dfad2daa5b4c1bd38b26e98cd0a6a72b1d0ae80`

## Decode path
- **k >= 2**: map support/received into the DECODING field, call
  `rs_code_guruswami_sudan_koetter(&c, rs, y, tau)` with `tau = target-radius`
  (the library chooses its own multiplicity / list size), map returned
  codewords back to canonical, then interpolate the degree-`<k` message
  polynomial by Lagrange over the canonical support and emit its coefficients
  constant-term-first (trailing zeros trimmed).
- **k == 1**: DECODING's GS parameter formulas divide by `(k-1)`, so the
  repetition code is decoded directly in the canonical field — every field
  constant whose codeword lies within `tau` of the received word. This is the
  exact complete set for the radius.

Because the library selects its own multiplicity/list size rather than the
fixture's `(s, ell)` geometry, every fixture is reported as `status=radius`
(honest: same code and radius, complete candidate set). The set still equals
the frozen expected set, so the controller classes each as **RadiusMatched**.

## Build flags
- Library: `make ... CUSTOM_FLAGS="-O3 -DNDEBUG -std=c89"` (backend `gf2n_word`,
  `MPFQ_GF2N_*` empty), links `-lgmp`.
- Adapter: `cc -O2 -std=gnu11 -Wall -Wextra -I<source>/include -o decoding-gs
  adapter.c <source>/libdecoding.a -lgmp -lm` (overridable via `ADAPTER_CFLAGS`).

## Correctness gate result
`cd external-bench/controller && cargo run --release -- run \
  $(pwd)/../decoding/decoding-gs ../fixtures`

| fixture                        | class          |
|--------------------------------|----------------|
| gf16-arbitrary-15-5-radius-4   | RadiusMatched  |
| gf16-arbitrary-15-5-radius-6   | RadiusMatched  |
| gf8-additive-4-1-radius-2      | RadiusMatched  |
| gf8-arbitrary-7-3-radius-1     | RadiusMatched  |

No Discrepancy / Error. Gate exit code 0.
