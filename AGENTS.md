# AGENTS.md — gs-engine

Contributor guide. For what the library *is* and how to use it, read the
rustdoc; this file covers the things a change can silently break.

## What this crate is

A reusable Guruswami–Sudan list-decoding engine for Reed–Solomon evaluation
codes. It provides checked parameter search, Hasse interpolation, complete
polynomial root extraction, and Hamming-radius filtering. The crate is
`no_std` with `alloc` when default features are disabled.

Field arithmetic and vector kernels come from [`fgf`](https://github.com/nanithefkuc/fgf);
the additive-FFT transform comes from [`cafft`](https://github.com/nanithefkuc/cafft);
GF linear algebra comes from [`gfm`](https://github.com/nanithefkuc/gfm). This
crate owns the GS algorithm itself — parameter search, interpolation, root
extraction, and candidate scoring.

## Invariants (do not break)

- The decoder returns **message polynomials**, not serialized codewords.
  Protocol-specific admissibility rules belong in adapters, not here.
- `GsPlan::decode_into` returns distinct normalized polynomials in
  deterministic coefficient order. Changing the order is a breaking change.
- Configuration is always caller-bounded: `ParameterLimits`,
  `AlekhnovichLimits`, `RothRuckensteinLimits`, and `DecodeScratch` cap every
  resource. A caller must be able to bound the worst case before calling.
- Repeating a decode with warmed scratch performs no internal heap allocation.
  This is tested, not assumed — `tests/decode_allocations.rs` counts allocations
  with a global allocator.

## Dependencies

- `fgf` — field arithmetic, packed-element kernels, backend dispatch. Pinned by
  git revision. The `SIMD_BACKEND` env var (owned by `simdispatch`) selects the
  runtime backend, downgrade-only.
- `cafft` — additive FFT for batch polynomial multiplication.
- `gfm` — GF linear algebra for module interpolation.

**Do not write `unsafe` SIMD in this crate.** All intrinsics live upstream in
`fgf` and `cafft`. The crate root carries `#![forbid(unsafe_code)]`.

## Public surface

The compatibility promise covers the crate root re-exports and the public
modules. The `internals` feature exposes the explicit reference interpolation
backend and its monomial/constraint helpers for benchmarking and research;
nothing behind it is a compatibility promise.

## Code conventions

- Rust 1.89 or newer, edition 2024.
- No `unsafe` code (`#![forbid(unsafe_code)]` at the crate root).
- Field arithmetic goes through `fgf`; never hand-roll a field loop.
- `mod.rs` and `lib.rs` hold declarations only — module docs, `mod`, `pub use`,
  and plain type declarations. No function bodies, no `impl` blocks.
- Do not put development history in doc comments: no milestone tags, no
  references to superseded designs, no phase numbering.

## Build & test

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

Any performance change MUST be measured through the criterion harness with
`--save-baseline` / `--baseline`. Do not land a performance change on the
strength of reasoning alone.