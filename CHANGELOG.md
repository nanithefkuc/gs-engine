# Changelog

All notable changes to gs-engine are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `GsPlan` now precomputes and owns the received-word-independent interpolation
  invariants (domain vanishing polynomial, its powers, module column shifts, and
  the Newton interpolation basis), exposed as `InterpolationPlan`, with
  `GsPlan::prepared_bytes` reporting the bounded prepared memory before a decode.
- `interpolate_module_into` plus a reusable `ModuleScratch`, and, under the
  `internals` feature, `DecodeScratch` capacity inspectors for benchmark
  diagnostics.

### Changed
- `GsPlan::decode_into` is now a streaming path: after preparation and warm-up
  it alternates between different received words with no internal heap
  allocation, executing interpolation, root extraction, and scoring every call.
  Root extraction reuses scratch pools throughout, and the previous hidden
  exact-repeat memoization was removed in favor of this always-on reuse.
- Polynomial products now reuse output buffers, defer normalization to operation
  boundaries, and avoid materializing coefficients outside the requested
  precision. Affine root transforms retain prefix-power, descriptor, product,
  and output storage between calls.
- Weak-Popov interpolation stores rows in one packed slab with cached degree
  and leading-term metadata. Product benchmarks now cover sparse, imbalanced,
  truncated, and batched geometries; interpolation benchmarks report both
  one-shot and reusable module execution.
- Updated `scripts/profiles/2026-08-10-intel-core-ultra-7-258v` records the
  resulting hot-loop instruction and allocation profile on the reference host.
- Replaced aggregate benchmark binaries with Criterion stage and end-to-end
  groups, explicit strategy comparisons, allocation/retained-memory reporting,
  reproducibility metadata, frozen GF8/GF16 candidate fixtures, and external
  decoder source pins.
- Moved shifted weak-Popov row reduction to `gfm::weak_popov`, retaining the
existing polynomial representation, deterministic leading-term order, and
interpolation results.
- Replaced the former `fff` dependency with `fgf` and updated `cafft` to the
shared field types and dispatch backend.
