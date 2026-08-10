# Changelog

All notable changes to gs-engine are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
releases follow [Semantic Versioning](https://semver.org/).

### Added
- Nonuniform-multiplicity interpolation problem under `internals`: new
  `MultiplicityPoint` and `InterpolationProblem` types carry a per-point
  multiplicity, the lower set a fast Kötter–Nielsen–Høholdt backend consumes
  directly. `interpolate_reference_nonuniform` is the explicit Hasse-matrix
  oracle for that problem, sharing monomial/constraint enumeration with the
  uniform reference backend and validating every per-point lower set. The
  existing Kötter and weak-Popov module backends remain differential oracles
  for the uniform case.
- External decoder comparison harness: a standalone controller crate
  (`external-bench/controller`) drives separate-process adapters over a frozen
  `.gsf` fixture corpus with a `.gso` result protocol, field-isomorphism
  verification, and a validate/run/aggregate pipeline. Three adapters are
  wired: DECODING (GPL) and Percy++ (GPL) build verified GF(2)-linear GF16
  isomorphisms and agree with `gs-engine` on the complete frozen set;
  Lambdaworks is labelled native-prime and rejects binary fixtures loudly. GPL
  code is linked only into standalone adapter executables and never into the
  MIT crate or the controller.
- Geometry-aware automatic strategy selection: every interpolation, product,
  candidate-scoring, and root-extraction crossover now resolves through pure,
  backend-explicit cost keys in the new `cost` module (`select_interpolation`,
  `select_product`, `select_scoring`, `select_root`, keyed on
  `BackendClass`/`DomainClass`). Selectors perform no CPU detection —
  classification happens once at stage entry — and the parameter search orders
  tuples with the same interpolation/root work model. Explicit strategy
  overrides are unchanged. Crossover values and their measurement provenance
  moved out of source comments into `BENCHMARKS.md`.
- Domain-specialized interpolation for additive-subspace and affine-coset
  domains: the vanishing polynomial `G(X)` is built from the `butterfly-fft`
  subspace polynomial, and the received-word interpolant `R` is computed by an
  inverse transform plus novel-to-monomial conversion (`O(n log n)`) instead
  of incremental Newton interpolation. Both paths produce byte-identical
  module interpolation results. `InterpolationPlan::new_with_domain` selects
  the strategy from the domain; `interpolate_module_into` accepts an optional
  domain to dispatch the transform path. Changed-word decode on an additive
  domain allocates zero after preparation.
- `GsPlan` now precomputes and owns the received-word-independent interpolation
  invariants (domain vanishing polynomial, its powers, module column shifts,
  and the Newton interpolation basis), exposed as `InterpolationPlan`, with
  `GsPlan::prepared_bytes` reporting the bounded prepared memory before a
  decode.
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
