# Changelog

All notable changes to gs-engine are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
releases follow [Semantic Versioning](https://semver.org/).
### Added
- Optional parallel batch decoding behind the new `parallel` feature
  (default-off, requires `std`). `GsPlan::decode_batch_into` decodes several
  received words against one shared immutable plan, spreading the words across
  the Rayon pool above `PARALLEL_BATCH_CROSSOVER` words and decoding in order
  below it. Each word owns its scratch and output buffer, so the jobs are fully
  independent and the result is byte-identical to per-word `decode_into` in
  order, regardless of the thread schedule (`tests/parallel.rs` checks this
  across repeated runs and thread counts). A new `DecodeError::BatchLengthMismatch`
  rejects mismatched `received`/`scratches`/`outputs` slices before any work
  runs. Single-word `decode_into` and the `no_std` core are unchanged when the
  feature is off; on the reference host a 16-word GF16 `n64` batch decodes
- External harness batch comparison: the controller gains a `bench-batch`
  subcommand and a `gs-engine-batch-v1` fixture format (documented in
  `external-bench/fixtures/FORMAT.md`) carrying several received words that
  share one geometry and expected candidate set. `bench-batch` times
  `GsPlan::decode_batch_into` (parallel, shared plan) against per-word
  `decode_into` in order (sequential, shared plan) and against the external
  adapters decoding each word in a separate process — the honest "no batch
  API" baseline for Percy++ and DECODING. `validate` now accepts batch
  fixtures and checks every word decodes to the shared set.
- High-rate re-encoding decode path: `GsPlan` now decodes through a
  factor-reduced Guruswami–Sudan module when the geometry is high-rate and long
  enough. It selects the first `k = w + 1` support points deterministically,
  interpolates a degree-`w` helper `e(X)` through them, shifts the received word
  so those coordinates vanish, and builds the interpolation module over the
  remaining `n - k` points from the reduced interpolant `R̃` and reduced
  vanishing polynomial `G_rem`. The shared re-encoding vanishing polynomial `Psi`
  is a prefactor of every reduced row (characteristic-two Sierpiński parity keeps
  the rows sparse); it is divided back out when reconstructing the full
  interpolation polynomial, whose roots are unshifted by `e(X)` before scoring.
  Transformed and direct paths return byte-identical candidate lists, verified
  against the direct module as a differential oracle, and the warmed changed-word
  path allocates nothing. A conservative pure selector (`select_reencode`,
  rate `k/n >= 3/4` and `n >= 32`) drives automatic use; `GsPlan::with_reencode`
  is the explicit override and `GsPlan::uses_reencode` reports the choice. On the
  reference host the path is up to ~1.4× faster end-to-end at high rate,
  including shift and unshift costs, while tiny and low-rate geometries stay on
  the direct module.
- Nonuniform-multiplicity interpolation problem under `internals`: new
  `MultiplicityPoint` and `InterpolationProblem` types carry a per-point
  multiplicity, the lower set a fast Kötter–Nielsen–Høholdt backend consumes
  directly. `interpolate_reference_nonuniform` is the explicit Hasse-matrix
  oracle for that problem, sharing monomial/constraint enumeration with the
  uniform reference backend and validating every per-point lower set. The
  existing Kötter and weak-Popov module backends remain differential oracles
- Fast Kötter–Nielsen–Høholdt interpolation under `internals`: a
  transformation-matrix KNH backend (`interpolate_fast_knh`,
  `interpolate_fast_knh_into`, `FastKnhScratch`) records the per-point
  elementary updates in an explicit polynomial-matrix transform and applies it
  to the identity basis at the end. It satisfies every Hasse constraint and
  matches the Kötter and module backends on the uniform problem. The classical
  Kötter and weak-Popov paths are retained below the measured crossover; the
  divide-and-conquer combine (`T₂·T₁` via polynomial-matrix multiplication over
  a product tree of vanishing polynomials) is scaffolded for the
  asymptotic speedup.
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
- Root extraction was refined for factor-heavy inputs. The Alekhnovich
  divide-and-conquer leaf now factors the base field through a pooled
  `FieldRootScratch` instead of allocating a fresh factorization workspace per
  leaf, and its modular Frobenius uses characteristic-two squaring
  (`P^2 = sum a_i^2 X^{2i}`) rather than a general product. Forced
  divide-and-conquer root extraction is 30–48% faster across the GF8/GF16
  synthetic and real-`Q` benchmark fixtures with no small-geometry regression.
  Per-family completion and `Q(X,f(X)) == 0` verification are isolated into an
  independent branch step prepared for optional parallel execution, and the
  Roth–Ruckenstein/Alekhnovich crossover is now validated against real
  interpolation polynomials, not only the synthetic four-root product. Output
  root sets, candidate-count bounds, and work/family/scratch failure behavior
  are unchanged.
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
