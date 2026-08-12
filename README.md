> [!WARNING]
> This library was made with the help of AI. While the library has tests
> to check for regressions, things may break. Audit the code yourself, or with
> your own agent before using.

# gs-engine

`gs-engine` provides checked parameter search, Hasse interpolation, complete
polynomial root extraction, and Hamming-radius filtering — a reusable
Guruswami–Sudan list-decoding engine for Reed–Solomon evaluation codes. The
crate is `no_std` with `alloc` when default features are disabled.

## Usage

The MSRV is Rust 1.89.

`gs-engine` is distributed through git only; it is not published to [crates.io](https://crates.io).

```toml
[dependencies]
gs-engine = { git = "https://github.com/nanithefkuc/gs-engine" }
```

Portable `no_std` builds are also available:

```toml
[dependencies]
gs-engine = { git = "https://github.com/nanithefkuc/gs-engine", default-features = false }
```

### Features

| Feature | Result |
| --- | --- |
| default (`std`, `simd`) | standard-library error integration and runtime-selected SIMD kernels |
| `std` without `simd` | portable kernels with allocation-backed plans |
| `--no-default-features` | `no_std`, `alloc`, portable kernels |
| `internals` | unstable implementation APIs for benchmarking and research; exempt from compatibility guarantees |
| `parallel` (implies `std`) | Rayon over independent decode jobs; default-off |

### Supported fields

The production decoder is validated for the binary extension fields `fgf::Gf8`
and `fgf::Gf16`. Generic APIs require the corresponding `fgf::FieldKernels` or
`butterfly_fft::core::kernel::ButterflyKernels` implementation. Root splitting
rejects fields whose order is not a power of two or whose stable element
representation exceeds 16 bytes; it never falls back to scanning the full field.

### Domain and received-symbol contract

`EvaluationDomain::arbitrary` preserves the supplied distinct-point order.
Additive-subspace and affine-coset constructors derive their point order
directly from the associated `butterfly-fft` plan. `EvaluationDomain::points()`
is the canonical order for encoding and for the received slice passed to
`GsPlan::decode_into`.

Received symbols are canonical `F::Elem` values, one per domain point. The core
decoder has no erasure or sentinel representation: adapters must normalize
external bytes, shortened coordinates, or reliability metadata before calling
it. A received slice with the wrong length is rejected before interpolation.

### Candidate semantics

A successful decode returns distinct normalized polynomials in deterministic
coefficient order. Every returned polynomial:

- has degree at most `GsParameters::max_degree()`;
- is an exact root of the interpolation polynomial after composition; and
- differs from the received word in at most `target_radius` domain positions.

Candidates are message polynomials, not serialized codewords. Protocol-specific
admissibility rules belong in adapters after decoding.

### Resource limits and scratch

Configuration is always caller-bounded:

- `ParameterLimits` caps multiplicity, interpolation `Y` degree, coefficient
  storage, and baseline scratch storage;
- `AlekhnovichLimits` caps work items, intermediate affine families, scratch
  bytes, and output roots;
- `RothRuckensteinLimits` caps fallback work and roots;
- `DecodeScratch` owns reusable interpolation, root, conversion, evaluation,
  distance, and candidate buffers.

`GsPlan::prepare_scratch` reserves geometry-dependent scoring and output
capacity. The first data-dependent interpolation/root pass may still grow
storage. Repeating the same decode with warmed scratch performs no internal heap
allocation.

### Minimal use

```rust,no_run
use fgf::Gf16;
use fgf::field::{Elem, Field};
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan,
    ParameterLimits,
};

let parameters = GsParameters::search::<Gf16>(
    16,
    4,
    6,
    ParameterLimits::new(8, 16, usize::MAX, usize::MAX),
)?;
let domain = EvaluationDomain::<Gf16>::additive_subspace(16)?;
let plan = GsPlan::new(
    parameters,
    domain,
    AlekhnovichLimits::new(1_000_000, 100_000, usize::MAX, usize::MAX, 128),
)?;
let received = vec![<Gf16 as Field>::Elem::ZERO; 16];
let mut scratch = DecodeScratch::new();
let mut candidates = Vec::new();
plan.decode_into(&received, &mut scratch, &mut candidates)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The primary decoding API is `GsPlan::decode_into(received, scratch, output)`.
`GsPlan::decode_scored_into(received, scratch, output, distances)` decodes
identically but also fills a caller-owned `distances` buffer parallel to
`output`, giving each accepted candidate's exact Hamming distance from the
received word without a second evaluation pass — useful when a caller narrows
the list with extra coordinates or an admissibility rule of its own.
For several received words sharing one geometry, `GsPlan::decode_batch_into`
shares the immutable plan across independent scratch instances and, with the
`parallel` feature, spreads the words across the Rayon pool above
`PARALLEL_BATCH_CROSSOVER` words. Output is byte-identical to per-word decoding
in order regardless of the thread schedule.

## Building

`gs-engine` builds on stable Rust (edition 2024, MSRV 1.89) with no extra
tooling or target-feature flags — SIMD kernels are selected at runtime:

```sh
cargo build                        # default: std + simd
cargo build --no-default-features  # portable no_std
cargo test --all-features
```

## Benchmarks and crossover policy

Each benchmark prints the field and the runtime-selected SIMD backend. Run the
matrix explicitly with:

```text
./scripts/run.sh
SIMD_BACKEND=scalar ./scripts/run.sh
```

The command runs all five Criterion groups with `internals` enabled and stores
confidence intervals, Criterion's machine-readable estimates, allocation
count/bytes, retained bytes, hardware, rustc, selected backend, field geometry,
and repository revisions under `target/benchmark-record/`. Arguments are
forwarded to Criterion, so a benchmark ID substring can select one workload.

`gfni` requests fall back to a supported backend on hosts that cannot execute
GFNI. Production crossover constants are exported by the crate. Product
thresholds count coefficients in the full, untruncated product; other thresholds
use code length, weighted input size, or domain points as shown.

| Decision | Scalar | Packed GFNI |
|---|---:|---:|
| Module interpolation vs. Kötter | code length 8 | code length 8 |
| Roth–Ruckenstein vs. Alekhnovich | Roth throughout measured range | weighted size above 20,000 |
| Schoolbook vs. AFFT, 1–3 GF16 products | 511 coefficients | schoolbook throughout field range |
| Schoolbook vs. AFFT, 4–7 GF16 products | 255 coefficients | 65,535 coefficients |
| Schoolbook vs. AFFT, 8–15 GF16 products | 255 coefficients | 32,767 coefficients |
| Schoolbook vs. AFFT, 16+ GF16 products | 127 coefficients | 8,191 coefficients |
| Horner vs. butterfly-fft, 1 candidate | 256 points | 256 points |
| Horner vs. butterfly-fft, 2–3 candidates | 64 points | 64 points |
| Horner vs. butterfly-fft, 4–7 candidates | 64 points | 64 points |
| Horner vs. butterfly-fft, 8–15 candidates | 32 points | 32 points |
| Horner vs. butterfly-fft, 16+ candidates | 16 points | 16 points |

GF8 AFFT products remain schoolbook because AFFT did not win before the
field-sized transform ceiling. `ProductStrategy::Auto` and the default root
crossover inspect the selected backend. An explicit product strategy or
`with_roth_ruckenstein_crossover` override remains backend-independent.

Packed AXPY is selected by `fgf` from the active backend and row width; scalar
fallback remains bit-exact and is continuously checked against the selected
backend. Measurement and reproduction notes are in [BENCHMARKS.md](BENCHMARKS.md).

## License

MIT - see [LICENSE](LICENSE)
