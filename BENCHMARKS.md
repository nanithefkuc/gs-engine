# Benchmark record

Every automatic strategy decision in `gs-engine` resolves through the pure
selectors in `src/cost.rs`. This file is the provenance for the crossover
constants those selectors compare against: what set each number, on which
hardware, and how to reproduce it. Source comments carry only a one-line
summary and a pointer here.

## Reproducing

The Criterion corpus and the machine/toolchain metadata are produced by one
command:

```sh
scripts/run.sh                      # full stage + end-to-end corpus
scripts/profile.sh                  # Linux perf counters, all three tiers
```

`scripts/run.sh` records, alongside the results, `uname -a`, `rustc -Vv`,
`cargo -V`, `lscpu`, the selected `fgf` backend, and the HEAD revision of every
ecosystem dependency (`fgf`, `butterfly-fft`, `gfm`, `simdispatch`). The forced
strategies used to bracket each crossover are exercised by the benchmark groups
below via the public override APIs:

- products — `ProductStrategy::{Schoolbook, Afft, Auto}`
  (`cargo bench --all-features --bench products`);
- scoring — `ScoringStrategy::{Horner, ButterflyFft, Auto}`
  (`cargo bench --all-features --bench scoring`);
- root extraction — `AlekhnovichLimits::with_roth_ruckenstein_crossover`
  (`cargo bench --all-features --bench root_extraction`);
- interpolation and end-to-end — `--bench interpolation`, `--bench decoder`.

## Reference host

The committed constants were set on the profile recorded under
`scripts/profiles/2026-08-10-intel-core-ultra-7-258v/`:

- CPU: Intel Core Ultra 7 258V (8 cores, 1 thread/core);
- selected `fgf` backend: `v3_gfni_crypto` (GFNI, 64-byte lanes);
- toolchain: rustc 1.93.0, LLVM 21.1.8, x86_64-unknown-linux-gnu;
- kernel: Linux 7.1.6 x86_64.

Scalar-backend constants were taken from the same host with the backend forced
to the scalar degrade rung. A constant that has not been re-measured on a given
backend keeps the conservative value (the one that switches to the heavier
algorithm later).

## Interpolation — Kötter vs weak-Popov module

Selector: `select_interpolation`. Axis: code length `n`.

| Constant | Value | Meaning |
| --- | ---: | --- |
| `MODULE_INTERPOLATION_CROSSOVER` | 8 | module backend at or above this `n` |

The module backend is ahead at `n = 8` for both GF8/GF16 and scalar/GFNI.
Kötter remains marginally faster for scalar GF16 at `n = 4`, so the crossover is
held at 8 rather than lowered.

## Products — schoolbook vs additive FFT

Selector: `select_product`. Axes: full-product coefficient count, batch size,
field order, backend. Binary-extension fields of order `<= 256` (GF8) never use
AFFT: packed GFNI schoolbook stays ahead across the whole GF8 transform range.
Values are the full-product coefficient count at or above which AFFT wins.

| Constant | Value | Batch | Backend |
| --- | ---: | --- | --- |
| `AFFT_PRODUCT_CROSSOVER` | `usize::MAX` | 1–3 | packed |
| `AFFT_BATCH4_CROSSOVER` | 65 535 | 4–7 | packed |
| `AFFT_BATCH8_CROSSOVER` | 32 767 | 8–15 | packed |
| `AFFT_BATCH16_CROSSOVER` | 8 191 | ≥16 | packed |
| `SCALAR_AFFT_PRODUCT_CROSSOVER` | 511 | 1–3 | scalar |
| `SCALAR_AFFT_BATCH4_CROSSOVER` | 255 | 4–7 | scalar |
| `SCALAR_AFFT_BATCH8_CROSSOVER` | 255 | 8–15 | scalar |
| `SCALAR_AFFT_BATCH16_CROSSOVER` | 127 | ≥16 | scalar |

For GF16 the packed schoolbook multiplication remained ahead over the full
single-product transform range, so `AFFT_PRODUCT_CROSSOVER` is `usize::MAX`
(AFFT is never auto-selected for a single packed product). Scalar crossovers are
lower than the packed ones in every batch bucket — a scalar host adopts AFFT no
later than a packed one — which is the ordering `cost::tests` asserts to guard
against a held-out-backend inversion.

## Candidate scoring — Horner vs butterfly FFT

Selector: `select_scoring`. Axes: points, candidate count, backend. Values are
the point count at or above which packed butterfly-FFT scoring wins; larger
candidate batches amortize the transform and lower the crossover.

| Constant | Value | Candidates |
| --- | ---: | --- |
| `BUTTERFLY_FFT_SINGLE_SCORING_CROSSOVER` | 256 | 1 |
| `BUTTERFLY_FFT_BATCH2_SCORING_CROSSOVER` | 64 | 2–3 |
| `BUTTERFLY_FFT_BATCH4_SCORING_CROSSOVER` | 64 | 4–7 |
| `BUTTERFLY_FFT_BATCH8_SCORING_CROSSOVER` | 32 | 8–15 |
| `BUTTERFLY_FFT_BATCH16_SCORING_CROSSOVER` | 16 | ≥16 |

## Root extraction — Roth–Ruckenstein vs Alekhnovich

Selector: `select_root`. Axes: weighted input coefficients, `Y`-coefficient
count, target precision, backend.

| Constant | Value | Meaning |
| --- | ---: | --- |
| `DEFAULT_ROTH_RUCKENSTEIN_CROSSOVER` | 20 000 | Roth–Ruckenstein at or below this weighted size |

GF16/GFNI first favors divide-and-conquer at weighted size 20 485; the default
keeps Roth–Ruckenstein through 20 000. On a scalar backend no divide-and-conquer
win was observed in the measured range, so the adaptive default keeps
Roth–Ruckenstein everywhere unless
`AlekhnovichLimits::with_roth_ruckenstein_crossover` overrides it.

## Parameter-search score

`GsParameters::search` orders enumerated `(s, ell, D)` tuples with
`cost::interpolation_work` + `cost::root_work` — the same interpolation and root
models the selectors are built around — then by storage, `Y`-degree,
multiplicity, and weighted degree. The score is an integer ordering key, not a
wall-clock estimate; it exists so search deterministically picks one feasible
tuple, never an infeasible one.
