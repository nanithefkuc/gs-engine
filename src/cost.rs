//! Backend-explicit cost keys and pure strategy selectors.
//!
//! Every automatic strategy decision in the decoder resolves through this
//! module. A selector is a *pure* function of a small cost key: it performs no
//! CPU detection and reads no environment variable, so it is deterministic and
//! directly testable across hypothetical backends. Detection happens exactly
//! once at a stage boundary via [`BackendClass::detect`]; the resulting class is
//! then threaded into the key.
//!
//! The crossover constants these selectors compare against, and the exact
//! measurement commands and hardware that set them, are recorded in
//! `BENCHMARKS.md`; source carries only a one-line pointer.

use fgf::kernel::{Backend, FieldKernels, backend_for};

use crate::domain::EvaluationBackend;

/// Backend capability class consumed by cost keys.
///
/// Derived once from the upstream `fgf`/`simdispatch` selected backend. It does
/// not perform detection itself; [`BackendClass::detect`] is the single seam
/// that queries the host, and the selectors below take a [`BackendClass`] by
/// value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendClass {
    lane_bytes: usize,
    scalar: bool,
}

impl BackendClass {
    /// Build a class from an explicit lane width and scalar flag.
    ///
    /// Prefer [`BackendClass::detect`] on a real decode path; this constructor
    /// exists so selectors can be exercised for held-out backends in tests.
    #[must_use]
    pub const fn new(lane_bytes: usize, scalar: bool) -> Self {
        Self { lane_bytes, scalar }
    }

    /// Classify the backend `fgf` selected for field `F`.
    ///
    /// This is the only cost-model function that observes the host.
    #[must_use]
    pub fn detect<F: FieldKernels>() -> Self {
        let backend = backend_for::<F>();
        Self {
            lane_bytes: backend.lane_bytes(),
            scalar: backend == Backend::Scalar,
        }
    }

    /// SIMD lane width in bytes reported by the selected backend.
    #[must_use]
    pub const fn lane_bytes(self) -> usize {
        self.lane_bytes
    }

    /// Whether the selected backend has no wide SIMD kernels.
    #[must_use]
    pub const fn is_scalar(self) -> bool {
        self.scalar
    }
}

/// Evaluation-domain shape used by interpolation cost keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainClass {
    /// Arbitrary distinct points; incremental interpolation and product `G`.
    Arbitrary,
    /// Additive subspace with a `butterfly-fft` transform plan.
    Additive,
    /// Affine coset with a shifted `butterfly-fft` transform plan.
    Affine,
}

impl From<EvaluationBackend> for DomainClass {
    fn from(backend: EvaluationBackend) -> Self {
        match backend {
            EvaluationBackend::Horner => DomainClass::Arbitrary,
            EvaluationBackend::ButterflyFftAdditive => DomainClass::Additive,
            EvaluationBackend::ButterflyFftAffineCoset => DomainClass::Affine,
        }
    }
}

/// Interpolation-backend choice for one geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationBackend {
    /// Iterative Kötter/KNH interpolation.
    Koetter,
    /// Weak-Popov module interpolation.
    Module,
}

/// Polynomial-product backend choice for one batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductBackend {
    /// Truncated schoolbook multiplication.
    Schoolbook,
    /// Packed additive-FFT multiplication.
    Afft,
}

/// Candidate-scoring backend choice for one candidate set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoringBackend {
    /// Per-candidate Horner evaluation.
    Horner,
    /// Packed butterfly-FFT evaluation.
    ButterflyFft,
}

/// Root-extraction backend choice for one interpolation polynomial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootBackend {
    /// Coefficient-prefix Roth–Ruckenstein lifting.
    RothRuckenstein,
    /// Divide-and-conquer Alekhnovich lifting.
    Alekhnovich,
}

/// Cost key for choosing an interpolation backend.
///
/// The decision axis is the number of evaluation points; the remaining fields
/// pin the geometry that a future retuning may weigh and keep the key aligned
/// with the parameter-search work model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterpolationCostKey {
    /// Field element width in bytes.
    pub field_bytes: usize,
    /// Selected backend class.
    pub backend: BackendClass,
    /// Domain shape.
    pub domain: DomainClass,
    /// Number of evaluation points `n`.
    pub points: usize,
    /// Interpolation multiplicity `s`.
    pub multiplicity: usize,
    /// Maximum interpolation `Y`-degree `ell`.
    pub y_degree: usize,
    /// `(1, max_degree)` weighted-degree bound `D`.
    pub weighted_degree: usize,
    /// Whether the plan prepared transform/vanishing data for this domain.
    pub prepared: bool,
}

/// Cost key for choosing a polynomial-product backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductCostKey {
    /// Coefficients in the longer batch left operand.
    pub left_coefficients: usize,
    /// Coefficients in the longer batch right operand.
    pub right_coefficients: usize,
    /// Full (untruncated) product coefficients driving transform size.
    pub output_coefficients: usize,
    /// Number of independent pairs in the batch.
    pub batch: usize,
    /// Field order; binary-extension fields of order `<= 256` never use AFFT.
    pub field_order: u128,
    /// Selected backend class.
    pub backend: BackendClass,
}

/// Cost key for choosing a candidate-scoring backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoringCostKey {
    /// Number of evaluation points.
    pub points: usize,
    /// Number of candidates to score.
    pub candidates: usize,
    /// Total coefficients across all candidates.
    pub total_coefficients: usize,
    /// Selected backend class.
    pub backend: BackendClass,
}

/// Cost key for choosing a root-extraction backend.
///
/// `roth_ruckenstein_crossover`/`backend_adaptive` carry the caller's root
/// policy so the selector stays pure while honoring explicit overrides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootCostKey {
    /// Weighted input coefficients (precision times `Y`-coefficient count).
    pub weighted_coefficients: usize,
    /// `Y`-coefficient count of the interpolation polynomial.
    pub y_degree: usize,
    /// Target composition precision.
    pub target_precision: usize,
    /// Selected backend class.
    pub backend: BackendClass,
    /// Weighted-size threshold at or below which Roth–Ruckenstein is used.
    pub roth_ruckenstein_crossover: usize,
    /// Whether the crossover adapts to a scalar backend.
    pub backend_adaptive: bool,
}

/// Choose the interpolation backend for one geometry. Pure. See `BENCHMARKS.md`.
#[must_use]
pub fn select_interpolation(key: InterpolationCostKey) -> InterpolationBackend {
    if key.points >= crate::interpolation::MODULE_INTERPOLATION_CROSSOVER {
        InterpolationBackend::Module
    } else {
        InterpolationBackend::Koetter
    }
}

/// Butterfly-FFT scoring crossover in points for a candidate-count bucket.
#[must_use]
pub fn scoring_crossover(candidates: usize) -> usize {
    use crate::evaluate::{
        BUTTERFLY_FFT_BATCH2_SCORING_CROSSOVER, BUTTERFLY_FFT_BATCH4_SCORING_CROSSOVER,
        BUTTERFLY_FFT_BATCH8_SCORING_CROSSOVER, BUTTERFLY_FFT_BATCH16_SCORING_CROSSOVER,
        BUTTERFLY_FFT_SINGLE_SCORING_CROSSOVER,
    };
    match candidates {
        0 | 1 => BUTTERFLY_FFT_SINGLE_SCORING_CROSSOVER,
        2 | 3 => BUTTERFLY_FFT_BATCH2_SCORING_CROSSOVER,
        4..=7 => BUTTERFLY_FFT_BATCH4_SCORING_CROSSOVER,
        8..=15 => BUTTERFLY_FFT_BATCH8_SCORING_CROSSOVER,
        _ => BUTTERFLY_FFT_BATCH16_SCORING_CROSSOVER,
    }
}

/// Choose the candidate-scoring backend. Pure. See `BENCHMARKS.md`.
#[must_use]
pub fn select_scoring(key: ScoringCostKey) -> ScoringBackend {
    if key.points >= scoring_crossover(key.candidates) {
        ScoringBackend::ButterflyFft
    } else {
        ScoringBackend::Horner
    }
}

/// AFFT product crossover in full-product coefficients for a batch bucket.
#[must_use]
pub fn product_crossover(scalar: bool, batch: usize) -> usize {
    use crate::poly::{
        AFFT_BATCH4_CROSSOVER, AFFT_BATCH8_CROSSOVER, AFFT_BATCH16_CROSSOVER, AFFT_PRODUCT_CROSSOVER,
        SCALAR_AFFT_BATCH4_CROSSOVER, SCALAR_AFFT_BATCH8_CROSSOVER, SCALAR_AFFT_BATCH16_CROSSOVER,
        SCALAR_AFFT_PRODUCT_CROSSOVER,
    };
    match (scalar, batch) {
        (true, 0..=3) => SCALAR_AFFT_PRODUCT_CROSSOVER,
        (true, 4..=7) => SCALAR_AFFT_BATCH4_CROSSOVER,
        (true, 8..=15) => SCALAR_AFFT_BATCH8_CROSSOVER,
        (true, _) => SCALAR_AFFT_BATCH16_CROSSOVER,
        (false, 0..=3) => AFFT_PRODUCT_CROSSOVER,
        (false, 4..=7) => AFFT_BATCH4_CROSSOVER,
        (false, 8..=15) => AFFT_BATCH8_CROSSOVER,
        (false, _) => AFFT_BATCH16_CROSSOVER,
    }
}

/// Choose the polynomial-product backend. Pure. See `BENCHMARKS.md`.
#[must_use]
pub fn select_product(key: ProductCostKey) -> ProductBackend {
    if key.field_order <= 256 {
        return ProductBackend::Schoolbook;
    }
    if key.output_coefficients >= product_crossover(key.backend.is_scalar(), key.batch) {
        ProductBackend::Afft
    } else {
        ProductBackend::Schoolbook
    }
}

/// Choose the root-extraction backend. Pure. See `BENCHMARKS.md`.
#[must_use]
pub fn select_root(key: RootCostKey) -> RootBackend {
    let crossover = if key.backend_adaptive && key.backend.is_scalar() {
        usize::MAX
    } else {
        key.roth_ruckenstein_crossover
    };
    if key.weighted_coefficients <= crossover {
        RootBackend::RothRuckenstein
    } else {
        RootBackend::Alekhnovich
    }
}

/// Interpolation work term of the parameter-search score.
///
/// Constraints times basis rows times SIMD vector blocks per row; the score
/// orders enumerated `(s, ell, D)` tuples and is not a wall-clock estimate.
#[must_use]
pub fn interpolation_work(constraints: usize, basis_rows: usize, vector_blocks: usize) -> u128 {
    (constraints as u128)
        .saturating_mul(basis_rows as u128)
        .saturating_mul(vector_blocks as u128)
}

/// Root work term of the parameter-search score.
#[must_use]
pub fn root_work(max_degree: usize, basis_rows: usize, multiplicity: usize) -> u128 {
    (max_degree as u128 + 1)
        .saturating_mul(basis_rows as u128)
        .saturating_mul(basis_rows as u128)
        .saturating_mul(multiplicity as u128)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCALAR: BackendClass = BackendClass::new(1, true);
    const SIMD: BackendClass = BackendClass::new(64, false);

    #[test]
    fn selectors_are_deterministic() {
        let key = ProductCostKey {
            left_coefficients: 100,
            right_coefficients: 100,
            output_coefficients: 100_000,
            batch: 8,
            field_order: 65_536,
            backend: SIMD,
        };
        assert_eq!(select_product(key), select_product(key));
    }

    #[test]
    fn binary_field_never_uses_afft() {
        let key = ProductCostKey {
            left_coefficients: usize::MAX / 2,
            right_coefficients: usize::MAX / 2,
            output_coefficients: usize::MAX,
            batch: 32,
            field_order: 256,
            backend: SIMD,
        };
        assert_eq!(select_product(key), ProductBackend::Schoolbook);
    }

    #[test]
    fn product_backend_ordering_has_no_inversion() {
        // A scalar host has slower schoolbook, so it adopts AFFT no later than a
        // packed host: its crossover is at most the packed one in every batch
        // bucket. This is the held-out-backend property that guards against a
        // selector inversion when the measured backend differs from the host.
        for batch in [1usize, 4, 8, 16, 64] {
            assert!(product_crossover(true, batch) <= product_crossover(false, batch));
        }
    }

    #[test]
    fn scoring_crossover_is_monotone_in_batch() {
        // Larger batches amortize the transform, so the crossover never rises
        // as the candidate bucket grows.
        let buckets = [1usize, 2, 4, 8, 16, 64];
        for pair in buckets.windows(2) {
            assert!(scoring_crossover(pair[0]) >= scoring_crossover(pair[1]));
        }
    }

    #[test]
    fn scalar_root_backend_prefers_roth_ruckenstein() {
        let key = RootCostKey {
            weighted_coefficients: usize::MAX,
            y_degree: 4,
            target_precision: 16,
            backend: SCALAR,
            roth_ruckenstein_crossover: 20_000,
            backend_adaptive: true,
        };
        assert_eq!(select_root(key), RootBackend::RothRuckenstein);
    }

    #[test]
    fn root_override_defeats_backend_adaptation() {
        let key = RootCostKey {
            weighted_coefficients: 1,
            y_degree: 4,
            target_precision: 16,
            backend: SCALAR,
            roth_ruckenstein_crossover: 0,
            backend_adaptive: false,
        };
        assert_eq!(select_root(key), RootBackend::Alekhnovich);
    }

    #[test]
    fn interpolation_switches_to_module_at_the_crossover() {
        let base = InterpolationCostKey {
            field_bytes: 2,
            backend: SIMD,
            domain: DomainClass::Additive,
            points: crate::interpolation::MODULE_INTERPOLATION_CROSSOVER,
            multiplicity: 2,
            y_degree: 4,
            weighted_degree: 17,
            prepared: true,
        };
        assert_eq!(select_interpolation(base), InterpolationBackend::Module);
        let below = InterpolationCostKey {
            points: base.points - 1,
            ..base
        };
        assert_eq!(select_interpolation(below), InterpolationBackend::Koetter);
    }
}
