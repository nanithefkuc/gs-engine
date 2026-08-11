//! Reusable Guruswami–Sudan list-decoding engine.
//!
//! `gs-engine` uses [`fgf`] for finite-field arithmetic and [`butterfly_fft`]
//! for additive transforms. Checked geometry helpers reject unrepresentable or
//! unallocatable decoder layouts before arithmetic begins.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

pub mod cost;
mod decoder;
pub mod domain;
mod error;
mod evaluate;
pub mod geometry;
pub mod interpolation;
pub mod params;
pub mod poly;
pub mod roots;
mod scratch;

pub use cost::{
    BackendClass, DomainClass, InterpolationBackend, InterpolationCostKey, ProductBackend,
    ProductCostKey, REENCODE_MIN_CODE_LENGTH, REENCODE_RATE_DENOMINATOR, REENCODE_RATE_NUMERATOR,
    ReencodeCostKey, RootBackend, RootCostKey, ScoringBackend, ScoringCostKey, select_interpolation,
    select_product, select_reencode, select_root, select_scoring,
};
pub use decoder::{DecodeError, GsPlan};
pub use domain::{DomainError, EvaluationBackend, EvaluationDomain};
pub use error::ConfigError;
pub use evaluate::{
    BUTTERFLY_FFT_BATCH2_SCORING_CROSSOVER, BUTTERFLY_FFT_BATCH4_SCORING_CROSSOVER,
    BUTTERFLY_FFT_BATCH8_SCORING_CROSSOVER, BUTTERFLY_FFT_BATCH16_SCORING_CROSSOVER,
    BUTTERFLY_FFT_SINGLE_SCORING_CROSSOVER,
};
#[cfg(feature = "internals")]
pub use evaluate::{ScoringStrategy, score_candidates_with_strategy};
#[cfg(feature = "internals")]
pub use interpolation::{
    FastKnhScratch, InterpolationConstraint, InterpolationMonomial, InterpolationProblem,
    MultiplicityPoint, ReferenceInterpolationLimits, interpolate_fast_knh,
    interpolate_fast_knh_into, interpolate_reference, interpolate_reference_nonuniform,
    reference_constraints, reference_monomials,
};
pub use interpolation::{
    InterpolationError, InterpolationPlan, KoetterScratch, MODULE_INTERPOLATION_CROSSOVER,
    ModuleScratch, interpolate_koetter, interpolate_koetter_into, interpolate_koetter_with_scratch,
    interpolate_module, interpolate_module_into,
};
pub use params::{GsParameters, ParameterLimits, ResourceEstimate};
pub use poly::{
    AFFT_BATCH4_CROSSOVER, AFFT_BATCH8_CROSSOVER, AFFT_BATCH16_CROSSOVER, AFFT_PRODUCT_CROSSOVER,
    BivariatePolynomial, Polynomial, PolynomialError, PolynomialProductScratch, ProductError,
    ProductStrategy, SCALAR_AFFT_BATCH4_CROSSOVER, SCALAR_AFFT_BATCH8_CROSSOVER,
    SCALAR_AFFT_BATCH16_CROSSOVER, SCALAR_AFFT_PRODUCT_CROSSOVER, WeightedTerm,
    multiply_batch_truncated,
};
pub use roots::{
    AffineRootFamily, AlekhnovichLimits, AlekhnovichScratch, BaseFieldRoots,
    DEFAULT_ROTH_RUCKENSTEIN_CROSSOVER, RootError, RothRuckensteinLimits, alekhnovich_roots,
    base_field_roots, roth_ruckenstein_roots,
};
pub use scratch::DecodeScratch;
