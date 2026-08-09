use alloc::vec::Vec;
use core::fmt;

use cafft::basis::{conversion_scratch_elements, monomial_to_novel_bytes, novel_to_monomial_bytes};
use cafft::core::kernel::ButterflyKernels;
use cafft::core::transform::TransformPlan;
use cafft::error::{PlanError, TransformLengthError};
use fgf::kernel::{Backend, backend_for};
use fgf::ops;

use crate::ConfigError;

use super::Polynomial;

/// Packed-kernel full-product threshold for a single product.
///
/// Packed GFNI schoolbook multiplication remained ahead over the full GF16
/// transform range, so automatic selection does not choose AFFT for this lane count.
pub const AFFT_PRODUCT_CROSSOVER: usize = usize::MAX;

/// Packed-kernel crossover for batches of four to seven products.
pub const AFFT_BATCH4_CROSSOVER: usize = 65_535;

/// Packed-kernel crossover for batches of eight to fifteen products.
pub const AFFT_BATCH8_CROSSOVER: usize = 32_767;

/// Packed-kernel crossover for batches of at least sixteen products.
pub const AFFT_BATCH16_CROSSOVER: usize = 8_191;

/// Scalar-kernel crossover for one to three products.
pub const SCALAR_AFFT_PRODUCT_CROSSOVER: usize = 511;

/// Scalar-kernel crossover for batches of four to seven products.
pub const SCALAR_AFFT_BATCH4_CROSSOVER: usize = 255;

/// Scalar-kernel crossover for batches of eight to fifteen products.
pub const SCALAR_AFFT_BATCH8_CROSSOVER: usize = 255;

/// Scalar-kernel crossover for batches of at least sixteen products.
pub const SCALAR_AFFT_BATCH16_CROSSOVER: usize = 127;

fn auto_crossover<F: ButterflyKernels>(pair_count: usize) -> usize {
    if F::ORDER <= 256 {
        return usize::MAX;
    }
    let scalar = backend_for::<F>() == Backend::Scalar;
    match (scalar, pair_count) {
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

/// Algorithm selection for polynomial product batches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductStrategy {
    /// Use the measured crossover and fall back when the transform is unavailable.
    Auto,
    /// Always use truncated schoolbook multiplication.
    Schoolbook,
    /// Require the AFFT backend.
    Afft,
}

/// Failure during a batched polynomial product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductError {
    /// Checked storage geometry or allocation failed.
    Config(ConfigError),
    /// The requested transform domain cannot be constructed.
    Plan(PlanError),
    /// A conversion or transform buffer has inconsistent geometry.
    Transform(TransformLengthError),
}

impl From<ConfigError> for ProductError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<PlanError> for ProductError {
    fn from(error: PlanError) -> Self {
        Self::Plan(error)
    }
}

impl From<TransformLengthError> for ProductError {
    fn from(error: TransformLengthError) -> Self {
        Self::Transform(error)
    }
}

impl fmt::Display for ProductError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::Plan(error) => error.fmt(formatter),
            Self::Transform(error) => error.fmt(formatter),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ProductError {}

/// Reusable transform and byte-row storage for polynomial products.
pub struct PolynomialProductScratch<F: ButterflyKernels> {
    plan: Option<TransformPlan<F>>,
    operands: Vec<u8>,
    products: Vec<u8>,
    conversion: Vec<u8>,
}

impl<F: ButterflyKernels> PolynomialProductScratch<F> {
    /// Construct empty product scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            plan: None,
            operands: Vec::new(),
            products: Vec::new(),
            conversion: Vec::new(),
        }
    }

    /// Retained operand-row capacity in bytes.
    #[must_use]
    pub fn operand_capacity_bytes(&self) -> usize {
        self.operands.capacity()
    }

    /// Retained product-row capacity in bytes.
    #[must_use]
    pub fn product_capacity_bytes(&self) -> usize {
        self.products.capacity()
    }
}

impl<F: ButterflyKernels> Default for PolynomialProductScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Multiply independent pairs, truncate every result to `coefficient_count`,
/// and write them to caller-owned output storage.
///
/// AFFT packs every left/right operand across byte-row columns, performs one
/// pair of forward transforms, uses FFF elementwise multiplication per point,
/// and inverse-transforms all products as a second packed batch.
pub fn multiply_batch_truncated<F: ButterflyKernels>(
    pairs: &[(&Polynomial<F>, &Polynomial<F>)],
    coefficient_count: usize,
    strategy: ProductStrategy,
    scratch: &mut PolynomialProductScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), ProductError> {
    output.clear();
    output
        .try_reserve(pairs.len())
        .map_err(|_| ConfigError::AllocationFailed {
            context: "polynomial product outputs",
            elements: pairs.len(),
            element_size: core::mem::size_of::<Polynomial<F>>(),
        })?;
    if pairs.is_empty() {
        return Ok(());
    }

    let mut max_full_count = 0_usize;
    for &(left, right) in pairs {
        let full_count = full_product_count(left, right)?;
        max_full_count = max_full_count.max(full_count);
    }
    if max_full_count == 0 || coefficient_count == 0 {
        output.resize_with(pairs.len(), Polynomial::zero);
        return Ok(());
    }

    let use_afft = match strategy {
        ProductStrategy::Schoolbook => false,
        ProductStrategy::Afft => true,
        ProductStrategy::Auto => {
            let crossover = auto_crossover::<F>(pairs.len());
            max_full_count >= crossover
        }
    };
    let Some(transform_size) = max_full_count.checked_next_power_of_two() else {
        if strategy == ProductStrategy::Afft {
            return Err(ConfigError::GeometryOverflow {
                context: "AFFT polynomial product size",
            }
            .into());
        }
        return schoolbook_batch(pairs, coefficient_count, output);
    };

    if !use_afft {
        return schoolbook_batch(pairs, coefficient_count, output);
    }
    if scratch.plan.as_ref().map(TransformPlan::size) != Some(transform_size) {
        match TransformPlan::<F>::new(transform_size) {
            Ok(plan) => scratch.plan = Some(plan),
            Err(error) if strategy == ProductStrategy::Auto => {
                let _ = error;
                return schoolbook_batch(pairs, coefficient_count, output);
            }
            Err(error) => return Err(error.into()),
        }
    }

    let pair_bytes = pairs
        .len()
        .checked_mul(F::BYTES)
        .ok_or(ConfigError::GeometryOverflow {
            context: "AFFT product row bytes",
        })?;
    let operand_row_bytes = pair_bytes
        .checked_mul(2)
        .ok_or(ConfigError::GeometryOverflow {
            context: "AFFT operand row bytes",
        })?;
    let operand_bytes =
        transform_size
            .checked_mul(operand_row_bytes)
            .ok_or(ConfigError::GeometryOverflow {
                context: "AFFT operand bytes",
            })?;
    let product_bytes =
        transform_size
            .checked_mul(pair_bytes)
            .ok_or(ConfigError::GeometryOverflow {
                context: "AFFT product bytes",
            })?;
    let conversion_bytes = conversion_scratch_elements(transform_size)
        .checked_mul(operand_row_bytes)
        .ok_or(ConfigError::GeometryOverflow {
            context: "AFFT conversion bytes",
        })?;
    ensure_len(&mut scratch.operands, operand_bytes, "AFFT operands")?;
    ensure_len(&mut scratch.products, product_bytes, "AFFT products")?;
    ensure_len(
        &mut scratch.conversion,
        conversion_bytes,
        "AFFT conversion scratch",
    )?;
    scratch.operands[..operand_bytes].fill(0);

    for (lane, &(left, right)) in pairs.iter().enumerate() {
        write_lane::<F>(
            &mut scratch.operands[..operand_bytes],
            operand_row_bytes,
            lane * F::BYTES,
            left,
        );
        write_lane::<F>(
            &mut scratch.operands[..operand_bytes],
            operand_row_bytes,
            pair_bytes + lane * F::BYTES,
            right,
        );
    }

    let plan = scratch
        .plan
        .as_ref()
        .expect("a matching AFFT plan was prepared");
    monomial_to_novel_bytes::<F>(
        &mut scratch.operands[..operand_bytes],
        operand_row_bytes,
        plan,
        &mut scratch.conversion[..conversion_bytes],
    )?;
    plan.forward_bytes(&mut scratch.operands[..operand_bytes], operand_row_bytes)?;

    for (operand_row, product_row) in scratch.operands[..operand_bytes]
        .chunks_exact(operand_row_bytes)
        .zip(scratch.products[..product_bytes].chunks_exact_mut(pair_bytes))
    {
        ops::mul_elementwise::<F>(
            product_row,
            &operand_row[..pair_bytes],
            &operand_row[pair_bytes..],
        );
    }
    plan.inverse_bytes(&mut scratch.products[..product_bytes], pair_bytes)?;
    novel_to_monomial_bytes::<F>(
        &mut scratch.products[..product_bytes],
        pair_bytes,
        plan,
        &mut scratch.conversion[..conversion_scratch_elements(transform_size) * pair_bytes],
    )?;

    for (lane, &(left, right)) in pairs.iter().enumerate() {
        let full_count = full_product_count(left, right)?;
        let result_count = full_count.min(coefficient_count);
        let byte_len = result_count
            .checked_mul(F::BYTES)
            .ok_or(ConfigError::GeometryOverflow {
                context: "AFFT result bytes",
            })?;
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(byte_len)
            .map_err(|_| ConfigError::AllocationFailed {
                context: "AFFT result coefficients",
                elements: byte_len,
                element_size: 1,
            })?;
        packed.resize(byte_len, 0);
        for degree in 0..result_count {
            let source = degree * pair_bytes + lane * F::BYTES;
            let destination = degree * F::BYTES;
            packed[destination..destination + F::BYTES]
                .copy_from_slice(&scratch.products[source..source + F::BYTES]);
        }
        output.push(Polynomial::from_packed(packed).expect("complete packed field elements"));
    }
    Ok(())
}

fn schoolbook_batch<F: ButterflyKernels>(
    pairs: &[(&Polynomial<F>, &Polynomial<F>)],
    coefficient_count: usize,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), ProductError> {
    for &(left, right) in pairs {
        output.push(left.multiply_truncated(right, coefficient_count)?);
    }
    Ok(())
}

fn full_product_count<F: ButterflyKernels>(
    left: &Polynomial<F>,
    right: &Polynomial<F>,
) -> Result<usize, ConfigError> {
    match (left.coefficient_count(), right.coefficient_count()) {
        (0, _) | (_, 0) => Ok(0),
        (left, right) => left
            .checked_add(right)
            .and_then(|sum| sum.checked_sub(1))
            .ok_or(ConfigError::GeometryOverflow {
                context: "polynomial product coefficients",
            }),
    }
}

fn write_lane<F: ButterflyKernels>(
    rows: &mut [u8],
    row_len: usize,
    lane_offset: usize,
    polynomial: &Polynomial<F>,
) {
    for (degree, coefficient) in polynomial.coefficients().enumerate() {
        let offset = degree * row_len + lane_offset;
        F::write(&mut rows[offset..offset + F::BYTES], coefficient);
    }
}

fn ensure_len(
    values: &mut Vec<u8>,
    required: usize,
    context: &'static str,
) -> Result<(), ConfigError> {
    if required > values.len() {
        values
            .try_reserve_exact(required - values.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context,
                elements: required,
                element_size: 1,
            })?;
        values.resize(required, 0);
    }
    Ok(())
}
