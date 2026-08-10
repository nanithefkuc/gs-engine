use alloc::vec::Vec;
use core::fmt;

use butterfly_fft::basis::{
    conversion_scratch_elements, monomial_to_novel_bytes, novel_to_monomial_bytes,
};
use butterfly_fft::core::kernel::ButterflyKernels;
use butterfly_fft::core::transform::TransformPlan;
use butterfly_fft::error::{PlanError, TransformLengthError};
use fgf::ops;

use crate::ConfigError;

use super::Polynomial;

/// AFFT product crossover in full-product coefficients, one to three packed products. See `BENCHMARKS.md`.
pub const AFFT_PRODUCT_CROSSOVER: usize = usize::MAX;

/// AFFT product crossover, four to seven packed products. See `BENCHMARKS.md`.
pub const AFFT_BATCH4_CROSSOVER: usize = 65_535;

/// AFFT product crossover, eight to fifteen packed products. See `BENCHMARKS.md`.
pub const AFFT_BATCH8_CROSSOVER: usize = 32_767;

/// AFFT product crossover, sixteen or more packed products. See `BENCHMARKS.md`.
pub const AFFT_BATCH16_CROSSOVER: usize = 8_191;

/// AFFT product crossover, one to three scalar products. See `BENCHMARKS.md`.
pub const SCALAR_AFFT_PRODUCT_CROSSOVER: usize = 511;

/// AFFT product crossover, four to seven scalar products. See `BENCHMARKS.md`.
pub const SCALAR_AFFT_BATCH4_CROSSOVER: usize = 255;

/// AFFT product crossover, eight to fifteen scalar products. See `BENCHMARKS.md`.
pub const SCALAR_AFFT_BATCH8_CROSSOVER: usize = 255;

/// AFFT product crossover, sixteen or more scalar products. See `BENCHMARKS.md`.
pub const SCALAR_AFFT_BATCH16_CROSSOVER: usize = 127;

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
    pub(crate) affine_powers: Vec<Polynomial<F>>,
    pub(crate) affine_products: Vec<Polynomial<F>>,
    pub(crate) affine_pairs: Vec<(usize, usize)>,
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
            affine_powers: Vec::new(),
            affine_products: Vec::new(),
            affine_pairs: Vec::new(),
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
/// pair of forward transforms, uses FGF elementwise multiplication per point,
/// and inverse-transforms all products as a second packed batch.
pub fn multiply_batch_truncated<F: ButterflyKernels>(
    pairs: &[(&Polynomial<F>, &Polynomial<F>)],
    coefficient_count: usize,
    strategy: ProductStrategy,
    scratch: &mut PolynomialProductScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), ProductError> {
    multiply_batch_truncated_with(
        pairs.len(),
        |index| pairs[index],
        coefficient_count,
        strategy,
        scratch,
        output,
    )
}

pub(crate) fn multiply_batch_truncated_with<'a, F, P>(
    pair_count: usize,
    pair: P,
    coefficient_count: usize,
    strategy: ProductStrategy,
    scratch: &mut PolynomialProductScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), ProductError>
where
    F: ButterflyKernels,
    P: Copy + Fn(usize) -> (&'a Polynomial<F>, &'a Polynomial<F>),
{
    prepare_output(output, pair_count)?;
    if pair_count == 0 {
        return Ok(());
    }

    let mut max_full_count = 0_usize;
    let mut max_left = 0_usize;
    let mut max_right = 0_usize;
    for index in 0..pair_count {
        let (left, right) = pair(index);
        let full_count = full_product_count(left, right)?;
        max_full_count = max_full_count.max(full_count);
        max_left = max_left.max(left.coefficient_count());
        max_right = max_right.max(right.coefficient_count());
    }
    if max_full_count == 0 || coefficient_count == 0 {
        for polynomial in output {
            polynomial.set_zero();
        }
        return Ok(());
    }

    let use_afft = match strategy {
        ProductStrategy::Schoolbook => false,
        ProductStrategy::Afft => true,
        ProductStrategy::Auto => {
            crate::cost::select_product(crate::cost::ProductCostKey {
                left_coefficients: max_left,
                right_coefficients: max_right,
                output_coefficients: max_full_count,
                batch: pair_count,
                field_order: F::ORDER,
                backend: crate::cost::BackendClass::detect::<F>(),
            }) == crate::cost::ProductBackend::Afft
        }
    };
    let Some(transform_size) = max_full_count.checked_next_power_of_two() else {
        if strategy == ProductStrategy::Afft {
            return Err(ConfigError::GeometryOverflow {
                context: "AFFT polynomial product size",
            }
            .into());
        }
        return schoolbook_batch_with(pair_count, pair, coefficient_count, output);
    };

    if !use_afft {
        return schoolbook_batch_with(pair_count, pair, coefficient_count, output);
    }
    if scratch.plan.as_ref().map(TransformPlan::size) != Some(transform_size) {
        match TransformPlan::<F>::new(transform_size) {
            Ok(plan) => scratch.plan = Some(plan),
            Err(error) if strategy == ProductStrategy::Auto => {
                let _ = error;
                return schoolbook_batch_with(pair_count, pair, coefficient_count, output);
            }
            Err(error) => return Err(error.into()),
        }
    }

    let pair_bytes = pair_count
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

    for lane in 0..pair_count {
        let (left, right) = pair(lane);
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

    for (lane, polynomial) in output.iter_mut().enumerate().take(pair_count) {
        let (left, right) = pair(lane);
        let full_count = full_product_count(left, right)?;
        let result_count = full_count.min(coefficient_count);
        let byte_len = result_count
            .checked_mul(F::BYTES)
            .ok_or(ConfigError::GeometryOverflow {
                context: "AFFT result bytes",
            })?;
        polynomial.set_zero();
        polynomial.resize_coefficients(result_count)?;
        for degree in 0..result_count {
            let source = degree * pair_bytes + lane * F::BYTES;
            let destination = degree * F::BYTES;
            polynomial.coefficients[destination..destination + F::BYTES]
                .copy_from_slice(&scratch.products[source..source + F::BYTES]);
        }
        debug_assert_eq!(polynomial.as_packed().len(), byte_len);
        polynomial.normalize();
    }
    Ok(())
}

fn schoolbook_batch_with<'a, F, P>(
    pair_count: usize,
    pair: P,
    coefficient_count: usize,
    output: &mut [Polynomial<F>],
) -> Result<(), ProductError>
where
    F: ButterflyKernels,
    P: Fn(usize) -> (&'a Polynomial<F>, &'a Polynomial<F>),
{
    for (index, polynomial) in output.iter_mut().enumerate().take(pair_count) {
        let (left, right) = pair(index);
        left.multiply_truncated_into(right, coefficient_count, polynomial)?;
    }
    Ok(())
}

fn prepare_output<F: ButterflyKernels>(
    output: &mut Vec<Polynomial<F>>,
    count: usize,
) -> Result<(), ConfigError> {
    if output.capacity() < count {
        output
            .try_reserve(count - output.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "polynomial product outputs",
                elements: count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })?;
    }
    while output.len() < count {
        output.push(Polynomial::zero());
    }
    output.truncate(count);
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
