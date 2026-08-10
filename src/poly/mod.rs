//! Dense univariate and bivariate polynomials over `fgf` fields.

mod afft;
mod arithmetic;
mod bivariate;

use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::ops;

use crate::ConfigError;
use crate::geometry::{checked_product, try_zeroed};

pub use afft::{
    AFFT_BATCH4_CROSSOVER, AFFT_BATCH8_CROSSOVER, AFFT_BATCH16_CROSSOVER, AFFT_PRODUCT_CROSSOVER,
    PolynomialProductScratch, ProductError, ProductStrategy, SCALAR_AFFT_BATCH4_CROSSOVER,
    SCALAR_AFFT_BATCH8_CROSSOVER, SCALAR_AFFT_BATCH16_CROSSOVER, SCALAR_AFFT_PRODUCT_CROSSOVER,
    multiply_batch_truncated,
};
pub use bivariate::{BivariatePolynomial, WeightedTerm};

/// Failure during polynomial arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolynomialError {
    /// Checked coefficient geometry or allocation failed.
    Config(ConfigError),
    /// Polynomial division was requested with the zero divisor.
    DivisionByZero,
    /// A division expected to have zero remainder did not.
    NonExactDivision,
}

impl fmt::Display for PolynomialError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::DivisionByZero => formatter.write_str("polynomial division by zero"),
            Self::NonExactDivision => formatter.write_str("polynomial division was not exact"),
        }
    }
}

impl From<ConfigError> for PolynomialError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PolynomialError {}

/// A normalized dense monomial-basis polynomial.
///
/// Coefficients use the field's packed little-endian representation, so wide
/// fixed-scalar operations can execute directly through `fgf` without unsafe
/// casts or representation copies. Zero is represented by an empty buffer;
/// every nonzero value ends in a nonzero coefficient.
#[derive(Clone)]
pub struct Polynomial<F: FieldKernels> {
    coefficients: Vec<u8>,
    field: PhantomData<F>,
}

impl<F: FieldKernels> Polynomial<F> {
    /// The zero polynomial.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            coefficients: Vec::new(),
            field: PhantomData,
        }
    }

    /// Construct a polynomial from low-to-high monomial coefficients.
    pub fn from_coefficients(coefficients: &[F::Elem]) -> Result<Self, ConfigError> {
        let byte_len =
            checked_product("polynomial coefficient bytes", coefficients.len(), F::BYTES)?;
        let mut packed = try_zeroed::<u8>("polynomial coefficients", byte_len)?;
        ops::pack::<F>(&mut packed, coefficients);
        let mut polynomial = Self {
            coefficients: packed,
            field: PhantomData,
        };
        polynomial.normalize();
        Ok(polynomial)
    }

    /// Construct a constant polynomial.
    pub fn constant(value: F::Elem) -> Result<Self, ConfigError> {
        Self::from_coefficients(&[value])
    }

    /// The multiplicative identity polynomial.
    pub fn one() -> Result<Self, ConfigError> {
        Self::constant(F::Elem::ONE)
    }

    /// Construct from packed field elements, or return `None` for a partial
    /// trailing element.
    #[must_use]
    pub fn from_packed(mut coefficients: Vec<u8>) -> Option<Self> {
        if !coefficients.len().is_multiple_of(F::BYTES) {
            return None;
        }
        normalize_bytes::<F>(&mut coefficients);
        Some(Self {
            coefficients,
            field: PhantomData,
        })
    }

    pub(crate) fn assign_packed(&mut self, coefficients: &[u8]) -> Result<(), ConfigError> {
        debug_assert!(coefficients.len().is_multiple_of(F::BYTES));
        let coefficient_count = coefficients.len() / F::BYTES;
        self.resize_coefficients(coefficient_count)?;
        self.coefficients[..coefficients.len()].copy_from_slice(coefficients);
        self.coefficients.truncate(coefficients.len());
        self.normalize();
        Ok(())
    }

    /// Packed low-to-high coefficient bytes.
    #[must_use]
    pub fn as_packed(&self) -> &[u8] {
        &self.coefficients
    }

    /// Number of stored coefficients, zero for the zero polynomial.
    #[must_use]
    pub fn coefficient_count(&self) -> usize {
        self.coefficients.len() / F::BYTES
    }

    /// Degree of a nonzero polynomial.
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        self.coefficient_count().checked_sub(1)
    }

    /// Whether this is the zero polynomial.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Coefficient of `X^degree`, returning zero beyond the stored degree.
    #[must_use]
    pub fn coefficient(&self, degree: usize) -> F::Elem {
        let Some((start, end)) = degree
            .checked_mul(F::BYTES)
            .and_then(|start| start.checked_add(F::BYTES).map(|end| (start, end)))
        else {
            return F::Elem::ZERO;
        };
        let Some(bytes) = self.coefficients.get(start..end) else {
            return F::Elem::ZERO;
        };
        F::read(bytes)
    }

    /// Stored coefficients in low-to-high order.
    pub fn coefficients(
        &self,
    ) -> impl DoubleEndedIterator<Item = F::Elem> + ExactSizeIterator + '_ {
        self.coefficients.chunks_exact(F::BYTES).map(F::read)
    }

    /// Leading coefficient of a nonzero polynomial.
    #[must_use]
    pub fn leading_coefficient(&self) -> Option<F::Elem> {
        self.coefficients().next_back()
    }

    /// Set one coefficient and restore the canonical representation.
    pub fn set_coefficient(&mut self, degree: usize, value: F::Elem) -> Result<(), ConfigError> {
        let required = degree.checked_add(1).ok_or(ConfigError::GeometryOverflow {
            context: "polynomial coefficient count",
        })?;
        self.resize_coefficients(required)?;
        let start = degree * F::BYTES;
        F::write(&mut self.coefficients[start..start + F::BYTES], value);
        self.normalize();
        Ok(())
    }

    /// Discard coefficients at degrees `>= coefficient_count`.
    pub fn truncate(&mut self, coefficient_count: usize) {
        let byte_len = coefficient_count.saturating_mul(F::BYTES);
        if byte_len < self.coefficients.len() {
            self.coefficients.truncate(byte_len);
            self.normalize();
        }
    }

    pub(crate) fn resize_coefficients(
        &mut self,
        coefficient_count: usize,
    ) -> Result<(), ConfigError> {
        let byte_len =
            checked_product("polynomial coefficient bytes", coefficient_count, F::BYTES)?;
        if byte_len > self.coefficients.len() {
            let additional = byte_len - self.coefficients.len();
            self.coefficients
                .try_reserve_exact(additional)
                .map_err(|_| ConfigError::AllocationFailed {
                    context: "polynomial coefficients",
                    elements: byte_len,
                    element_size: 1,
                })?;
            self.coefficients.resize(byte_len, 0);
        }
        Ok(())
    }

    pub(crate) fn normalize(&mut self) {
        normalize_bytes::<F>(&mut self.coefficients);
    }
}

impl<F: FieldKernels> Default for Polynomial<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: FieldKernels> PartialEq for Polynomial<F> {
    fn eq(&self, other: &Self) -> bool {
        self.coefficients == other.coefficients
    }
}

impl<F: FieldKernels> Eq for Polynomial<F> {}

impl<F: FieldKernels> fmt::Debug for Polynomial<F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Polynomial")
            .field(&self.coefficients().collect::<Vec<_>>())
            .finish()
    }
}

fn normalize_bytes<F: Field>(coefficients: &mut Vec<u8>) {
    while !coefficients.is_empty()
        && coefficients[coefficients.len() - F::BYTES..]
            .iter()
            .all(|&byte| byte == 0)
    {
        coefficients.truncate(coefficients.len() - F::BYTES);
    }
}
