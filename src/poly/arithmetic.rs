use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::{FieldKernels, backend_for};
use fgf::ops;

use crate::ConfigError;
use crate::geometry::{checked_product, try_zeroed};

use super::{Polynomial, PolynomialError};

impl<F: FieldKernels> Polynomial<F> {
    /// Add `other` in place.
    pub fn add_assign(&mut self, other: &Self) -> Result<(), ConfigError> {
        self.add_scaled_assign(F::Elem::ONE, other)
    }

    /// Return `self + other`.
    pub fn add(&self, other: &Self) -> Result<Self, ConfigError> {
        let mut result = self.clone();
        result.add_assign(other)?;
        Ok(result)
    }

    /// Add `scale * other` in place.
    pub fn add_scaled_assign(&mut self, scale: F::Elem, other: &Self) -> Result<(), ConfigError> {
        self.add_scaled_packed_at(scale, other.as_packed(), 0)
    }

    pub(crate) fn add_scaled_shifted_assign(
        &mut self,
        scale: F::Elem,
        other: &Self,
        shift: usize,
    ) -> Result<(), ConfigError> {
        self.add_scaled_packed_at(scale, other.as_packed(), shift)
    }

    /// Return `self + scale * other`.
    pub fn add_scaled(&self, scale: F::Elem, other: &Self) -> Result<Self, ConfigError> {
        let mut result = self.clone();
        result.add_scaled_assign(scale, other)?;
        Ok(result)
    }

    /// Multiply every coefficient by `scale` in place.
    pub fn scale_assign(&mut self, scale: F::Elem) {
        if self.is_zero() || scale.is_one() {
            return;
        }
        if scale.is_zero() {
            self.coefficients.clear();
            return;
        }
        if use_packed_kernel::<F>(self.coefficients.len()) {
            ops::mul_assign::<F>(&mut self.coefficients, scale);
        } else {
            for coefficient in self.coefficients.chunks_exact_mut(F::BYTES) {
                F::write(coefficient, F::read(coefficient).mul(scale));
            }
        }
        self.normalize();
    }

    /// Return `scale * self`.
    #[must_use]
    pub fn scaled(&self, scale: F::Elem) -> Self {
        let mut result = self.clone();
        result.scale_assign(scale);
        result
    }

    /// Return `X^amount * self`.
    pub fn shifted(&self, amount: usize) -> Result<Self, ConfigError> {
        if self.is_zero() {
            return Ok(Self::zero());
        }
        let mut result = Self::zero();
        result.add_scaled_packed_at(F::Elem::ONE, self.as_packed(), amount)?;
        Ok(result)
    }

    /// Return `(X + constant) * self`.
    pub fn multiply_x_plus(&self, constant: F::Elem) -> Result<Self, ConfigError> {
        let mut result = self.shifted(1)?;
        result.add_scaled_assign(constant, self)?;
        Ok(result)
    }

    /// Return the schoolbook product.
    pub fn multiply(&self, other: &Self) -> Result<Self, ConfigError> {
        let output_count = match (self.coefficient_count(), other.coefficient_count()) {
            (0, _) | (_, 0) => return Ok(Self::zero()),
            (left, right) => left
                .checked_add(right)
                .and_then(|sum| sum.checked_sub(1))
                .ok_or(ConfigError::GeometryOverflow {
                    context: "polynomial product coefficients",
                })?,
        };
        self.multiply_truncated(other, output_count)
    }

    /// Return the product truncated to coefficients below `coefficient_count`.
    pub fn multiply_truncated(
        &self,
        other: &Self,
        coefficient_count: usize,
    ) -> Result<Self, ConfigError> {
        if self.is_zero() || other.is_zero() || coefficient_count == 0 {
            return Ok(Self::zero());
        }
        let full_count = self
            .coefficient_count()
            .checked_add(other.coefficient_count())
            .and_then(|sum| sum.checked_sub(1))
            .ok_or(ConfigError::GeometryOverflow {
                context: "polynomial product coefficients",
            })?;
        let output_count = coefficient_count.min(full_count);
        let mut result = Self::zero();
        result.resize_coefficients(output_count)?;

        let (source, factors) = if self.coefficient_count() >= other.coefficient_count() {
            (self, other)
        } else {
            (other, self)
        };
        for (shift, scale) in factors.coefficients().enumerate() {
            if shift >= output_count || scale.is_zero() {
                continue;
            }
            let source_count = source.coefficient_count().min(output_count - shift);
            result.add_scaled_packed_at_raw(
                scale,
                &source.as_packed()[..source_count * F::BYTES],
                shift,
            )?;
        }
        result.normalize();
        Ok(result)
    }

    /// Evaluate at one field element with Horner's rule.
    #[must_use]
    pub fn evaluate(&self, point: F::Elem) -> F::Elem {
        self.coefficients()
            .rev()
            .fold(F::Elem::ZERO, |value, coefficient| {
                value.mul(point).add(coefficient)
            })
    }

    /// Evaluate independently at every supplied point.
    pub fn evaluate_many(&self, points: &[F::Elem]) -> Result<Vec<F::Elem>, ConfigError> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(points.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "polynomial evaluations",
                elements: points.len(),
                element_size: core::mem::size_of::<F::Elem>(),
            })?;
        values.extend(points.iter().copied().map(|point| self.evaluate(point)));
        Ok(values)
    }

    /// Evaluate the Hasse derivative of `order` at `point` without allocating.
    #[must_use]
    pub fn evaluate_hasse(&self, point: F::Elem, order: usize) -> F::Elem {
        if order >= self.coefficient_count() {
            return F::Elem::ZERO;
        }
        let mut power = F::Elem::ONE;
        let mut value = F::Elem::ZERO;
        for degree in order..self.coefficient_count() {
            if binomial_odd(degree, order) {
                value = value.add(self.coefficient(degree).mul(power));
            }
            power = power.mul(point);
        }
        value
    }

    /// Return the Hasse derivative of the requested order.
    pub fn hasse_derivative(&self, order: usize) -> Result<Self, ConfigError> {
        if order >= self.coefficient_count() {
            return Ok(Self::zero());
        }
        let output_count = self.coefficient_count() - order;
        let mut coefficients = try_zeroed::<F::Elem>("Hasse derivative", output_count)?;
        for source_degree in order..self.coefficient_count() {
            if binomial_odd(source_degree, order) {
                coefficients[source_degree - order] = self.coefficient(source_degree);
            }
        }
        Self::from_coefficients(&coefficients)
    }

    /// Return the first formal derivative.
    pub fn formal_derivative(&self) -> Result<Self, ConfigError> {
        self.hasse_derivative(1)
    }

    /// Divide by `divisor`, returning quotient and remainder.
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), PolynomialError> {
        let Some(divisor_degree) = divisor.degree() else {
            return Err(PolynomialError::DivisionByZero);
        };
        let Some(dividend_degree) = self.degree() else {
            return Ok((Self::zero(), Self::zero()));
        };
        if dividend_degree < divisor_degree {
            return Ok((Self::zero(), self.clone()));
        }

        let quotient_count = dividend_degree - divisor_degree + 1;
        let mut quotient = try_zeroed::<F::Elem>("polynomial quotient", quotient_count)?;
        let mut remainder = self.clone();
        let divisor_leading_inverse = divisor
            .leading_coefficient()
            .expect("nonzero divisor has a leading coefficient")
            .inv();

        while let Some(remainder_degree) = remainder.degree() {
            if remainder_degree < divisor_degree {
                break;
            }
            let shift = remainder_degree - divisor_degree;
            let scale = remainder
                .leading_coefficient()
                .expect("nonzero remainder has a leading coefficient")
                .mul(divisor_leading_inverse);
            quotient[shift] = quotient[shift].add(scale);
            remainder.add_scaled_packed_at(scale, divisor.as_packed(), shift)?;
        }
        Ok((Self::from_coefficients(&quotient)?, remainder))
    }

    /// Divide exactly, returning an error when the remainder is nonzero.
    pub fn exact_divide(&self, divisor: &Self) -> Result<Self, PolynomialError> {
        let (quotient, remainder) = self.div_rem(divisor)?;
        if remainder.is_zero() {
            Ok(quotient)
        } else {
            Err(PolynomialError::NonExactDivision)
        }
    }

    /// Return a monic copy, leaving zero unchanged.
    #[must_use]
    pub fn monic(&self) -> Self {
        let Some(leading) = self.leading_coefficient() else {
            return Self::zero();
        };
        self.scaled(leading.inv())
    }

    /// Return the monic greatest common divisor.
    pub fn gcd(&self, other: &Self) -> Result<Self, PolynomialError> {
        let mut left = self.clone();
        let mut right = other.clone();
        while !right.is_zero() {
            let (_, remainder) = left.div_rem(&right)?;
            left = right;
            right = remainder;
        }
        Ok(left.monic())
    }

    /// Return `self mod modulus`.
    pub fn remainder(&self, modulus: &Self) -> Result<Self, PolynomialError> {
        self.div_rem(modulus).map(|(_, remainder)| remainder)
    }

    /// Multiply and reduce modulo `modulus`.
    pub fn multiply_mod(&self, other: &Self, modulus: &Self) -> Result<Self, PolynomialError> {
        self.multiply(other)?.remainder(modulus)
    }

    /// Square and reduce modulo `modulus`.
    pub fn square_mod(&self, modulus: &Self) -> Result<Self, PolynomialError> {
        self.multiply_mod(self, modulus)
    }

    /// Raise to `exponent` modulo `modulus` by square-and-multiply.
    pub fn pow_mod(&self, mut exponent: u128, modulus: &Self) -> Result<Self, PolynomialError> {
        if modulus.is_zero() {
            return Err(PolynomialError::DivisionByZero);
        }
        let mut result = Self::one()?.remainder(modulus)?;
        let mut base = self.remainder(modulus)?;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.multiply_mod(&base, modulus)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.square_mod(modulus)?;
            }
        }
        Ok(result)
    }

    /// Compose with the affine polynomial `constant + linear * X`.
    pub fn compose_linear(&self, constant: F::Elem, linear: F::Elem) -> Result<Self, ConfigError> {
        let affine = Self::from_coefficients(&[constant, linear])?;
        let mut result = Self::zero();
        for coefficient in self.coefficients().rev() {
            result = result.multiply(&affine)?;
            if !coefficient.is_zero() {
                let value = result.coefficient(0).add(coefficient);
                result.set_coefficient(0, value)?;
            }
        }
        Ok(result)
    }

    /// Smallest exponent with a nonzero coefficient.
    #[must_use]
    pub fn x_valuation(&self) -> Option<usize> {
        self.coefficients()
            .position(|coefficient| !coefficient.is_zero())
    }

    /// Divide exactly by `X^power`.
    pub fn divide_by_x_power(&self, power: usize) -> Result<Self, PolynomialError> {
        if self.is_zero() || power == 0 {
            return Ok(self.clone());
        }
        if self.x_valuation().is_none_or(|valuation| valuation < power) {
            return Err(PolynomialError::NonExactDivision);
        }
        let byte_offset = power
            .checked_mul(F::BYTES)
            .ok_or(ConfigError::GeometryOverflow {
                context: "polynomial X-power offset",
            })?;
        let packed = self.coefficients[byte_offset..].to_vec();
        Ok(Self::from_packed(packed).expect("coefficient-aligned suffix"))
    }

    /// Return `scale * X^shift * self`.
    pub fn scaled_shifted(&self, scale: F::Elem, shift: usize) -> Result<Self, ConfigError> {
        let mut result = Self::zero();
        result.add_scaled_packed_at(scale, self.as_packed(), shift)?;
        Ok(result)
    }

    fn add_scaled_packed_at(
        &mut self,
        scale: F::Elem,
        source: &[u8],
        shift: usize,
    ) -> Result<(), ConfigError> {
        self.add_scaled_packed_at_raw(scale, source, shift)?;
        self.normalize();
        Ok(())
    }

    #[inline]
    fn add_scaled_packed_at_raw(
        &mut self,
        scale: F::Elem,
        source: &[u8],
        shift: usize,
    ) -> Result<(), ConfigError> {
        if source.is_empty() || scale.is_zero() {
            return Ok(());
        }
        debug_assert_eq!(source.len() % F::BYTES, 0);
        let source_count = source.len() / F::BYTES;
        let required = shift
            .checked_add(source_count)
            .ok_or(ConfigError::GeometryOverflow {
                context: "shifted polynomial coefficient count",
            })?;
        self.resize_coefficients(required)?;
        let start = shift
            .checked_mul(F::BYTES)
            .ok_or(ConfigError::GeometryOverflow {
                context: "shifted polynomial byte offset",
            })?;
        let destination = &mut self.coefficients[start..start + source.len()];
        if use_packed_kernel::<F>(source.len()) {
            ops::mul_add::<F>(destination, scale, source);
        } else {
            for (output, input) in destination
                .chunks_exact_mut(F::BYTES)
                .zip(source.chunks_exact(F::BYTES))
            {
                F::write(output, F::read(output).add(scale.mul(F::read(input))));
            }
        }
        Ok(())
    }

    /// Reuse this polynomial's buffer to hold a copy of `source`.
    pub(crate) fn assign_from(&mut self, source: &Self) {
        self.coefficients.clone_from(&source.coefficients);
    }

    /// Reset to the zero polynomial while retaining allocated capacity.
    pub(crate) fn set_zero(&mut self) {
        self.coefficients.clear();
    }

    /// Write the schoolbook product into reusable output storage.
    pub(crate) fn multiply_into(&self, other: &Self, out: &mut Self) -> Result<(), ConfigError> {
        let output_count = match (self.coefficient_count(), other.coefficient_count()) {
            (0, _) | (_, 0) => {
                out.set_zero();
                return Ok(());
            }
            (left, right) => left
                .checked_add(right)
                .and_then(|sum| sum.checked_sub(1))
                .ok_or(ConfigError::GeometryOverflow {
                    context: "polynomial product coefficients",
                })?,
        };
        self.multiply_truncated_into(other, output_count, out)
    }

    /// Write the characteristic-two square `self^2` into reusable `out`.
    ///
    /// In characteristic two `(sum a_i X^i)^2 = sum a_i^2 X^{2i}`: the cross
    /// terms cancel, so squaring spreads each coefficient to twice its degree
    /// and squares it in the field. This is `O(deg)` rather than the `O(deg^2)`
    /// of a general product, and underlies the modular Frobenius in base-field
    /// factorization.
    pub(crate) fn square_into(&self, out: &mut Self) -> Result<(), ConfigError> {
        out.set_zero();
        let count = self.coefficient_count();
        if count == 0 {
            return Ok(());
        }
        let output_count = 2 * count - 1;
        out.resize_coefficients(output_count)?;
        for degree in 0..count {
            let coefficient = self.coefficient(degree);
            if coefficient.is_zero() {
                continue;
            }
            let squared = coefficient.mul(coefficient);
            let start = 2 * degree * F::BYTES;
            F::write(&mut out.coefficients[start..start + F::BYTES], squared);
        }
        out.normalize();
        Ok(())
    }

    /// Write the truncated product into reusable output storage.
    pub(crate) fn multiply_truncated_into(
        &self,
        other: &Self,
        coefficient_count: usize,
        out: &mut Self,
    ) -> Result<(), ConfigError> {
        out.set_zero();
        if self.is_zero() || other.is_zero() || coefficient_count == 0 {
            return Ok(());
        }
        let full_count = self
            .coefficient_count()
            .checked_add(other.coefficient_count())
            .and_then(|sum| sum.checked_sub(1))
            .ok_or(ConfigError::GeometryOverflow {
                context: "polynomial product coefficients",
            })?;
        let output_count = coefficient_count.min(full_count);
        out.resize_coefficients(output_count)?;

        let (source, factors) = if self.coefficient_count() >= other.coefficient_count() {
            (self, other)
        } else {
            (other, self)
        };
        for (shift, scale) in factors.coefficients().enumerate() {
            if shift >= output_count || scale.is_zero() {
                continue;
            }
            let source_count = source.coefficient_count().min(output_count - shift);
            out.add_scaled_packed_at_raw(
                scale,
                &source.as_packed()[..source_count * F::BYTES],
                shift,
            )?;
        }
        out.normalize();
        Ok(())
    }

    /// Write quotient and remainder into reusable output storage.
    pub(crate) fn div_rem_into(
        &self,
        divisor: &Self,
        quotient: &mut Self,
        remainder: &mut Self,
    ) -> Result<(), PolynomialError> {
        let Some(divisor_degree) = divisor.degree() else {
            return Err(PolynomialError::DivisionByZero);
        };
        quotient.set_zero();
        let Some(dividend_degree) = self.degree() else {
            remainder.set_zero();
            return Ok(());
        };
        remainder.assign_from(self);
        if dividend_degree < divisor_degree {
            return Ok(());
        }
        quotient.resize_coefficients(dividend_degree - divisor_degree + 1)?;
        let divisor_leading_inverse = divisor
            .leading_coefficient()
            .expect("nonzero divisor has a leading coefficient")
            .inv();
        while let Some(remainder_degree) = remainder.degree() {
            if remainder_degree < divisor_degree {
                break;
            }
            let shift = remainder_degree - divisor_degree;
            let scale = remainder
                .leading_coefficient()
                .expect("nonzero remainder has a leading coefficient")
                .mul(divisor_leading_inverse);
            let start = shift * F::BYTES;
            let updated = F::read(&quotient.coefficients[start..start + F::BYTES]).add(scale);
            F::write(&mut quotient.coefficients[start..start + F::BYTES], updated);
            remainder.add_scaled_shifted_assign(scale, divisor, shift)?;
        }
        quotient.normalize();
        Ok(())
    }

    /// Divide exactly by `X^power` into reusable output storage.
    pub(crate) fn divide_by_x_power_into(
        &self,
        power: usize,
        out: &mut Self,
    ) -> Result<(), PolynomialError> {
        if self.is_zero() || power == 0 {
            out.assign_from(self);
            return Ok(());
        }
        if self.x_valuation().is_none_or(|valuation| valuation < power) {
            return Err(PolynomialError::NonExactDivision);
        }
        let byte_offset = power
            .checked_mul(F::BYTES)
            .ok_or(ConfigError::GeometryOverflow {
                context: "polynomial X-power offset",
            })?;
        out.assign_packed(&self.coefficients[byte_offset..])?;
        Ok(())
    }

    /// Overwrite with low-to-high coefficients, reusing existing capacity.
    pub(crate) fn assign_coefficients(
        &mut self,
        coefficients: &[F::Elem],
    ) -> Result<(), ConfigError> {
        self.set_zero();
        let byte_len =
            checked_product("polynomial coefficient bytes", coefficients.len(), F::BYTES)?;
        self.resize_coefficients(coefficients.len())?;
        ops::pack::<F>(&mut self.coefficients[..byte_len], coefficients);
        self.normalize();
        Ok(())
    }
}

pub(crate) const fn binomial_odd(upper: usize, lower: usize) -> bool {
    lower <= upper && (upper & lower) == lower
}

fn use_packed_kernel<F: FieldKernels>(byte_len: usize) -> bool {
    byte_len >= backend_for::<F>().lane_bytes()
}
