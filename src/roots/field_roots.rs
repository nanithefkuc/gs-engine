use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use crate::{ConfigError, Polynomial};

use super::{BaseFieldRoots, RootError};

/// Caller-owned reusable storage for base-field root factorization.
///
/// Every intermediate polynomial in `gcd(p, X^|F| + X)`, the deterministic
/// trace splitting, and the factor stack is drawn from these buffers, so a
/// warmed extraction over a changed input performs no heap allocation.
pub(crate) struct FieldRootScratch<F: FieldKernels> {
    x: Polynomial<F>,
    pow_result: Polynomial<F>,
    pow_result_next: Polynomial<F>,
    pow_base: Polynomial<F>,
    pow_base_next: Polynomial<F>,
    mul_tmp: Polynomial<F>,
    quot: Polynomial<F>,
    vanishing: Polynomial<F>,
    base_factor: Polynomial<F>,
    gcd_left: Polynomial<F>,
    gcd_right: Polynomial<F>,
    gcd_rem: Polynomial<F>,
    trace_acc: Polynomial<F>,
    trace_term: Polynomial<F>,
    trace_term_next: Polynomial<F>,
    trace_poly: Polynomial<F>,
    split_left: Polynomial<F>,
    split_right: Polynomial<F>,
    factors: Vec<Polynomial<F>>,
    pool: Vec<Polynomial<F>>,
}

impl<F: FieldKernels> FieldRootScratch<F> {
    pub(crate) const fn new() -> Self {
        Self {
            x: Polynomial::zero(),
            pow_result: Polynomial::zero(),
            pow_result_next: Polynomial::zero(),
            pow_base: Polynomial::zero(),
            pow_base_next: Polynomial::zero(),
            mul_tmp: Polynomial::zero(),
            quot: Polynomial::zero(),
            vanishing: Polynomial::zero(),
            base_factor: Polynomial::zero(),
            gcd_left: Polynomial::zero(),
            gcd_right: Polynomial::zero(),
            gcd_rem: Polynomial::zero(),
            trace_acc: Polynomial::zero(),
            trace_term: Polynomial::zero(),
            trace_term_next: Polynomial::zero(),
            trace_poly: Polynomial::zero(),
            split_left: Polynomial::zero(),
            split_right: Polynomial::zero(),
            factors: Vec::new(),
            pool: Vec::new(),
        }
    }

    /// Retained factor-stack and buffer-pool capacity.
    pub(crate) fn capacity(&self) -> usize {
        self.factors.capacity() + self.pool.capacity()
    }

    fn recycle_factors(&mut self) {
        while let Some(mut factor) = self.factors.pop() {
            factor.set_zero();
            self.pool.push(factor);
        }
    }
}

impl<F: FieldKernels> Default for FieldRootScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Return every distinct root in the polynomial's coefficient field.
///
/// For a nonzero polynomial this computes
/// `gcd(p, X^|F| + X)`, obtaining the square-free product of exactly its
/// base-field linear factors. Deterministic characteristic-two trace maps then
/// split that product. No field-wide evaluation scan is used.
pub fn base_field_roots<F: FieldKernels>(
    polynomial: &Polynomial<F>,
) -> Result<BaseFieldRoots<F::Elem>, RootError> {
    let mut scratch = FieldRootScratch::new();
    let mut roots = Vec::new();
    if base_field_roots_into(polynomial, &mut scratch, &mut roots)? {
        Ok(BaseFieldRoots::All)
    } else {
        Ok(BaseFieldRoots::Finite(roots))
    }
}

/// Write every distinct base-field root of `polynomial` into `roots`, reusing
/// `scratch`. Returns `true` when every field element is a root (the zero
/// polynomial); otherwise `roots` holds the sorted, deduplicated finite set.
pub(crate) fn base_field_roots_into<F: FieldKernels>(
    polynomial: &Polynomial<F>,
    scratch: &mut FieldRootScratch<F>,
    roots: &mut Vec<F::Elem>,
) -> Result<bool, RootError> {
    validate_binary_field::<F>()?;
    scratch.recycle_factors();
    roots.clear();
    let Some(degree) = polynomial.degree() else {
        return Ok(true);
    };
    if degree == 0 {
        return Ok(false);
    }

    scratch
        .x
        .assign_coefficients(&[F::Elem::ZERO, F::Elem::ONE])?;
    pow_x_field_order_mod(scratch, polynomial)?;
    scratch.vanishing.assign_from(&scratch.pow_result);
    scratch.vanishing.add_assign(&scratch.x)?;
    // `vanishing = X^|F| + X`; addition is in characteristic two.
    gcd_into(
        polynomial,
        &scratch.vanishing,
        &mut scratch.base_factor,
        &mut scratch.gcd_left,
        &mut scratch.gcd_right,
        &mut scratch.gcd_rem,
        &mut scratch.quot,
    )?;
    let Some(base_degree) = scratch.base_factor.degree() else {
        return Ok(false);
    };
    if base_degree == 0 {
        return Ok(false);
    }

    let capacity = degree.min(base_degree);
    reserve_polynomials(&mut scratch.factors, capacity, "base-field factor stack")?;
    reserve_polynomials(&mut scratch.pool, capacity, "base-field factor pool")?;
    reserve_elements(roots, capacity, "base-field roots")?;

    let mut base_factor_buffer = scratch.pool.pop().unwrap_or_default();
    base_factor_buffer.assign_from(&scratch.base_factor);
    scratch.factors.push(base_factor_buffer);

    let extension_degree = F::ORDER.trailing_zeros() as usize;
    while let Some(mut factor) = scratch.factors.pop() {
        let outcome = process_factor(polynomial, &factor, extension_degree, scratch, roots);
        factor.set_zero();
        scratch.pool.push(factor);
        outcome?;
    }

    roots.sort_by_key(|root| element_key::<F>(*root));
    roots.dedup();
    if roots
        .iter()
        .any(|root| !polynomial.evaluate(*root).is_zero())
    {
        return Err(RootError::FactorizationInvariant {
            reason: "the final root list contains a nonroot",
        });
    }
    Ok(false)
}

fn process_factor<F: FieldKernels>(
    polynomial: &Polynomial<F>,
    factor: &Polynomial<F>,
    extension_degree: usize,
    scratch: &mut FieldRootScratch<F>,
    roots: &mut Vec<F::Elem>,
) -> Result<(), RootError> {
    let Some(factor_degree) = factor.degree() else {
        return Err(RootError::FactorizationInvariant {
            reason: "the factor stack contained zero",
        });
    };
    if factor_degree == 0 {
        return Ok(());
    }
    if factor_degree == 1 {
        let linear = factor.coefficient(1);
        if linear.is_zero() {
            return Err(RootError::FactorizationInvariant {
                reason: "a degree-one factor has zero leading coefficient",
            });
        }
        let root = factor.coefficient(0).mul(linear.inv());
        if !polynomial.evaluate(root).is_zero() {
            return Err(RootError::FactorizationInvariant {
                reason: "an extracted linear root does not vanish in the input",
            });
        }
        roots.push(root);
        return Ok(());
    }

    split_factor_into(factor, factor_degree, extension_degree, scratch)?;
    let mut right = scratch.pool.pop().unwrap_or_default();
    right.assign_from(&scratch.split_right);
    let mut left = scratch.pool.pop().unwrap_or_default();
    left.assign_from(&scratch.split_left);
    scratch.factors.push(right);
    scratch.factors.push(left);
    Ok(())
}

/// Compute `X^|F| mod polynomial` into `scratch.pow_result` by
/// square-and-multiply over reusable ping-pong buffers.
fn pow_x_field_order_mod<F: FieldKernels>(
    scratch: &mut FieldRootScratch<F>,
    modulus: &Polynomial<F>,
) -> Result<(), RootError> {
    scratch.pow_result.assign_coefficients(&[F::Elem::ONE])?;
    scratch
        .x
        .div_rem_into(modulus, &mut scratch.quot, &mut scratch.pow_base)?;
    let mut exponent = F::ORDER;
    while exponent != 0 {
        if exponent & 1 != 0 {
            mul_mod(
                &scratch.pow_result,
                &scratch.pow_base,
                modulus,
                &mut scratch.pow_result_next,
                &mut scratch.mul_tmp,
                &mut scratch.quot,
            )?;
            core::mem::swap(&mut scratch.pow_result, &mut scratch.pow_result_next);
        }
        exponent >>= 1;
        if exponent != 0 {
            mul_mod(
                &scratch.pow_base,
                &scratch.pow_base,
                modulus,
                &mut scratch.pow_base_next,
                &mut scratch.mul_tmp,
                &mut scratch.quot,
            )?;
            core::mem::swap(&mut scratch.pow_base, &mut scratch.pow_base_next);
        }
    }
    Ok(())
}

/// Split a square-free product of at least two base-field linear factors,
/// leaving the two factors in `scratch.split_left` and `scratch.split_right`.
///
/// The powers `1, generator, ..., generator^(m-1)` form a basis of
/// `GF(2^m)` over `GF(2)`. For two distinct roots, nondegeneracy of the trace
/// pairing guarantees that one basis seed separates them. Thus at most `m`
/// deterministic trace attempts are required; field-element enumeration is
/// never needed.
fn split_factor_into<F: FieldKernels>(
    factor: &Polynomial<F>,
    factor_degree: usize,
    extension_degree: usize,
    scratch: &mut FieldRootScratch<F>,
) -> Result<(), RootError> {
    let mut seed = F::Elem::ONE;
    for _ in 0..extension_degree {
        trace_polynomial_into(factor, seed, extension_degree, scratch)?;
        gcd_into(
            factor,
            &scratch.trace_poly,
            &mut scratch.split_left,
            &mut scratch.gcd_left,
            &mut scratch.gcd_right,
            &mut scratch.gcd_rem,
            &mut scratch.quot,
        )?;
        let left_degree = scratch.split_left.degree().unwrap_or(0);
        if left_degree != 0 && left_degree != factor_degree {
            factor.div_rem_into(
                &scratch.split_left,
                &mut scratch.split_right,
                &mut scratch.gcd_rem,
            )?;
            if !scratch.gcd_rem.is_zero() {
                return Err(RootError::FactorizationInvariant {
                    reason: "a trace factor did not divide its parent",
                });
            }
            return Ok(());
        }
        seed = seed.mul(F::GENERATOR);
    }
    Err(RootError::FactorizationInvariant {
        reason: "the trace basis did not separate distinct roots",
    })
}

/// Compute the trace polynomial `sum_{i<m} (seed*X)^(2^i) mod modulus` into
/// `scratch.trace_poly`.
fn trace_polynomial_into<F: FieldKernels>(
    modulus: &Polynomial<F>,
    seed: F::Elem,
    extension_degree: usize,
    scratch: &mut FieldRootScratch<F>,
) -> Result<(), RootError> {
    scratch
        .trace_poly
        .assign_coefficients(&[F::Elem::ZERO, seed])?;
    scratch
        .trace_poly
        .div_rem_into(modulus, &mut scratch.quot, &mut scratch.trace_term)?;
    scratch.trace_acc.set_zero();
    for round in 0..extension_degree {
        scratch.trace_acc.add_assign(&scratch.trace_term)?;
        if round + 1 != extension_degree {
            mul_mod(
                &scratch.trace_term,
                &scratch.trace_term,
                modulus,
                &mut scratch.trace_term_next,
                &mut scratch.mul_tmp,
                &mut scratch.quot,
            )?;
            core::mem::swap(&mut scratch.trace_term, &mut scratch.trace_term_next);
        }
    }
    scratch.trace_poly.assign_from(&scratch.trace_acc);
    Ok(())
}

fn mul_mod<F: FieldKernels>(
    left: &Polynomial<F>,
    right: &Polynomial<F>,
    modulus: &Polynomial<F>,
    out: &mut Polynomial<F>,
    product: &mut Polynomial<F>,
    quotient: &mut Polynomial<F>,
) -> Result<(), RootError> {
    left.multiply_into(right, product)?;
    product.div_rem_into(modulus, quotient, out)?;
    Ok(())
}

fn gcd_into<F: FieldKernels>(
    left_input: &Polynomial<F>,
    right_input: &Polynomial<F>,
    out: &mut Polynomial<F>,
    left: &mut Polynomial<F>,
    right: &mut Polynomial<F>,
    remainder: &mut Polynomial<F>,
    quotient: &mut Polynomial<F>,
) -> Result<(), RootError> {
    left.assign_from(left_input);
    right.assign_from(right_input);
    while !right.is_zero() {
        left.div_rem_into(right, quotient, remainder)?;
        core::mem::swap(left, right);
        core::mem::swap(right, remainder);
    }
    out.assign_from(left);
    if let Some(leading) = out.leading_coefficient() {
        out.scale_assign(leading.inv());
    }
    Ok(())
}

fn reserve_polynomials<F: FieldKernels>(
    values: &mut Vec<Polynomial<F>>,
    capacity: usize,
    context: &'static str,
) -> Result<(), ConfigError> {
    if values.capacity() < capacity {
        values
            .try_reserve(capacity - values.capacity())
            .map_err(|_| ConfigError::AllocationFailed {
                context,
                elements: capacity,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })?;
    }
    Ok(())
}

fn reserve_elements<E>(
    values: &mut Vec<E>,
    capacity: usize,
    context: &'static str,
) -> Result<(), ConfigError> {
    if values.capacity() < capacity {
        values
            .try_reserve(capacity - values.capacity())
            .map_err(|_| ConfigError::AllocationFailed {
                context,
                elements: capacity,
                element_size: core::mem::size_of::<E>(),
            })?;
    }
    Ok(())
}

fn validate_binary_field<F: FieldKernels>() -> Result<(), RootError> {
    if !F::ORDER.is_power_of_two() || F::BYTES == 0 || F::BYTES > 16 {
        Err(RootError::UnsupportedField {
            field_order: F::ORDER,
            element_bytes: F::BYTES,
        })
    } else {
        Ok(())
    }
}

pub(super) fn element_key<F: FieldKernels>(element: F::Elem) -> u128 {
    debug_assert!(F::BYTES <= 16);
    let mut bytes = [0_u8; 16];
    F::write(&mut bytes[..F::BYTES], element);
    u128::from_le_bytes(bytes)
}
