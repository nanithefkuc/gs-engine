use alloc::vec::Vec;

use fff::field::Elem;
use fff::kernel::FieldKernels;

use crate::{ConfigError, Polynomial};

use super::{BaseFieldRoots, RootError};

/// Return every distinct root in the polynomial's coefficient field.
///
/// For a nonzero polynomial this computes
/// `gcd(p, X^|F| + X)`, obtaining the square-free product of exactly its
/// base-field linear factors. Deterministic characteristic-two trace maps then
/// split that product. No field-wide evaluation scan is used.
pub fn base_field_roots<F: FieldKernels>(
    polynomial: &Polynomial<F>,
) -> Result<BaseFieldRoots<F::Elem>, RootError> {
    validate_binary_field::<F>()?;
    let Some(degree) = polynomial.degree() else {
        return Ok(BaseFieldRoots::All);
    };
    if degree == 0 {
        return Ok(BaseFieldRoots::Finite(Vec::new()));
    }

    let x = Polynomial::<F>::from_coefficients(&[F::Elem::ZERO, F::Elem::ONE])?;
    let x_to_field_order = x.pow_mod(F::ORDER, polynomial)?;
    let vanishing = x_to_field_order.add(&x)?;
    let base_factor = polynomial.gcd(&vanishing)?;
    let Some(base_degree) = base_factor.degree() else {
        return Ok(BaseFieldRoots::Finite(Vec::new()));
    };
    if base_degree == 0 {
        return Ok(BaseFieldRoots::Finite(Vec::new()));
    }

    let capacity = degree.min(base_degree);
    let mut factors = Vec::new();
    factors
        .try_reserve_exact(capacity)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "base-field factor stack",
            elements: capacity,
            element_size: core::mem::size_of::<Polynomial<F>>(),
        })?;
    factors.push(base_factor);
    let mut roots = Vec::new();
    roots
        .try_reserve_exact(capacity)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "base-field roots",
            elements: capacity,
            element_size: core::mem::size_of::<F::Elem>(),
        })?;

    let extension_degree = F::ORDER.trailing_zeros() as usize;
    while let Some(factor) = factors.pop() {
        let Some(factor_degree) = factor.degree() else {
            return Err(RootError::FactorizationInvariant {
                reason: "the factor stack contained zero",
            });
        };
        if factor_degree == 0 {
            continue;
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
            continue;
        }

        let (left, right) = split_factor::<F>(&factor, extension_degree)?;
        factors.push(right);
        factors.push(left);
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
    Ok(BaseFieldRoots::Finite(roots))
}

/// Split a square-free product of at least two base-field linear factors.
///
/// The powers `1, generator, ..., generator^(m-1)` form a basis of
/// `GF(2^m)` over `GF(2)`. For two distinct roots, nondegeneracy of the trace
/// pairing guarantees that one basis seed separates them. Thus at most `m`
/// deterministic trace attempts are required; field-element enumeration is
/// never needed.
fn split_factor<F: FieldKernels>(
    factor: &Polynomial<F>,
    extension_degree: usize,
) -> Result<(Polynomial<F>, Polynomial<F>), RootError> {
    let factor_degree = factor.degree().ok_or(RootError::FactorizationInvariant {
        reason: "attempted to split the zero polynomial",
    })?;
    let mut seed = F::Elem::ONE;
    for _ in 0..extension_degree {
        let trace = trace_polynomial::<F>(factor, seed, extension_degree)?;
        let left = factor.gcd(&trace)?;
        let left_degree = left.degree().unwrap_or(0);
        if left_degree != 0 && left_degree != factor_degree {
            let right = factor.exact_divide(&left)?;
            return Ok((left, right));
        }
        seed = seed.mul(F::GENERATOR);
    }
    Err(RootError::FactorizationInvariant {
        reason: "the trace basis did not separate distinct roots",
    })
}

fn trace_polynomial<F: FieldKernels>(
    modulus: &Polynomial<F>,
    seed: F::Elem,
    extension_degree: usize,
) -> Result<Polynomial<F>, RootError> {
    let mut term =
        Polynomial::<F>::from_coefficients(&[F::Elem::ZERO, seed])?.remainder(modulus)?;
    let mut trace = Polynomial::<F>::zero();
    for round in 0..extension_degree {
        trace.add_assign(&term)?;
        if round + 1 != extension_degree {
            term = term.square_mod(modulus)?;
        }
    }
    Ok(trace)
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
