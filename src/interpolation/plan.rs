//! Prepared, received-word-independent interpolation data owned by `GsPlan`.
//!
//! The Guruswami–Sudan module basis over an arbitrary support depends on the
//! domain vanishing polynomial `G(X)`, its powers, the module column shifts, and
//! the Newton basis and denominators that turn a received word into its
//! interpolant. All of these are fixed once the parameters and domain are
//! chosen, so they are precomputed once and reused across every decode.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use crate::{ConfigError, GsParameters, Polynomial};

use super::InterpolationError;

#[derive(Clone, Debug)]
/// Received-word-independent interpolation invariants for one plan geometry.
pub struct InterpolationPlan<F: FieldKernels> {
    /// Domain vanishing polynomial `G(X) = prod_i (X - alpha_i)`.
    pub(crate) vanishing: Polynomial<F>,
    /// Powers `G^0, G^1, ..., G^s`.
    pub(crate) vanishing_powers: Vec<Polynomial<F>>,
    /// Weak-Popov column shifts `0, D, 2D, ..., ell*D`.
    pub(crate) column_shifts: Vec<usize>,
    /// Newton basis `N_i(X) = prod_{j<i} (X - alpha_j)`, one per support point.
    pub(crate) newton_partials: Vec<Polynomial<F>>,
    /// Inverse Newton denominators `N_i(alpha_i)^{-1}`.
    pub(crate) newton_denominators: Vec<F::Elem>,
}

impl<F: FieldKernels> InterpolationPlan<F> {
    /// Precompute the vanishing polynomial, its powers, the column shifts, and
    /// the Newton interpolation basis for the given parameters and support.
    pub fn new(parameters: GsParameters, points: &[F::Elem]) -> Result<Self, InterpolationError> {
        let multiplicity = parameters.multiplicity();
        let max_degree = parameters.max_degree();
        let row_count =
            parameters
                .y_degree()
                .checked_add(1)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "interpolation plan row count",
                })?;

        let mut column_shifts = Vec::new();
        reserve(
            &mut column_shifts,
            row_count,
            "interpolation plan column shifts",
        )?;
        for column in 0..row_count {
            column_shifts.push(column.checked_mul(max_degree).ok_or(
                ConfigError::GeometryOverflow {
                    context: "interpolation plan column shift",
                },
            )?);
        }

        let mut newton_partials = Vec::new();
        let mut newton_denominators = Vec::new();
        reserve(
            &mut newton_partials,
            points.len(),
            "interpolation plan Newton basis",
        )?;
        reserve(
            &mut newton_denominators,
            points.len(),
            "interpolation plan Newton denominators",
        )?;
        let mut current = Polynomial::<F>::one()?;
        for &point in points {
            let denominator = current.evaluate(point);
            if denominator.is_zero() {
                return Err(InterpolationError::InvalidResult {
                    reason: "validated interpolation points became singular",
                });
            }
            newton_denominators.push(denominator.inv());
            let mut partial = Polynomial::zero();
            partial.assign_packed(current.as_packed())?;
            newton_partials.push(partial);
            current = current.multiply_x_plus(point)?;
        }
        let vanishing = current;

        let mut vanishing_powers = Vec::new();
        reserve(
            &mut vanishing_powers,
            multiplicity
                .checked_add(1)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "interpolation plan power count",
                })?,
            "interpolation plan vanishing powers",
        )?;
        vanishing_powers.push(Polynomial::one()?);
        for exponent in 1..=multiplicity {
            let power = vanishing_powers[exponent - 1].multiply(&vanishing)?;
            vanishing_powers.push(power);
        }

        Ok(Self {
            vanishing,
            vanishing_powers,
            column_shifts,
            newton_partials,
            newton_denominators,
        })
    }

    /// Heap-backed vector capacity retained by the prepared interpolation data.
    #[must_use]
    pub fn prepared_bytes(&self) -> usize {
        let polynomial_storage = |polynomials: &Vec<Polynomial<F>>| {
            polynomials.capacity() * core::mem::size_of::<Polynomial<F>>()
                + polynomials
                    .iter()
                    .map(Polynomial::retained_capacity_bytes)
                    .sum::<usize>()
        };
        self.vanishing.retained_capacity_bytes()
            + polynomial_storage(&self.vanishing_powers)
            + polynomial_storage(&self.newton_partials)
            + self.column_shifts.capacity() * core::mem::size_of::<usize>()
            + self.newton_denominators.capacity() * core::mem::size_of::<F::Elem>()
    }
}

fn reserve<T>(
    values: &mut Vec<T>,
    capacity: usize,
    context: &'static str,
) -> Result<(), ConfigError> {
    values
        .try_reserve_exact(capacity)
        .map_err(|_| ConfigError::AllocationFailed {
            context,
            elements: capacity,
            element_size: core::mem::size_of::<T>(),
        })
}
