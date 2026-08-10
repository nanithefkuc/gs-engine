//! Prepared, received-word-independent interpolation data owned by `GsPlan`.
//!
//! The Guruswami–Sudan module basis over an arbitrary support depends on the
//! domain vanishing polynomial `G(X)`, its powers, the module column shifts,
//! and the Newton basis and denominators that turn a received word into its
//! interpolant. All of these are fixed once the parameters and domain are
//! chosen, so they are precomputed once and reused across every decode.
//!
//! For additive-subspace and affine-coset domains, the vanishing polynomial
//! is obtained from the `butterfly-fft` subspace polynomial and the received
//! word is interpolated by an inverse transform plus novel-to-monomial
//! conversion, both `O(n log n)`. The Newton basis is still stored for the
//! arbitrary-domain path and for differential testing.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use crate::{ConfigError, GsParameters, Polynomial};

use super::InterpolationError;

/// How received-word interpolation and the vanishing polynomial are computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DomainStrategy {
    /// Arbitrary points: incremental Newton interpolation and product `G`.
    Newton,
    /// Additive subspace or affine coset: inverse transform and subspace `G`.
    Transform,
}

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
    /// Selected received-word interpolation and vanishing-polynomial strategy.
    pub(crate) strategy: DomainStrategy,
}

/// Newton basis, denominators, and incremental vanishing polynomial.
type NewtonBasis<F> = (
    Polynomial<F>,
    Vec<Polynomial<F>>,
    Vec<<F as fgf::field::Field>::Elem>,
);

impl<F: FieldKernels> InterpolationPlan<F> {
    /// Precompute the vanishing polynomial, its powers, the column shifts, and
    /// the Newton interpolation basis for the given parameters and support.
    ///
    /// Uses the incremental Newton path for received-word interpolation. For
    /// additive-subspace and affine-coset domains, prefer
    /// [`InterpolationPlan::new_with_domain`], which builds `G` from the
    /// subspace polynomial and selects the `O(n log n)` inverse-transform
    /// received-word path.
    pub fn new(parameters: GsParameters, points: &[F::Elem]) -> Result<Self, InterpolationError> {
        let (vanishing, newton_partials, newton_denominators) =
            build_newton_basis::<F>(parameters, points)?;
        Self::finish(
            parameters,
            &vanishing,
            newton_partials,
            newton_denominators,
            DomainStrategy::Newton,
        )
    }

    /// Precompute interpolation invariants, selecting the transform path when
    /// the domain carries a `butterfly-fft` plan.
    ///
    /// For additive-subspace and affine-coset domains, the vanishing
    /// polynomial is built from the plan's subspace polynomial in `O(n)` rather
    /// than `O(n²)` incremental multiplication, and received-word
    /// interpolation will use an inverse transform. The Newton basis is still
    /// computed for differential testing and the arbitrary fallback.
    pub fn new_with_domain(
        parameters: GsParameters,
        domain: &crate::domain::EvaluationDomain<F>,
    ) -> Result<Self, InterpolationError>
    where
        F: butterfly_fft::core::kernel::ButterflyKernels,
    {
        let points = domain.points();
        let (vanishing, newton_partials, newton_denominators) =
            build_newton_basis::<F>(parameters, points)?;
        let vanishing = match domain.transform_plan() {
            Some(plan) => {
                let subspace = plan.vanishing_polynomial();
                Polynomial::<F>::from_coefficients(&subspace)?
            }
            None => vanishing,
        };
        let strategy = if domain.transform_plan().is_some() {
            DomainStrategy::Transform
        } else {
            DomainStrategy::Newton
        };
        Self::finish(
            parameters,
            &vanishing,
            newton_partials,
            newton_denominators,
            strategy,
        )
    }

    fn finish(
        parameters: GsParameters,
        vanishing: &Polynomial<F>,
        newton_partials: Vec<Polynomial<F>>,
        newton_denominators: Vec<F::Elem>,
        strategy: DomainStrategy,
    ) -> Result<Self, InterpolationError> {
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
            let power = vanishing_powers[exponent - 1].multiply(vanishing)?;
            vanishing_powers.push(power);
        }

        Ok(Self {
            vanishing: vanishing.clone(),
            vanishing_powers,
            column_shifts,
            newton_partials,
            newton_denominators,
            strategy,
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

/// Build the Newton basis `N_i(X)`, denominators, and the incremental
/// vanishing polynomial `G(X) = ∏(X + α_i)`.
fn build_newton_basis<F: FieldKernels>(
    _parameters: GsParameters,
    points: &[F::Elem],
) -> Result<NewtonBasis<F>, InterpolationError> {
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
    Ok((current, newton_partials, newton_denominators))
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
