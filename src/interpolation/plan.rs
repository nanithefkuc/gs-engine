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

/// Received-word-independent data for the re-encoding decode path.
///
/// Re-encoding zeroes the first `k = w + 1` received coordinates by adding a
/// degree-`w` helper polynomial `e(X)` that interpolates them, then decodes the
/// shifted word over the remaining `n - k` support points with a
/// factor-reduced interpolation module. Every quantity here depends only on the
/// support and parameters, so it is prepared once and reused across decodes.
///
/// The shifted received word `r'` vanishes on the re-encoded points, so its
/// interpolant is divisible by `Psi(X) = prod_{j < k} (X + alpha_j)`. Writing
/// `R' = Psi * R̃` and `G = Psi * G_rem` lets the module be built over the
/// remaining points from the reduced interpolant `R̃` and the reduced vanishing
/// polynomial `G_rem`, whose degrees are `n - k` smaller than the direct
/// module's.
#[derive(Clone, Debug)]
pub struct ReencodePlan<F: FieldKernels> {
    /// Number of re-encoded (zeroed) leading positions, `k = w + 1`.
    pub(crate) message_len: usize,
    /// Powers `Psi^0 .. Psi^ell` of the re-encoding vanishing polynomial.
    pub(crate) psi_powers: Vec<Polynomial<F>>,
    /// Powers `G_rem^0 .. G_rem^s` over the remaining support points.
    pub(crate) grem_powers: Vec<Polynomial<F>>,
    /// `Psi(alpha_i)^{-1}` for each remaining support point `i`.
    pub(crate) inv_psi_at_remaining: Vec<F::Elem>,
    /// Newton basis over the re-encoded points, for the helper interpolant.
    pub(crate) helper_partials: Vec<Polynomial<F>>,
    /// Inverse Newton denominators over the re-encoded points.
    pub(crate) helper_denominators: Vec<F::Elem>,
    /// Newton basis over the remaining points, for the reduced interpolant.
    pub(crate) reduced_partials: Vec<Polynomial<F>>,
    /// Inverse Newton denominators over the remaining points.
    pub(crate) reduced_denominators: Vec<F::Elem>,
    /// Reduced weak-Popov column shifts `ell, ell-1, ..., 0`.
    pub(crate) reduced_shifts: Vec<usize>,
}

impl<F: FieldKernels> ReencodePlan<F> {
    /// Prepare the re-encoding invariants for the given parameters and support.
    ///
    /// The first `k = w + 1` support points are the deterministic re-encoding
    /// set; the rest form the reduced support. Returns an error when there is no
    /// remaining support (`k >= n`); callers gate on the conservative selector
    /// before constructing this.
    pub(crate) fn new(
        parameters: GsParameters,
        points: &[F::Elem],
    ) -> Result<Self, InterpolationError> {
        let n = parameters.code_length();
        let message_len =
            parameters
                .max_degree()
                .checked_add(1)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "re-encoding message length",
                })?;
        if message_len >= n {
            return Err(InterpolationError::InvalidResult {
                reason: "re-encoding requires a nonempty remaining support",
            });
        }
        let multiplicity = parameters.multiplicity();
        let y_degree = parameters.y_degree();
        let (psi, helper_partials, helper_denominators) =
            build_newton_basis::<F>(parameters, &points[..message_len])?;
        let (grem, reduced_partials, reduced_denominators) =
            build_newton_basis::<F>(parameters, &points[message_len..])?;

        // Characteristic-two Sierpiński parity plus the prefactor split mean the
        // reduced rows and reconstruction only reference `Psi` up to exponent
        // `max(s, ell - s)`; higher powers are never built.
        let psi_max = multiplicity.max(y_degree.saturating_sub(multiplicity));
        let psi_powers = powers_of(&psi, psi_max, "re-encoding Psi powers")?;
        let grem_powers = powers_of(&grem, multiplicity, "re-encoding G_rem powers")?;

        let mut inv_psi_at_remaining = Vec::new();
        reserve(
            &mut inv_psi_at_remaining,
            n - message_len,
            "re-encoding inverse Psi values",
        )?;
        for &point in &points[message_len..] {
            let value = psi.evaluate(point);
            if value.is_zero() {
                return Err(InterpolationError::InvalidResult {
                    reason: "re-encoding vanishing polynomial has a repeated support point",
                });
            }
            inv_psi_at_remaining.push(value.inv());
        }

        let mut reduced_shifts = Vec::new();
        reserve(
            &mut reduced_shifts,
            y_degree
                .checked_add(1)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "re-encoding reduced shift count",
                })?,
            "re-encoding reduced shifts",
        )?;
        for column in 0..=y_degree {
            reduced_shifts.push(y_degree - column);
        }

        Ok(Self {
            message_len,
            psi_powers,
            grem_powers,
            inv_psi_at_remaining,
            helper_partials,
            helper_denominators,
            reduced_partials,
            reduced_denominators,
            reduced_shifts,
        })
    }

    /// Heap-backed vector capacity retained by the prepared re-encoding data.
    #[must_use]
    pub(crate) fn prepared_bytes(&self) -> usize {
        let polynomial_storage = |polynomials: &Vec<Polynomial<F>>| {
            polynomials.capacity() * core::mem::size_of::<Polynomial<F>>()
                + polynomials
                    .iter()
                    .map(Polynomial::retained_capacity_bytes)
                    .sum::<usize>()
        };
        polynomial_storage(&self.psi_powers)
            + polynomial_storage(&self.grem_powers)
            + polynomial_storage(&self.helper_partials)
            + polynomial_storage(&self.reduced_partials)
            + self.inv_psi_at_remaining.capacity() * core::mem::size_of::<F::Elem>()
            + self.helper_denominators.capacity() * core::mem::size_of::<F::Elem>()
            + self.reduced_denominators.capacity() * core::mem::size_of::<F::Elem>()
            + self.reduced_shifts.capacity() * core::mem::size_of::<usize>()
    }
}

/// Build `base^0 .. base^maximum` with checked reservation.
fn powers_of<F: FieldKernels>(
    base: &Polynomial<F>,
    maximum: usize,
    context: &'static str,
) -> Result<Vec<Polynomial<F>>, InterpolationError> {
    let mut powers = Vec::new();
    reserve(
        &mut powers,
        maximum
            .checked_add(1)
            .ok_or(ConfigError::GeometryOverflow {
                context: "re-encoding power count",
            })?,
        context,
    )?;
    powers.push(Polynomial::one()?);
    for exponent in 1..=maximum {
        let power = powers[exponent - 1].multiply(base)?;
        powers.push(power);
    }
    Ok(powers)
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
