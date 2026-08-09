use alloc::{vec, vec::Vec};

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use gfm::{WeakPopovRow, weak_popov};

use crate::{BivariatePolynomial, ConfigError, GsParameters, Polynomial};

use super::{InterpolationError, binomial_odd, validate_inputs, validate_result};

/// Conservative measured crossover in code length for the weak-Popov module backend.
///
/// The module backend is ahead at length eight for both GF8/GF16 and
/// scalar/GFNI. Kötter remains marginally faster for scalar GF16 at length four.
pub const MODULE_INTERPOLATION_CROSSOVER: usize = 8;

/// Construct an interpolation polynomial by reducing the Guruswami–Sudan
/// interpolation module to weak Popov form.
pub fn interpolate_module<F: FieldKernels>(
    parameters: GsParameters,
    points: &[F::Elem],
    values: &[F::Elem],
) -> Result<BivariatePolynomial<F>, InterpolationError> {
    validate_inputs(parameters, points, values)?;
    let (received, vanishing) = interpolate_received(points, values)?;
    let multiplicity = parameters.multiplicity();
    let y_degree = parameters.y_degree();
    let received_powers = polynomial_powers(&received, multiplicity)?;
    let vanishing_powers = polynomial_powers(&vanishing, multiplicity)?;

    let row_count = y_degree
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "interpolation module row count",
        })?;
    let mut basis = Vec::new();
    basis
        .try_reserve_exact(row_count)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "interpolation module basis",
            elements: row_count,
            element_size: core::mem::size_of::<BivariatePolynomial<F>>(),
        })?;
    for row in 0..row_count {
        basis.push(module_row(
            row,
            multiplicity,
            y_degree,
            &received_powers,
            &vanishing_powers,
        )?);
    }

    let mut shifts = Vec::new();
    shifts
        .try_reserve_exact(row_count)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "interpolation module shift",
            elements: row_count,
            element_size: core::mem::size_of::<usize>(),
        })?;
    for column in 0..row_count {
        shifts.push(column.checked_mul(parameters.max_degree()).ok_or(
            ConfigError::GeometryOverflow {
                context: "interpolation module column shift",
            },
        )?);
    }
    weak_popov::<F, _>(&mut basis, &shifts)?;
    let mut selected = None;
    for polynomial in basis {
        let Some(degree) = polynomial.weighted_degree(parameters.max_degree())? else {
            continue;
        };
        if degree > parameters.weighted_degree() {
            continue;
        }
        let key = (degree, polynomial.y_degree().unwrap_or(0));
        if selected
            .as_ref()
            .is_none_or(|(selected_key, _)| key < *selected_key)
        {
            selected = Some((key, polynomial));
        }
    }
    let selected =
        selected
            .map(|(_, polynomial)| polynomial)
            .ok_or(InterpolationError::InvalidResult {
                reason: "reduced interpolation module has no row within the weighted-degree bound",
            })?;
    validate_result(parameters, points, values, &selected)?;
    Ok(selected)
}

fn interpolate_received<F: FieldKernels>(
    points: &[F::Elem],
    values: &[F::Elem],
) -> Result<(Polynomial<F>, Polynomial<F>), InterpolationError> {
    let mut received = Polynomial::<F>::zero();
    let mut vanishing = Polynomial::<F>::one()?;
    for (&point, &value) in points.iter().zip(values) {
        let denominator = vanishing.evaluate(point);
        if denominator.is_zero() {
            return Err(InterpolationError::InvalidResult {
                reason: "validated interpolation points became singular",
            });
        }
        let discrepancy = value.add(received.evaluate(point));
        received.add_scaled_assign(discrepancy.mul(denominator.inv()), &vanishing)?;
        vanishing = vanishing.multiply_x_plus(point)?;
    }
    Ok((received, vanishing))
}

fn polynomial_powers<F: FieldKernels>(
    polynomial: &Polynomial<F>,
    maximum: usize,
) -> Result<Vec<Polynomial<F>>, ConfigError> {
    let count = maximum
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "interpolation module power count",
        })?;
    let mut powers = Vec::new();
    powers
        .try_reserve_exact(count)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "interpolation module powers",
            elements: count,
            element_size: core::mem::size_of::<Polynomial<F>>(),
        })?;
    powers.push(Polynomial::one()?);
    for exponent in 1..=maximum {
        powers.push(powers[exponent - 1].multiply(polynomial)?);
    }
    Ok(powers)
}

fn module_row<F: FieldKernels>(
    row: usize,
    multiplicity: usize,
    y_degree: usize,
    received_powers: &[Polynomial<F>],
    vanishing_powers: &[Polynomial<F>],
) -> Result<BivariatePolynomial<F>, ConfigError> {
    let mut coefficients = vec![Polynomial::zero(); y_degree + 1];
    if row < multiplicity {
        let vanishing = &vanishing_powers[multiplicity - row];
        for y in 0..=row {
            if binomial_odd(row, y) {
                coefficients[y] = received_powers[row - y].multiply(vanishing)?;
            }
        }
    } else {
        let shift = row - multiplicity;
        for y in 0..=multiplicity {
            if binomial_odd(multiplicity, y) {
                coefficients[shift + y] = received_powers[multiplicity - y].clone();
            }
        }
    }
    Ok(BivariatePolynomial::from_y_coefficients(coefficients))
}

impl<F: FieldKernels> WeakPopovRow<F> for BivariatePolynomial<F> {
    type Error = InterpolationError;

    fn column_count(&self) -> usize {
        self.y_coefficient_count()
    }

    fn degree(&self, column: usize) -> Option<usize> {
        self.y_coefficient(column).and_then(Polynomial::degree)
    }

    fn coefficient(&self, column: usize, degree: usize) -> F::Elem {
        self.y_coefficient(column)
            .map_or(F::Elem::ZERO, |polynomial| polynomial.coefficient(degree))
    }

    fn add_scaled_shifted_assign(
        &mut self,
        scale: F::Elem,
        pivot: &Self,
        shift: usize,
    ) -> Result<(), Self::Error> {
        self.add_scaled_x_shifted_assign(scale, pivot, shift)
            .map_err(InterpolationError::from)
    }
}
