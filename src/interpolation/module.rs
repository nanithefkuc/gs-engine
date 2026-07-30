use alloc::{vec, vec::Vec};

use fff::field::Elem;
use fff::kernel::FieldKernels;

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

    reduce_to_weak_popov(&mut basis, parameters.max_degree())?;
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

fn reduce_to_weak_popov<F: FieldKernels>(
    basis: &mut [BivariatePolynomial<F>],
    y_weight: usize,
) -> Result<(), InterpolationError> {
    let mut leading_rows = vec![None; basis.len()];
    loop {
        leading_rows.fill(None);
        let mut collision = None;
        for (row, polynomial) in basis.iter().enumerate() {
            let Some(leading) = polynomial.weighted_leading_term(y_weight)? else {
                continue;
            };
            if let Some(previous) = leading_rows[leading.y_degree] {
                collision = Some((previous, row));
                break;
            }
            leading_rows[leading.y_degree] = Some(row);
        }
        let Some((left, right)) = collision else {
            return Ok(());
        };
        reduce_pair(basis, left, right, y_weight)?;
    }
}

fn reduce_pair<F: FieldKernels>(
    basis: &mut [BivariatePolynomial<F>],
    left: usize,
    right: usize,
    y_weight: usize,
) -> Result<(), InterpolationError> {
    let left_leading =
        basis[left]
            .weighted_leading_term(y_weight)?
            .ok_or(InterpolationError::InvalidResult {
                reason: "interpolation module reduction selected a zero row",
            })?;
    let right_leading =
        basis[right]
            .weighted_leading_term(y_weight)?
            .ok_or(InterpolationError::InvalidResult {
                reason: "interpolation module reduction selected a zero row",
            })?;
    let (target, pivot, target_leading, pivot_leading) =
        if left_leading.x_degree >= right_leading.x_degree {
            (left, right, left_leading, right_leading)
        } else {
            (right, left, right_leading, left_leading)
        };
    let target_coefficient = basis[target]
        .y_coefficient(target_leading.y_degree)
        .ok_or(InterpolationError::InvalidResult {
            reason: "interpolation module leading row disappeared",
        })?
        .coefficient(target_leading.x_degree);
    let pivot_coefficient = basis[pivot]
        .y_coefficient(pivot_leading.y_degree)
        .ok_or(InterpolationError::InvalidResult {
            reason: "interpolation module leading row disappeared",
        })?
        .coefficient(pivot_leading.x_degree);
    let scale = target_coefficient.mul(pivot_coefficient.inv());
    let shift = target_leading.x_degree - pivot_leading.x_degree;
    let (target_polynomial, pivot_polynomial) = if target < pivot {
        let (lower, upper) = basis.split_at_mut(pivot);
        (&mut lower[target], &upper[0])
    } else {
        let (lower, upper) = basis.split_at_mut(target);
        (&mut upper[0], &lower[pivot])
    };
    target_polynomial
        .add_scaled_x_shifted_assign(scale, pivot_polynomial, shift)
        .map_err(InterpolationError::from)
}
