use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use gfm::{WeakPopovRow, WeakPopovScratch, weak_popov_with_scratch};

use crate::{BivariatePolynomial, ConfigError, GsParameters, Polynomial};

use super::{
    InterpolationError, InterpolationPlan, binomial_odd, validate_inputs, validate_result,
};

/// Conservative measured crossover in code length for the weak-Popov module backend.
///
/// The module backend is ahead at length eight for both GF8/GF16 and
/// scalar/GFNI. Kötter remains marginally faster for scalar GF16 at length four.
pub const MODULE_INTERPOLATION_CROSSOVER: usize = 8;

/// Reusable received-interpolant, `R` power, and basis-row storage for the
/// weak-Popov module interpolation backend. The vanishing polynomial, its
/// powers, and the column shifts are supplied by the prepared plan.
pub struct ModuleScratch<F: FieldKernels> {
    received: Polynomial<F>,
    received_powers: Vec<Polynomial<F>>,
    basis: Vec<ModuleRow<F>>,
    reduction: WeakPopovScratch,
}

struct ModuleRow<F: FieldKernels> {
    polynomial: BivariatePolynomial<F>,
    row_pool: Vec<Polynomial<F>>,
}

impl<F: FieldKernels> Default for ModuleRow<F> {
    fn default() -> Self {
        Self {
            polynomial: BivariatePolynomial::zero(),
            row_pool: Vec::new(),
        }
    }
}

impl<F: FieldKernels> ModuleScratch<F> {
    /// Construct empty reusable module-interpolation scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            received: Polynomial::zero(),
            received_powers: Vec::new(),
            basis: Vec::new(),
            reduction: WeakPopovScratch::new(),
        }
    }

    /// Retained basis-row and `R`-power capacity available to a later interpolation.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.basis.capacity()
            + self.received_powers.capacity()
            + self.reduction.capacity()
            + self
                .basis
                .iter()
                .map(|row| row.row_pool.capacity())
                .sum::<usize>()
    }
}

impl<F: FieldKernels> Default for ModuleScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Construct an interpolation polynomial by reducing the Guruswami–Sudan
/// interpolation module to weak Popov form.
pub fn interpolate_module<F: FieldKernels>(
    parameters: GsParameters,
    points: &[F::Elem],
    values: &[F::Elem],
) -> Result<BivariatePolynomial<F>, InterpolationError> {
    let plan = InterpolationPlan::new(parameters, points)?;
    let mut scratch = ModuleScratch::new();
    let mut output = BivariatePolynomial::zero();
    interpolate_module_into(parameters, points, values, &plan, &mut scratch, &mut output)?;
    Ok(output)
}

/// Reduce the interpolation module into reusable `output` storage, recycling the
/// basis rows and `R` powers held by `scratch` and the vanishing polynomial,
/// vanishing powers, and column shifts prepared in `plan`.
pub fn interpolate_module_into<F: FieldKernels>(
    parameters: GsParameters,
    points: &[F::Elem],
    values: &[F::Elem],
    plan: &InterpolationPlan<F>,
    scratch: &mut ModuleScratch<F>,
    output: &mut BivariatePolynomial<F>,
) -> Result<(), InterpolationError> {
    validate_inputs(parameters, points, values)?;
    interpolate_received_into(
        points,
        values,
        &plan.newton_partials,
        &plan.newton_denominators,
        &mut scratch.received,
    )?;
    let multiplicity = parameters.multiplicity();
    let y_degree = parameters.y_degree();
    fill_polynomial_powers(
        &scratch.received,
        multiplicity,
        &mut scratch.received_powers,
    )?;

    let row_count = y_degree
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "interpolation module row count",
        })?;
    reserve_basis(&mut scratch.basis, row_count)?;
    for row in 0..row_count {
        module_row_into(
            row,
            multiplicity,
            y_degree,
            &scratch.received_powers,
            &plan.vanishing_powers,
            &mut scratch.basis[row],
        )?;
    }

    weak_popov_with_scratch::<F, _>(
        &mut scratch.basis[..row_count],
        &plan.column_shifts,
        &mut scratch.reduction,
    )?;

    let mut selected: Option<(usize, (usize, usize))> = None;
    for (index, row) in scratch.basis[..row_count].iter().enumerate() {
        let polynomial = &row.polynomial;
        let Some(degree) = polynomial.weighted_degree(parameters.max_degree())? else {
            continue;
        };
        if degree > parameters.weighted_degree() {
            continue;
        }
        let key = (degree, polynomial.y_degree().unwrap_or(0));
        if selected.is_none_or(|(_, selected_key)| key < selected_key) {
            selected = Some((index, key));
        }
    }
    let index = selected
        .map(|(index, _)| index)
        .ok_or(InterpolationError::InvalidResult {
            reason: "reduced interpolation module has no row within the weighted-degree bound",
        })?;
    output.try_assign_from(&scratch.basis[index].polynomial)?;
    validate_result(parameters, points, values, output)?;
    Ok(())
}

fn reserve_basis<F: FieldKernels>(
    basis: &mut Vec<ModuleRow<F>>,
    row_count: usize,
) -> Result<(), ConfigError> {
    if basis.capacity() < row_count {
        basis
            .try_reserve(row_count - basis.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "interpolation module basis",
                elements: row_count,
                element_size: core::mem::size_of::<ModuleRow<F>>(),
            })?;
    }
    while basis.len() < row_count {
        basis.push(ModuleRow::default());
    }
    Ok(())
}

fn interpolate_received_into<F: FieldKernels>(
    points: &[F::Elem],
    values: &[F::Elem],
    newton_partials: &[Polynomial<F>],
    newton_denominators: &[F::Elem],
    received: &mut Polynomial<F>,
) -> Result<(), InterpolationError> {
    debug_assert_eq!(newton_partials.len(), points.len());
    debug_assert_eq!(newton_denominators.len(), points.len());
    received.set_zero();
    for index in 0..points.len() {
        let discrepancy = values[index].add(received.evaluate(points[index]));
        let coefficient = discrepancy.mul(newton_denominators[index]);
        received.add_scaled_assign(coefficient, &newton_partials[index])?;
    }
    Ok(())
}

fn fill_polynomial_powers<F: FieldKernels>(
    polynomial: &Polynomial<F>,
    maximum: usize,
    powers: &mut Vec<Polynomial<F>>,
) -> Result<(), ConfigError> {
    let count = maximum
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "interpolation module power count",
        })?;
    if powers.capacity() < count {
        powers
            .try_reserve(count - powers.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "interpolation module powers",
                elements: count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })?;
    }
    while powers.len() < count {
        powers.push(Polynomial::zero());
    }
    powers[0].assign_coefficients(&[F::Elem::ONE])?;
    for exponent in 1..=maximum {
        let (previous, current) = powers.split_at_mut(exponent);
        previous[exponent - 1].multiply_into(polynomial, &mut current[0])?;
    }
    Ok(())
}

fn module_row_into<F: FieldKernels>(
    row: usize,
    multiplicity: usize,
    y_degree: usize,
    received_powers: &[Polynomial<F>],
    vanishing_powers: &[Polynomial<F>],
    output: &mut ModuleRow<F>,
) -> Result<(), ConfigError> {
    output
        .polynomial
        .resize_and_zero_rows(y_degree + 1, &mut output.row_pool)?;
    if row < multiplicity {
        let vanishing = &vanishing_powers[multiplicity - row];
        for y in 0..=row {
            if binomial_odd(row, y) {
                received_powers[row - y]
                    .multiply_into(vanishing, output.polynomial.y_coefficient_mut(y))?;
            }
        }
    } else {
        let shift = row - multiplicity;
        for y in 0..=multiplicity {
            if binomial_odd(multiplicity, y) {
                output
                    .polynomial
                    .y_coefficient_mut(shift + y)
                    .assign_packed(received_powers[multiplicity - y].as_packed())?;
            }
        }
    }
    output.polynomial.normalize_pooled(&mut output.row_pool);
    Ok(())
}

impl<F: FieldKernels> WeakPopovRow<F> for ModuleRow<F> {
    type Error = InterpolationError;

    fn column_count(&self) -> usize {
        self.polynomial.y_coefficient_count()
    }

    fn degree(&self, column: usize) -> Option<usize> {
        self.polynomial
            .y_coefficient(column)
            .and_then(Polynomial::degree)
    }

    fn coefficient(&self, column: usize, degree: usize) -> F::Elem {
        self.polynomial
            .y_coefficient(column)
            .map_or(F::Elem::ZERO, |polynomial| polynomial.coefficient(degree))
    }

    fn add_scaled_shifted_assign(
        &mut self,
        scale: F::Elem,
        pivot: &Self,
        shift: usize,
    ) -> Result<(), Self::Error> {
        self.polynomial
            .add_scaled_x_shifted_assign_pooled(scale, &pivot.polynomial, shift, &mut self.row_pool)
            .map_err(InterpolationError::from)
    }
}
