use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use fgf::ops;
use gfm::{PopovLeadingTerm, WeakPopovBasis, WeakPopovScratch, weak_popov_basis_with_scratch};

use crate::{BivariatePolynomial, ConfigError, GsParameters, Polynomial};

use super::{
    InterpolationError, InterpolationPlan, binomial_odd, validate_inputs, validate_result,
};

/// Conservative measured crossover in code length for the weak-Popov module backend.
///
/// The module backend is ahead at length eight for both GF8/GF16 and
/// scalar/GFNI. Kötter remains marginally faster for scalar GF16 at length four.
pub const MODULE_INTERPOLATION_CROSSOVER: usize = 8;

/// Reusable received-interpolant, power, packed-basis, and reduction storage for
/// the weak-Popov module interpolation backend.
pub struct ModuleScratch<F: FieldKernels> {
    received: Polynomial<F>,
    received_powers: Vec<Polynomial<F>>,
    product: Polynomial<F>,
    basis: ModuleSlab<F>,
    reduction: WeakPopovScratch,
}

struct ModuleSlab<F: FieldKernels> {
    coefficients: Vec<u8>,
    degrees: Vec<Option<usize>>,
    leading: Vec<Option<PopovLeadingTerm>>,
    rows: usize,
    columns: usize,
    x_capacity: usize,
    field: core::marker::PhantomData<F>,
}

impl<F: FieldKernels> ModuleSlab<F> {
    const fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            degrees: Vec::new(),
            leading: Vec::new(),
            rows: 0,
            columns: 0,
            x_capacity: 0,
            field: core::marker::PhantomData,
        }
    }

    fn prepare(&mut self, parameters: GsParameters) -> Result<(), ConfigError> {
        let rows = parameters
            .y_degree()
            .checked_add(1)
            .ok_or(ConfigError::GeometryOverflow {
                context: "interpolation module row count",
            })?;
        // A weak-Popov update cannot increase a row's shifted leading degree:
        // every shifted pivot term is bounded by the target leading term it
        // cancels. The largest initial shifted degree is therefore a bound on
        // every X degree reached during reduction.
        let code_length = parameters.code_length();
        let multiplicity = parameters.multiplicity();
        let y_degree = parameters.y_degree();
        let max_degree = parameters.max_degree();
        let low_row_bound =
            code_length
                .checked_mul(multiplicity)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "interpolation module low-row X capacity",
                })?;
        let high_row_bound = code_length
            .checked_sub(1)
            .and_then(|degree| degree.checked_mul(multiplicity))
            .and_then(|base| {
                y_degree
                    .checked_sub(multiplicity)
                    .and_then(|tail| tail.checked_mul(max_degree))
                    .and_then(|tail| base.checked_add(tail))
            })
            .ok_or(ConfigError::GeometryOverflow {
                context: "interpolation module high-row X capacity",
            })?;
        let x_capacity = low_row_bound.max(high_row_bound).checked_add(1).ok_or(
            ConfigError::GeometryOverflow {
                context: "interpolation module X capacity",
            },
        )?;
        let columns = rows;
        let column_count = rows
            .checked_mul(columns)
            .ok_or(ConfigError::GeometryOverflow {
                context: "interpolation module slab columns",
            })?;
        let coefficient_count =
            column_count
                .checked_mul(x_capacity)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "interpolation module slab coefficients",
                })?;
        let byte_count =
            coefficient_count
                .checked_mul(F::BYTES)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "interpolation module slab bytes",
                })?;

        resize_zeroed(
            &mut self.coefficients,
            byte_count,
            "interpolation module coefficient slab",
        )?;
        resize_zeroed(
            &mut self.degrees,
            column_count,
            "interpolation module degree metadata",
        )?;
        resize_zeroed(
            &mut self.leading,
            rows,
            "interpolation module leading metadata",
        )?;
        self.coefficients.fill(0);
        self.degrees.fill(None);
        self.leading.fill(None);
        self.rows = rows;
        self.columns = columns;
        self.x_capacity = x_capacity;
        Ok(())
    }

    fn retained_bytes(&self) -> usize {
        self.coefficients.capacity()
            + self.degrees.capacity() * core::mem::size_of::<Option<usize>>()
            + self.leading.capacity() * core::mem::size_of::<Option<PopovLeadingTerm>>()
    }

    fn column_index(&self, row: usize, column: usize) -> usize {
        row * self.columns + column
    }

    fn column_byte_offset(&self, row: usize, column: usize) -> usize {
        self.column_index(row, column) * self.x_capacity * F::BYTES
    }

    fn set_column(
        &mut self,
        row: usize,
        column: usize,
        polynomial: &Polynomial<F>,
    ) -> Result<(), ConfigError> {
        if polynomial.coefficient_count() > self.x_capacity {
            return Err(ConfigError::ResourceLimitExceeded {
                resource: "interpolation module X coefficients",
                required: polynomial.coefficient_count(),
                limit: self.x_capacity,
            });
        }
        let start = self.column_byte_offset(row, column);
        let source = polynomial.as_packed();
        self.coefficients[start..start + source.len()].copy_from_slice(source);
        let index = self.column_index(row, column);
        self.degrees[index] = polynomial.degree();
        Ok(())
    }

    fn recompute_leading(
        &mut self,
        row: usize,
        shifts: &[usize],
    ) -> Result<(), InterpolationError> {
        let mut leading = None;
        for (column, &shift) in shifts.iter().enumerate() {
            let Some(degree) = self.degrees[self.column_index(row, column)] else {
                continue;
            };
            let shifted_degree = degree
                .checked_add(shift)
                .ok_or(gfm::ReduceError::DegreeOverflow { degree, shift })?;
            let candidate = PopovLeadingTerm {
                degree,
                column,
                shifted_degree,
            };
            if leading.is_none_or(|current: PopovLeadingTerm| {
                (candidate.shifted_degree, candidate.column)
                    > (current.shifted_degree, current.column)
            }) {
                leading = Some(candidate);
            }
        }
        self.leading[row] = leading;
        Ok(())
    }

    fn recompute_degree(&mut self, row: usize, column: usize, maximum: usize) {
        let start = self.column_byte_offset(row, column);
        let bytes = &self.coefficients[start..start + self.x_capacity * F::BYTES];
        let degree = (0..maximum).rev().find(|&degree| {
            let offset = degree * F::BYTES;
            !F::read(&bytes[offset..offset + F::BYTES]).is_zero()
        });
        let index = self.column_index(row, column);
        self.degrees[index] = degree;
    }

    fn materialize(
        &self,
        row: usize,
        output: &mut BivariatePolynomial<F>,
    ) -> Result<(), ConfigError> {
        output.prepare_y_rows(self.columns)?;
        for column in 0..self.columns {
            let target = output.y_coefficient_mut(column);
            let Some(degree) = self.degrees[self.column_index(row, column)] else {
                target.set_zero();
                continue;
            };
            let byte_count = (degree + 1) * F::BYTES;
            let start = self.column_byte_offset(row, column);
            target.assign_packed(&self.coefficients[start..start + byte_count])?;
        }
        output.normalize();
        Ok(())
    }
}

impl<F: FieldKernels> ModuleScratch<F> {
    /// Construct empty reusable module-interpolation scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            received: Polynomial::zero(),
            received_powers: Vec::new(),
            product: Polynomial::zero(),
            basis: ModuleSlab::new(),
            reduction: WeakPopovScratch::new(),
        }
    }

    /// Retained packed-basis, metadata, power, and reduction capacity in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.basis.retained_bytes()
            + self.received.retained_capacity_bytes()
            + self.product.retained_capacity_bytes()
            + self
                .received_powers
                .iter()
                .map(Polynomial::retained_capacity_bytes)
                .sum::<usize>()
            + self.received_powers.capacity() * core::mem::size_of::<Polynomial<F>>()
            + self.reduction.capacity() * core::mem::size_of::<Option<usize>>()
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
/// packed basis and `R` powers held by `scratch` and the plan-owned invariants.
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
    fill_polynomial_powers(
        &scratch.received,
        multiplicity,
        &mut scratch.received_powers,
    )?;

    scratch.basis.prepare(parameters)?;
    for row in 0..scratch.basis.rows {
        module_row_into(
            row,
            multiplicity,
            &scratch.received_powers,
            &plan.vanishing_powers,
            &mut scratch.product,
            &mut scratch.basis,
        )?;
        scratch.basis.recompute_leading(row, &plan.column_shifts)?;
    }

    weak_popov_basis_with_scratch::<F, _>(
        &mut scratch.basis,
        &plan.column_shifts,
        &mut scratch.reduction,
    )?;

    let mut selected: Option<(usize, (usize, usize))> = None;
    for (row, leading) in scratch.basis.leading.iter().copied().enumerate() {
        let Some(leading) = leading else {
            continue;
        };
        if leading.shifted_degree > parameters.weighted_degree() {
            continue;
        }
        let key = (leading.shifted_degree, leading.column);
        if selected.is_none_or(|(_, selected_key)| key < selected_key) {
            selected = Some((row, key));
        }
    }
    let row = selected
        .map(|(row, _)| row)
        .ok_or(InterpolationError::InvalidResult {
            reason: "reduced interpolation module has no row within the weighted-degree bound",
        })?;
    scratch.basis.materialize(row, output)?;
    validate_result(parameters, points, values, output)?;
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
    received_powers: &[Polynomial<F>],
    vanishing_powers: &[Polynomial<F>],
    product: &mut Polynomial<F>,
    basis: &mut ModuleSlab<F>,
) -> Result<(), ConfigError> {
    if row < multiplicity {
        let vanishing = &vanishing_powers[multiplicity - row];
        for y in 0..=row {
            if binomial_odd(row, y) {
                received_powers[row - y].multiply_into(vanishing, product)?;
                basis.set_column(row, y, product)?;
            }
        }
    } else {
        let shift = row - multiplicity;
        for y in 0..=multiplicity {
            if binomial_odd(multiplicity, y) {
                basis.set_column(row, shift + y, &received_powers[multiplicity - y])?;
            }
        }
    }
    Ok(())
}

impl<F: FieldKernels> WeakPopovBasis<F> for ModuleSlab<F> {
    type Error = InterpolationError;

    fn row_count(&self) -> usize {
        self.rows
    }

    fn column_count(&self, _row: usize) -> usize {
        self.columns
    }

    fn degree(&self, row: usize, column: usize) -> Option<usize> {
        self.degrees[self.column_index(row, column)]
    }

    fn coefficient(&self, row: usize, column: usize, degree: usize) -> F::Elem {
        if degree >= self.x_capacity {
            return F::Elem::ZERO;
        }
        let start = self.column_byte_offset(row, column) + degree * F::BYTES;
        F::read(&self.coefficients[start..start + F::BYTES])
    }

    fn leading_term(
        &self,
        row: usize,
        _shifts: &[usize],
    ) -> Result<Option<PopovLeadingTerm>, Self::Error> {
        Ok(self.leading[row])
    }

    fn add_scaled_shifted_assign(
        &mut self,
        target: usize,
        pivot: usize,
        scale: F::Elem,
        shift: usize,
        shifts: &[usize],
    ) -> Result<(), Self::Error> {
        let row_bytes = self.columns * self.x_capacity * F::BYTES;
        let column_bytes = self.x_capacity * F::BYTES;
        for column in 0..self.columns {
            let target_count = self.degree(target, column).map_or(0, |degree| degree + 1);
            let pivot_count = self
                .degree(pivot, column)
                .and_then(|degree| degree.checked_add(shift))
                .map_or(0, |degree| degree + 1);
            let required = target_count.max(pivot_count);
            if required > self.x_capacity {
                return Err(ConfigError::ResourceLimitExceeded {
                    resource: "interpolation module X coefficients",
                    required,
                    limit: self.x_capacity,
                }
                .into());
            }
        }

        let prepared_scale = ops::Coeff::<F>::new(scale);

        {
            let degrees = &self.degrees;
            let columns = self.columns;
            let (target_row, pivot_row) = if target < pivot {
                let (lower, upper) = self.coefficients.split_at_mut(pivot * row_bytes);
                (
                    &mut lower[target * row_bytes..(target + 1) * row_bytes],
                    &upper[..row_bytes],
                )
            } else {
                let (lower, upper) = self.coefficients.split_at_mut(target * row_bytes);
                (
                    &mut upper[..row_bytes],
                    &lower[pivot * row_bytes..(pivot + 1) * row_bytes],
                )
            };
            for column in 0..columns {
                let Some(pivot_degree) = degrees[pivot * columns + column] else {
                    continue;
                };
                let source_count = pivot_degree + 1;
                let source_start = column * column_bytes;
                let target_start = source_start + shift * F::BYTES;
                ops::mul_add_with::<F>(
                    &mut target_row[target_start..target_start + source_count * F::BYTES],
                    &prepared_scale,
                    &pivot_row[source_start..source_start + source_count * F::BYTES],
                );
            }
        }

        for column in 0..self.columns {
            let target_degree = self.degree(target, column);
            let Some(pivot_degree) = self.degree(pivot, column) else {
                continue;
            };
            let shifted_pivot_degree =
                pivot_degree
                    .checked_add(shift)
                    .ok_or(ConfigError::GeometryOverflow {
                        context: "interpolation module shifted pivot degree",
                    })?;
            if target_degree != Some(shifted_pivot_degree) {
                let degree = target_degree.map_or(shifted_pivot_degree, |degree| {
                    degree.max(shifted_pivot_degree)
                });
                let index = self.column_index(target, column);
                self.degrees[index] = Some(degree);
                continue;
            }

            let start = self.column_byte_offset(target, column) + shifted_pivot_degree * F::BYTES;
            if F::read(&self.coefficients[start..start + F::BYTES]).is_zero() {
                self.recompute_degree(target, column, shifted_pivot_degree);
            }
        }
        self.recompute_leading(target, shifts)
    }
}

fn resize_zeroed<T: Clone + Default>(
    values: &mut Vec<T>,
    required: usize,
    context: &'static str,
) -> Result<(), ConfigError> {
    if values.capacity() < required {
        values
            .try_reserve_exact(required - values.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context,
                elements: required,
                element_size: core::mem::size_of::<T>(),
            })?;
    }
    values.resize(required, T::default());
    Ok(())
}
