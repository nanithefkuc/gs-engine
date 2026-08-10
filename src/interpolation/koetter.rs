//! Iterative Kötter/KNH multiplicity interpolation.
//!
//! Constraints are processed as a lower set: point-major, increasing total
//! Hasse order, and increasing `Y` order within a total order. Multiplication
//! of the pivot by `X + alpha` clears the current discrepancy because its
//! `(a,b)` Hasse value becomes the pivot's already-processed `(a-1,b)` value
//! (or zero when `a == 0`). It preserves every earlier constraint for the same
//! reason. Cross-multiplied row updates are linear combinations, so they also
//! preserve the processed constraint space without field inversion.
//!
//! Each basis polynomial starts with `X`-degree zero. Only one pivot is
//! multiplied by a linear `X` factor per constraint; row combinations do not
//! increase the maximum degree. With `C` constraints, `C + 1` `X`
//! coefficients per `Y` row are therefore a checked upper bound.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::{FieldKernels, backend_for};
use fgf::ops::{self, Coeff};

use crate::{BivariatePolynomial, ConfigError, GsParameters};

use super::{InterpolationError, binomial_odd, fill_powers, validate_inputs, validate_result};

/// Reusable coefficient slabs and local-jet buffers for Kötter interpolation.
pub struct KoetterScratch<F: FieldKernels> {
    basis: Vec<DenseBasisPolynomial<F>>,
    discrepancies: Vec<F::Elem>,
    x_powers: Vec<F::Elem>,
    y_powers: Vec<F::Elem>,
    jets: Vec<F::Elem>,
    row: Vec<u8>,
}

impl<F: FieldKernels> KoetterScratch<F> {
    /// Construct empty interpolation scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            basis: Vec::new(),
            discrepancies: Vec::new(),
            x_powers: Vec::new(),
            y_powers: Vec::new(),
            jets: Vec::new(),
            row: Vec::new(),
        }
    }
}

impl<F: FieldKernels> Default for KoetterScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Construct a nonzero interpolation polynomial with temporary scratch.
pub fn interpolate_koetter<F: FieldKernels>(
    parameters: GsParameters,
    points: &[F::Elem],
    values: &[F::Elem],
) -> Result<BivariatePolynomial<F>, InterpolationError> {
    interpolate_koetter_with_scratch(parameters, points, values, &mut KoetterScratch::new())
}

/// Construct a nonzero interpolation polynomial with reusable scratch.
pub fn interpolate_koetter_with_scratch<F: FieldKernels>(
    parameters: GsParameters,
    points: &[F::Elem],
    values: &[F::Elem],
    scratch: &mut KoetterScratch<F>,
) -> Result<BivariatePolynomial<F>, InterpolationError> {
    let mut output = BivariatePolynomial::zero();
    interpolate_koetter_into(parameters, points, values, scratch, &mut output)?;
    Ok(output)
}

/// Write a nonzero interpolation polynomial into reusable output storage.
pub fn interpolate_koetter_into<F: FieldKernels>(
    parameters: GsParameters,
    points: &[F::Elem],
    values: &[F::Elem],
    scratch: &mut KoetterScratch<F>,
    output: &mut BivariatePolynomial<F>,
) -> Result<(), InterpolationError> {
    validate_inputs(parameters, points, values)?;
    let geometry = prepare_scratch(parameters, scratch)?;
    let basis_count = geometry.basis_count;
    let x_capacity = geometry.x_capacity;
    let jet_count = geometry.jet_count;
    let jet_elements = geometry.jet_elements;
    let row_bytes = geometry.row_bytes;
    let basis = &mut scratch.basis[..basis_count];
    let discrepancies = &mut scratch.discrepancies[..basis_count];
    let x_powers = &mut scratch.x_powers[..x_capacity];
    let y_powers = &mut scratch.y_powers[..basis_count];
    let jets = &mut scratch.jets[..jet_elements];
    let row_scratch = &mut scratch.row[..row_bytes];

    for (&point, &value) in points.iter().zip(values) {
        fill_powers(x_powers, point);
        fill_powers(y_powers, value);
        let point_coefficient = Coeff::<F>::new(point);
        for (basis_index, polynomial) in basis.iter().enumerate() {
            let row = &mut jets[basis_index * jet_count..(basis_index + 1) * jet_count];
            for total_order in 0..parameters.multiplicity() {
                for y_order in 0..=total_order {
                    let x_order = total_order - y_order;
                    row[jet_index(total_order, y_order)] =
                        polynomial.discrepancy(x_order, y_order, x_powers, y_powers);
                }
            }
        }
        for total_order in 0..parameters.multiplicity() {
            for y_order in 0..=total_order {
                let constraint = jet_index(total_order, y_order);
                for (basis_index, discrepancy) in discrepancies.iter_mut().enumerate() {
                    *discrepancy = jets[basis_index * jet_count + constraint];
                }
                let Some(pivot) = select_pivot(basis, discrepancies) else {
                    continue;
                };
                let pivot_discrepancy = discrepancies[pivot];
                let pivot_scale = Coeff::<F>::new(pivot_discrepancy);

                for (target, target_discrepancy) in discrepancies.iter().copied().enumerate() {
                    if target == pivot || target_discrepancy.is_zero() {
                        continue;
                    }
                    cross_update_jets(
                        jets,
                        jet_count,
                        target,
                        pivot,
                        constraint,
                        pivot_discrepancy,
                        target_discrepancy,
                    );
                    let target_scale = Coeff::<F>::new(target_discrepancy);
                    let (target_polynomial, pivot_polynomial) =
                        target_and_pivot(basis, target, pivot);
                    target_polynomial.cross_update(
                        pivot_polynomial,
                        pivot_discrepancy,
                        &pivot_scale,
                        target_discrepancy,
                        &target_scale,
                    );
                }
                basis[pivot].multiply_x_plus(point, &point_coefficient, row_scratch);
                shift_pivot_jets(
                    &mut jets[pivot * jet_count..(pivot + 1) * jet_count],
                    parameters.multiplicity(),
                );
            }
        }
    }

    let selected = basis
        .iter()
        .enumerate()
        .filter_map(|(index, polynomial)| {
            polynomial
                .leading
                .filter(|term| term.weighted_degree <= parameters.weighted_degree() as u128)
                .map(|term| (term_key(term, index), index))
        })
        .min_by_key(|(key, _)| *key)
        .map(|(_, index)| index)
        .ok_or(InterpolationError::InvalidResult {
            reason: "Kötter basis contains no polynomial within the weighted-degree bound",
        })?;
    basis[selected].write_bivariate(output)?;
    validate_result(parameters, points, values, output)?;
    Ok(())
}

#[derive(Clone, Copy)]
struct ScratchGeometry {
    basis_count: usize,
    x_capacity: usize,
    jet_count: usize,
    jet_elements: usize,
    row_bytes: usize,
}

fn prepare_scratch<F: FieldKernels>(
    parameters: GsParameters,
    scratch: &mut KoetterScratch<F>,
) -> Result<ScratchGeometry, InterpolationError> {
    let basis_count =
        parameters
            .y_degree()
            .checked_add(1)
            .ok_or(ConfigError::GeometryOverflow {
                context: "Kötter basis count",
            })?;
    let x_capacity = parameters.resources().constraints().checked_add(1).ok_or(
        ConfigError::GeometryOverflow {
            context: "Kötter X capacity",
        },
    )?;
    let row_elements =
        basis_count
            .checked_mul(x_capacity)
            .ok_or(ConfigError::GeometryOverflow {
                context: "Kötter basis row elements",
            })?;
    let slab_bytes = row_elements
        .checked_mul(F::BYTES)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Kötter basis slab bytes",
        })?;
    let total_bytes = slab_bytes
        .checked_mul(basis_count)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Kötter coefficient bytes",
        })?;
    if total_bytes != parameters.resources().coefficient_bytes() {
        return Err(InterpolationError::InvalidResult {
            reason: "Kötter storage disagrees with the parameter estimate",
        });
    }

    if scratch.basis.len() < basis_count {
        scratch
            .basis
            .try_reserve_exact(basis_count - scratch.basis.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "Kötter interpolation basis",
                elements: basis_count,
                element_size: core::mem::size_of::<DenseBasisPolynomial<F>>(),
            })?;
        scratch
            .basis
            .resize_with(basis_count, DenseBasisPolynomial::empty);
    }
    for (basis_index, polynomial) in scratch.basis[..basis_count].iter_mut().enumerate() {
        polynomial.reset_monomial(
            basis_count,
            x_capacity,
            parameters.max_degree(),
            basis_index,
        )?;
    }

    let multiplicity_plus_one =
        parameters
            .multiplicity()
            .checked_add(1)
            .ok_or(ConfigError::GeometryOverflow {
                context: "Kötter Hasse jet count",
            })?;
    let jet_count = parameters
        .multiplicity()
        .checked_mul(multiplicity_plus_one)
        .map(|product| product / 2)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Kötter Hasse jet count",
        })?;
    let jet_elements = basis_count
        .checked_mul(jet_count)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Kötter Hasse jet elements",
        })?;
    let row_bytes = x_capacity
        .checked_mul(F::BYTES)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Kötter row scratch bytes",
        })?;
    ensure_len(
        &mut scratch.discrepancies,
        basis_count,
        "Kötter discrepancies",
    )?;
    ensure_len(&mut scratch.x_powers, x_capacity, "Kötter X powers")?;
    ensure_len(&mut scratch.y_powers, basis_count, "Kötter Y powers")?;
    ensure_len(&mut scratch.jets, jet_elements, "Kötter Hasse jets")?;
    ensure_len(&mut scratch.row, row_bytes, "Kötter row scratch")?;
    Ok(ScratchGeometry {
        basis_count,
        x_capacity,
        jet_count,
        jet_elements,
        row_bytes,
    })
}

#[derive(Clone, Copy)]
struct DenseLeadingTerm {
    x_degree: usize,
    y_degree: usize,
    weighted_degree: u128,
}

struct DenseBasisPolynomial<F: FieldKernels> {
    coefficients: Vec<u8>,
    active: Vec<usize>,
    y_rows: usize,
    x_capacity: usize,
    y_weight: usize,
    leading: Option<DenseLeadingTerm>,
    field: core::marker::PhantomData<F>,
}

impl<F: FieldKernels> DenseBasisPolynomial<F> {
    fn empty() -> Self {
        Self {
            coefficients: Vec::new(),
            active: Vec::new(),
            y_rows: 0,
            x_capacity: 0,
            y_weight: 0,
            leading: None,
            field: core::marker::PhantomData,
        }
    }

    fn reset_monomial(
        &mut self,
        y_rows: usize,
        x_capacity: usize,
        y_weight: usize,
        y_degree: usize,
    ) -> Result<(), ConfigError> {
        let elements = y_rows
            .checked_mul(x_capacity)
            .ok_or(ConfigError::GeometryOverflow {
                context: "Kötter basis polynomial elements",
            })?;
        let bytes = elements
            .checked_mul(F::BYTES)
            .ok_or(ConfigError::GeometryOverflow {
                context: "Kötter basis polynomial bytes",
            })?;
        ensure_len(&mut self.coefficients, bytes, "Kötter basis coefficients")?;
        ensure_len(&mut self.active, y_rows, "Kötter active row lengths")?;
        self.coefficients[..bytes].fill(0);
        self.active[..y_rows].fill(0);
        self.y_rows = y_rows;
        self.x_capacity = x_capacity;
        self.y_weight = y_weight;
        self.leading = None;
        self.write(y_degree, 0, F::Elem::ONE);
        self.active[y_degree] = 1;
        self.refresh_leading();
        Ok(())
    }

    fn discrepancy(
        &self,
        x_order: usize,
        y_order: usize,
        x_powers: &[F::Elem],
        y_powers: &[F::Elem],
    ) -> F::Elem {
        if y_order >= self.y_rows {
            return F::Elem::ZERO;
        }
        let mut value = F::Elem::ZERO;
        for y_degree in y_order..self.y_rows {
            if !binomial_odd(y_degree, y_order) {
                continue;
            }
            for x_degree in x_order..self.active[y_degree] {
                if binomial_odd(x_degree, x_order) {
                    value = value.add(
                        self.read(y_degree, x_degree)
                            .mul(x_powers[x_degree - x_order])
                            .mul(y_powers[y_degree - y_order]),
                    );
                }
            }
        }
        value
    }

    fn cross_update(
        &mut self,
        pivot: &Self,
        pivot_discrepancy: F::Elem,
        pivot_scale: &Coeff<F>,
        target_discrepancy: F::Elem,
        target_scale: &Coeff<F>,
    ) {
        for y_degree in 0..self.y_rows {
            let target_active = self.active[y_degree];
            let pivot_active = pivot.active[y_degree];
            if target_active != 0 {
                scale_row::<F>(
                    self.row_prefix_mut(y_degree, target_active),
                    pivot_discrepancy,
                    pivot_scale,
                );
            }
            if pivot_active != 0 {
                axpy_row::<F>(
                    self.row_prefix_mut(y_degree, pivot_active),
                    target_discrepancy,
                    target_scale,
                    pivot.row_prefix(y_degree, pivot_active),
                );
            }
            self.active[y_degree] = self.trimmed_active(y_degree, target_active.max(pivot_active));
        }
        self.refresh_leading();
    }

    fn multiply_x_plus(&mut self, point: F::Elem, scale: &Coeff<F>, scratch: &mut [u8]) {
        for y_degree in 0..self.y_rows {
            let active = self.active[y_degree];
            if active == 0 {
                continue;
            }
            debug_assert!(active < self.x_capacity);
            let active_bytes = active * F::BYTES;
            scratch[..active_bytes].copy_from_slice(self.row_prefix(y_degree, active));
            scale_into_row::<F>(
                self.row_prefix_mut(y_degree, active),
                point,
                scale,
                &scratch[..active_bytes],
            );
            let row_start = self.row_start(y_degree);
            self.coefficients[row_start + active_bytes..row_start + active_bytes + F::BYTES]
                .fill(0);
            ops::add_assign::<F>(
                &mut self.coefficients[row_start + F::BYTES..row_start + F::BYTES + active_bytes],
                &scratch[..active_bytes],
            );
            self.active[y_degree] = active + 1;
        }
        self.refresh_leading();
    }

    fn write_bivariate(
        &self,
        output: &mut BivariatePolynomial<F>,
    ) -> Result<(), InterpolationError> {
        output.prepare_y_rows(self.y_rows)?;
        for y_degree in 0..self.y_rows {
            let active = self.active[y_degree];
            output
                .y_coefficient_mut(y_degree)
                .assign_packed(self.row_prefix(y_degree, active))?;
        }
        output.normalize();
        Ok(())
    }

    fn refresh_leading(&mut self) {
        self.leading = None;
        for (y_degree, &active) in self.active.iter().enumerate() {
            let Some(x_degree) = active.checked_sub(1) else {
                continue;
            };
            let weighted_degree = x_degree as u128 + (y_degree as u128) * (self.y_weight as u128);
            let candidate = DenseLeadingTerm {
                x_degree,
                y_degree,
                weighted_degree,
            };
            if self.leading.is_none_or(|current| {
                (
                    candidate.weighted_degree,
                    candidate.y_degree,
                    candidate.x_degree,
                ) > (current.weighted_degree, current.y_degree, current.x_degree)
            }) {
                self.leading = Some(candidate);
            }
        }
    }

    fn trimmed_active(&self, y_degree: usize, mut upper_bound: usize) -> usize {
        while upper_bound != 0 && self.read(y_degree, upper_bound - 1).is_zero() {
            upper_bound -= 1;
        }
        upper_bound
    }

    fn read(&self, y_degree: usize, x_degree: usize) -> F::Elem {
        let start = self.row_start(y_degree) + x_degree * F::BYTES;
        F::read(&self.coefficients[start..start + F::BYTES])
    }

    fn write(&mut self, y_degree: usize, x_degree: usize, value: F::Elem) {
        let start = self.row_start(y_degree) + x_degree * F::BYTES;
        F::write(&mut self.coefficients[start..start + F::BYTES], value);
    }

    fn row_start(&self, y_degree: usize) -> usize {
        y_degree * self.x_capacity * F::BYTES
    }

    fn row_prefix(&self, y_degree: usize, active: usize) -> &[u8] {
        let start = self.row_start(y_degree);
        &self.coefficients[start..start + active * F::BYTES]
    }

    fn row_prefix_mut(&mut self, y_degree: usize, active: usize) -> &mut [u8] {
        let start = self.row_start(y_degree);
        &mut self.coefficients[start..start + active * F::BYTES]
    }
}

fn ensure_len<T: Default + Clone>(
    values: &mut Vec<T>,
    required: usize,
    context: &'static str,
) -> Result<(), ConfigError> {
    if required > values.len() {
        values
            .try_reserve_exact(required - values.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context,
                elements: required,
                element_size: core::mem::size_of::<T>(),
            })?;
        values.resize(required, T::default());
    }
    Ok(())
}

fn select_pivot<F: FieldKernels>(
    basis: &[DenseBasisPolynomial<F>],
    discrepancies: &[F::Elem],
) -> Option<usize> {
    basis
        .iter()
        .zip(discrepancies)
        .enumerate()
        .filter(|(_, (_, discrepancy))| !discrepancy.is_zero())
        .filter_map(|(index, (polynomial, _))| {
            polynomial
                .leading
                .map(|term| (term_key(term, index), index))
        })
        .min_by_key(|(key, _)| *key)
        .map(|(_, index)| index)
}

fn term_key(term: DenseLeadingTerm, basis_index: usize) -> (u128, usize, usize, usize) {
    (
        term.weighted_degree,
        term.y_degree,
        term.x_degree,
        basis_index,
    )
}

fn target_and_pivot<F: FieldKernels>(
    basis: &mut [DenseBasisPolynomial<F>],
    target: usize,
    pivot: usize,
) -> (&mut DenseBasisPolynomial<F>, &DenseBasisPolynomial<F>) {
    debug_assert_ne!(target, pivot);
    if target < pivot {
        let (left, right) = basis.split_at_mut(pivot);
        (&mut left[target], &right[0])
    } else {
        let (left, right) = basis.split_at_mut(target);
        (&mut right[0], &left[pivot])
    }
}

const fn jet_index(total_order: usize, y_order: usize) -> usize {
    total_order * (total_order + 1) / 2 + y_order
}

fn cross_update_jets<E: Elem>(
    jets: &mut [E],
    jet_count: usize,
    target: usize,
    pivot: usize,
    constraint: usize,
    pivot_discrepancy: E,
    target_discrepancy: E,
) {
    debug_assert_ne!(target, pivot);
    let (target_row, pivot_row) = if target < pivot {
        let (left, right) = jets.split_at_mut(pivot * jet_count);
        (
            &mut left[target * jet_count..(target + 1) * jet_count],
            &right[..jet_count],
        )
    } else {
        let (left, right) = jets.split_at_mut(target * jet_count);
        (
            &mut right[..jet_count],
            &left[pivot * jet_count..(pivot + 1) * jet_count],
        )
    };
    target_row[constraint] = E::ZERO;
    for index in constraint + 1..jet_count {
        target_row[index] = target_row[index]
            .mul(pivot_discrepancy)
            .add(pivot_row[index].mul(target_discrepancy));
    }
}

fn shift_pivot_jets<E: Elem>(jets: &mut [E], multiplicity: usize) {
    for total_order in (0..multiplicity).rev() {
        for y_order in 0..=total_order {
            let x_order = total_order - y_order;
            let destination = jet_index(total_order, y_order);
            jets[destination] = if x_order == 0 {
                E::ZERO
            } else {
                jets[jet_index(total_order - 1, y_order)]
            };
        }
    }
}

fn scale_row<F: FieldKernels>(row: &mut [u8], scale: F::Elem, prepared: &Coeff<F>) {
    if row.len() >= backend_for::<F>().lane_bytes() {
        ops::mul_assign_with::<F>(row, prepared);
    } else {
        for coefficient in row.chunks_exact_mut(F::BYTES) {
            let value = F::read(coefficient).mul(scale);
            F::write(coefficient, value);
        }
    }
}

fn axpy_row<F: FieldKernels>(
    destination: &mut [u8],
    scale: F::Elem,
    prepared: &Coeff<F>,
    source: &[u8],
) {
    if source.len() >= backend_for::<F>().lane_bytes() {
        ops::mul_add_with::<F>(destination, prepared, source);
    } else {
        for (output, input) in destination
            .chunks_exact_mut(F::BYTES)
            .zip(source.chunks_exact(F::BYTES))
        {
            let value = F::read(output).add(scale.mul(F::read(input)));
            F::write(output, value);
        }
    }
}

fn scale_into_row<F: FieldKernels>(
    destination: &mut [u8],
    scale: F::Elem,
    prepared: &Coeff<F>,
    source: &[u8],
) {
    if source.len() >= backend_for::<F>().lane_bytes() {
        ops::mul_into_with::<F>(destination, prepared, source);
    } else {
        for (output, input) in destination
            .chunks_exact_mut(F::BYTES)
            .zip(source.chunks_exact(F::BYTES))
        {
            F::write(output, scale.mul(F::read(input)));
        }
    }
}
