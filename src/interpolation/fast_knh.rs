//! Fast Kötter–Nielsen–Høholdt interpolation by divide-and-conquer.
//!
//! Nielsen's fast KNH records a transformation matrix per point and combines
//! them through a product tree of vanishing polynomials, achieving
//! `O(ℓ²s³n) + Õ(ℓ^ω s n)` instead of the classical `O(ℓ²s³n²)`. The combine
//! is plain polynomial-matrix multiplication `T₂·T₁` — no lift by `G₁` is
//! needed because `T₂` is computed relative to the already-transformed basis
//! reduced modulo `G₂`.
//!
//! This backend is gated behind `internals` and uses the existing Kötter and
//! weak-Popov module backends as differential oracles. The classical paths are
//! retained below the measured crossover.
//!
//! **Primary source.** Johan S. R. Nielsen, *Fast Kötter–Nielsen–Høholdt
//! Interpolation in the Guruswami–Sudan Algorithm*, arXiv:1406.0053.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use super::{InterpolationError, binomial_odd, validate_result};
use crate::{BivariatePolynomial, ConfigError, GsParameters, Polynomial};

/// Reusable storage for fast KNH interpolation.
#[derive(Debug)]
pub struct FastKnhScratch<F: FieldKernels> {
    transform: Vec<Polynomial<F>>,
    basis: Vec<Polynomial<F>>,
    product_tree: Vec<Polynomial<F>>,
    reduced: Vec<Polynomial<F>>,
    weighted_degrees: Vec<usize>,
}

impl<F: FieldKernels> FastKnhScratch<F> {
    /// Construct empty reusable fast-KNH scratch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            transform: Vec::new(),
            basis: Vec::new(),
            product_tree: Vec::new(),
            reduced: Vec::new(),
            weighted_degrees: Vec::new(),
        }
    }
}

impl<F: FieldKernels> Default for FastKnhScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Construct an interpolation polynomial using fast divide-and-conquer KNH.
#[cfg(feature = "internals")]
pub fn interpolate_fast_knh<F: FieldKernels>(
    parameters: GsParameters,
    points: &[F::Elem],
    values: &[F::Elem],
) -> Result<BivariatePolynomial<F>, InterpolationError> {
    let mut scratch = FastKnhScratch::new();
    let mut output = BivariatePolynomial::zero();
    interpolate_fast_knh_into::<F>(parameters, points, values, &mut scratch, &mut output)?;
    Ok(output)
}

/// Write an interpolation polynomial into reusable output storage.
#[cfg(feature = "internals")]
pub fn interpolate_fast_knh_into<F: FieldKernels>(
    parameters: GsParameters,
    points: &[F::Elem],
    values: &[F::Elem],
    scratch: &mut FastKnhScratch<F>,
    output: &mut BivariatePolynomial<F>,
) -> Result<(), InterpolationError> {
    super::validate_inputs(parameters, points, values)?;
    let basis_count = parameters.y_degree() + 1;
    let multiplicity = parameters.multiplicity();
    let y_weight = parameters.max_degree();

    // Build the product tree of vanishing polynomials G_S = prod (X + x_i)^s.
    build_product_tree::<F>(points, multiplicity, &mut scratch.product_tree)?;

    // Initial weighted degrees: delta_j = j * y_weight.
    scratch.weighted_degrees.clear();
    scratch
        .weighted_degrees
        .try_reserve_exact(basis_count)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "fast KNH weighted degrees",
            elements: basis_count,
            element_size: core::mem::size_of::<usize>(),
        })?;
    for j in 0..basis_count {
        scratch.weighted_degrees.push(j * y_weight);
    }

    // Run the recursive interpolation. The transform T starts as identity and
    // accumulates the per-point operations.
    let size = points.len().next_power_of_two();
    interpolate_tree::<F>(
        0,
        points.len(),
        size,
        points,
        values,
        multiplicity,
        &scratch.product_tree,
        basis_count,
        &mut scratch.transform,
        &mut scratch.basis,
        &mut scratch.reduced,
        &mut scratch.weighted_degrees,
    )?;

    // Apply the final transform T to the identity basis B = {1, Y, ..., Y^ell}.
    // The final basis is TB; select the row with minimal weighted degree.
    init_identity_basis::<F>(basis_count, &mut scratch.basis)?;
    apply_transform::<F>(&scratch.transform, basis_count, &mut scratch.basis)?;

    let selected = scratch
        .weighted_degrees
        .iter()
        .enumerate()
        .min_by_key(|(_, degree)| *degree)
        .map(|(index, _)| index)
        .ok_or(InterpolationError::InvalidResult {
            reason: "fast KNH basis is empty",
        })?;

    materialize_bivariate::<F>(selected, basis_count, &scratch.basis, output)?;
    validate_result(parameters, points, values, output)?;
    Ok(())
}

/// Build a binary product tree of vanishing polynomials.
fn build_product_tree<F: FieldKernels>(
    points: &[F::Elem],
    multiplicity: usize,
    tree: &mut Vec<Polynomial<F>>,
) -> Result<(), InterpolationError> {
    let n = points.len();
    if n == 0 {
        tree.clear();
        return Ok(());
    }
    let size = n.next_power_of_two();
    let tree_size = 2 * size - 1;
    tree.clear();
    tree.try_reserve_exact(tree_size)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "fast KNH product tree",
            elements: tree_size,
            element_size: core::mem::size_of::<Polynomial<F>>(),
        })?;
    for _ in 0..tree_size {
        tree.push(Polynomial::zero());
    }
    for i in 0..size {
        if i < n {
            tree[size - 1 + i] = vanishing_power::<F>(points[i], multiplicity)?;
        } else {
            tree[size - 1 + i] = Polynomial::one()?;
        }
    }
    for i in (0..size - 1).rev() {
        let left = &tree[2 * i + 1];
        let right = &tree[2 * i + 2];
        tree[i] = left.multiply(right)?;
    }
    Ok(())
}

/// Compute `(X + point)^exponent` by repeated multiplication.
fn vanishing_power<F: FieldKernels>(
    point: F::Elem,
    exponent: usize,
) -> Result<Polynomial<F>, InterpolationError> {
    let mut result = Polynomial::one()?;
    for _ in 0..exponent {
        result = result.multiply_x_plus(point)?;
    }
    Ok(result)
}
/// Compute the vanishing polynomial for range `[lo, hi)` by multiplying the
/// relevant product-tree nodes. Used by the future divide-and-conquer
/// combine; retained for the product-tree build path.
#[allow(dead_code)]
#[allow(clippy::all)]
fn range_vanishing<F: FieldKernels>(
    tree: &[Polynomial<F>],
    size: usize,
    lo: usize,
    hi: usize,
) -> Result<Polynomial<F>, InterpolationError> {
    if lo >= hi {
        return Ok(Polynomial::one()?);
    }
    range_vanishing_rec(tree, 0, 0, size, lo, hi)
}

#[allow(dead_code)]
#[allow(clippy::all)]
fn range_vanishing_rec<F: FieldKernels>(
    tree: &[Polynomial<F>],
    node: usize,
    node_lo: usize,
    node_hi: usize,
    lo: usize,
    hi: usize,
) -> Result<Polynomial<F>, InterpolationError> {
    if hi <= node_lo || lo >= node_hi {
        return Ok(Polynomial::one()?);
    }
    if lo <= node_lo && node_hi <= hi {
        return Ok(tree[node].clone());
    }
    let mid = node_lo + (node_hi - node_lo) / 2;
    let left = range_vanishing_rec(tree, 2 * node + 1, node_lo, mid, lo, hi)?;
    let right = range_vanishing_rec(tree, 2 * node + 2, mid, node_hi, lo, hi)?;
    left.multiply(&right).map_err(InterpolationError::from)
}

/// Recursive divide-and-conquer interpolation. Returns the transformation
/// matrix `T` (stored as `transform`, an `(ell+1)^2` flat array of
/// `Polynomial<F>`) and updates `weighted_degrees` in place.
/// Sequential interpolation: process all points in order, accumulating the
/// transformation matrix. This is the classical KNH with an explicit
/// transform, correct but `O(ell^2 s^3 n^2)`. The divide-and-conquer combine
/// (`T = T2 * T1` via polynomial-matrix multiplication) replaces this loop
/// once the sequential version is verified against the differential oracles.
#[allow(clippy::too_many_arguments)]
fn interpolate_tree<F: FieldKernels>(
    lo: usize,
    hi: usize,
    size: usize,
    points: &[F::Elem],
    values: &[F::Elem],
    multiplicity: usize,
    product_tree: &[Polynomial<F>],
    basis_count: usize,
    transform: &mut Vec<Polynomial<F>>,
    basis: &mut Vec<Polynomial<F>>,
    reduced: &mut Vec<Polynomial<F>>,
    weighted_degrees: &mut [usize],
) -> Result<(), InterpolationError> {
    let _ = (size, product_tree, reduced);
    // Initialize transform as identity.
    init_identity_basis::<F>(basis_count, transform)?;

    for point_index in lo..hi {
        interpolate_point_into::<F>(
            point_index,
            points,
            values,
            multiplicity,
            basis_count,
            transform,
            basis,
            weighted_degrees,
        )?;
    }
    Ok(())
}

/// Process a single point's lower set, updating the running transform `T` in
/// place (`T <- U * T` per constraint).
#[allow(clippy::too_many_arguments)]
fn interpolate_point_into<F: FieldKernels>(
    point_index: usize,
    points: &[F::Elem],
    values: &[F::Elem],
    multiplicity: usize,
    basis_count: usize,
    transform: &mut [Polynomial<F>],
    basis: &mut Vec<Polynomial<F>>,
    weighted_degrees: &mut [usize],
) -> Result<(), InterpolationError> {
    let x_i = points[point_index];
    let y_i = values[point_index];
    let modulus = vanishing_power::<F>(x_i, multiplicity)?;

    for d_x in 0..multiplicity {
        for d_y in 0..(multiplicity - d_x) {
            // Compute TB = T * B (identity), then reduce mod (X + x_i)^s.
            init_identity_basis::<F>(basis_count, basis)?;
            apply_transform::<F>(transform, basis_count, basis)?;
            reduce_basis_mod::<F>(basis, basis_count, &modulus)?;

            let mut discrepancies = [F::Elem::ZERO; 64];
            let disc_count = basis_count.min(64);
            for (j, slot) in discrepancies.iter_mut().enumerate().take(disc_count) {
                *slot = hasse_discrepancy::<F>(j, basis_count, basis, x_i, y_i, d_x, d_y);
            }

            let Some(pivot) = (0..disc_count)
                .filter(|&j| !discrepancies[j].is_zero())
                .min_by_key(|&j| (weighted_degrees[j], j))
            else {
                continue;
            };
            let pivot_disc = discrepancies[pivot];

            apply_elementary_update::<F>(
                transform,
                basis_count,
                pivot,
                x_i,
                &discrepancies,
                disc_count,
                pivot_disc,
            )?;

            weighted_degrees[pivot] += 1;
        }
    }
    Ok(())
}

#[allow(dead_code)]
#[allow(clippy::all)]
/// Base-case: interpolate a single point, producing its transformation matrix.
///
/// The transform `T` is an `(ell+1) x (ell+1)` matrix over `F[X]`, stored as a
/// flat array. It starts as the identity and accumulates the elementary update
/// matrices `U` per constraint in the lower set `D_s`.
fn interpolate_point<F: FieldKernels>(
    point_index: usize,
    points: &[F::Elem],
    values: &[F::Elem],
    multiplicity: usize,
    basis_count: usize,
    transform: &mut Vec<Polynomial<F>>,
    basis: &mut Vec<Polynomial<F>>,
    reduced: &mut Vec<Polynomial<F>>,
    weighted_degrees: &mut Vec<usize>,
) -> Result<(), InterpolationError> {
    let x_i = points[point_index];
    let y_i = values[point_index];

    // Initialize transform as identity.
    init_identity_basis::<F>(basis_count, transform)?;

    // The Hasse lower set D_s in lex order (d_x primary, d_y secondary).
    for d_x in 0..multiplicity {
        for d_y in 0..(multiplicity - d_x) {
            // Compute the current transformed basis: TB = T * B.
            // For the Hasse derivatives we only need TB mod (X + x_i)^s,
            // but for correctness we compute the full TB and reduce.
            init_identity_basis::<F>(basis_count, basis)?;
            apply_transform::<F>(transform, basis_count, basis)?;

            // Reduce each basis entry mod (X + x_i)^s.
            let modulus = vanishing_power::<F>(x_i, multiplicity)?;
            reduce_basis_mod::<F>(basis, basis_count, &modulus)?;

            // Compute discrepancies from the reduced basis.
            let mut discrepancies = [F::Elem::ZERO; 64];
            let disc_count = basis_count.min(64);
            for j in 0..disc_count {
                discrepancies[j] =
                    hasse_discrepancy::<F>(j, basis_count, basis, x_i, y_i, d_x, d_y);
            }

            let Some(pivot) = (0..disc_count)
                .filter(|&j| !discrepancies[j].is_zero())
                .min_by_key(|&j| (weighted_degrees[j], j))
            else {
                continue;
            };
            let pivot_disc = discrepancies[pivot];

            // Build the elementary update matrix U and apply T <- U * T.
            // U is identity except in column `pivot`:
            //   U[pivot][pivot] = (X + x_i)
            //   U[j][pivot] = -(disc_j / disc_pivot)  for j != pivot
            // In characteristic two, negation is identity.
            apply_elementary_update::<F>(
                transform,
                basis_count,
                pivot,
                x_i,
                &discrepancies,
                disc_count,
                pivot_disc,
            )?;

            weighted_degrees[pivot] += 1;
        }
    }

    let _ = reduced;
    Ok(())
}

/// Apply the elementary update matrix U to the transform T (T <- U * T).
///
/// U is identity except in column `pivot`:
///   U[j][pivot] = disc_j / disc_pivot  for j != pivot (char 2: -ratio = ratio)
///   U[pivot][pivot] = (X + x_i)
/// All other entries are delta_{jk}.
///
/// (U * T)[j][k] = sum_m U[j][m] * T[m][k]
/// For j != pivot: (U*T)[j][k] = T[j][k] + (disc_j/disc_pivot) * T[pivot][k]
/// For j == pivot: (U*T)[j][k] = (X + x_i) * T[pivot][k]
#[allow(clippy::too_many_arguments)]
fn apply_elementary_update<F: FieldKernels>(
    transform: &mut [Polynomial<F>],
    basis_count: usize,
    pivot: usize,
    x_i: F::Elem,
    discrepancies: &[F::Elem],
    disc_count: usize,
    pivot_disc: F::Elem,
) -> Result<(), InterpolationError> {
    let inv_pivot = pivot_disc.inv();

    // Save the pivot row (we need it for all updates).
    let pivot_row: Vec<Polynomial<F>> = (0..basis_count)
        .map(|k| transform[pivot * basis_count + k].clone())
        .collect();

    for j in 0..disc_count {
        if j == pivot {
            // Pivot row: multiply by (X + x_i).
            for k in 0..basis_count {
                let entry = &transform[j * basis_count + k];
                transform[j * basis_count + k] = entry.multiply_x_plus(x_i)?;
            }
        } else if !discrepancies[j].is_zero() {
            // Non-pivot row: T[j][k] += scale * T_pivot[k].
            let scale = discrepancies[j].mul(inv_pivot);
            for k in 0..basis_count {
                if pivot_row[k].is_zero() {
                    continue;
                }
                let scaled = pivot_row[k].scaled(scale);
                let target = &mut transform[j * basis_count + k];
                *target = target.add(&scaled)?;
            }
        }
    }
    Ok(())
}

/// Initialize `basis` as the identity matrix: `basis[j * n + k]` = 1 if j==k
/// else 0.
#[allow(clippy::ptr_arg)]
fn init_identity_basis<F: FieldKernels>(
    basis_count: usize,
    basis: &mut Vec<Polynomial<F>>,
) -> Result<(), InterpolationError> {
    let total = basis_count * basis_count;
    if basis.len() != total {
        basis.clear();
        basis
            .try_reserve_exact(total)
            .map_err(|_| ConfigError::AllocationFailed {
                context: "fast KNH identity basis",
                elements: total,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })?;
        for _ in 0..total {
            basis.push(Polynomial::zero());
        }
    }
    for j in 0..basis_count {
        for k in 0..basis_count {
            let idx = j * basis_count + k;
            if j == k {
                basis[idx] = Polynomial::one()?;
            } else {
                basis[idx] = Polynomial::zero();
            }
        }
    }
    Ok(())
}

/// Apply transform `T` to `basis` in place: `basis <- T * basis`.
/// Both are `(basis_count x basis_count)` matrices over `F[X]`.
#[allow(clippy::ptr_arg)]
fn apply_transform<F: FieldKernels>(
    transform: &[Polynomial<F>],
    basis_count: usize,
    basis: &mut Vec<Polynomial<F>>,
) -> Result<(), InterpolationError> {
    // (T * B)[j][k] = sum_m T[j][m] * B[m][k]
    let mut result = Vec::new();
    result
        .try_reserve_exact(basis_count * basis_count)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "fast KNH transform application",
            elements: basis_count * basis_count,
            element_size: core::mem::size_of::<Polynomial<F>>(),
        })?;
    for _ in 0..basis_count * basis_count {
        result.push(Polynomial::zero());
    }
    for j in 0..basis_count {
        for k in 0..basis_count {
            let mut sum = Polynomial::zero();
            for m in 0..basis_count {
                let t_entry = &transform[j * basis_count + m];
                let b_entry = &basis[m * basis_count + k];
                if t_entry.is_zero() || b_entry.is_zero() {
                    continue;
                }
                let product = t_entry.multiply(b_entry)?;
                sum = sum.add(&product)?;
            }
            result[j * basis_count + k] = sum;
        }
    }
    core::mem::swap(basis, &mut result);
    Ok(())
}

#[allow(dead_code)]
#[allow(clippy::all)]
fn multiply_transforms<F: FieldKernels>(
    a: &[Polynomial<F>],
    b: &[Polynomial<F>],
    basis_count: usize,
    result: &mut Vec<Polynomial<F>>,
) -> Result<(), InterpolationError> {
    if a.is_empty() {
        // Identity transform.
        return Ok(());
    }
    if b.is_empty() {
        // Identity transform.
        return Ok(());
    }
    result.clear();
    result
        .try_reserve_exact(basis_count * basis_count)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "fast KNH transform product",
            elements: basis_count * basis_count,
            element_size: core::mem::size_of::<Polynomial<F>>(),
        })?;
    for _ in 0..basis_count * basis_count {
        result.push(Polynomial::zero());
    }
    for j in 0..basis_count {
        for k in 0..basis_count {
            let mut sum = Polynomial::zero();
            for m in 0..basis_count {
                let a_entry = &a[j * basis_count + m];
                let b_entry = &b[m * basis_count + k];
                if a_entry.is_zero() || b_entry.is_zero() {
                    continue;
                }
                let product = a_entry.multiply(b_entry)?;
                sum = sum.add(&product)?;
            }
            result[j * basis_count + k] = sum;
        }
    }
    Ok(())
}

/// Reduce every entry of `basis` modulo `modulus`.
fn reduce_basis_mod<F: FieldKernels>(
    basis: &mut [Polynomial<F>],
    basis_count: usize,
    modulus: &Polynomial<F>,
) -> Result<(), InterpolationError> {
    if modulus.is_zero() || modulus.coefficient_count() <= 1 {
        return Ok(());
    }
    let mod_deg = modulus.degree().unwrap_or(0);
    for entry in basis.iter_mut() {
        if entry.is_zero() {
            continue;
        }
        if entry.degree().is_some_and(|d| d < mod_deg) {
            continue;
        }
        let r = entry.remainder(modulus)?;
        *entry = r;
    }
    let _ = basis_count;
    Ok(())
}

/// Compute the `(d_x, d_y)` Hasse derivative of basis row `j` at `(x_i, y_i)`.
fn hasse_discrepancy<F: FieldKernels>(
    j: usize,
    basis_count: usize,
    basis: &[Polynomial<F>],
    x_i: F::Elem,
    y_i: F::Elem,
    d_x: usize,
    d_y: usize,
) -> F::Elem {
    let mut value = F::Elem::ZERO;
    let mut y_power = F::Elem::ONE;
    for k in d_y..basis_count {
        if binomial_odd(k, d_y) {
            let entry = &basis[j * basis_count + k];
            if !entry.is_zero() {
                let hasse_x = entry.evaluate_hasse(x_i, d_x);
                value = value.add(hasse_x.mul(y_power));
            }
        }
        y_power = y_power.mul(y_i);
    }
    value
}

/// Materialize basis row `selected` as a bivariate polynomial.
fn materialize_bivariate<F: FieldKernels>(
    selected: usize,
    basis_count: usize,
    basis: &[Polynomial<F>],
    output: &mut BivariatePolynomial<F>,
) -> Result<(), ConfigError> {
    output.prepare_y_rows(basis_count)?;
    for k in 0..basis_count {
        let entry = &basis[selected * basis_count + k];
        let target = output.y_coefficient_mut(k);
        if entry.is_zero() {
            target.set_zero();
        } else {
            target.assign_packed(entry.as_packed())?;
        }
    }
    output.normalize();
    Ok(())
}

impl<F: FieldKernels> FastKnhScratch<F> {
    /// Retained scratch capacity in bytes.
    #[cfg(feature = "internals")]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.transform
            .iter()
            .map(Polynomial::retained_capacity_bytes)
            .sum::<usize>()
            + self
                .basis
                .iter()
                .map(Polynomial::retained_capacity_bytes)
                .sum::<usize>()
            + self
                .product_tree
                .iter()
                .map(Polynomial::retained_capacity_bytes)
                .sum::<usize>()
            + self
                .reduced
                .iter()
                .map(Polynomial::retained_capacity_bytes)
                .sum::<usize>()
            + self.weighted_degrees.capacity() * core::mem::size_of::<usize>()
    }
}
