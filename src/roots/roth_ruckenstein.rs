use alloc::vec::Vec;
use core::cmp::Ordering;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use crate::geometry::try_zeroed;
use crate::{BivariatePolynomial, ConfigError, Polynomial};

use super::RootError;
use super::field_roots::{FieldRootScratch, base_field_roots_into, element_key};

/// Caller-provided limits for Roth–Ruckenstein prefix lifting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RothRuckensteinLimits {
    max_work_items: usize,
    max_output_roots: usize,
}

impl RothRuckensteinLimits {
    /// Construct limits on transformed prefix nodes and returned roots.
    #[must_use]
    pub const fn new(max_work_items: usize, max_output_roots: usize) -> Self {
        Self {
            max_work_items,
            max_output_roots,
        }
    }

    /// Maximum number of transformed prefix nodes created during extraction.
    #[must_use]
    pub const fn max_work_items(self) -> usize {
        self.max_work_items
    }

    /// Maximum number of distinct verified output roots.
    #[must_use]
    pub const fn max_output_roots(self) -> usize {
        self.max_output_roots
    }
}

/// Caller-owned reusable storage for Roth–Ruckenstein prefix lifting.
///
/// The frame stack, transformed-node rows, base-field factorization, and
/// candidate polynomials are all recycled through internal pools, so a warmed
/// extraction over a changed input performs no heap allocation.
pub(crate) struct RothRuckensteinScratch<F: FieldKernels> {
    field_roots: FieldRootScratch<F>,
    prefix: Vec<F::Elem>,
    frames: Vec<Frame<F>>,
    frame_pool: Vec<Frame<F>>,
    row_pool: Vec<Polynomial<F>>,
    sub_powers: Vec<F::Elem>,
    shifted: BivariatePolynomial<F>,
    constant_coeffs: Vec<F::Elem>,
    constant_y: Polynomial<F>,
    compose_acc: Polynomial<F>,
    compose_product: Polynomial<F>,
    candidate: Polynomial<F>,
    candidate_pool: Vec<Polynomial<F>>,
}

impl<F: FieldKernels> RothRuckensteinScratch<F> {
    /// Construct empty reusable lifting scratch.
    pub(crate) const fn new() -> Self {
        Self {
            field_roots: FieldRootScratch::new(),
            prefix: Vec::new(),
            frames: Vec::new(),
            frame_pool: Vec::new(),
            row_pool: Vec::new(),
            sub_powers: Vec::new(),
            shifted: BivariatePolynomial::zero(),
            constant_coeffs: Vec::new(),
            constant_y: Polynomial::zero(),
            compose_acc: Polynomial::zero(),
            compose_product: Polynomial::zero(),
            candidate: Polynomial::zero(),
            candidate_pool: Vec::new(),
        }
    }

    /// Retained frame-stack and pool capacity available to a subsequent lift.
    pub(crate) fn capacity(&self) -> usize {
        self.frames.capacity()
            + self.frame_pool.capacity()
            + self.row_pool.capacity()
            + self.candidate_pool.capacity()
            + self.field_roots.capacity()
    }

    fn recycle_frames(&mut self) {
        while let Some(mut frame) = self.frames.pop() {
            frame.roots.clear();
            self.frame_pool.push(frame);
        }
    }
}

impl<F: FieldKernels> Default for RothRuckensteinScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

struct Frame<F: FieldKernels> {
    transformed: BivariatePolynomial<F>,
    roots: Vec<F::Elem>,
    next_root: usize,
    depth: usize,
}

impl<F: FieldKernels> Frame<F> {
    const fn empty() -> Self {
        Self {
            transformed: BivariatePolynomial::zero(),
            roots: Vec::new(),
            next_root: 0,
            depth: 0,
        }
    }

    fn next_root(&mut self) -> Option<F::Elem> {
        let root = self.roots.get(self.next_root).copied()?;
        self.next_root += 1;
        Some(root)
    }
}

/// Find every polynomial `f` of degree at most `max_degree` satisfying
/// `Q(X, f(X)) == 0`.
///
/// The traversal is iterative. One fixed coefficient-prefix buffer is reused
/// by every branch, while the explicit frame stack contains at most one
/// transformed bivariate polynomial per coefficient depth.
pub fn roth_ruckenstein_roots<F: FieldKernels>(
    polynomial: &BivariatePolynomial<F>,
    max_degree: usize,
    limits: RothRuckensteinLimits,
) -> Result<Vec<Polynomial<F>>, RootError> {
    let mut scratch = RothRuckensteinScratch::new();
    let mut output = Vec::new();
    roth_ruckenstein_roots_into(polynomial, max_degree, limits, &mut scratch, &mut output)?;
    Ok(output)
}

/// Write every bounded-degree polynomial root into reusable `output` storage.
///
/// Existing `output` entries are recycled into an internal pool first. After a
/// warm-up over the same geometry, alternating a changed `polynomial` performs
/// no heap allocation.
pub(crate) fn roth_ruckenstein_roots_into<F: FieldKernels>(
    polynomial: &BivariatePolynomial<F>,
    max_degree: usize,
    limits: RothRuckensteinLimits,
    scratch: &mut RothRuckensteinScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), RootError> {
    scratch.recycle_frames();
    while let Some(mut candidate) = output.pop() {
        candidate.set_zero();
        scratch.candidate_pool.push(candidate);
    }
    if polynomial.is_zero() {
        return Err(RootError::ZeroBivariatePolynomial);
    }
    let coefficient_count = max_degree
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Roth–Ruckenstein coefficient count",
        })?;
    let y_degree = polynomial
        .y_degree()
        .ok_or(RootError::ZeroBivariatePolynomial)?;
    if y_degree == 0 {
        return Ok(());
    }
    enforce_limit("Roth–Ruckenstein work items", 1, limits.max_work_items)?;

    if scratch.prefix.len() < coefficient_count {
        scratch
            .prefix
            .try_reserve(coefficient_count - scratch.prefix.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "Roth–Ruckenstein coefficient prefix",
                elements: coefficient_count,
                element_size: core::mem::size_of::<F::Elem>(),
            })?;
        scratch.prefix.resize(coefficient_count, F::Elem::ZERO);
    }
    scratch.prefix[..coefficient_count].fill(F::Elem::ZERO);
    let initial_valuation = polynomial
        .x_valuation()
        .ok_or(RootError::ZeroBivariatePolynomial)?;
    let mut initial = scratch.frame_pool.pop().unwrap_or_else(Frame::empty);
    polynomial.divide_by_x_power_into(
        initial_valuation,
        &mut scratch.row_pool,
        &mut initial.transformed,
    )?;
    fill_frame_roots(
        &initial.transformed,
        &mut scratch.constant_coeffs,
        &mut scratch.constant_y,
        &mut scratch.field_roots,
        &mut initial.roots,
    )?;
    initial.depth = 0;
    initial.next_root = 0;
    if initial.roots.is_empty() {
        initial.roots.clear();
        scratch.frame_pool.push(initial);
        return Ok(());
    }
    if scratch.frames.capacity() < coefficient_count {
        let additional = coefficient_count - scratch.frames.capacity();
        scratch
            .frames
            .try_reserve(additional)
            .map_err(|_| ConfigError::AllocationFailed {
                context: "Roth–Ruckenstein frame stack",
                elements: coefficient_count,
                element_size: core::mem::size_of::<Frame<F>>(),
            })?;
    }
    scratch.frames.push(initial);
    let mut work_items = 1_usize;

    loop {
        let (root, depth) = {
            let Some(frame) = scratch.frames.last_mut() else {
                break;
            };
            match frame.next_root() {
                Some(root) => (root, frame.depth),
                None => {
                    let mut done = scratch.frames.pop().expect("nonempty frame stack");
                    done.roots.clear();
                    scratch.frame_pool.push(done);
                    continue;
                }
            }
        };
        scratch.prefix[depth] = root;

        if depth + 1 == coefficient_count {
            scratch
                .candidate
                .assign_coefficients(&scratch.prefix[..coefficient_count])?;
            let is_root = polynomial.has_root_with(
                &scratch.candidate,
                &mut scratch.compose_acc,
                &mut scratch.compose_product,
            )?;
            if is_root && !output.iter().any(|existing| existing == &scratch.candidate) {
                if output.len() >= y_degree {
                    return Err(RootError::FactorizationInvariant {
                        reason: "verified polynomial roots exceed the bivariate Y-degree",
                    });
                }
                enforce_limit(
                    "Roth–Ruckenstein output roots",
                    output.len() + 1,
                    limits.max_output_roots,
                )?;
                let mut buffer = scratch.candidate_pool.pop().unwrap_or_default();
                buffer.assign_from(&scratch.candidate);
                output.push(buffer);
            }
            continue;
        }

        let required_work_items =
            work_items
                .checked_add(1)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "Roth–Ruckenstein work item count",
                })?;
        enforce_limit(
            "Roth–Ruckenstein work items",
            required_work_items,
            limits.max_work_items,
        )?;

        let index = scratch.frames.len() - 1;
        scratch.frames[index].transformed.substitute_y_linear_into(
            root,
            &mut scratch.sub_powers,
            &mut scratch.row_pool,
            &mut scratch.shifted,
        )?;
        let valuation = scratch
            .shifted
            .x_valuation()
            .ok_or(RootError::FactorizationInvariant {
                reason: "a nonzero Y substitution produced zero",
            })?;
        let mut child = scratch.frame_pool.pop().unwrap_or_else(Frame::empty);
        scratch.shifted.divide_by_x_power_into(
            valuation,
            &mut scratch.row_pool,
            &mut child.transformed,
        )?;
        fill_frame_roots(
            &child.transformed,
            &mut scratch.constant_coeffs,
            &mut scratch.constant_y,
            &mut scratch.field_roots,
            &mut child.roots,
        )?;
        child.depth = depth + 1;
        child.next_root = 0;
        work_items = required_work_items;
        if child.roots.is_empty() {
            child.roots.clear();
            scratch.frame_pool.push(child);
        } else {
            scratch.frames.push(child);
        }
    }

    output.sort_by(|left, right| compare_polynomials::<F>(left, right));
    output.dedup();
    if output.len() > y_degree {
        return Err(RootError::FactorizationInvariant {
            reason: "deduplicated polynomial roots exceed the bivariate Y-degree",
        });
    }
    for candidate in output.iter() {
        if !polynomial.has_root_with(
            candidate,
            &mut scratch.compose_acc,
            &mut scratch.compose_product,
        )? {
            return Err(RootError::FactorizationInvariant {
                reason: "the final candidate list contains a nonroot",
            });
        }
    }
    Ok(())
}

/// Extract the constant-`X` polynomial and its base-field roots into `roots`.
fn fill_frame_roots<F: FieldKernels>(
    transformed: &BivariatePolynomial<F>,
    coeffs: &mut Vec<F::Elem>,
    constant_y: &mut Polynomial<F>,
    field_roots: &mut FieldRootScratch<F>,
    roots: &mut Vec<F::Elem>,
) -> Result<(), RootError> {
    constant_y_polynomial_into(transformed, coeffs, constant_y)?;
    if base_field_roots_into(constant_y, field_roots, roots)? {
        return Err(RootError::FactorizationInvariant {
            reason: "an X-normalized transformed polynomial has zero constant-X row",
        });
    }
    Ok(())
}

/// Extract the constant-`X` coefficient of each `Y` row into `out`.
fn constant_y_polynomial_into<F: FieldKernels>(
    polynomial: &BivariatePolynomial<F>,
    coeffs: &mut Vec<F::Elem>,
    out: &mut Polynomial<F>,
) -> Result<(), RootError> {
    coeffs.clear();
    let count = polynomial.y_coefficient_count();
    if coeffs.capacity() < count {
        coeffs.try_reserve(count - coeffs.capacity()).map_err(|_| {
            ConfigError::AllocationFailed {
                context: "Roth–Ruckenstein constant-X polynomial",
                elements: count,
                element_size: core::mem::size_of::<F::Elem>(),
            }
        })?;
    }
    for row in polynomial.y_coefficients() {
        coeffs.push(row.coefficient(0));
    }
    out.assign_coefficients(coeffs)?;
    if out.is_zero() {
        Err(RootError::FactorizationInvariant {
            reason: "an X-normalized polynomial yielded a zero constant-X polynomial",
        })
    } else {
        Ok(())
    }
}

pub(super) fn constant_y_polynomial<F: FieldKernels>(
    polynomial: &BivariatePolynomial<F>,
) -> Result<Polynomial<F>, RootError> {
    let mut coefficients = try_zeroed::<F::Elem>(
        "Roth–Ruckenstein constant-X polynomial",
        polynomial.y_coefficient_count(),
    )?;
    for (coefficient, row) in coefficients.iter_mut().zip(polynomial.y_coefficients()) {
        *coefficient = row.coefficient(0);
    }
    let result = Polynomial::from_coefficients(&coefficients)?;
    if result.is_zero() {
        Err(RootError::FactorizationInvariant {
            reason: "an X-normalized polynomial yielded a zero constant-X polynomial",
        })
    } else {
        Ok(result)
    }
}

pub(super) fn compare_polynomials<F: FieldKernels>(
    left: &Polynomial<F>,
    right: &Polynomial<F>,
) -> Ordering {
    let shared = left.coefficient_count().min(right.coefficient_count());
    for degree in 0..shared {
        let ordering = element_key::<F>(left.coefficient(degree))
            .cmp(&element_key::<F>(right.coefficient(degree)));
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.coefficient_count().cmp(&right.coefficient_count())
}

pub(super) fn enforce_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), RootError> {
    if required > limit {
        Err(RootError::ResourceLimitExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}
