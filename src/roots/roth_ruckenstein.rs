use alloc::vec::Vec;
use core::cmp::Ordering;

use fgf::kernel::FieldKernels;

use crate::geometry::try_zeroed;
use crate::{BivariatePolynomial, ConfigError, Polynomial};

use super::field_roots::element_key;
use super::{BaseFieldRoots, RootError, base_field_roots};

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
        return Ok(Vec::new());
    }
    enforce_limit("Roth–Ruckenstein work items", 1, limits.max_work_items)?;

    let initial_valuation = polynomial
        .x_valuation()
        .ok_or(RootError::ZeroBivariatePolynomial)?;
    let initial = polynomial.divide_by_x_power(initial_valuation)?;
    let initial_frame = make_frame(initial, 0)?;
    if initial_frame.roots.is_empty() {
        return Ok(Vec::new());
    }

    let mut prefix =
        try_zeroed::<F::Elem>("Roth–Ruckenstein coefficient prefix", coefficient_count)?;
    let mut frames = Vec::new();
    frames
        .try_reserve_exact(coefficient_count)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "Roth–Ruckenstein frame stack",
            elements: coefficient_count,
            element_size: core::mem::size_of::<Frame<F>>(),
        })?;
    frames.push(initial_frame);
    let output_capacity = y_degree.min(limits.max_output_roots);
    let mut candidates = Vec::new();
    candidates
        .try_reserve_exact(output_capacity)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "Roth–Ruckenstein candidates",
            elements: output_capacity,
            element_size: core::mem::size_of::<Polynomial<F>>(),
        })?;
    let mut work_items = 1_usize;

    while let Some(frame) = frames.last_mut() {
        let Some(root) = frame.next_root() else {
            frames.pop();
            continue;
        };
        let depth = frame.depth;
        prefix[depth] = root;

        if depth + 1 == coefficient_count {
            let candidate = Polynomial::<F>::from_coefficients(&prefix)?;
            if polynomial.has_root(&candidate)?
                && !candidates.iter().any(|existing| existing == &candidate)
            {
                if candidates.len() >= y_degree {
                    return Err(RootError::FactorizationInvariant {
                        reason: "verified polynomial roots exceed the bivariate Y-degree",
                    });
                }
                enforce_limit(
                    "Roth–Ruckenstein output roots",
                    candidates.len() + 1,
                    limits.max_output_roots,
                )?;
                candidates.push(candidate);
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
        let child = {
            let transformed = &frames
                .last()
                .ok_or(RootError::FactorizationInvariant {
                    reason: "the active Roth–Ruckenstein frame disappeared",
                })?
                .transformed;
            let shifted = transformed.substitute_y_linear(root)?;
            let valuation = shifted
                .x_valuation()
                .ok_or(RootError::FactorizationInvariant {
                    reason: "a nonzero Y substitution produced zero",
                })?;
            shifted.divide_by_x_power(valuation)?
        };
        let child_frame = make_frame(child, depth + 1)?;
        work_items = required_work_items;
        if !child_frame.roots.is_empty() {
            frames.push(child_frame);
        }
    }

    candidates.sort_by(|left, right| compare_polynomials::<F>(left, right));
    candidates.dedup();
    if candidates.len() > y_degree {
        return Err(RootError::FactorizationInvariant {
            reason: "deduplicated polynomial roots exceed the bivariate Y-degree",
        });
    }
    for candidate in &candidates {
        if !polynomial.has_root(candidate)? {
            return Err(RootError::FactorizationInvariant {
                reason: "the final candidate list contains a nonroot",
            });
        }
    }
    Ok(candidates)
}

struct Frame<F: FieldKernels> {
    transformed: BivariatePolynomial<F>,
    roots: Vec<F::Elem>,
    next_root: usize,
    depth: usize,
}

impl<F: FieldKernels> Frame<F> {
    fn next_root(&mut self) -> Option<F::Elem> {
        let root = self.roots.get(self.next_root).copied()?;
        self.next_root += 1;
        Some(root)
    }
}

fn make_frame<F: FieldKernels>(
    transformed: BivariatePolynomial<F>,
    depth: usize,
) -> Result<Frame<F>, RootError> {
    let constant_y = constant_y_polynomial(&transformed)?;
    let roots = match base_field_roots(&constant_y)? {
        BaseFieldRoots::All => {
            return Err(RootError::FactorizationInvariant {
                reason: "an X-normalized transformed polynomial has zero constant-X row",
            });
        }
        BaseFieldRoots::Finite(roots) => roots,
    };
    Ok(Frame {
        transformed,
        roots,
        next_root: 0,
        depth,
    })
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
