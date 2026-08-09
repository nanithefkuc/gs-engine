use alloc::vec::Vec;
use core::cmp::Ordering;

use cafft::core::kernel::ButterflyKernels;
use fgf::field::Field;
use fgf::kernel::{Backend, backend_for};

use crate::{BivariatePolynomial, ConfigError, Polynomial, PolynomialProductScratch};

use super::roth_ruckenstein::{
    compare_polynomials, constant_y_polynomial, enforce_limit, roth_ruckenstein_roots,
};
use super::{BaseFieldRoots, RootError, RothRuckensteinLimits, base_field_roots};

/// Default measured packed-kernel crossover in weighted input coefficients.
///
/// GF16/GFNI first favors divide-and-conquer at weighted size 20,485. The
/// default keeps Roth–Ruckenstein through 20,000; scalar kernels retain
/// Roth–Ruckenstein because no divide-and-conquer win was observed in the
/// measured range. Use [`AlekhnovichLimits::with_roth_ruckenstein_crossover`]
/// to force a backend-independent threshold.
pub const DEFAULT_ROTH_RUCKENSTEIN_CROSSOVER: usize = 20_000;

/// Caller-provided bounds for Alekhnovich root extraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlekhnovichLimits {
    max_work_items: usize,
    max_intermediate_families: usize,
    max_coefficients: usize,
    max_scratch_bytes: usize,
    max_output_roots: usize,
    roth_ruckenstein_crossover: usize,
    backend_adaptive_crossover: bool,
}

impl AlekhnovichLimits {
    /// Construct extraction limits.
    #[must_use]
    pub const fn new(
        max_work_items: usize,
        max_intermediate_families: usize,
        max_coefficients: usize,
        max_scratch_bytes: usize,
        max_output_roots: usize,
    ) -> Self {
        Self {
            max_work_items,
            max_intermediate_families,
            max_coefficients,
            max_scratch_bytes,
            max_output_roots,
            roth_ruckenstein_crossover: DEFAULT_ROTH_RUCKENSTEIN_CROSSOVER,
            backend_adaptive_crossover: true,
        }
    }

    /// Override the weighted-size crossover to Roth–Ruckenstein.
    #[must_use]
    pub const fn with_roth_ruckenstein_crossover(mut self, crossover: usize) -> Self {
        self.roth_ruckenstein_crossover = crossover;
        self.backend_adaptive_crossover = false;
        self
    }

    /// Maximum number of explicit divide-and-conquer nodes.
    #[must_use]
    pub const fn max_work_items(self) -> usize {
        self.max_work_items
    }

    /// Maximum number of affine families materialized during extraction.
    #[must_use]
    pub const fn max_intermediate_families(self) -> usize {
        self.max_intermediate_families
    }

    /// Maximum cumulative coefficient capacity charged to an extraction.
    #[must_use]
    pub const fn max_coefficients(self) -> usize {
        self.max_coefficients
    }

    /// Maximum cumulative temporary storage charged to an extraction.
    #[must_use]
    pub const fn max_scratch_bytes(self) -> usize {
        self.max_scratch_bytes
    }

    /// Maximum number of distinct verified output roots.
    #[must_use]
    pub const fn max_output_roots(self) -> usize {
        self.max_output_roots
    }

    /// Weighted-size crossover at or below which Roth–Ruckenstein is used.
    #[must_use]
    pub const fn roth_ruckenstein_crossover(self) -> usize {
        self.roth_ruckenstein_crossover
    }
}

/// An affine family `prefix(X) + X^tail_degree h(X)` of power-series roots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineRootFamily<F: ButterflyKernels> {
    prefix: Polynomial<F>,
    tail_degree: usize,
}

impl<F: ButterflyKernels> AffineRootFamily<F> {
    fn new(mut prefix: Polynomial<F>, tail_degree: usize) -> Self {
        prefix.truncate(tail_degree);
        Self {
            prefix,
            tail_degree,
        }
    }

    /// Fixed low-degree prefix.
    #[must_use]
    pub fn prefix(&self) -> &Polynomial<F> {
        &self.prefix
    }

    /// First coefficient belonging to the free tail.
    #[must_use]
    pub const fn tail_degree(&self) -> usize {
        self.tail_degree
    }
}

/// Caller-owned reusable stack storage for Alekhnovich extraction.
pub struct AlekhnovichScratch<F: ButterflyKernels> {
    frames: Vec<DncFrame<F>>,
    completed: Option<Vec<AffineRootFamily<F>>>,
    products: PolynomialProductScratch<F>,
}

impl<F: ButterflyKernels> AlekhnovichScratch<F> {
    /// Construct empty reusable extraction scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            frames: Vec::new(),
            completed: None,
            products: PolynomialProductScratch::new(),
        }
    }

    /// Retained explicit-frame capacity available to a subsequent extraction.
    #[must_use]
    pub fn frame_capacity(&self) -> usize {
        self.frames.capacity()
    }

    fn clear(&mut self) {
        self.frames.clear();
        self.completed = None;
    }
}

impl<F: ButterflyKernels> Default for AlekhnovichScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Find every polynomial `f` of degree at most `max_degree` satisfying
/// `Q(X, f(X)) == 0`.
///
/// The extractor computes reduced affine prefix families with an explicit
/// divide-and-conquer work stack. Each recursive stage first solves at half
/// precision, then transforms `Q(X, prefix + X^d Y)` and solves only the
/// surviving residual precision. Every transformed row is truncated to the
/// precision of its node. Small weighted inputs use the configured
/// Roth–Ruckenstein crossover.
pub fn alekhnovich_roots<F: ButterflyKernels>(
    polynomial: &BivariatePolynomial<F>,
    max_degree: usize,
    limits: AlekhnovichLimits,
    scratch: &mut AlekhnovichScratch<F>,
) -> Result<Vec<Polynomial<F>>, RootError> {
    scratch.clear();
    let result = alekhnovich_roots_inner(polynomial, max_degree, limits, scratch);
    scratch.clear();
    result
}

fn alekhnovich_roots_inner<F: ButterflyKernels>(
    polynomial: &BivariatePolynomial<F>,
    max_degree: usize,
    limits: AlekhnovichLimits,
    scratch: &mut AlekhnovichScratch<F>,
) -> Result<Vec<Polynomial<F>>, RootError> {
    if polynomial.is_zero() {
        return Err(RootError::ZeroBivariatePolynomial);
    }
    let y_degree = polynomial
        .y_degree()
        .ok_or(RootError::ZeroBivariatePolynomial)?;
    if y_degree == 0 {
        return Ok(Vec::new());
    }

    let initial_valuation = polynomial
        .x_valuation()
        .ok_or(RootError::ZeroBivariatePolynomial)?;
    let initial = polynomial.divide_by_x_power(initial_valuation)?;
    let composition_degree = initial
        .weighted_degree(max_degree)?
        .ok_or(RootError::ZeroBivariatePolynomial)?;
    let precision = composition_degree
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Alekhnovich exact-composition precision",
        })?;
    let weighted_size = precision.checked_mul(initial.y_coefficient_count()).ok_or(
        ConfigError::GeometryOverflow {
            context: "Alekhnovich weighted input size",
        },
    )?;

    enforce_limit(
        "Alekhnovich coefficients",
        weighted_size,
        limits.max_coefficients,
    )?;
    let initial_bytes = weighted_size
        .checked_mul(F::BYTES)
        .and_then(|bytes| {
            initial
                .y_coefficient_count()
                .checked_mul(core::mem::size_of::<Polynomial<F>>())
                .and_then(|rows| bytes.checked_add(rows))
        })
        .and_then(|bytes| bytes.checked_add(core::mem::size_of::<DncFrame<F>>()))
        .ok_or(ConfigError::GeometryOverflow {
            context: "Alekhnovich initial scratch bytes",
        })?;
    enforce_limit(
        "Alekhnovich scratch bytes",
        initial_bytes,
        limits.max_scratch_bytes,
    )?;

    let crossover = if limits.backend_adaptive_crossover && backend_for::<F>() == Backend::Scalar {
        usize::MAX
    } else {
        limits.roth_ruckenstein_crossover
    };
    if weighted_size <= crossover {
        return roth_ruckenstein_roots(
            polynomial,
            max_degree,
            RothRuckensteinLimits::new(limits.max_work_items, limits.max_output_roots),
        );
    }

    let mut budget = Budget::new(weighted_size, initial_bytes);
    push_frame(
        scratch,
        DncFrame::new(initial, precision),
        &mut budget,
        limits,
    )?;

    while let Some(mut frame) = scratch.frames.pop() {
        match frame.state {
            FrameState::Enter => {
                if frame.precision == 1 {
                    let constant_y_count = frame.polynomial.y_coefficient_count();
                    let splitter_coefficients = constant_y_count
                        .checked_mul(constant_y_count)
                        .and_then(|count| count.checked_mul(4))
                        .ok_or(ConfigError::GeometryOverflow {
                            context: "Alekhnovich scalar splitter coefficients",
                        })?;
                    let family_bytes = core::mem::size_of::<AffineRootFamily<F>>()
                        .checked_mul(frame.polynomial.y_degree().unwrap_or(0))
                        .ok_or(ConfigError::GeometryOverflow {
                            context: "Alekhnovich scalar family bytes",
                        })?;
                    budget.charge_materialization::<F>(
                        splitter_coefficients,
                        family_bytes,
                        limits,
                    )?;
                    let constant_y = constant_y_polynomial(&frame.polynomial)?;
                    let roots = match base_field_roots(&constant_y)? {
                        BaseFieldRoots::All => {
                            return Err(RootError::FactorizationInvariant {
                                reason: "an X-normalized Alekhnovich node has zero constant-X row",
                            });
                        }
                        BaseFieldRoots::Finite(roots) => roots,
                    };
                    let mut families = Vec::new();
                    reserve_exact::<AffineRootFamily<F>>(
                        &mut families,
                        roots.len(),
                        "Alekhnovich scalar root families",
                    )?;
                    for root in roots {
                        insert_family(
                            &mut families,
                            AffineRootFamily::new(Polynomial::constant(root)?, 1),
                            &mut budget,
                            limits,
                        )?;
                    }
                    finish_frame(scratch, families);
                } else {
                    let coarse_precision = frame.precision.div_ceil(2);
                    let coarse_coefficients = frame
                        .polynomial
                        .y_coefficient_count()
                        .checked_mul(coarse_precision)
                        .ok_or(ConfigError::GeometryOverflow {
                            context: "Alekhnovich coarse coefficient bound",
                        })?;
                    budget.charge_materialization::<F>(
                        coarse_coefficients,
                        core::mem::size_of::<DncFrame<F>>(),
                        limits,
                    )?;
                    let coarse_polynomial = frame.polynomial.truncated_x(coarse_precision);
                    frame.state = FrameState::AwaitCoarse;
                    scratch.frames.push(frame);
                    push_frame(
                        scratch,
                        DncFrame::new(coarse_polynomial, coarse_precision),
                        &mut budget,
                        limits,
                    )?;
                }
            }
            FrameState::AwaitCoarse => {
                let coarse = take_completed(scratch)?;
                frame.state = FrameState::Refine {
                    coarse,
                    next: 0,
                    refined: Vec::new(),
                };
                scratch.frames.push(frame);
            }
            FrameState::Refine {
                coarse,
                mut next,
                mut refined,
            } => {
                let Some(family) = coarse.get(next).cloned() else {
                    finish_frame(scratch, refined);
                    continue;
                };
                next += 1;
                let transform_bound = frame
                    .polynomial
                    .y_coefficient_count()
                    .checked_mul(frame.precision)
                    .ok_or(ConfigError::GeometryOverflow {
                        context: "Alekhnovich transformed coefficient bound",
                    })?;
                budget.charge_materialization::<F>(transform_bound, 0, limits)?;
                let transformed = frame.polynomial.substitute_y_affine_truncated_fast(
                    &family.prefix,
                    family.tail_degree,
                    frame.precision,
                    &mut scratch.products,
                )?;
                let Some(valuation) = transformed.x_valuation() else {
                    insert_family(&mut refined, family, &mut budget, limits)?;
                    frame.state = FrameState::Refine {
                        coarse,
                        next,
                        refined,
                    };
                    scratch.frames.push(frame);
                    continue;
                };
                let coarse_precision = frame.precision.div_ceil(2);
                if valuation < coarse_precision {
                    return Err(RootError::FactorizationInvariant {
                        reason: "an affine family failed its established coarse precision",
                    });
                }
                if valuation >= frame.precision {
                    insert_family(&mut refined, family, &mut budget, limits)?;
                    frame.state = FrameState::Refine {
                        coarse,
                        next,
                        refined,
                    };
                    scratch.frames.push(frame);
                    continue;
                }
                let residual_precision = frame.precision - valuation;
                budget.charge_materialization::<F>(transform_bound, 0, limits)?;
                let residual = transformed.divide_by_x_power(valuation)?;
                frame.state = FrameState::AwaitTail {
                    coarse,
                    next,
                    refined,
                    family,
                };
                scratch.frames.push(frame);
                push_frame(
                    scratch,
                    DncFrame::new(residual, residual_precision),
                    &mut budget,
                    limits,
                )?;
            }
            FrameState::AwaitTail {
                coarse,
                next,
                mut refined,
                family,
            } => {
                let tails = take_completed(scratch)?;
                for tail in tails {
                    let tail_degree = family.tail_degree.checked_add(tail.tail_degree).ok_or(
                        ConfigError::GeometryOverflow {
                            context: "Alekhnovich affine tail degree",
                        },
                    )?;
                    budget.charge_materialization::<F>(tail_degree, 0, limits)?;
                    let mut prefix = family.prefix.clone();
                    prefix.add_assign(&tail.prefix.shifted(family.tail_degree)?)?;
                    insert_family(
                        &mut refined,
                        AffineRootFamily::new(prefix, tail_degree),
                        &mut budget,
                        limits,
                    )?;
                }
                frame.state = FrameState::Refine {
                    coarse,
                    next,
                    refined,
                };
                scratch.frames.push(frame);
            }
        }
    }

    let families = take_completed(scratch)?;
    materialize_candidates(polynomial, max_degree, y_degree, families, limits)
}

struct DncFrame<F: ButterflyKernels> {
    polynomial: BivariatePolynomial<F>,
    precision: usize,
    state: FrameState<F>,
}

impl<F: ButterflyKernels> DncFrame<F> {
    fn new(polynomial: BivariatePolynomial<F>, precision: usize) -> Self {
        Self {
            polynomial,
            precision,
            state: FrameState::Enter,
        }
    }
}

enum FrameState<F: ButterflyKernels> {
    Enter,
    AwaitCoarse,
    Refine {
        coarse: Vec<AffineRootFamily<F>>,
        next: usize,
        refined: Vec<AffineRootFamily<F>>,
    },
    AwaitTail {
        coarse: Vec<AffineRootFamily<F>>,
        next: usize,
        refined: Vec<AffineRootFamily<F>>,
        family: AffineRootFamily<F>,
    },
}

struct Budget {
    work_items: usize,
    families: usize,
    coefficients: usize,
    scratch_bytes: usize,
}

impl Budget {
    const fn new(coefficients: usize, scratch_bytes: usize) -> Self {
        Self {
            work_items: 0,
            families: 0,
            coefficients,
            scratch_bytes,
        }
    }

    fn charge_materialization<F: ButterflyKernels>(
        &mut self,
        coefficients: usize,
        structural_bytes: usize,
        limits: AlekhnovichLimits,
    ) -> Result<(), RootError> {
        let required_coefficients =
            self.coefficients
                .checked_add(coefficients)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "Alekhnovich cumulative coefficients",
                })?;
        enforce_limit(
            "Alekhnovich coefficients",
            required_coefficients,
            limits.max_coefficients,
        )?;
        let coefficient_bytes =
            coefficients
                .checked_mul(F::BYTES)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "Alekhnovich coefficient scratch bytes",
                })?;
        let required_bytes = self
            .scratch_bytes
            .checked_add(coefficient_bytes)
            .and_then(|bytes| bytes.checked_add(structural_bytes))
            .ok_or(ConfigError::GeometryOverflow {
                context: "Alekhnovich cumulative scratch bytes",
            })?;
        enforce_limit(
            "Alekhnovich scratch bytes",
            required_bytes,
            limits.max_scratch_bytes,
        )?;
        self.coefficients = required_coefficients;
        self.scratch_bytes = required_bytes;
        Ok(())
    }
}

fn push_frame<F: ButterflyKernels>(
    scratch: &mut AlekhnovichScratch<F>,
    frame: DncFrame<F>,
    budget: &mut Budget,
    limits: AlekhnovichLimits,
) -> Result<(), RootError> {
    let required = budget
        .work_items
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Alekhnovich work item count",
        })?;
    enforce_limit("Alekhnovich work items", required, limits.max_work_items)?;
    scratch
        .frames
        .try_reserve(1)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "Alekhnovich explicit work stack",
            elements: scratch.frames.len() + 1,
            element_size: core::mem::size_of::<DncFrame<F>>(),
        })?;
    budget.work_items = required;
    scratch.frames.push(frame);
    Ok(())
}

fn finish_frame<F: ButterflyKernels>(
    scratch: &mut AlekhnovichScratch<F>,
    mut families: Vec<AffineRootFamily<F>>,
) {
    families.sort_by(compare_families::<F>);
    scratch.completed = Some(families);
}

fn take_completed<F: ButterflyKernels>(
    scratch: &mut AlekhnovichScratch<F>,
) -> Result<Vec<AffineRootFamily<F>>, RootError> {
    scratch
        .completed
        .take()
        .ok_or(RootError::FactorizationInvariant {
            reason: "an Alekhnovich frame resumed without a child result",
        })
}

fn insert_family<F: ButterflyKernels>(
    families: &mut Vec<AffineRootFamily<F>>,
    family: AffineRootFamily<F>,
    budget: &mut Budget,
    limits: AlekhnovichLimits,
) -> Result<(), RootError> {
    if families
        .iter()
        .any(|existing| family_contains(existing, &family))
    {
        return Ok(());
    }
    families.retain(|existing| !family_contains(&family, existing));
    let required = budget
        .families
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Alekhnovich affine family count",
        })?;
    enforce_limit(
        "Alekhnovich intermediate families",
        required,
        limits.max_intermediate_families,
    )?;
    families
        .try_reserve(1)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "Alekhnovich affine families",
            elements: families.len() + 1,
            element_size: core::mem::size_of::<AffineRootFamily<F>>(),
        })?;
    budget.families = required;
    families.push(family);
    Ok(())
}

fn family_contains<F: ButterflyKernels>(
    outer: &AffineRootFamily<F>,
    inner: &AffineRootFamily<F>,
) -> bool {
    outer.tail_degree <= inner.tail_degree
        && (0..outer.tail_degree)
            .all(|degree| outer.prefix.coefficient(degree) == inner.prefix.coefficient(degree))
}

fn compare_families<F: ButterflyKernels>(
    left: &AffineRootFamily<F>,
    right: &AffineRootFamily<F>,
) -> Ordering {
    left.tail_degree
        .cmp(&right.tail_degree)
        .then_with(|| compare_polynomials::<F>(&left.prefix, &right.prefix))
}

fn materialize_candidates<F: ButterflyKernels>(
    polynomial: &BivariatePolynomial<F>,
    max_degree: usize,
    y_degree: usize,
    families: Vec<AffineRootFamily<F>>,
    limits: AlekhnovichLimits,
) -> Result<Vec<Polynomial<F>>, RootError> {
    let coefficient_count = max_degree
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Alekhnovich output coefficient count",
        })?;
    let field_order = usize::try_from(F::ORDER).map_err(|_| ConfigError::GeometryOverflow {
        context: "Alekhnovich field order",
    })?;
    let mut candidates = Vec::new();

    for family in families {
        if family
            .prefix
            .degree()
            .is_some_and(|degree| degree > max_degree)
        {
            continue;
        }
        let free_count = coefficient_count.saturating_sub(family.tail_degree);
        let completion_count = checked_power(field_order, free_count)?;
        let required_outputs = candidates.len().checked_add(completion_count).ok_or(
            ConfigError::GeometryOverflow {
                context: "Alekhnovich output root count",
            },
        )?;
        enforce_limit(
            "Alekhnovich output roots",
            required_outputs,
            limits.max_output_roots,
        )?;
        enforce_limit(
            "Alekhnovich Y-degree root bound",
            required_outputs,
            y_degree,
        )?;

        for ordinal in 0..completion_count {
            let mut candidate = family.prefix.clone();
            let mut digits = ordinal;
            for degree in family.tail_degree..coefficient_count {
                let key = digits % field_order;
                digits /= field_order;
                candidate.set_coefficient(degree, element_from_key::<F>(key))?;
            }
            if !polynomial.has_root(&candidate)? {
                return Err(RootError::FactorizationInvariant {
                    reason: "a final Alekhnovich affine family contained a nonroot",
                });
            }
            if !candidates.iter().any(|existing| existing == &candidate) {
                candidates
                    .try_reserve(1)
                    .map_err(|_| ConfigError::AllocationFailed {
                        context: "Alekhnovich output roots",
                        elements: candidates.len() + 1,
                        element_size: core::mem::size_of::<Polynomial<F>>(),
                    })?;
                candidates.push(candidate);
            }
        }
    }

    candidates.sort_by(compare_polynomials::<F>);
    candidates.dedup();
    if candidates.len() > y_degree {
        return Err(RootError::FactorizationInvariant {
            reason: "verified Alekhnovich roots exceed the bivariate Y-degree",
        });
    }
    for candidate in &candidates {
        if !polynomial.has_root(candidate)? {
            return Err(RootError::FactorizationInvariant {
                reason: "the final Alekhnovich candidate list contains a nonroot",
            });
        }
    }
    Ok(candidates)
}

fn checked_power(base: usize, exponent: usize) -> Result<usize, RootError> {
    let mut value = 1_usize;
    for _ in 0..exponent {
        value = value
            .checked_mul(base)
            .ok_or(ConfigError::GeometryOverflow {
                context: "Alekhnovich affine completion count",
            })?;
    }
    Ok(value)
}

fn element_from_key<F: Field>(key: usize) -> F::Elem {
    let bytes = (key as u128).to_le_bytes();
    F::read(&bytes[..F::BYTES])
}

fn reserve_exact<T>(
    values: &mut Vec<T>,
    additional: usize,
    context: &'static str,
) -> Result<(), ConfigError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| ConfigError::AllocationFailed {
            context,
            elements: additional,
            element_size: core::mem::size_of::<T>(),
        })
}
