//! Guruswami–Sudan interpolation backends.

mod koetter;
mod module;
mod plan;

pub use koetter::{
    KoetterScratch, interpolate_koetter, interpolate_koetter_into, interpolate_koetter_with_scratch,
};
pub use module::{
    MODULE_INTERPOLATION_CROSSOVER, ModuleScratch, interpolate_module, interpolate_module_into,
};
pub use plan::InterpolationPlan;

#[cfg(feature = "internals")]
use alloc::vec::Vec;
use core::fmt;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

#[cfg(feature = "internals")]
use crate::Polynomial;
#[cfg(feature = "internals")]
use crate::geometry::try_zeroed;
use crate::{BivariatePolynomial, ConfigError, GsParameters};

#[cfg(feature = "internals")]
/// One monomial in the weighted interpolation basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterpolationMonomial {
    /// Exponent of `X`.
    pub x_degree: usize,
    /// Exponent of `Y`.
    pub y_degree: usize,
    /// `(1, max_degree)` weighted degree.
    pub weighted_degree: usize,
}

#[cfg(feature = "internals")]
/// One Hasse constraint in the fixed lower-set traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterpolationConstraint {
    /// Index of the interpolation point.
    pub point_index: usize,
    /// Hasse derivative order in `X`.
    pub x_order: usize,
    /// Hasse derivative order in `Y`.
    pub y_order: usize,
}

#[cfg(feature = "internals")]
/// Hard caps for the explicit reference interpolation matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReferenceInterpolationLimits {
    max_matrix_elements: usize,
    max_matrix_bytes: usize,
}

#[cfg(feature = "internals")]
impl ReferenceInterpolationLimits {
    /// Construct explicit element and byte caps.
    #[must_use]
    pub const fn new(max_matrix_elements: usize, max_matrix_bytes: usize) -> Self {
        Self {
            max_matrix_elements,
            max_matrix_bytes,
        }
    }

    /// Maximum matrix entries.
    #[must_use]
    pub const fn max_matrix_elements(self) -> usize {
        self.max_matrix_elements
    }

    /// Maximum matrix storage in bytes.
    #[must_use]
    pub const fn max_matrix_bytes(self) -> usize {
        self.max_matrix_bytes
    }
}

/// Failure while constructing or solving the reference interpolation system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationError {
    /// Checked geometry or allocation failed.
    Config(ConfigError),
    /// The shared weak-Popov reducer rejected the row geometry or termination
    /// measure.
    Reduction(gfm::ReduceError),
    /// Point or received-value length differs from the planned code length.
    LengthMismatch {
        /// Planned number of points.
        expected: usize,
        /// Supplied evaluation points.
        points: usize,
        /// Supplied received values.
        values: usize,
    },
    /// Two evaluation points are equal.
    DuplicatePoint {
        /// Index of the first occurrence.
        first: usize,
        /// Index of the duplicate occurrence.
        second: usize,
    },
    #[cfg(feature = "internals")]
    /// The explicit reference matrix exceeds a hard caller limit.
    ReferenceLimitExceeded {
        /// Bounded resource name.
        resource: &'static str,
        /// Amount required.
        required: usize,
        /// Configured maximum.
        limit: usize,
    },
    #[cfg(feature = "internals")]
    /// Elimination found no nonzero homogeneous solution.
    NoNonzeroSolution,
    /// A reconstructed polynomial violated an interpolation invariant.
    InvalidResult {
        /// Failed invariant.
        reason: &'static str,
    },
    /// A reconstructed polynomial violates one Hasse constraint.
    ConstraintViolation {
        /// Interpolation point index.
        point_index: usize,
        /// Hasse order in `X`.
        x_order: usize,
        /// Hasse order in `Y`.
        y_order: usize,
    },
}

impl fmt::Display for InterpolationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Config(error) => error.fmt(formatter),
            Self::LengthMismatch {
                expected,
                points,
                values,
            } => write!(
                formatter,
                "interpolation expected {expected} points, got {points} points and {values} values"
            ),
            Self::DuplicatePoint { first, second } => {
                write!(
                    formatter,
                    "evaluation points {first} and {second} are equal"
                )
            }
            #[cfg(feature = "internals")]
            Self::ReferenceLimitExceeded {
                resource,
                required,
                limit,
            } => write!(
                formatter,
                "reference {resource} requires {required}, exceeding limit {limit}"
            ),
            #[cfg(feature = "internals")]
            Self::NoNonzeroSolution => {
                formatter.write_str("reference interpolation has no nonzero nullspace vector")
            }
            Self::InvalidResult { reason } => {
                write!(
                    formatter,
                    "interpolation produced an invalid result: {reason}"
                )
            }
            Self::Reduction(error) => error.fmt(formatter),
            Self::ConstraintViolation {
                point_index,
                x_order,
                y_order,
            } => write!(
                formatter,
                "interpolation violates constraint ({point_index}, {x_order}, {y_order})"
            ),
        }
    }
}

impl From<ConfigError> for InterpolationError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<gfm::ReduceError> for InterpolationError {
    fn from(error: gfm::ReduceError) -> Self {
        Self::Reduction(error)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InterpolationError {}

#[cfg(feature = "internals")]
/// Enumerate all allowed monomials in deterministic `Y`-major order.
pub fn reference_monomials(
    parameters: GsParameters,
) -> Result<Vec<InterpolationMonomial>, InterpolationError> {
    let expected = parameters.resources().monomials();
    let mut monomials = Vec::new();
    monomials
        .try_reserve_exact(expected)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "reference interpolation monomials",
            elements: expected,
            element_size: core::mem::size_of::<InterpolationMonomial>(),
        })?;
    for y_degree in 0..=parameters.y_degree() {
        let y_weight =
            y_degree
                .checked_mul(parameters.max_degree())
                .ok_or(ConfigError::GeometryOverflow {
                    context: "reference monomial Y weight",
                })?;
        if y_weight > parameters.weighted_degree() {
            continue;
        }
        let max_x_degree = parameters.weighted_degree() - y_weight;
        for x_degree in 0..=max_x_degree {
            monomials.push(InterpolationMonomial {
                x_degree,
                y_degree,
                weighted_degree: x_degree + y_weight,
            });
        }
    }
    if monomials.len() != expected {
        return Err(InterpolationError::InvalidResult {
            reason: "monomial enumeration disagrees with the parameter count",
        });
    }
    Ok(monomials)
}

/// Enumerate constraints point-major, then by increasing total Hasse order.
#[cfg(feature = "internals")]
/// Within one total order, `X` order decreases while `Y` order increases.
pub fn reference_constraints(
    parameters: GsParameters,
) -> Result<Vec<InterpolationConstraint>, InterpolationError> {
    let expected = parameters.resources().constraints();
    let mut constraints = Vec::new();
    constraints
        .try_reserve_exact(expected)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "reference interpolation constraints",
            elements: expected,
            element_size: core::mem::size_of::<InterpolationConstraint>(),
        })?;
    for point_index in 0..parameters.code_length() {
        for total_order in 0..parameters.multiplicity() {
            for y_order in 0..=total_order {
                constraints.push(InterpolationConstraint {
                    point_index,
                    x_order: total_order - y_order,
                    y_order,
                });
            }
        }
    }
    if constraints.len() != expected {
        return Err(InterpolationError::InvalidResult {
            reason: "constraint enumeration disagrees with the parameter count",
        });
    }
    Ok(constraints)
}

#[cfg(feature = "internals")]
/// Construct a nonzero interpolation polynomial with an explicit Hasse matrix.
///
/// This backend is intentionally limited to small geometries and is intended
/// as a correctness oracle for production interpolation algorithms.
pub fn interpolate_reference<F: FieldKernels>(
    parameters: GsParameters,
    points: &[F::Elem],
    values: &[F::Elem],
    limits: ReferenceInterpolationLimits,
) -> Result<BivariatePolynomial<F>, InterpolationError> {
    validate_inputs(parameters, points, values)?;
    let matrix_elements = parameters.resources().reference_matrix_elements();
    enforce_limit(
        "matrix elements",
        matrix_elements,
        limits.max_matrix_elements,
    )?;
    let matrix_bytes = matrix_elements
        .checked_mul(core::mem::size_of::<F::Elem>())
        .ok_or(ConfigError::GeometryOverflow {
            context: "reference interpolation matrix bytes",
        })?;
    enforce_limit("matrix bytes", matrix_bytes, limits.max_matrix_bytes)?;

    let monomials = reference_monomials(parameters)?;
    let constraints = reference_constraints(parameters)?;
    let mut matrix = materialize_matrix::<F>(&monomials, &constraints, points, values)?;
    let solution = nonzero_nullspace_vector(&mut matrix, constraints.len(), monomials.len())?;
    let polynomial = reconstruct::<F>(parameters, &monomials, &solution)?;
    validate_result(parameters, points, values, &polynomial)?;
    Ok(polynomial)
}

fn validate_inputs<E: Elem>(
    parameters: GsParameters,
    points: &[E],
    values: &[E],
) -> Result<(), InterpolationError> {
    if points.len() != parameters.code_length() || values.len() != parameters.code_length() {
        return Err(InterpolationError::LengthMismatch {
            expected: parameters.code_length(),
            points: points.len(),
            values: values.len(),
        });
    }
    for second in 0..points.len() {
        if let Some(first) = points[..second]
            .iter()
            .position(|point| *point == points[second])
        {
            return Err(InterpolationError::DuplicatePoint { first, second });
        }
    }
    Ok(())
}

#[cfg(feature = "internals")]
fn enforce_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), InterpolationError> {
    if required > limit {
        Err(InterpolationError::ReferenceLimitExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

#[cfg(feature = "internals")]
fn materialize_matrix<F: FieldKernels>(
    monomials: &[InterpolationMonomial],
    constraints: &[InterpolationConstraint],
    points: &[F::Elem],
    values: &[F::Elem],
) -> Result<Vec<F::Elem>, InterpolationError> {
    let element_count =
        constraints
            .len()
            .checked_mul(monomials.len())
            .ok_or(ConfigError::GeometryOverflow {
                context: "reference interpolation matrix elements",
            })?;
    let mut matrix = try_zeroed::<F::Elem>("reference interpolation matrix", element_count)?;
    let max_x_degree = monomials
        .iter()
        .map(|monomial| monomial.x_degree)
        .max()
        .unwrap_or(0);
    let max_y_degree = monomials
        .iter()
        .map(|monomial| monomial.y_degree)
        .max()
        .unwrap_or(0);
    let x_power_count = max_x_degree
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "reference X power count",
        })?;
    let y_power_count = max_y_degree
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "reference Y power count",
        })?;
    let mut x_powers = try_zeroed::<F::Elem>("reference X powers", x_power_count)?;
    let mut y_powers = try_zeroed::<F::Elem>("reference Y powers", y_power_count)?;
    let mut cached_point = None;

    for (row, constraint) in constraints.iter().enumerate() {
        if cached_point != Some(constraint.point_index) {
            fill_powers(&mut x_powers, points[constraint.point_index]);
            fill_powers(&mut y_powers, values[constraint.point_index]);
            cached_point = Some(constraint.point_index);
        }
        for (column, monomial) in monomials.iter().enumerate() {
            if binomial_odd(monomial.x_degree, constraint.x_order)
                && binomial_odd(monomial.y_degree, constraint.y_order)
            {
                matrix[row * monomials.len() + column] = x_powers
                    [monomial.x_degree - constraint.x_order]
                    .mul(y_powers[monomial.y_degree - constraint.y_order]);
            }
        }
    }
    Ok(matrix)
}

fn fill_powers<E: Elem>(powers: &mut [E], value: E) {
    if powers.is_empty() {
        return;
    }
    powers[0] = E::ONE;
    for exponent in 1..powers.len() {
        powers[exponent] = powers[exponent - 1].mul(value);
    }
}

#[cfg(feature = "internals")]
fn nonzero_nullspace_vector<E: Elem>(
    matrix: &mut [E],
    rows: usize,
    columns: usize,
) -> Result<Vec<E>, InterpolationError> {
    debug_assert_eq!(matrix.len(), rows * columns);
    let mut pivot_columns = Vec::new();
    pivot_columns
        .try_reserve_exact(rows.min(columns))
        .map_err(|_| ConfigError::AllocationFailed {
            context: "reference interpolation pivots",
            elements: rows.min(columns),
            element_size: core::mem::size_of::<usize>(),
        })?;
    let mut rank = 0;
    for column in 0..columns {
        if rank == rows {
            break;
        }
        let Some(pivot) = (rank..rows).find(|&row| !matrix[row * columns + column].is_zero())
        else {
            continue;
        };
        if pivot != rank {
            for entry in 0..columns {
                matrix.swap(rank * columns + entry, pivot * columns + entry);
            }
        }
        let inverse = matrix[rank * columns + column].inv();
        for entry in column..columns {
            matrix[rank * columns + entry] = matrix[rank * columns + entry].mul(inverse);
        }
        for row in 0..rows {
            if row == rank {
                continue;
            }
            let scale = matrix[row * columns + column];
            if scale.is_zero() {
                continue;
            }
            for entry in column..columns {
                matrix[row * columns + entry] =
                    matrix[row * columns + entry].add(scale.mul(matrix[rank * columns + entry]));
            }
        }
        pivot_columns.push(column);
        rank += 1;
    }

    let Some(free_column) = (0..columns).find(|column| !pivot_columns.contains(column)) else {
        return Err(InterpolationError::NoNonzeroSolution);
    };
    let mut solution = try_zeroed::<E>("reference nullspace vector", columns)?;
    solution[free_column] = E::ONE;
    for (row, &pivot_column) in pivot_columns.iter().enumerate() {
        solution[pivot_column] = matrix[row * columns + free_column];
    }
    Ok(solution)
}

#[cfg(feature = "internals")]
fn reconstruct<F: FieldKernels>(
    parameters: GsParameters,
    monomials: &[InterpolationMonomial],
    solution: &[F::Elem],
) -> Result<BivariatePolynomial<F>, InterpolationError> {
    let row_count = parameters
        .y_degree()
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "reference interpolation Y rows",
        })?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(row_count)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "reference interpolation Y rows",
            elements: row_count,
            element_size: core::mem::size_of::<Polynomial<F>>(),
        })?;
    rows.resize_with(row_count, Polynomial::zero);
    for (monomial, &coefficient) in monomials.iter().zip(solution) {
        if !coefficient.is_zero() {
            rows[monomial.y_degree].set_coefficient(monomial.x_degree, coefficient)?;
        }
    }
    Ok(BivariatePolynomial::from_y_coefficients(rows))
}

fn validate_result<F: FieldKernels>(
    parameters: GsParameters,
    points: &[F::Elem],
    values: &[F::Elem],
    polynomial: &BivariatePolynomial<F>,
) -> Result<(), InterpolationError> {
    if polynomial.is_zero() {
        return Err(InterpolationError::InvalidResult {
            reason: "the nullspace vector reconstructed to zero",
        });
    }
    if polynomial
        .y_degree()
        .is_some_and(|degree| degree > parameters.y_degree())
    {
        return Err(InterpolationError::InvalidResult {
            reason: "Y-degree exceeds the parameter bound",
        });
    }
    if polynomial.weighted_degree(parameters.max_degree())? > Some(parameters.weighted_degree()) {
        return Err(InterpolationError::InvalidResult {
            reason: "weighted degree exceeds the parameter bound",
        });
    }
    for point_index in 0..parameters.code_length() {
        for total_order in 0..parameters.multiplicity() {
            for y_order in 0..=total_order {
                let x_order = total_order - y_order;
                if !polynomial
                    .hasse_discrepancy(points[point_index], values[point_index], x_order, y_order)
                    .is_zero()
                {
                    return Err(InterpolationError::ConstraintViolation {
                        point_index,
                        x_order,
                        y_order,
                    });
                }
            }
        }
    }
    Ok(())
}

const fn binomial_odd(upper: usize, lower: usize) -> bool {
    lower <= upper && (upper & lower) == lower
}
