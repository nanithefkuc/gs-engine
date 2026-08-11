//! Guruswami–Sudan interpolation backends.

#[cfg(feature = "internals")]
mod fast_knh;
mod koetter;
mod module;
mod plan;

#[cfg(feature = "internals")]
pub use fast_knh::{FastKnhScratch, interpolate_fast_knh, interpolate_fast_knh_into};
pub use koetter::{
    KoetterScratch, interpolate_koetter, interpolate_koetter_into, interpolate_koetter_with_scratch,
};
pub use module::{
    MODULE_INTERPOLATION_CROSSOVER, ModuleScratch, interpolate_module, interpolate_module_into,
};
pub(crate) use module::{ReencodeScratch, interpolate_reencoded_into};
pub use plan::InterpolationPlan;
pub(crate) use plan::ReencodePlan;

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
/// One interpolation point carrying its own multiplicity.
///
/// Uniform Guruswami–Sudan assigns the same multiplicity to every point.
/// Fast Kötter–Nielsen–Høholdt interpolation consumes a lower set where each
/// point may carry a different multiplicity; this type is the entry seam for
/// that representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiplicityPoint<E> {
    /// Evaluation point `X = x`.
    pub x: E,
    /// Received value `Y = y` at `x`.
    pub y: E,
    /// Hasse multiplicity `s` at `(x, y)`; the lower set is `a + b < s`.
    pub multiplicity: usize,
}

#[cfg(feature = "internals")]
/// A nonuniform-multiplicity Guruswami–Sudan interpolation problem.
///
/// Each point may carry a different multiplicity, matching the lower set a
/// fast KNH backend consumes directly. The `(1, y_weight)` weighted-degree
/// bound and `Y`-degree bound are global, exactly as in the uniform problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterpolationProblem<'a, E: Elem> {
    /// Interpolation points with per-point multiplicities, in canonical order.
    pub points: &'a [MultiplicityPoint<E>],
    /// `(1, max_degree)` weight applied to `Y` exponents.
    pub y_weight: usize,
    /// Maximum `Y` degree `ell`.
    pub y_degree: usize,
    /// Weighted-degree bound `D`.
    pub weighted_degree: usize,
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
    /// A `butterfly-fft` transform buffer had inconsistent geometry.
    Transform(butterfly_fft::error::TransformLengthError),
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
            Self::Transform(error) => error.fmt(formatter),
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

impl From<butterfly_fft::error::TransformLengthError> for InterpolationError {
    fn from(error: butterfly_fft::error::TransformLengthError) -> Self {
        Self::Transform(error)
    }
}
impl From<crate::poly::PolynomialError> for InterpolationError {
    fn from(error: crate::poly::PolynomialError) -> Self {
        use crate::poly::PolynomialError;
        match error {
            PolynomialError::DivisionByZero => Self::InvalidResult {
                reason: "polynomial division by zero in fast KNH",
            },
            PolynomialError::NonExactDivision => Self::InvalidResult {
                reason: "non-exact polynomial division in fast KNH",
            },
            PolynomialError::Config(config) => Self::Config(config),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InterpolationError {}

#[cfg(feature = "internals")]
/// Enumerate all allowed monomials in deterministic `Y`-major order.
pub fn reference_monomials(
    parameters: GsParameters,
) -> Result<Vec<InterpolationMonomial>, InterpolationError> {
    let monomials = enumerate_monomials(
        parameters.y_degree(),
        parameters.max_degree(),
        parameters.weighted_degree(),
    )?;
    if monomials.len() != parameters.resources().monomials() {
        return Err(InterpolationError::InvalidResult {
            reason: "monomial enumeration disagrees with the parameter count",
        });
    }
    Ok(monomials)
}

#[cfg(feature = "internals")]
/// Enumerate every monomial `X^a Y^b` with `b <= y_degree` and
/// `a + b * y_weight <= weighted_degree`, in `Y`-major order.
fn enumerate_monomials(
    y_degree: usize,
    y_weight: usize,
    weighted_degree: usize,
) -> Result<Vec<InterpolationMonomial>, InterpolationError> {
    let expected = crate::params::interpolation_monomials(weighted_degree, y_degree, y_weight)?;
    let mut monomials = Vec::new();
    monomials
        .try_reserve_exact(expected)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "reference interpolation monomials",
            elements: expected,
            element_size: core::mem::size_of::<InterpolationMonomial>(),
        })?;
    for b in 0..=y_degree {
        let weight = b
            .checked_mul(y_weight)
            .ok_or(ConfigError::GeometryOverflow {
                context: "reference monomial Y weight",
            })?;
        if weight > weighted_degree {
            continue;
        }
        let max_a = weighted_degree - weight;
        for a in 0..=max_a {
            monomials.push(InterpolationMonomial {
                x_degree: a,
                y_degree: b,
                weighted_degree: a + weight,
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
    let multiplicities = alloc::vec![parameters.multiplicity(); parameters.code_length()];
    let constraints = enumerate_constraints(&multiplicities)?;
    if constraints.len() != parameters.resources().constraints() {
        return Err(InterpolationError::InvalidResult {
            reason: "constraint enumeration disagrees with the parameter count",
        });
    }
    Ok(constraints)
}

#[cfg(feature = "internals")]
/// Enumerate the nonuniform lower set: for each point `i` with multiplicity
/// `s_i`, emit `(i, a, b)` for `a + b < s_i`, point-major, then by increasing
/// total order, with `X` order decreasing and `Y` order increasing within one
/// total order — the same ordering the uniform lower set uses.
fn enumerate_constraints(
    multiplicities: &[usize],
) -> Result<Vec<InterpolationConstraint>, InterpolationError> {
    let total: usize = multiplicities
        .iter()
        .map(|&s| s.checked_mul(s + 1).map(|p| p / 2))
        .try_fold(0usize, |acc, next| {
            next.and_then(|add| acc.checked_add(add))
        })
        .ok_or(ConfigError::GeometryOverflow {
            context: "reference interpolation constraint count",
        })?;
    let mut constraints = Vec::new();
    constraints
        .try_reserve_exact(total)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "reference interpolation constraints",
            elements: total,
            element_size: core::mem::size_of::<InterpolationConstraint>(),
        })?;
    for (point_index, &multiplicity) in multiplicities.iter().enumerate() {
        for total_order in 0..multiplicity {
            for y_order in 0..=total_order {
                constraints.push(InterpolationConstraint {
                    point_index,
                    x_order: total_order - y_order,
                    y_order,
                });
            }
        }
    }
    if constraints.len() != total {
        return Err(InterpolationError::InvalidResult {
            reason: "constraint enumeration disagrees with the expected count",
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
    materialize_matrix_with::<F>(monomials, constraints, |index| {
        (points[index], values[index])
    })
}

/// Fill the Hasse matrix rows: `matrix[row, col]` is the `(x_order, y_order)`
/// Hasse derivative of monomial `col` evaluated at point `point_index`. The
/// `coordinate` closure supplies `(x, y)` per point index so the uniform and
/// nonuniform kernels share the fill loop.
#[cfg(feature = "internals")]
fn materialize_matrix_with<F: FieldKernels>(
    monomials: &[InterpolationMonomial],
    constraints: &[InterpolationConstraint],
    coordinate: impl Fn(usize) -> (F::Elem, F::Elem),
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
            let (x, y) = coordinate(constraint.point_index);
            fill_powers(&mut x_powers, x);
            fill_powers(&mut y_powers, y);
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
    reconstruct_y_rows::<F>(parameters.y_degree(), monomials, solution)
}

#[cfg(feature = "internals")]
fn reconstruct_y_rows<F: FieldKernels>(
    y_degree: usize,
    monomials: &[InterpolationMonomial],
    solution: &[F::Elem],
) -> Result<BivariatePolynomial<F>, InterpolationError> {
    let row_count = y_degree
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

#[cfg(feature = "internals")]
/// Construct a nonzero interpolation polynomial for a nonuniform-multiplicity
/// problem with an explicit Hasse matrix.
///
/// This is the reference oracle for fast KNH, which consumes a lower set where
/// each point carries its own multiplicity. The `(1, y_weight)` weighted-degree
/// and `Y`-degree bounds are exactly those of [`InterpolationProblem`]. The
/// caller-supplied limits cap the explicit matrix size, keeping this backend
/// on small geometries like the uniform reference.
///
/// When every point shares the same multiplicity `s`, `y_weight = max_degree`,
/// and the bounds match a [`GsParameters`] tuple, this returns a polynomial
/// satisfying the same Hasse constraints as [`interpolate_reference`].
pub fn interpolate_reference_nonuniform<F: FieldKernels>(
    problem: InterpolationProblem<F::Elem>,
    limits: ReferenceInterpolationLimits,
) -> Result<BivariatePolynomial<F>, InterpolationError> {
    validate_problem(problem)?;
    let monomials =
        enumerate_monomials(problem.y_degree, problem.y_weight, problem.weighted_degree)?;
    let multiplicities = problem
        .points
        .iter()
        .map(|point| point.multiplicity)
        .collect::<alloc::vec::Vec<_>>();
    let constraints = enumerate_constraints(&multiplicities)?;
    let matrix_elements =
        constraints
            .len()
            .checked_mul(monomials.len())
            .ok_or(ConfigError::GeometryOverflow {
                context: "reference interpolation matrix elements",
            })?;
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

    let mut matrix = materialize_matrix_with::<F>(&monomials, &constraints, |index| {
        (problem.points[index].x, problem.points[index].y)
    })?;
    let solution = nonzero_nullspace_vector(&mut matrix, constraints.len(), monomials.len())?;
    let polynomial = reconstruct_y_rows::<F>(problem.y_degree, &monomials, &solution)?;
    validate_nonuniform_result(problem, &polynomial)?;
    Ok(polynomial)
}

#[cfg(feature = "internals")]
fn validate_problem<E: Elem>(problem: InterpolationProblem<E>) -> Result<(), InterpolationError> {
    if problem.points.is_empty() {
        return Err(InterpolationError::LengthMismatch {
            expected: 1,
            points: 0,
            values: 0,
        });
    }
    for (index, point) in problem.points.iter().enumerate() {
        if point.multiplicity == 0 {
            return Err(InterpolationError::InvalidResult {
                reason: "interpolation multiplicity must be positive",
            });
        }
        if problem
            .points
            .iter()
            .take(index)
            .any(|other| other.x == point.x)
        {
            return Err(InterpolationError::DuplicatePoint {
                first: problem
                    .points
                    .iter()
                    .position(|other| other.x == point.x)
                    .unwrap_or(0),
                second: index,
            });
        }
    }
    Ok(())
}

#[cfg(feature = "internals")]
fn validate_nonuniform_result<F: FieldKernels>(
    problem: InterpolationProblem<F::Elem>,
    polynomial: &BivariatePolynomial<F>,
) -> Result<(), InterpolationError> {
    if polynomial.is_zero() {
        return Err(InterpolationError::InvalidResult {
            reason: "the nullspace vector reconstructed to zero",
        });
    }
    if polynomial
        .y_degree()
        .is_some_and(|degree| degree > problem.y_degree)
    {
        return Err(InterpolationError::InvalidResult {
            reason: "Y-degree exceeds the parameter bound",
        });
    }
    if polynomial.weighted_degree(problem.y_weight)? > Some(problem.weighted_degree) {
        return Err(InterpolationError::InvalidResult {
            reason: "weighted degree exceeds the parameter bound",
        });
    }
    for (point_index, point) in problem.points.iter().enumerate() {
        for total_order in 0..point.multiplicity {
            for y_order in 0..=total_order {
                let x_order = total_order - y_order;
                if !polynomial
                    .hasse_discrepancy(point.x, point.y, x_order, y_order)
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
