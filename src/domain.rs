//! Validated arbitrary, additive-subspace, and affine-coset domains.

use alloc::vec::Vec;
use core::fmt;

use butterfly_fft::core::kernel::ButterflyKernels;
use butterfly_fft::core::transform::TransformPlan;
use butterfly_fft::error::PlanError;
use butterfly_fft::shifted::ShiftedPlan;

use crate::ConfigError;

/// Evaluation implementation selected by a validated domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationBackend {
    /// Scalar Horner evaluation at arbitrary points.
    Horner,
    /// butterfly-fft evaluation over an additive subspace.
    ButterflyFftAdditive,
    /// butterfly-fft evaluation over an affine coset.
    ButterflyFftAffineCoset,
}

/// Failure while constructing or matching an evaluation domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainError {
    /// General checked geometry or allocation failure.
    Config(ConfigError),
    /// Two arbitrary evaluation points are equal.
    DuplicatePoint {
        /// Index of the first occurrence.
        first: usize,
        /// Index of the duplicate occurrence.
        second: usize,
    },
    /// A butterfly-fft plan could not represent the requested domain.
    TransformPlan(PlanError),
    /// The domain size does not match the decoder parameters.
    LengthMismatch {
        /// Length required by the parameters.
        expected: usize,
        /// Number of points in the domain.
        got: usize,
    },
}

impl From<ConfigError> for DomainError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<PlanError> for DomainError {
    fn from(error: PlanError) -> Self {
        Self::TransformPlan(error)
    }
}

impl fmt::Display for DomainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::DuplicatePoint { first, second } => write!(
                formatter,
                "evaluation points at indices {first} and {second} are equal"
            ),
            Self::TransformPlan(error) => error.fmt(formatter),
            Self::LengthMismatch { expected, got } => write!(
                formatter,
                "evaluation domain has {got} points, but decoder parameters require {expected}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DomainError {}

/// Distinct evaluation points with an optional butterfly-fft execution plan.
#[derive(Clone, Debug)]
pub struct EvaluationDomain<F: ButterflyKernels> {
    points: Vec<F::Elem>,
    kind: DomainKind<F>,
}

#[derive(Clone, Debug)]
enum DomainKind<F: ButterflyKernels> {
    Arbitrary,
    Additive(TransformPlan<F>),
    Affine(ShiftedPlan<F>),
}

impl<F: ButterflyKernels> EvaluationDomain<F> {
    /// Construct a domain from arbitrary distinct points.
    pub fn arbitrary(points: Vec<F::Elem>) -> Result<Self, DomainError> {
        validate_points::<F>(&points)?;
        Ok(Self {
            points,
            kind: DomainKind::Arbitrary,
        })
    }

    /// Construct the default bit-basis additive subspace of `size` points.
    pub fn additive_subspace(size: usize) -> Result<Self, DomainError> {
        Self::from_additive_plan(TransformPlan::<F>::new(size)?)
    }

    /// Construct an additive subspace from an explicit ordered basis prefix.
    pub fn additive_subspace_with_basis(
        size: usize,
        basis: &[F::Elem],
    ) -> Result<Self, DomainError> {
        Self::from_additive_plan(TransformPlan::<F>::with_basis(size, basis)?)
    }

    /// Construct a default bit-basis affine coset.
    pub fn affine_coset(size: usize, shift: F::Elem) -> Result<Self, DomainError> {
        Self::from_shifted_plan(ShiftedPlan::<F>::new(size, shift)?)
    }

    /// Construct an affine coset from an explicit ordered basis prefix.
    pub fn affine_coset_with_basis(
        size: usize,
        basis: &[F::Elem],
        shift: F::Elem,
    ) -> Result<Self, DomainError> {
        Self::from_shifted_plan(ShiftedPlan::<F>::from_elements(size, basis, shift)?)
    }

    /// Construct from an already prepared unshifted plan.
    pub fn from_additive_plan(plan: TransformPlan<F>) -> Result<Self, DomainError> {
        let mut points = Vec::new();
        reserve_exact::<F::Elem>(&mut points, plan.size(), "additive-domain points")?;
        points.extend((0..plan.size()).map(|index| plan.point_element(index)));
        Ok(Self {
            points,
            kind: DomainKind::Additive(plan),
        })
    }

    /// Construct from an already prepared shifted plan.
    pub fn from_shifted_plan(plan: ShiftedPlan<F>) -> Result<Self, DomainError> {
        let mut points = Vec::new();
        reserve_exact::<F::Elem>(&mut points, plan.size(), "affine-domain points")?;
        points.extend((0..plan.size()).map(|index| plan.point_element(index)));
        Ok(Self {
            points,
            kind: DomainKind::Affine(plan),
        })
    }

    /// Evaluation points in decoder/transform order.
    #[must_use]
    pub fn points(&self) -> &[F::Elem] {
        &self.points
    }

    /// Number of evaluation points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the domain contains no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Evaluation backend selected by this domain.
    #[must_use]
    pub const fn backend(&self) -> EvaluationBackend {
        match self.kind {
            DomainKind::Arbitrary => EvaluationBackend::Horner,
            DomainKind::Additive(_) => EvaluationBackend::ButterflyFftAdditive,
            DomainKind::Affine(_) => EvaluationBackend::ButterflyFftAffineCoset,
        }
    }

    pub(crate) fn transform_plan(&self) -> Option<&TransformPlan<F>> {
        match &self.kind {
            DomainKind::Arbitrary => None,
            DomainKind::Additive(plan) => Some(plan),
            DomainKind::Affine(plan) => Some(plan.plan()),
        }
    }

    pub(crate) fn forward_bytes(
        &self,
        rows: &mut [u8],
        row_len: usize,
    ) -> Result<(), butterfly_fft::error::TransformLengthError> {
        match &self.kind {
            DomainKind::Arbitrary => Ok(()),
            DomainKind::Additive(plan) => plan.forward_bytes(rows, row_len),
            DomainKind::Affine(plan) => plan.forward_bytes(rows, row_len),
        }
    }
}

fn validate_points<F: ButterflyKernels>(points: &[F::Elem]) -> Result<(), DomainError> {
    if points.is_empty() {
        return Err(ConfigError::ZeroParameter {
            parameter: "evaluation-domain length",
        }
        .into());
    }
    if (points.len() as u128) > F::ORDER {
        return Err(ConfigError::FieldCapacityExceeded {
            code_length: points.len(),
            field_order: F::ORDER,
        }
        .into());
    }
    for second in 1..points.len() {
        if let Some(first) = points[..second]
            .iter()
            .position(|point| point == &points[second])
        {
            return Err(DomainError::DuplicatePoint { first, second });
        }
    }
    Ok(())
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
