use alloc::vec::Vec;
use core::fmt;

use butterfly_fft::core::kernel::ButterflyKernels;
use butterfly_fft::error::TransformLengthError;

use crate::evaluate::score_candidates;
use crate::roots::alekhnovich_roots_into;
use crate::{
    AlekhnovichLimits, ConfigError, DecodeScratch, DomainError, EvaluationDomain, GsParameters,
    InterpolationError, InterpolationPlan, Polynomial, RootError, interpolate_koetter_into,
    interpolate_module_into,
};

/// Failure while constructing or executing an end-to-end GS decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// Decoder or workspace geometry is invalid.
    Config(ConfigError),
    /// Evaluation-domain construction or matching failed.
    Domain(DomainError),
    /// The received word has the wrong length.
    ReceivedLength {
        /// Length required by the plan.
        expected: usize,
        /// Supplied received-word length.
        got: usize,
    },
    /// Multiplicity interpolation failed.
    Interpolation(InterpolationError),
    /// Polynomial root extraction failed.
    Roots(RootError),
    /// A butterfly-fft execution buffer had inconsistent geometry.
    Transform(TransformLengthError),
    /// A decoder-internal postcondition was violated.
    InternalInvariant {
        /// Static explanation of the invariant.
        reason: &'static str,
    },
}

impl From<ConfigError> for DecodeError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<DomainError> for DecodeError {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

impl From<InterpolationError> for DecodeError {
    fn from(error: InterpolationError) -> Self {
        Self::Interpolation(error)
    }
}

impl From<RootError> for DecodeError {
    fn from(error: RootError) -> Self {
        Self::Roots(error)
    }
}

impl From<TransformLengthError> for DecodeError {
    fn from(error: TransformLengthError) -> Self {
        Self::Transform(error)
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::Domain(error) => error.fmt(formatter),
            Self::ReceivedLength { expected, got } => write!(
                formatter,
                "received word has length {got}, but decoder plan requires {expected}"
            ),
            Self::Interpolation(error) => error.fmt(formatter),
            Self::Roots(error) => error.fmt(formatter),
            Self::Transform(error) => error.fmt(formatter),
            Self::InternalInvariant { reason } => {
                write!(formatter, "decoder invariant failed: {reason}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecodeError {}

/// Immutable validated plan joining GS parameters, a domain, and root limits.
#[derive(Clone, Debug)]
pub struct GsPlan<F: ButterflyKernels> {
    parameters: GsParameters,
    domain: EvaluationDomain<F>,
    root_limits: AlekhnovichLimits,
    interpolation: InterpolationPlan<F>,
}

impl<F: ButterflyKernels> GsPlan<F> {
    /// Construct a decoder plan and require exact parameter/domain agreement.
    pub fn new(
        parameters: GsParameters,
        domain: EvaluationDomain<F>,
        root_limits: AlekhnovichLimits,
    ) -> Result<Self, DecodeError> {
        if domain.len() != parameters.code_length() {
            return Err(DomainError::LengthMismatch {
                expected: parameters.code_length(),
                got: domain.len(),
            }
            .into());
        }
        let interpolation = InterpolationPlan::new_with_domain(parameters, &domain)?;
        Ok(Self {
            parameters,
            domain,
            root_limits,
            interpolation,
        })
    }

    /// Validated GS interpolation and radius parameters.
    #[must_use]
    pub const fn parameters(&self) -> GsParameters {
        self.parameters
    }

    /// Evaluation domain in received-word order.
    #[must_use]
    pub const fn domain(&self) -> &EvaluationDomain<F> {
        &self.domain
    }

    /// Root-extraction resource limits.
    #[must_use]
    pub const fn root_limits(&self) -> AlekhnovichLimits {
        self.root_limits
    }

    /// Total heap bytes retained by the prepared, received-word-independent
    /// interpolation data. Report this to bound plan memory before decoding.
    #[must_use]
    pub fn prepared_bytes(&self) -> usize {
        self.interpolation.prepared_bytes()
    }

    /// Reserve the geometry-dependent decoder workspace and output capacity.
    ///
    /// Call this once when construction-time allocation is preferable to
    /// first-use allocation. Interpolation and data-dependent root-factor
    /// storage may still grow for a new received word.
    pub fn prepare_scratch(
        &self,
        scratch: &mut DecodeScratch<F>,
        output: &mut Vec<Polynomial<F>>,
    ) -> Result<(), DecodeError> {
        if self.domain.transform_plan().is_some() {
            scratch.reserve_evaluation(self.domain.len(), self.parameters.y_degree())?;
            scratch.module.reserve_transform(self.domain.len())?;
        }
        let root_capacity = self
            .parameters
            .y_degree()
            .min(self.root_limits.max_output_roots());
        if scratch.root_candidates.capacity() < root_capacity {
            scratch
                .root_candidates
                .try_reserve(root_capacity - scratch.root_candidates.capacity())
                .map_err(|_| ConfigError::AllocationFailed {
                    context: "planned root candidates",
                    elements: root_capacity,
                    element_size: core::mem::size_of::<Polynomial<F>>(),
                })?;
        }
        if output.capacity() < self.parameters.y_degree() {
            output
                .try_reserve(self.parameters.y_degree().saturating_sub(output.len()))
                .map_err(|_| ConfigError::AllocationFailed {
                    context: "planned decoded candidates",
                    elements: self.parameters.y_degree(),
                    element_size: core::mem::size_of::<Polynomial<F>>(),
                })?;
        }
        Ok(())
    }

    /// Decode into caller-owned output storage.
    ///
    /// Existing output entries are removed first. On success the output holds
    /// exactly the distinct bounded-degree polynomial roots whose evaluations
    /// are within the configured target radius. Received symbols are borrowed
    /// only for this call and are never retained by the plan.
    pub fn decode_into(
        &self,
        received: &[F::Elem],
        scratch: &mut DecodeScratch<F>,
        output: &mut Vec<Polynomial<F>>,
    ) -> Result<usize, DecodeError> {
        if received.len() != self.parameters.code_length() {
            output.clear();
            return Err(DecodeError::ReceivedLength {
                expected: self.parameters.code_length(),
                got: received.len(),
            });
        }

        let interpolation_backend = crate::cost::select_interpolation(crate::cost::InterpolationCostKey {
            field_bytes: F::BYTES,
            backend: crate::cost::BackendClass::detect::<F>(),
            domain: self.domain.backend().into(),
            points: self.parameters.code_length(),
            multiplicity: self.parameters.multiplicity(),
            y_degree: self.parameters.y_degree(),
            weighted_degree: self.parameters.weighted_degree(),
            prepared: self.domain.transform_plan().is_some(),
        });
        let interpolation = if interpolation_backend == crate::cost::InterpolationBackend::Module {
            interpolate_module_into::<F>(
                self.parameters,
                self.domain.points(),
                received,
                &self.interpolation,
                Some(&self.domain),
                &mut scratch.module,
                &mut scratch.interpolation_output,
            )
            .map_err(DecodeError::from)
        } else {
            interpolate_koetter_into::<F>(
                self.parameters,
                self.domain.points(),
                received,
                &mut scratch.interpolation,
                &mut scratch.interpolation_output,
            )
            .map_err(DecodeError::from)
        };
        if let Err(error) = interpolation {
            output.clear();
            return Err(error);
        }
        if let Err(error) = alekhnovich_roots_into(
            &scratch.interpolation_output,
            self.parameters.max_degree(),
            self.root_limits,
            &mut scratch.roots,
            &mut scratch.root_candidates,
        ) {
            output.clear();
            return Err(error.into());
        }
        let candidates = core::mem::take(&mut scratch.root_candidates);
        let scoring = score_candidates(
            &self.domain,
            received,
            &candidates,
            self.parameters.target_radius(),
            scratch,
            output,
        );
        scratch.root_candidates = candidates;
        if let Err(error) = scoring {
            output.clear();
            return Err(error);
        }
        Ok(output.len())
    }
}
