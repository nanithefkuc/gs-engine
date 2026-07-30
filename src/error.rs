use core::fmt;

/// Failure while validating a decoder configuration or reserving its storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigError {
    /// A required count is zero.
    ZeroParameter {
        /// Name of the zero-valued parameter.
        parameter: &'static str,
    },
    /// The candidate polynomial degree is incompatible with the code length.
    DegreeOutOfRange {
        /// Maximum allowed candidate degree.
        max_degree: usize,
        /// Number of evaluation points.
        code_length: usize,
    },
    /// The requested radius leaves no agreement position.
    RadiusOutOfRange {
        /// Requested Hamming radius.
        target_radius: usize,
        /// Number of evaluation points.
        code_length: usize,
    },
    /// The requested code has more points than the field contains.
    FieldCapacityExceeded {
        /// Number of requested evaluation points.
        code_length: usize,
        /// Number of elements in the selected field.
        field_order: u128,
    },
    /// A derived length or byte count cannot be represented by [`usize`].
    GeometryOverflow {
        /// Name of the value whose calculation overflowed.
        context: &'static str,
    },
    /// A feasible geometry exceeds a caller-provided resource limit.
    ResourceLimitExceeded {
        /// Name of the bounded resource.
        resource: &'static str,
        /// Amount required by the parameter tuple.
        required: usize,
        /// Maximum amount permitted by the caller.
        limit: usize,
    },
    /// The interpolation monomial space cannot satisfy all constraints.
    InsufficientInterpolationSpace {
        /// Available interpolation monomials.
        monomials: usize,
        /// Multiplicity constraints to satisfy.
        constraints: usize,
    },
    /// The weighted-degree bound does not force an agreeing candidate to be a root.
    InsufficientAgreement {
        /// Weighted-degree bound of the interpolation polynomial.
        weighted_degree: usize,
        /// Root multiplicity supplied by the target agreement.
        agreement_multiplicity: usize,
    },
    /// No tuple inside the search bounds reaches the requested radius.
    NoFeasibleParameters {
        /// Requested Hamming radius.
        target_radius: usize,
        /// Largest multiplicity searched.
        max_multiplicity: usize,
        /// Largest interpolation `Y`-degree searched.
        max_y_degree: usize,
    },
    /// Storage for a validated geometry could not be reserved.
    AllocationFailed {
        /// Name of the buffer being allocated.
        context: &'static str,
        /// Number of elements requested.
        elements: usize,
        /// Size of each element in bytes.
        element_size: usize,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ZeroParameter { parameter } => {
                write!(formatter, "{parameter} must be nonzero")
            }
            Self::DegreeOutOfRange {
                max_degree,
                code_length,
            } => write!(
                formatter,
                "maximum candidate degree {max_degree} must be below code length {code_length}"
            ),
            Self::RadiusOutOfRange {
                target_radius,
                code_length,
            } => write!(
                formatter,
                "target radius {target_radius} must be below code length {code_length}"
            ),
            Self::FieldCapacityExceeded {
                code_length,
                field_order,
            } => write!(
                formatter,
                "code length {code_length} exceeds the field order {field_order}"
            ),
            Self::GeometryOverflow { context } => {
                write!(
                    formatter,
                    "{context} exceeds the platform's addressable geometry"
                )
            }
            Self::ResourceLimitExceeded {
                resource,
                required,
                limit,
            } => write!(
                formatter,
                "{resource} requires {required}, exceeding the configured limit {limit}"
            ),
            Self::InsufficientInterpolationSpace {
                monomials,
                constraints,
            } => write!(
                formatter,
                "{monomials} interpolation monomials cannot satisfy {constraints} constraints"
            ),
            Self::InsufficientAgreement {
                weighted_degree,
                agreement_multiplicity,
            } => write!(
                formatter,
                "weighted degree {weighted_degree} is not below agreement multiplicity {agreement_multiplicity}"
            ),
            Self::NoFeasibleParameters {
                target_radius,
                max_multiplicity,
                max_y_degree,
            } => write!(
                formatter,
                "no parameters reach radius {target_radius} with multiplicity <= {max_multiplicity} and Y-degree <= {max_y_degree}"
            ),
            Self::AllocationFailed {
                context,
                elements,
                element_size,
            } => write!(
                formatter,
                "could not reserve {elements} elements of {element_size} bytes for {context}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ConfigError {}
