//! Exact Guruswami–Sudan parameter validation and bounded search.

use core::cmp::min;

use fgf::kernel::{FieldKernels, backend_for};

use crate::ConfigError;

/// Caller-provided bounds for parameter search and interpolation storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterLimits {
    max_multiplicity: usize,
    max_y_degree: usize,
    max_coefficient_bytes: usize,
    max_scratch_bytes: usize,
}

impl ParameterLimits {
    /// Construct explicit search and storage limits.
    #[must_use]
    pub const fn new(
        max_multiplicity: usize,
        max_y_degree: usize,
        max_coefficient_bytes: usize,
        max_scratch_bytes: usize,
    ) -> Self {
        Self {
            max_multiplicity,
            max_y_degree,
            max_coefficient_bytes,
            max_scratch_bytes,
        }
    }

    /// Largest interpolation multiplicity considered by automatic search.
    #[must_use]
    pub const fn max_multiplicity(self) -> usize {
        self.max_multiplicity
    }

    /// Largest interpolation `Y`-degree considered by automatic search.
    #[must_use]
    pub const fn max_y_degree(self) -> usize {
        self.max_y_degree
    }

    /// Maximum dense interpolation-basis capacity in bytes.
    #[must_use]
    pub const fn max_coefficient_bytes(self) -> usize {
        self.max_coefficient_bytes
    }

    /// Maximum baseline interpolation scratch capacity in bytes.
    #[must_use]
    pub const fn max_scratch_bytes(self) -> usize {
        self.max_scratch_bytes
    }
}

/// Checked resource geometry associated with one GS parameter tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceEstimate {
    monomials: usize,
    constraints: usize,
    reference_matrix_elements: usize,
    koetter_coefficient_elements: usize,
    coefficient_bytes: usize,
    scratch_elements: usize,
    scratch_bytes: usize,
    estimated_work: u128,
    lane_bytes: usize,
}

impl ResourceEstimate {
    /// Number of weighted-degree-bounded interpolation monomials.
    #[must_use]
    pub const fn monomials(self) -> usize {
        self.monomials
    }

    /// Number of Hasse multiplicity constraints.
    #[must_use]
    pub const fn constraints(self) -> usize {
        self.constraints
    }

    /// Elements in the explicit reference interpolation matrix.
    #[must_use]
    pub const fn reference_matrix_elements(self) -> usize {
        self.reference_matrix_elements
    }

    /// Conservative elements reserved for the dense Kötter interpolation basis.
    #[must_use]
    pub const fn koetter_coefficient_elements(self) -> usize {
        self.koetter_coefficient_elements
    }

    /// Bytes reserved for the dense Kötter interpolation basis.
    #[must_use]
    pub const fn coefficient_bytes(self) -> usize {
        self.coefficient_bytes
    }

    /// Elements in the baseline row-update and discrepancy scratch.
    #[must_use]
    pub const fn scratch_elements(self) -> usize {
        self.scratch_elements
    }

    /// Bytes in the baseline row-update and discrepancy scratch.
    #[must_use]
    pub const fn scratch_bytes(self) -> usize {
        self.scratch_bytes
    }

    /// Deterministic interpolation-plus-root work score used by automatic search.
    #[must_use]
    pub const fn estimated_work(self) -> u128 {
        self.estimated_work
    }

    /// SIMD lane width used by the work estimate.
    #[must_use]
    pub const fn lane_bytes(self) -> usize {
        self.lane_bytes
    }
}

/// A validated Guruswami–Sudan interpolation parameter tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GsParameters {
    code_length: usize,
    max_degree: usize,
    target_radius: usize,
    multiplicity: usize,
    y_degree: usize,
    weighted_degree: usize,
    guaranteed_agreement: usize,
    guaranteed_radius: usize,
    resources: ResourceEstimate,
}

impl GsParameters {
    /// Validate an explicit `(s, ell, D)` tuple for a target radius.
    ///
    /// The tuple must have more interpolation monomials than Hasse constraints
    /// and must satisfy `s * (n - target_radius) > D`.
    pub fn new<F: FieldKernels>(
        code_length: usize,
        max_degree: usize,
        target_radius: usize,
        multiplicity: usize,
        y_degree: usize,
        weighted_degree: usize,
        limits: ParameterLimits,
    ) -> Result<Self, ConfigError> {
        validate_code_geometry(code_length, max_degree, target_radius)?;
        validate_field_capacity::<F>(code_length)?;
        require_nonzero("multiplicity", multiplicity)?;
        require_nonzero("Y-degree", y_degree)?;
        check_limit("multiplicity", multiplicity, limits.max_multiplicity)?;
        check_limit("Y-degree", y_degree, limits.max_y_degree)?;

        let monomials = interpolation_monomials(weighted_degree, y_degree, max_degree)?;
        let constraints = interpolation_constraints(code_length, multiplicity)?;
        if monomials <= constraints {
            return Err(ConfigError::InsufficientInterpolationSpace {
                monomials,
                constraints,
            });
        }

        let target_agreement = code_length - target_radius;
        let agreement_multiplicity =
            checked_product_to_usize("agreement multiplicity", multiplicity, target_agreement)?;
        if weighted_degree >= agreement_multiplicity {
            return Err(ConfigError::InsufficientAgreement {
                weighted_degree,
                agreement_multiplicity,
            });
        }

        let resources =
            estimate_resources::<F>(max_degree, multiplicity, y_degree, monomials, constraints)?;
        check_limit(
            "interpolation coefficient bytes",
            resources.coefficient_bytes,
            limits.max_coefficient_bytes,
        )?;
        check_limit(
            "interpolation scratch bytes",
            resources.scratch_bytes,
            limits.max_scratch_bytes,
        )?;

        let guaranteed_agreement = weighted_degree
            .checked_div(multiplicity)
            .and_then(|quotient| quotient.checked_add(1))
            .ok_or(ConfigError::GeometryOverflow {
                context: "guaranteed agreement",
            })?;
        let guaranteed_radius = code_length - guaranteed_agreement;

        Ok(Self {
            code_length,
            max_degree,
            target_radius,
            multiplicity,
            y_degree,
            weighted_degree,
            guaranteed_agreement,
            guaranteed_radius,
            resources,
        })
    }

    /// Search every tuple inside `limits` and choose the lowest-cost feasible one.
    ///
    /// For each `(s, ell)`, search chooses the smallest `D` with sufficient
    /// interpolation monomials. Candidates are ordered by a deterministic work
    /// score based on the active FFF backend's SIMD lane width, then by storage,
    /// `Y`-degree, multiplicity, and weighted degree.
    pub fn search<F: FieldKernels>(
        code_length: usize,
        max_degree: usize,
        target_radius: usize,
        limits: ParameterLimits,
    ) -> Result<Self, ConfigError> {
        validate_code_geometry(code_length, max_degree, target_radius)?;
        validate_field_capacity::<F>(code_length)?;
        require_nonzero("maximum multiplicity", limits.max_multiplicity)?;
        require_nonzero("maximum Y-degree", limits.max_y_degree)?;

        let target_agreement = code_length - target_radius;
        let mut best: Option<(SearchScore, Self)> = None;
        let mut first_capacity_error = None;

        for multiplicity in 1..=limits.max_multiplicity {
            let constraints = match interpolation_constraints(code_length, multiplicity) {
                Ok(value) => value,
                Err(error) => {
                    first_capacity_error.get_or_insert(error);
                    break;
                }
            };
            for y_degree in 1..=limits.max_y_degree {
                let Some(weighted_degree) = minimum_weighted_degree(
                    max_degree,
                    target_agreement,
                    multiplicity,
                    y_degree,
                    constraints,
                )?
                else {
                    continue;
                };

                match Self::new::<F>(
                    code_length,
                    max_degree,
                    target_radius,
                    multiplicity,
                    y_degree,
                    weighted_degree,
                    limits,
                ) {
                    Ok(candidate) => {
                        let score = SearchScore::of(candidate);
                        if best.as_ref().is_none_or(|(current, _)| score < *current) {
                            best = Some((score, candidate));
                        }
                    }
                    Err(error @ ConfigError::ResourceLimitExceeded { .. })
                    | Err(error @ ConfigError::GeometryOverflow { .. }) => {
                        first_capacity_error.get_or_insert(error);
                    }
                    Err(ConfigError::InsufficientInterpolationSpace { .. })
                    | Err(ConfigError::InsufficientAgreement { .. }) => {}
                    Err(error) => return Err(error),
                }
            }
        }

        if let Some((_, parameters)) = best {
            return Ok(parameters);
        }
        if let Some(error) = first_capacity_error {
            return Err(error);
        }
        Err(ConfigError::NoFeasibleParameters {
            target_radius,
            max_multiplicity: limits.max_multiplicity,
            max_y_degree: limits.max_y_degree,
        })
    }

    /// Number of evaluation points.
    #[must_use]
    pub const fn code_length(self) -> usize {
        self.code_length
    }

    /// Maximum degree allowed for candidate message polynomials.
    #[must_use]
    pub const fn max_degree(self) -> usize {
        self.max_degree
    }

    /// Radius requested when the tuple was validated.
    #[must_use]
    pub const fn target_radius(self) -> usize {
        self.target_radius
    }

    /// Interpolation multiplicity `s`.
    #[must_use]
    pub const fn multiplicity(self) -> usize {
        self.multiplicity
    }

    /// Maximum interpolation `Y`-degree `ell`.
    #[must_use]
    pub const fn y_degree(self) -> usize {
        self.y_degree
    }

    /// `(1, max_degree)` weighted-degree bound `D`.
    #[must_use]
    pub const fn weighted_degree(self) -> usize {
        self.weighted_degree
    }

    /// Minimum agreement forced to be a polynomial root.
    #[must_use]
    pub const fn guaranteed_agreement(self) -> usize {
        self.guaranteed_agreement
    }

    /// Maximum radius guaranteed by this exact tuple.
    #[must_use]
    pub const fn guaranteed_radius(self) -> usize {
        self.guaranteed_radius
    }

    /// Checked storage and work estimates.
    #[must_use]
    pub const fn resources(self) -> ResourceEstimate {
        self.resources
    }
}

/// Count monomials `X^a Y^b` with `b <= y_degree` and
/// `a + b * max_degree <= weighted_degree`.
pub fn interpolation_monomials(
    weighted_degree: usize,
    y_degree: usize,
    max_degree: usize,
) -> Result<usize, ConfigError> {
    to_usize(
        "interpolation monomial count",
        interpolation_monomials_u128(
            weighted_degree as u128,
            y_degree as u128,
            max_degree as u128,
        )?,
    )
}

/// Count Hasse constraints imposed by `code_length` points at `multiplicity`.
pub fn interpolation_constraints(
    code_length: usize,
    multiplicity: usize,
) -> Result<usize, ConfigError> {
    require_nonzero("code length", code_length)?;
    require_nonzero("multiplicity", multiplicity)?;
    let multiplicity = multiplicity as u128;
    let pairs = half_product(multiplicity, multiplicity + 1)?;
    let constraints =
        (code_length as u128)
            .checked_mul(pairs)
            .ok_or(ConfigError::GeometryOverflow {
                context: "interpolation constraint count",
            })?;
    to_usize("interpolation constraint count", constraints)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SearchScore {
    work: u128,
    coefficient_bytes: usize,
    scratch_bytes: usize,
    y_degree: usize,
    multiplicity: usize,
    weighted_degree: usize,
}

impl SearchScore {
    fn of(parameters: GsParameters) -> Self {
        Self {
            work: parameters.resources.estimated_work,
            coefficient_bytes: parameters.resources.coefficient_bytes,
            scratch_bytes: parameters.resources.scratch_bytes,
            y_degree: parameters.y_degree,
            multiplicity: parameters.multiplicity,
            weighted_degree: parameters.weighted_degree,
        }
    }
}

fn validate_code_geometry(
    code_length: usize,
    max_degree: usize,
    target_radius: usize,
) -> Result<(), ConfigError> {
    require_nonzero("code length", code_length)?;
    if max_degree >= code_length {
        return Err(ConfigError::DegreeOutOfRange {
            max_degree,
            code_length,
        });
    }
    if target_radius >= code_length {
        return Err(ConfigError::RadiusOutOfRange {
            target_radius,
            code_length,
        });
    }
    Ok(())
}

fn validate_field_capacity<F: FieldKernels>(code_length: usize) -> Result<(), ConfigError> {
    if code_length as u128 > F::ORDER {
        Err(ConfigError::FieldCapacityExceeded {
            code_length,
            field_order: F::ORDER,
        })
    } else {
        Ok(())
    }
}

fn require_nonzero(parameter: &'static str, value: usize) -> Result<(), ConfigError> {
    if value == 0 {
        Err(ConfigError::ZeroParameter { parameter })
    } else {
        Ok(())
    }
}

fn check_limit(resource: &'static str, required: usize, limit: usize) -> Result<(), ConfigError> {
    if required > limit {
        Err(ConfigError::ResourceLimitExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

fn minimum_weighted_degree(
    max_degree: usize,
    target_agreement: usize,
    multiplicity: usize,
    y_degree: usize,
    constraints: usize,
) -> Result<Option<usize>, ConfigError> {
    let root_multiplicity = (multiplicity as u128)
        .checked_mul(target_agreement as u128)
        .ok_or(ConfigError::GeometryOverflow {
            context: "agreement multiplicity",
        })?;
    let maximum = root_multiplicity - 1;
    if interpolation_monomials_u128(maximum, y_degree as u128, max_degree as u128)?
        <= constraints as u128
    {
        return Ok(None);
    }

    let mut low = 0u128;
    let mut high = maximum;
    while low < high {
        let middle = low + (high - low) / 2;
        if interpolation_monomials_u128(middle, y_degree as u128, max_degree as u128)?
            > constraints as u128
        {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    to_usize("weighted-degree bound", low).map(Some)
}

fn interpolation_monomials_u128(
    weighted_degree: u128,
    y_degree: u128,
    max_degree: u128,
) -> Result<u128, ConfigError> {
    let active_rows = if max_degree == 0 {
        y_degree
            .checked_add(1)
            .ok_or(ConfigError::GeometryOverflow {
                context: "active interpolation rows",
            })?
    } else {
        min(y_degree, weighted_degree / max_degree)
            .checked_add(1)
            .ok_or(ConfigError::GeometryOverflow {
                context: "active interpolation rows",
            })?
    };
    let rectangle =
        active_rows
            .checked_mul(weighted_degree + 1)
            .ok_or(ConfigError::GeometryOverflow {
                context: "interpolation monomial count",
            })?;
    let staircase = half_product(active_rows, active_rows - 1)?
        .checked_mul(max_degree)
        .ok_or(ConfigError::GeometryOverflow {
            context: "interpolation monomial count",
        })?;
    rectangle
        .checked_sub(staircase)
        .ok_or(ConfigError::GeometryOverflow {
            context: "interpolation monomial count",
        })
}

fn estimate_resources<F: FieldKernels>(
    max_degree: usize,
    multiplicity: usize,
    y_degree: usize,
    monomials: usize,
    constraints: usize,
) -> Result<ResourceEstimate, ConfigError> {
    let basis_rows = checked_add_to_usize("interpolation basis rows", y_degree, 1)?;
    let x_capacity = checked_add_to_usize("interpolation X capacity", constraints, 1)?;
    let row_elements =
        checked_product_to_usize("interpolation basis row elements", basis_rows, x_capacity)?;
    let koetter_coefficient_elements =
        checked_product_to_usize("Kötter coefficient elements", basis_rows, row_elements)?;
    let coefficient_bytes = checked_product_to_usize(
        "interpolation coefficient bytes",
        koetter_coefficient_elements,
        F::BYTES,
    )?;
    let scratch_elements =
        checked_add_to_usize("interpolation scratch elements", row_elements, basis_rows)?;
    let scratch_bytes =
        checked_product_to_usize("interpolation scratch bytes", scratch_elements, F::BYTES)?;
    let reference_matrix_elements = checked_product_to_usize(
        "reference interpolation matrix elements",
        constraints,
        monomials,
    )?;

    let lane_bytes = backend_for::<F>().lane_bytes();
    let row_bytes =
        checked_product_to_usize("interpolation basis row bytes", row_elements, F::BYTES)?;
    let vector_blocks = row_bytes.div_ceil(lane_bytes);
    let interpolation_work = (constraints as u128)
        .saturating_mul(basis_rows as u128)
        .saturating_mul(vector_blocks as u128);
    let root_work = (max_degree as u128 + 1)
        .saturating_mul(basis_rows as u128)
        .saturating_mul(basis_rows as u128)
        .saturating_mul(multiplicity as u128);
    let estimated_work = interpolation_work.saturating_add(root_work);

    Ok(ResourceEstimate {
        monomials,
        constraints,
        reference_matrix_elements,
        koetter_coefficient_elements,
        coefficient_bytes,
        scratch_elements,
        scratch_bytes,
        estimated_work,
        lane_bytes,
    })
}

fn half_product(left: u128, right: u128) -> Result<u128, ConfigError> {
    let (left, right) = if left.is_multiple_of(2) {
        (left / 2, right)
    } else {
        (left, right / 2)
    };
    left.checked_mul(right)
        .ok_or(ConfigError::GeometryOverflow {
            context: "triangular count",
        })
}

fn checked_product_to_usize(
    context: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, ConfigError> {
    left.checked_mul(right)
        .ok_or(ConfigError::GeometryOverflow { context })
}

fn checked_add_to_usize(
    context: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, ConfigError> {
    left.checked_add(right)
        .ok_or(ConfigError::GeometryOverflow { context })
}

fn to_usize(context: &'static str, value: u128) -> Result<usize, ConfigError> {
    usize::try_from(value).map_err(|_| ConfigError::GeometryOverflow { context })
}
