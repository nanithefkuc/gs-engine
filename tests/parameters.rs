use fgf::{Gf8, Gf16};
use gs_engine::params::{interpolation_constraints, interpolation_monomials};
use gs_engine::{ConfigError, GsParameters, ParameterLimits};

const GENEROUS: ParameterLimits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);

#[test]
fn validates_radius_six_parameters_exactly() {
    let parameters =
        GsParameters::new::<Gf16>(15, 4, 6, 2, 4, 17, GENEROUS).expect("feasible tuple");

    assert_eq!(parameters.code_length(), 15);
    assert_eq!(parameters.max_degree(), 4);
    assert_eq!(parameters.target_radius(), 6);
    assert_eq!(parameters.multiplicity(), 2);
    assert_eq!(parameters.y_degree(), 4);
    assert_eq!(parameters.weighted_degree(), 17);
    assert_eq!(parameters.guaranteed_agreement(), 9);
    assert_eq!(parameters.guaranteed_radius(), 6);

    let resources = parameters.resources();
    assert_eq!(resources.monomials(), 50);
    assert_eq!(resources.constraints(), 45);
    assert_eq!(resources.reference_matrix_elements(), 2_250);
    assert_eq!(resources.koetter_coefficient_elements(), 1_150);
    assert_eq!(resources.coefficient_bytes(), 2_300);
    assert_eq!(resources.scratch_elements(), 235);
    assert_eq!(resources.scratch_bytes(), 470);
    assert!(resources.estimated_work() > 0);
    assert!(resources.lane_bytes() > 0);
}

#[test]
fn bounded_search_finds_the_minimum_known_tuple() {
    let parameters =
        GsParameters::search::<Gf16>(15, 4, 6, GENEROUS).expect("bounded feasible search");

    assert_eq!(parameters.multiplicity(), 2);
    assert_eq!(parameters.y_degree(), 3);
    assert_eq!(parameters.weighted_degree(), 17);
    assert_eq!(
        parameters,
        GsParameters::search::<Gf16>(15, 4, 6, GENEROUS).expect("deterministic search")
    );
}

#[test]
fn search_only_returns_feasible_tuples_across_a_grid() {
    for radius in 1..=6 {
        let Ok(parameters) = GsParameters::search::<Gf16>(15, 4, radius, GENEROUS) else {
            continue;
        };
        let resources = parameters.resources();
        assert!(
            resources.monomials() > resources.constraints(),
            "search returned an infeasible tuple at radius {radius}"
        );
        assert!(parameters.guaranteed_radius() >= radius);
    }
}

#[test]
fn exact_counts_include_zero_weight_y_rows() {
    assert_eq!(interpolation_monomials(17, 4, 4), Ok(50));
    assert_eq!(interpolation_constraints(15, 2), Ok(45));
    assert_eq!(interpolation_monomials(3, 4, 0), Ok(20));
}

#[test]
fn closed_form_counts_match_direct_enumeration() {
    for weighted_degree in 0usize..=12 {
        for y_degree in 0..=6 {
            for max_degree in 0..=5 {
                let direct = (0..=y_degree)
                    .map(|y| {
                        let y_weight = y * max_degree;
                        usize::from(y_weight <= weighted_degree)
                            * (weighted_degree.saturating_sub(y_weight) + 1)
                    })
                    .sum();
                assert_eq!(
                    interpolation_monomials(weighted_degree, y_degree, max_degree),
                    Ok(direct)
                );
            }
        }
    }

    for code_length in 1usize..=16 {
        for multiplicity in 1..=8 {
            let direct = code_length
                * (0..multiplicity)
                    .map(|total_order| total_order + 1)
                    .sum::<usize>();
            assert_eq!(
                interpolation_constraints(code_length, multiplicity),
                Ok(direct)
            );
        }
    }
}

#[test]
fn explicit_validation_rejects_each_failed_inequality() {
    assert_eq!(
        GsParameters::new::<Gf8>(15, 4, 6, 2, 4, 16, GENEROUS),
        Err(ConfigError::InsufficientInterpolationSpace {
            monomials: 45,
            constraints: 45,
        })
    );
    assert_eq!(
        GsParameters::new::<Gf8>(15, 4, 6, 2, 4, 18, GENEROUS),
        Err(ConfigError::InsufficientAgreement {
            weighted_degree: 18,
            agreement_multiplicity: 18,
        })
    );
}

#[test]
fn search_distinguishes_math_and_resource_failures() {
    let too_shallow = ParameterLimits::new(1, 2, usize::MAX, usize::MAX);
    assert_eq!(
        GsParameters::search::<Gf16>(15, 4, 6, too_shallow),
        Err(ConfigError::NoFeasibleParameters {
            target_radius: 6,
            max_multiplicity: 1,
            max_y_degree: 2,
        })
    );

    let too_small = ParameterLimits::new(2, 4, 1_471, usize::MAX);
    assert_eq!(
        GsParameters::search::<Gf16>(15, 4, 6, too_small),
        Err(ConfigError::ResourceLimitExceeded {
            resource: "interpolation coefficient bytes",
            required: 1_472,
            limit: 1_471,
        })
    );
}

#[test]
fn invalid_and_overflowing_geometries_are_reported() {
    assert_eq!(
        GsParameters::search::<Gf8>(0, 0, 0, GENEROUS),
        Err(ConfigError::ZeroParameter {
            parameter: "code length"
        })
    );
    assert_eq!(
        GsParameters::search::<Gf8>(15, 15, 0, GENEROUS),
        Err(ConfigError::DegreeOutOfRange {
            max_degree: 15,
            code_length: 15,
        })
    );
    assert_eq!(
        GsParameters::search::<Gf8>(15, 4, 15, GENEROUS),
        Err(ConfigError::RadiusOutOfRange {
            target_radius: 15,
            code_length: 15,
        })
    );
    assert_eq!(
        GsParameters::search::<Gf8>(257, 4, 0, GENEROUS),
        Err(ConfigError::FieldCapacityExceeded {
            code_length: 257,
            field_order: 256,
        })
    );
    assert!(matches!(
        interpolation_constraints(usize::MAX, usize::MAX),
        Err(ConfigError::GeometryOverflow { .. })
    ));
    assert!(matches!(
        interpolation_monomials(usize::MAX, usize::MAX, 0),
        Err(ConfigError::GeometryOverflow { .. })
    ));
}

#[test]
fn exact_resource_boundaries_are_inclusive_and_one_byte_less_is_rejected() {
    let exact = ParameterLimits::new(2, 4, 2_300, 470);
    assert!(GsParameters::new::<Gf16>(15, 4, 6, 2, 4, 17, exact).is_ok());

    assert_eq!(
        GsParameters::new::<Gf16>(15, 4, 6, 2, 4, 17, ParameterLimits::new(2, 4, 2_299, 470),),
        Err(ConfigError::ResourceLimitExceeded {
            resource: "interpolation coefficient bytes",
            required: 2_300,
            limit: 2_299,
        })
    );
    assert_eq!(
        GsParameters::new::<Gf16>(15, 4, 6, 2, 4, 17, ParameterLimits::new(2, 4, 2_300, 469),),
        Err(ConfigError::ResourceLimitExceeded {
            resource: "interpolation scratch bytes",
            required: 470,
            limit: 469,
        })
    );
}
