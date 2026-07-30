use fff::field::{Elem, Field};
use fff::{Gf8, Gf16};
use gs_engine::{
    BivariatePolynomial, GsParameters, ParameterLimits, Polynomial, interpolate_koetter,
    interpolate_module,
};
#[cfg(feature = "diagnostic")]
use gs_engine::{
    InterpolationConstraint, InterpolationError, InterpolationMonomial,
    ReferenceInterpolationLimits, interpolate_reference, reference_constraints,
    reference_monomials,
};

const PARAMETER_LIMITS: ParameterLimits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);

fn gf16(value: u16) -> <Gf16 as Field>::Elem {
    Gf16::read(&value.to_le_bytes())
}

fn gf8(value: u8) -> <Gf8 as Field>::Elem {
    Gf8::read(&[value])
}

fn assert_hasse_constraints<F: fff::kernel::FieldKernels>(
    parameters: GsParameters,
    points: &[F::Elem],
    values: &[F::Elem],
    polynomial: &BivariatePolynomial<F>,
) {
    for point_index in 0..parameters.code_length() {
        for total_order in 0..parameters.multiplicity() {
            for y_order in 0..=total_order {
                let x_order = total_order - y_order;
                assert!(
                    polynomial
                        .hasse_discrepancy(
                            points[point_index],
                            values[point_index],
                            x_order,
                            y_order,
                        )
                        .is_zero()
                );
            }
        }
    }
}

#[cfg(feature = "diagnostic")]
#[test]
fn monomial_and_constraint_orders_are_exact() {
    let parameters = GsParameters::new::<Gf16>(15, 4, 6, 2, 3, 17, PARAMETER_LIMITS).unwrap();
    let monomials = reference_monomials(parameters).unwrap();
    assert_eq!(monomials.len(), 48);
    assert_eq!(
        &monomials[..3],
        &[
            InterpolationMonomial {
                x_degree: 0,
                y_degree: 0,
                weighted_degree: 0,
            },
            InterpolationMonomial {
                x_degree: 1,
                y_degree: 0,
                weighted_degree: 1,
            },
            InterpolationMonomial {
                x_degree: 2,
                y_degree: 0,
                weighted_degree: 2,
            },
        ]
    );
    assert_eq!(
        monomials.last(),
        Some(&InterpolationMonomial {
            x_degree: 5,
            y_degree: 3,
            weighted_degree: 17,
        })
    );

    let constraints = reference_constraints(parameters).unwrap();
    assert_eq!(constraints.len(), 45);
    assert_eq!(
        &constraints[..3],
        &[
            InterpolationConstraint {
                point_index: 0,
                x_order: 0,
                y_order: 0,
            },
            InterpolationConstraint {
                point_index: 0,
                x_order: 1,
                y_order: 0,
            },
            InterpolationConstraint {
                point_index: 0,
                x_order: 0,
                y_order: 1,
            },
        ]
    );
    assert_eq!(constraints.last().unwrap().point_index, 14);
}

#[cfg(feature = "diagnostic")]
#[test]
fn reference_interpolation_recovers_the_guaranteed_root() {
    let parameters = GsParameters::new::<Gf16>(15, 4, 6, 2, 3, 17, PARAMETER_LIMITS).unwrap();
    let points: Vec<_> = (1..=15).map(gf16).collect();
    let message = Polynomial::<Gf16>::from_coefficients(&[
        gf16(0x1234),
        gf16(0xabcd),
        gf16(0x0108),
        gf16(0xbeef),
        gf16(0x2222),
    ])
    .unwrap();
    let mut received = message.evaluate_many(&points).unwrap();
    for (offset, value) in received[9..].iter_mut().enumerate() {
        *value = value.add(gf16((offset + 1) as u16));
    }

    let interpolation = interpolate_reference::<Gf16>(
        parameters,
        &points,
        &received,
        ReferenceInterpolationLimits::new(3_000, 6_000),
    )
    .unwrap();

    assert!(!interpolation.is_zero());
    assert!(interpolation.y_degree().unwrap() <= 3);
    assert!(interpolation.weighted_degree(4).unwrap().unwrap() <= 17);
    assert!(interpolation.has_root(&message).unwrap());
    for constraint in reference_constraints(parameters).unwrap() {
        assert!(
            interpolation
                .hasse_discrepancy(
                    points[constraint.point_index],
                    received[constraint.point_index],
                    constraint.x_order,
                    constraint.y_order,
                )
                .is_zero()
        );
    }
}

#[cfg(feature = "diagnostic")]
#[test]
fn reference_interpolation_works_over_gf8() {
    let parameters = GsParameters::new::<Gf8>(7, 2, 2, 1, 2, 4, PARAMETER_LIMITS).unwrap();
    let points: Vec<_> = (1..=7).map(gf8).collect();
    let values: Vec<_> = points
        .iter()
        .copied()
        .map(|point| point.square().add(gf8(3)))
        .collect();
    let interpolation = interpolate_reference::<Gf8>(
        parameters,
        &points,
        &values,
        ReferenceInterpolationLimits::new(100, 100),
    )
    .unwrap();

    assert!(!interpolation.is_zero());
    for constraint in reference_constraints(parameters).unwrap() {
        assert!(
            interpolation
                .hasse_discrepancy(
                    points[constraint.point_index],
                    values[constraint.point_index],
                    constraint.x_order,
                    constraint.y_order,
                )
                .is_zero()
        );
    }
}

#[test]
fn koetter_interpolation_recovers_the_guaranteed_root() {
    let parameters = GsParameters::new::<Gf16>(15, 4, 6, 2, 3, 17, PARAMETER_LIMITS).unwrap();
    let points: Vec<_> = (0..15).map(gf16).collect();
    let message = Polynomial::<Gf16>::from_coefficients(&[
        gf16(0x1234),
        gf16(0xabcd),
        gf16(0x0108),
        gf16(0xbeef),
        gf16(0x2222),
    ])
    .unwrap();
    let mut received = message.evaluate_many(&points).unwrap();
    for (offset, value) in received[9..].iter_mut().enumerate() {
        *value = value.add(gf16((offset + 1) as u16));
    }

    let interpolation = interpolate_koetter::<Gf16>(parameters, &points, &received).unwrap();
    let module = interpolate_module::<Gf16>(parameters, &points, &received).unwrap();

    assert!(!interpolation.is_zero());
    assert!(interpolation.y_degree().unwrap() <= 3);
    assert!(interpolation.weighted_degree(4).unwrap().unwrap() <= 17);
    assert!(interpolation.has_root(&message).unwrap());
    assert!(!module.is_zero());
    assert!(module.y_degree().unwrap() <= 3);
    assert!(module.weighted_degree(4).unwrap().unwrap() <= 17);
    assert!(module.has_root(&message).unwrap());
    assert_hasse_constraints(parameters, &points, &received, &interpolation);
    assert_hasse_constraints(parameters, &points, &received, &module);
}

#[test]
fn high_multiplicity_koetter_jets_match_module_constraints() {
    let parameters = GsParameters::search::<Gf16>(15, 5, 6, PARAMETER_LIMITS).unwrap();
    assert_eq!(parameters.multiplicity(), 6);
    assert_eq!(parameters.y_degree(), 10);
    let points: Vec<_> = (0..15).map(gf16).collect();
    let message = Polynomial::<Gf16>::from_coefficients(&[
        gf16(3),
        gf16(17),
        gf16(29),
        gf16(43),
        gf16(71),
        gf16(113),
    ])
    .unwrap();
    let mut received = message.evaluate_many(&points).unwrap();
    for (offset, value) in received[9..].iter_mut().enumerate() {
        *value = value.add(gf16((offset + 31) as u16));
    }

    let koetter = interpolate_koetter::<Gf16>(parameters, &points, &received).unwrap();
    let module = interpolate_module::<Gf16>(parameters, &points, &received).unwrap();

    assert_hasse_constraints(parameters, &points, &received, &koetter);
    assert_hasse_constraints(parameters, &points, &received, &module);
    assert!(koetter.has_root(&message).unwrap());
    assert!(module.has_root(&message).unwrap());
}

#[cfg(feature = "diagnostic")]
#[test]
fn reference_and_koetter_yield_the_same_filtered_candidates() {
    let parameters = GsParameters::new::<Gf8>(5, 0, 2, 1, 1, 2, PARAMETER_LIMITS).unwrap();
    let points: Vec<_> = (1..=5).map(gf8).collect();
    let received = vec![gf8(7), gf8(7), gf8(7), gf8(9), gf8(9)];
    let reference = interpolate_reference::<Gf8>(
        parameters,
        &points,
        &received,
        ReferenceInterpolationLimits::new(100, 100),
    )
    .unwrap();
    let production = interpolate_koetter::<Gf8>(parameters, &points, &received).unwrap();
    let optimized = interpolate_module::<Gf8>(parameters, &points, &received).unwrap();

    let filtered_candidates = |interpolation: &gs_engine::BivariatePolynomial<Gf8>| {
        (u8::MIN..=u8::MAX)
            .filter_map(|raw| {
                let value = gf8(raw);
                let candidate = Polynomial::<Gf8>::from_coefficients(&[value]).unwrap();
                let distance = points
                    .iter()
                    .zip(&received)
                    .filter(|(point, received_value)| {
                        candidate.evaluate(**point) != **received_value
                    })
                    .count();
                (distance <= parameters.target_radius()
                    && interpolation.has_root(&candidate).unwrap())
                .then_some(value)
            })
            .collect::<Vec<_>>()
    };

    let reference_candidates = filtered_candidates(&reference);
    let production_candidates = filtered_candidates(&production);
    let optimized_candidates = filtered_candidates(&optimized);
    assert_eq!(reference_candidates, vec![gf8(7)]);
    assert_eq!(production_candidates, reference_candidates);
    assert_eq!(optimized_candidates, reference_candidates);
}

#[cfg(feature = "diagnostic")]
#[test]
fn reference_limits_and_inputs_fail_before_solving() {
    let parameters = GsParameters::new::<Gf16>(15, 4, 6, 2, 3, 17, PARAMETER_LIMITS).unwrap();
    let points: Vec<_> = (1..=15).map(gf16).collect();
    let values = vec![gf16(1); 15];

    assert_eq!(
        interpolate_reference::<Gf16>(
            parameters,
            &points,
            &values,
            ReferenceInterpolationLimits::new(2_159, usize::MAX),
        ),
        Err(InterpolationError::ReferenceLimitExceeded {
            resource: "matrix elements",
            required: 2_160,
            limit: 2_159,
        })
    );
    assert_eq!(
        interpolate_reference::<Gf16>(
            parameters,
            &points,
            &values,
            ReferenceInterpolationLimits::new(usize::MAX, 4_319),
        ),
        Err(InterpolationError::ReferenceLimitExceeded {
            resource: "matrix bytes",
            required: 4_320,
            limit: 4_319,
        })
    );
    assert_eq!(
        interpolate_reference::<Gf16>(
            parameters,
            &points[..14],
            &values,
            ReferenceInterpolationLimits::new(usize::MAX, usize::MAX),
        ),
        Err(InterpolationError::LengthMismatch {
            expected: 15,
            points: 14,
            values: 15,
        })
    );

    let mut duplicate = points.clone();
    duplicate[7] = duplicate[2];
    assert_eq!(
        interpolate_reference::<Gf16>(
            parameters,
            &duplicate,
            &values,
            ReferenceInterpolationLimits::new(usize::MAX, usize::MAX),
        ),
        Err(InterpolationError::DuplicatePoint {
            first: 2,
            second: 7,
        })
    );
}
