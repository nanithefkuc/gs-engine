//! Nonuniform-multiplicity interpolation and differential oracle tests.
//!
//! Work package 6 introduces a per-point multiplicity representation behind
//! `internals` and uses the existing Kötter and module backends as
//! differential oracles. These tests cover:
//!
//! - the nonuniform reference oracle matches the uniform reference oracle
//!   when every point shares the same multiplicity;
//! - the nonuniform oracle satisfies its per-point Hasse lower sets directly;
//! - nonuniform multiplicities agree with a reference constraint-matrix
//!   kernel on small cases;
//! - the existing Kötter and module backends remain differential oracles for
//!   the uniform problem.

#![cfg(feature = "internals")]

use fgf::Gf8;
use fgf::Gf16;
use fgf::field::{Elem, Field};
use gs_engine::{
    BivariatePolynomial, GsParameters, InterpolationProblem, MultiplicityPoint, ParameterLimits,
    Polynomial, ReferenceInterpolationLimits, interpolate_koetter, interpolate_module,
    interpolate_reference, interpolate_reference_nonuniform,
};

const PARAMETER_LIMITS: ParameterLimits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);

fn gf8(value: u8) -> <Gf8 as Field>::Elem {
    Gf8::read(&[value])
}

fn gf16(value: u16) -> <Gf16 as Field>::Elem {
    Gf16::read(&value.to_le_bytes())
}

fn assert_nonuniform_hasse_gf8(
    points: &[MultiplicityPoint<<Gf8 as Field>::Elem>],
    polynomial: &BivariatePolynomial<Gf8>,
) {
    for (point_index, point) in points.iter().enumerate() {
        for total_order in 0..point.multiplicity {
            for y_order in 0..=total_order {
                let x_order = total_order - y_order;
                assert!(
                    polynomial
                        .hasse_discrepancy(point.x, point.y, x_order, y_order)
                        .is_zero(),
                    "constraint ({point_index}, x={x_order}, y={y_order}) violated"
                );
            }
        }
    }
}

fn assert_hasse_constraints_gf8(
    parameters: GsParameters,
    points: &[<Gf8 as Field>::Elem],
    values: &[<Gf8 as Field>::Elem],
    polynomial: &BivariatePolynomial<Gf8>,
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

fn assert_hasse_constraints_gf16(
    parameters: GsParameters,
    points: &[<Gf16 as Field>::Elem],
    values: &[<Gf16 as Field>::Elem],
    polynomial: &BivariatePolynomial<Gf16>,
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

/// Exhaustive GF8 candidate scan: returns every degree-0 polynomial whose
/// evaluation is within the target radius and is a root of `interpolation`.
fn filtered_candidates_gf8(
    parameters: GsParameters,
    points: &[<Gf8 as Field>::Elem],
    received: &[<Gf8 as Field>::Elem],
    interpolation: &BivariatePolynomial<Gf8>,
) -> Vec<<Gf8 as Field>::Elem> {
    (0u8..=u8::MAX)
        .filter_map(|raw| {
            let value = gf8(raw);
            let candidate = Polynomial::<Gf8>::from_coefficients(&[value]).unwrap();
            let distance = points
                .iter()
                .zip(received)
                .filter(|(point, received_value)| candidate.evaluate(**point) != **received_value)
                .count();
            (distance <= parameters.target_radius() && interpolation.has_root(&candidate).unwrap())
                .then_some(value)
        })
        .collect()
}

#[test]
fn nonuniform_uniform_matches_reference() {
    let parameters = GsParameters::new::<Gf8>(7, 2, 2, 1, 2, 4, PARAMETER_LIMITS).unwrap();
    let points: Vec<_> = (1..=7).map(gf8).collect();
    let values: Vec<_> = points
        .iter()
        .copied()
        .map(|point| point.square().add(gf8(3)))
        .collect();

    let uniform = interpolate_reference::<Gf8>(
        parameters,
        &points,
        &values,
        ReferenceInterpolationLimits::new(100, 100),
    )
    .unwrap();

    let nonuniform_points = points
        .iter()
        .zip(&values)
        .map(|(&x, &y)| MultiplicityPoint {
            x,
            y,
            multiplicity: parameters.multiplicity(),
        })
        .collect::<Vec<_>>();
    let problem = InterpolationProblem {
        points: &nonuniform_points,
        y_weight: parameters.max_degree(),
        y_degree: parameters.y_degree(),
        weighted_degree: parameters.weighted_degree(),
    };
    let nonuniform = interpolate_reference_nonuniform::<Gf8>(
        problem,
        ReferenceInterpolationLimits::new(100, 100),
    )
    .unwrap();

    assert_nonuniform_hasse_gf8(&nonuniform_points, &nonuniform);
    assert_hasse_constraints_gf8(parameters, &points, &values, &uniform);
    let uniform_candidates = filtered_candidates_gf8(parameters, &points, &values, &uniform);
    let nonuniform_candidates = filtered_candidates_gf8(parameters, &points, &values, &nonuniform);
    assert_eq!(uniform_candidates, nonuniform_candidates);
}

#[test]
fn nonuniform_varying_multiplicities_satisfy_lower_sets() {
    let points = [
        MultiplicityPoint {
            x: gf8(1),
            y: gf8(2),
            multiplicity: 1,
        },
        MultiplicityPoint {
            x: gf8(2),
            y: gf8(4),
            multiplicity: 2,
        },
        MultiplicityPoint {
            x: gf8(3),
            y: gf8(6),
            multiplicity: 1,
        },
    ];
    let problem = InterpolationProblem {
        points: &points,
        y_weight: 2,
        y_degree: 2,
        weighted_degree: 6,
    };
    let polynomial = interpolate_reference_nonuniform::<Gf8>(
        problem,
        ReferenceInterpolationLimits::new(10_000, 10_000),
    )
    .unwrap();

    assert!(!polynomial.is_zero());
    assert_nonuniform_hasse_gf8(&points, &polynomial);
}

#[test]
fn nonuniform_matches_explicit_constraint_kernel() {
    let points = [
        MultiplicityPoint {
            x: gf8(1),
            y: gf8(1),
            multiplicity: 2,
        },
        MultiplicityPoint {
            x: gf8(2),
            y: gf8(1),
            multiplicity: 1,
        },
    ];
    let problem = InterpolationProblem {
        points: &points,
        y_weight: 1,
        y_degree: 1,
        weighted_degree: 3,
    };
    let polynomial = interpolate_reference_nonuniform::<Gf8>(
        problem,
        ReferenceInterpolationLimits::new(10_000, 10_000),
    )
    .unwrap();

    assert!(polynomial.y_degree().unwrap() <= 1);
    assert!(polynomial.weighted_degree(1).unwrap().unwrap() <= 3);
    assert_nonuniform_hasse_gf8(&points, &polynomial);
}

#[test]
fn koetter_and_module_remain_differential_oracles() {
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

    let reference = interpolate_reference::<Gf16>(
        parameters,
        &points,
        &received,
        ReferenceInterpolationLimits::new(3_000, 6_000),
    )
    .unwrap();
    let koetter = interpolate_koetter::<Gf16>(parameters, &points, &received).unwrap();
    let module = interpolate_module::<Gf16>(parameters, &points, &received).unwrap();

    assert_hasse_constraints_gf16(parameters, &points, &received, &reference);
    assert_hasse_constraints_gf16(parameters, &points, &received, &koetter);
    assert_hasse_constraints_gf16(parameters, &points, &received, &module);
    assert!(reference.has_root(&message).unwrap());
    assert!(koetter.has_root(&message).unwrap());
    assert!(module.has_root(&message).unwrap());
}

#[test]
fn nonuniform_rejects_zero_multiplicity_and_duplicates() {
    let zero_mult = [
        MultiplicityPoint {
            x: gf8(1),
            y: gf8(1),
            multiplicity: 1,
        },
        MultiplicityPoint {
            x: gf8(2),
            y: gf8(1),
            multiplicity: 0,
        },
    ];
    let problem = InterpolationProblem {
        points: &zero_mult,
        y_weight: 1,
        y_degree: 1,
        weighted_degree: 2,
    };
    assert!(
        interpolate_reference_nonuniform::<Gf8>(
            problem,
            ReferenceInterpolationLimits::new(usize::MAX, usize::MAX)
        )
        .is_err()
    );

    let dup = [
        MultiplicityPoint {
            x: gf8(1),
            y: gf8(1),
            multiplicity: 1,
        },
        MultiplicityPoint {
            x: gf8(1),
            y: gf8(2),
            multiplicity: 1,
        },
    ];
    let problem = InterpolationProblem {
        points: &dup,
        y_weight: 1,
        y_degree: 1,
        weighted_degree: 2,
    };
    assert!(
        interpolate_reference_nonuniform::<Gf8>(
            problem,
            ReferenceInterpolationLimits::new(usize::MAX, usize::MAX)
        )
        .is_err()
    );
}
