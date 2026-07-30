use fff::field::{Elem, Field};
use fff::kernel::FieldKernels;
use fff::{Gf8, Gf16};
use gs_engine::{
    AlekhnovichLimits, AlekhnovichScratch, BivariatePolynomial, GsParameters, ParameterLimits,
    Polynomial, RootError, RothRuckensteinLimits, alekhnovich_roots, interpolate_koetter,
    roth_ruckenstein_roots,
};

const GENEROUS: RothRuckensteinLimits = RothRuckensteinLimits::new(100_000, 128);

const ALEKHNOVICH_GENEROUS: AlekhnovichLimits =
    AlekhnovichLimits::new(1_000_000, 100_000, 100_000_000, 400_000_000, 128)
        .with_roth_ruckenstein_crossover(0);

fn gf8(value: u8) -> <Gf8 as Field>::Elem {
    Gf8::read(&[value])
}

fn gf16(value: u16) -> <Gf16 as Field>::Elem {
    Gf16::read(&value.to_le_bytes())
}

fn polynomial<F: FieldKernels>(coefficients: &[F::Elem]) -> Polynomial<F> {
    Polynomial::from_coefficients(coefficients).unwrap()
}

fn product_of_y_plus<F: FieldKernels>(roots: &[Polynomial<F>]) -> BivariatePolynomial<F> {
    let mut rows = vec![Polynomial::<F>::one().unwrap()];
    for root in roots {
        let mut product = vec![Polynomial::<F>::zero(); rows.len() + 1];
        for (y_degree, row) in rows.iter().enumerate() {
            product[y_degree]
                .add_assign(&row.multiply(root).unwrap())
                .unwrap();
            product[y_degree + 1].add_assign(row).unwrap();
        }
        rows = product;
    }
    BivariatePolynomial::from_y_coefficients(rows)
}

fn assert_exact_roots<F: FieldKernels>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
    expected: &[Polynomial<F>],
) {
    let actual = roth_ruckenstein_roots(q, max_degree, GENEROUS).unwrap();
    assert_eq!(actual.len(), expected.len());
    assert!(expected.iter().all(|root| actual.contains(root)));
    assert!(actual.iter().all(|root| q.has_root(root).unwrap()));
    assert!(actual.len() <= q.y_degree().unwrap());
}

#[test]
fn extracts_multiple_shared_and_short_roots() {
    let roots = vec![
        Polynomial::<Gf16>::zero(),
        polynomial::<Gf16>(&[gf16(5)]),
        polynomial::<Gf16>(&[gf16(1), gf16(2), gf16(3)]),
        polynomial::<Gf16>(&[gf16(1), gf16(2), gf16(4)]),
        polynomial::<Gf16>(&[gf16(1), gf16(2), gf16(3), gf16(0), gf16(9)]),
    ];
    let mut q = product_of_y_plus(&roots);
    for _ in 0..3 {
        q = q.multiply_x_plus(gf16(0)).unwrap();
    }

    assert_exact_roots(&q, 4, &roots);
}

#[test]
fn repeated_factors_are_returned_once_and_order_is_canonical() {
    let first = polynomial::<Gf8>(&[gf8(11), gf8(22), gf8(33)]);
    let second = polynomial::<Gf8>(&[gf8(11), gf8(22), gf8(44)]);
    let forward = product_of_y_plus(&[first.clone(), first.clone(), second.clone()]);
    let reverse = product_of_y_plus(&[second.clone(), first.clone(), first.clone()]);

    let forward_roots = roth_ruckenstein_roots(&forward, 2, GENEROUS).unwrap();
    let reverse_roots = roth_ruckenstein_roots(&reverse, 2, GENEROUS).unwrap();
    assert_eq!(forward_roots, reverse_roots);
    assert_eq!(forward_roots.len(), 2);
    assert!(forward_roots.contains(&first));
    assert!(forward_roots.contains(&second));
}

#[test]
fn gf8_bounded_roots_match_brute_force() {
    let expected = [
        polynomial::<Gf8>(&[gf8(7), gf8(19)]),
        polynomial::<Gf8>(&[gf8(41), gf8(3)]),
        polynomial::<Gf8>(&[gf8(200)]),
    ];
    let q = product_of_y_plus(&expected);
    let actual = roth_ruckenstein_roots(&q, 1, GENEROUS).unwrap();
    let mut exhaustive = Vec::new();
    for constant in u8::MIN..=u8::MAX {
        for linear in u8::MIN..=u8::MAX {
            let candidate = polynomial::<Gf8>(&[gf8(constant), gf8(linear)]);
            if q.has_root(&candidate).unwrap() {
                exhaustive.push(candidate);
            }
        }
    }

    assert_eq!(actual, exhaustive);
}

#[test]
fn generated_gf8_bivariates_match_exact_linear_root_enumeration() {
    let mut state = 0xa341_316c_u32;
    for case in 0..8 {
        let mut rows = Vec::new();
        for y_degree in 0..=3 {
            let mut coefficients = Vec::new();
            for _ in 0..=3 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                coefficients.push(gf8((state >> 24) as u8));
            }
            if y_degree == 3 && coefficients.iter().all(|coefficient| coefficient.is_zero()) {
                coefficients[0] = gf8(1);
            }
            rows.push(polynomial::<Gf8>(&coefficients));
        }
        let q = BivariatePolynomial::from_y_coefficients(rows);
        let identity_degree = q
            .y_coefficients()
            .iter()
            .enumerate()
            .filter_map(|(y_degree, row)| row.degree().map(|x_degree| x_degree + y_degree))
            .max()
            .unwrap();
        let actual = roth_ruckenstein_roots(&q, 1, GENEROUS).unwrap();
        let mut scratch = AlekhnovichScratch::new();
        let divide_and_conquer =
            alekhnovich_roots(&q, 1, ALEKHNOVICH_GENEROUS, &mut scratch).unwrap();
        let mut exhaustive = Vec::new();
        for constant in u8::MIN..=u8::MAX {
            for linear in u8::MIN..=u8::MAX {
                let constant = gf8(constant);
                let linear = gf8(linear);
                let is_root = (0..=identity_degree).all(|raw_x| {
                    let x = gf8(raw_x as u8);
                    q.evaluate(x, constant.add(linear.mul(x))).is_zero()
                });
                if is_root {
                    exhaustive.push(polynomial::<Gf8>(&[constant, linear]));
                }
            }
        }
        assert_eq!(actual, exhaustive, "generated bivariate case {case}");
        assert_eq!(
            divide_and_conquer, exhaustive,
            "Alekhnovich generated bivariate case {case}"
        );
    }
}

#[test]
fn production_interpolation_root_is_extracted() {
    let parameter_limits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);
    let parameters = GsParameters::new::<Gf16>(15, 4, 6, 2, 3, 17, parameter_limits).unwrap();
    let points: Vec<_> = (0..15).map(gf16).collect();
    let message = polynomial::<Gf16>(&[
        gf16(0x1234),
        gf16(0xabcd),
        gf16(0x0108),
        gf16(0xbeef),
        gf16(0x2222),
    ]);
    let mut received = message.evaluate_many(&points).unwrap();
    for (offset, value) in received[9..].iter_mut().enumerate() {
        *value = value.add(gf16((offset + 1) as u16));
    }

    let q = interpolate_koetter::<Gf16>(parameters, &points, &received).unwrap();
    let roots = roth_ruckenstein_roots(&q, parameters.max_degree(), GENEROUS).unwrap();
    let mut scratch = AlekhnovichScratch::new();
    let divide_and_conquer = alekhnovich_roots(
        &q,
        parameters.max_degree(),
        ALEKHNOVICH_GENEROUS,
        &mut scratch,
    )
    .unwrap();

    assert!(roots.contains(&message));
    assert!(roots.iter().all(|root| q.has_root(root).unwrap()));
    assert_eq!(divide_and_conquer, roots);
}

#[test]
fn zero_constant_and_resource_boundaries_are_explicit() {
    assert_eq!(
        roth_ruckenstein_roots(&BivariatePolynomial::<Gf8>::zero(), 2, GENEROUS,),
        Err(RootError::ZeroBivariatePolynomial)
    );
    let y_independent =
        BivariatePolynomial::from_y_coefficients(vec![polynomial::<Gf8>(&[gf8(1), gf8(1)])]);
    assert!(
        roth_ruckenstein_roots(&y_independent, 2, RothRuckensteinLimits::new(0, 0))
            .unwrap()
            .is_empty()
    );

    let root = polynomial::<Gf8>(&[gf8(7), gf8(9)]);
    let q = product_of_y_plus(&[root]);
    assert_eq!(
        roth_ruckenstein_roots(&q, 1, RothRuckensteinLimits::new(0, 1)),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein work items",
            required: 1,
            limit: 0,
        })
    );
    assert_eq!(
        roth_ruckenstein_roots(&q, 1, RothRuckensteinLimits::new(10, 0)),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein output roots",
            required: 1,
            limit: 0,
        })
    );
}

#[test]
fn alekhnovich_matches_roth_on_shared_repeated_gf8_roots() {
    let first = polynomial::<Gf8>(&[gf8(0), gf8(17), gf8(0), gf8(91)]);
    let second = polynomial::<Gf8>(&[gf8(0), gf8(17), gf8(0), gf8(33)]);
    let short = polynomial::<Gf8>(&[gf8(0)]);
    let q = product_of_y_plus(&[first.clone(), second.clone(), first, short]);
    let expected = roth_ruckenstein_roots(&q, 3, GENEROUS).unwrap();
    let mut scratch = AlekhnovichScratch::new();
    let actual = alekhnovich_roots(&q, 3, ALEKHNOVICH_GENEROUS, &mut scratch).unwrap();

    assert_eq!(actual, expected);
    assert!(actual.iter().all(|root| q.has_root(root).unwrap()));
    assert!(actual.len() <= q.y_degree().unwrap());
    assert!(scratch.frame_capacity() > 0);
}

#[test]
fn sampled_gf16_factors_match_roth_and_are_order_independent() {
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    for case in 0..6 {
        let mut roots = Vec::new();
        for _ in 0..4 {
            let mut coefficients = Vec::new();
            for _ in 0..=5 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                coefficients.push(gf16((state >> 32) as u16));
            }
            roots.push(polynomial::<Gf16>(&coefficients));
        }
        if case % 2 == 0 {
            roots[3] = roots[0].clone();
        }
        let forward = product_of_y_plus(&roots);
        roots.reverse();
        let reverse = product_of_y_plus(&roots);
        let expected = roth_ruckenstein_roots(&forward, 5, GENEROUS).unwrap();
        let mut scratch = AlekhnovichScratch::new();
        let actual = alekhnovich_roots(&forward, 5, ALEKHNOVICH_GENEROUS, &mut scratch).unwrap();
        let reverse_actual =
            alekhnovich_roots(&reverse, 5, ALEKHNOVICH_GENEROUS, &mut scratch).unwrap();

        assert_eq!(actual, expected, "sampled GF16 case {case}");
        assert_eq!(reverse_actual, expected, "reversed GF16 case {case}");
        assert!(actual.len() <= forward.y_degree().unwrap());
    }
}

#[test]
fn alekhnovich_resource_limits_and_small_crossover_are_explicit() {
    let root = polynomial::<Gf8>(&[gf8(1), gf8(2), gf8(3), gf8(4)]);
    let q = product_of_y_plus(core::slice::from_ref(&root));
    let mut scratch = AlekhnovichScratch::new();
    let limits =
        AlekhnovichLimits::new(100, 100, 1_000, 1_000, 10).with_roth_ruckenstein_crossover(0);

    assert_eq!(
        alekhnovich_roots(
            &q,
            3,
            AlekhnovichLimits::new(100, 100, 7, 1_000, 10).with_roth_ruckenstein_crossover(0),
            &mut scratch,
        ),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich coefficients",
            required: 8,
            limit: 7,
        })
    );
    assert!(matches!(
        alekhnovich_roots(
            &q,
            3,
            AlekhnovichLimits::new(100, 100, 1_000, 7, 10)
                .with_roth_ruckenstein_crossover(0),
            &mut scratch,
        ),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich scratch bytes",
            required,
            limit: 7,
        }) if required > 7
    ));
    assert_eq!(
        alekhnovich_roots(
            &q,
            3,
            AlekhnovichLimits::new(0, 100, 1_000, 1_000, 10).with_roth_ruckenstein_crossover(0),
            &mut scratch,
        ),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich work items",
            required: 1,
            limit: 0,
        })
    );
    assert!(matches!(
        alekhnovich_roots(
            &q,
            3,
            AlekhnovichLimits::new(100, 0, 1_000, 1_000, 10).with_roth_ruckenstein_crossover(0),
            &mut scratch,
        ),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich intermediate families",
            ..
        })
    ));
    assert_eq!(
        alekhnovich_roots(
            &q,
            3,
            AlekhnovichLimits::new(100, 100, 1_000, 1_000, 0).with_roth_ruckenstein_crossover(0),
            &mut scratch,
        ),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich output roots",
            required: 1,
            limit: 0,
        })
    );

    let crossover_limits = AlekhnovichLimits::new(100, 0, 1_000, 1_000, 10);
    assert_eq!(
        alekhnovich_roots(&q, 3, crossover_limits, &mut scratch).unwrap(),
        vec![root]
    );
    assert!(limits.roth_ruckenstein_crossover() == 0);
}
