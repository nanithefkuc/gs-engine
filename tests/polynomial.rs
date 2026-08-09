use cafft::core::kernel::ButterflyKernels;
use fgf::field::{Elem, Field};
use fgf::kernel::backend_for;
use fgf::{Gf8, Gf16};
use gs_engine::{BivariatePolynomial, Polynomial, PolynomialError, WeightedTerm};

fn powers<F: ButterflyKernels>(start: usize, count: usize) -> Vec<F::Elem> {
    (start..start + count)
        .map(|exponent| F::GENERATOR.pow(exponent as u64))
        .collect()
}

fn polynomial<F: ButterflyKernels>(coefficients: &[F::Elem]) -> Polynomial<F> {
    Polynomial::from_coefficients(coefficients).expect("small polynomial")
}

fn assert_univariate_identities<F: ButterflyKernels>() {
    let p = polynomial::<F>(&powers::<F>(1, 5));
    let q = polynomial::<F>(&powers::<F>(11, 3));
    let scale = F::GENERATOR.pow(29);
    let points = powers::<F>(41, 7);

    let sum = p.add(&q).unwrap();
    let axpy = p.add_scaled(scale, &q).unwrap();
    let product = p.multiply(&q).unwrap();
    let mut truncated = product.clone();
    truncated.truncate(5);
    assert_eq!(p.multiply_truncated(&q, 5).unwrap(), truncated);
    assert_eq!(
        p.evaluate_many(&points).unwrap(),
        points
            .iter()
            .copied()
            .map(|point| p.evaluate(point))
            .collect::<Vec<_>>()
    );

    for &point in &points {
        assert_eq!(
            sum.evaluate(point),
            p.evaluate(point).add(q.evaluate(point))
        );
        assert_eq!(
            axpy.evaluate(point),
            p.evaluate(point).add(scale.mul(q.evaluate(point)))
        );
        assert_eq!(
            product.evaluate(point),
            p.evaluate(point).mul(q.evaluate(point))
        );
        assert_eq!(
            p.shifted(3).unwrap().evaluate(point),
            point.pow(3).mul(p.evaluate(point))
        );
        assert_eq!(
            p.multiply_x_plus(scale).unwrap().evaluate(point),
            point.add(scale).mul(p.evaluate(point))
        );
    }

    assert_eq!(product.exact_divide(&p).unwrap(), q);
    assert_eq!(product.exact_divide(&q).unwrap(), p);
    assert_eq!(p.exact_divide(&q), Err(PolynomialError::NonExactDivision));
    assert_eq!(
        p.div_rem(&Polynomial::zero()),
        Err(PolynomialError::DivisionByZero)
    );

    let common = polynomial::<F>(&[scale, F::Elem::ONE]);
    let left = common
        .multiply(&polynomial::<F>(&powers::<F>(70, 3)))
        .unwrap();
    let right = common
        .multiply(&polynomial::<F>(&powers::<F>(90, 2)))
        .unwrap();
    assert_eq!(left.gcd(&right).unwrap(), common.monic());

    let modulus = polynomial::<F>(&[
        F::Elem::ONE,
        F::GENERATOR.pow(3),
        F::Elem::ZERO,
        F::Elem::ONE,
    ]);
    let squared = p.multiply_mod(&p, &modulus).unwrap();
    assert_eq!(p.square_mod(&modulus).unwrap(), squared);
    assert_eq!(p.pow_mod(2, &modulus).unwrap(), squared);
    assert_eq!(p.pow_mod(0, &modulus).unwrap(), Polynomial::one().unwrap());
}

#[test]
fn univariate_identities_hold_over_both_fields() {
    assert_univariate_identities::<Gf8>();
    assert_univariate_identities::<Gf16>();
}

#[test]
fn representation_derivatives_and_composition_are_canonical() {
    type F = Gf16;
    let one = <F as Field>::Elem::ONE;
    let zero = <F as Field>::Elem::ZERO;
    let a = F::GENERATOR.pow(7);
    let b = F::GENERATOR.pow(13);
    let p = polynomial::<F>(&[one, a, b, one, zero, zero]);

    assert_eq!(p.coefficient_count(), 4);
    assert_eq!(p.degree(), Some(3));
    assert_eq!(Polynomial::<F>::from_packed(vec![0]), None);

    let first = p.formal_derivative().unwrap();
    assert_eq!(first, polynomial::<F>(&[a, zero, one]));
    let second_hasse = p.hasse_derivative(2).unwrap();
    assert_eq!(second_hasse, polynomial::<F>(&[b, one]));
    for point in powers::<F>(20, 5) {
        assert_eq!(first.evaluate(point), p.evaluate_hasse(point, 1));
        assert_eq!(second_hasse.evaluate(point), p.evaluate_hasse(point, 2));
    }

    let composed = p.compose_linear(a, b).unwrap();
    for point in powers::<F>(30, 5) {
        assert_eq!(composed.evaluate(point), p.evaluate(a.add(b.mul(point))));
    }

    let shifted = polynomial::<F>(&[zero, zero, a, b]);
    assert_eq!(shifted.x_valuation(), Some(2));
    assert_eq!(
        shifted.divide_by_x_power(2).unwrap(),
        polynomial::<F>(&[a, b])
    );
    assert_eq!(
        shifted.divide_by_x_power(3),
        Err(PolynomialError::NonExactDivision)
    );

    let mut normalized = p.clone();
    normalized.set_coefficient(3, zero).unwrap();
    assert_eq!(normalized.degree(), Some(2));
}

fn assert_axpy_boundaries<F: ButterflyKernels>() {
    let lane_elements = backend_for::<F>().lane_bytes().div_ceil(F::BYTES);
    let counts = [
        1,
        lane_elements.saturating_sub(1).max(1),
        lane_elements,
        lane_elements + 1,
        lane_elements * 2 + 1,
    ];
    let scale = F::GENERATOR.pow(37);

    for count in counts {
        let left = powers::<F>(1, count);
        let right = powers::<F>(count + 11, count);
        let expected: Vec<_> = left
            .iter()
            .zip(&right)
            .map(|(&a, &b)| a.add(scale.mul(b)))
            .collect();
        let mut actual = polynomial::<F>(&left);
        actual
            .add_scaled_assign(scale, &polynomial::<F>(&right))
            .unwrap();
        assert_eq!(actual, polynomial::<F>(&expected), "count {count}");
    }
}

#[test]
fn scalar_and_packed_axpy_agree_at_lane_boundaries() {
    assert_axpy_boundaries::<Gf8>();
    assert_axpy_boundaries::<Gf16>();
}

fn direct_hasse<F: ButterflyKernels>(
    polynomial: &BivariatePolynomial<F>,
    x: F::Elem,
    y: F::Elem,
    x_order: usize,
    y_order: usize,
) -> F::Elem {
    let mut result = F::Elem::ZERO;
    for (y_degree, row) in polynomial.y_coefficients().iter().enumerate() {
        for (x_degree, coefficient) in row.coefficients().enumerate() {
            if (x_degree & x_order) == x_order && (y_degree & y_order) == y_order {
                result = result.add(
                    coefficient
                        .mul(x.pow((x_degree - x_order) as u64))
                        .mul(y.pow((y_degree - y_order) as u64)),
                );
            }
        }
    }
    result
}

#[test]
fn bivariate_weight_hasse_substitution_and_roots_agree_directly() {
    type F = Gf16;
    let rows = vec![
        polynomial::<F>(&powers::<F>(1, 4)),
        polynomial::<F>(&powers::<F>(10, 2)),
        polynomial::<F>(&powers::<F>(20, 3)),
    ];
    let q = BivariatePolynomial::from_y_coefficients(rows);
    assert_eq!(
        q.weighted_leading_term(4).unwrap(),
        Some(WeightedTerm {
            x_degree: 2,
            y_degree: 2,
            weighted_degree: 10,
        })
    );

    let x = F::GENERATOR.pow(31);
    let y = F::GENERATOR.pow(47);
    for x_order in 0..=3 {
        for y_order in 0..=2 {
            assert_eq!(
                q.hasse_discrepancy(x, y, x_order, y_order),
                direct_hasse(&q, x, y, x_order, y_order)
            );
        }
    }

    let constant = F::GENERATOR.pow(59);
    let multiplied = q.multiply_x_plus(constant).unwrap();
    assert_eq!(
        multiplied.evaluate(x, y),
        x.add(constant).mul(q.evaluate(x, y))
    );
    let substituted = q.substitute_y_linear(constant).unwrap();
    for x in powers::<F>(70, 4) {
        for z in powers::<F>(90, 3) {
            assert_eq!(
                substituted.evaluate(x, z),
                q.evaluate(x, constant.add(x.mul(z)))
            );
        }
    }

    let candidate = polynomial::<F>(&powers::<F>(100, 4));
    let root_polynomial = BivariatePolynomial::from_y_coefficients(vec![
        candidate.clone(),
        Polynomial::one().unwrap(),
    ]);
    assert!(root_polynomial.has_root(&candidate).unwrap());
    assert_eq!(
        root_polynomial.compose_y(&candidate).unwrap(),
        Polynomial::zero()
    );

    let shifted = BivariatePolynomial::from_y_coefficients(vec![
        candidate.shifted(3).unwrap(),
        candidate.shifted(2).unwrap(),
    ]);
    assert_eq!(shifted.x_valuation(), Some(2));
    assert_eq!(shifted.divide_by_x_power(2).unwrap().x_valuation(), Some(0));
}
