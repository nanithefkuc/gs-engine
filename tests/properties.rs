use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8, Gf16};
use gs_engine::{BaseFieldRoots, Polynomial, base_field_roots};

fn element<F: Field>(value: u64) -> F::Elem {
    let bytes = value.to_le_bytes();
    F::read(&bytes[..F::BYTES])
}

fn generated<F: FieldKernels>(mut state: u64, count: usize) -> Polynomial<F> {
    let coefficients: Vec<_> = (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            element::<F>(state)
        })
        .collect();
    Polynomial::from_coefficients(&coefficients).unwrap()
}

fn assert_evaluation_homomorphisms<F: FieldKernels>() {
    for seed in 1_u64..=128 {
        let left = generated::<F>(seed, (seed as usize % 9) + 1);
        let right = generated::<F>(seed ^ 0xa5a5_5a5a, (seed as usize % 7) + 1);
        let point = element::<F>(seed.wrapping_mul(257).wrapping_add(19));
        let sum = left.add(&right).unwrap();
        let product = left.multiply(&right).unwrap();

        assert_eq!(
            sum.evaluate(point),
            left.evaluate(point).add(right.evaluate(point))
        );
        assert_eq!(
            product.evaluate(point),
            left.evaluate(point).mul(right.evaluate(point))
        );
        if !left.is_zero() {
            assert_eq!(product.exact_divide(&left).unwrap(), right);
        }
    }
}

#[test]
fn generated_polynomial_arithmetic_preserves_evaluation_over_gf8_and_gf16() {
    assert_evaluation_homomorphisms::<Gf8>();
    assert_evaluation_homomorphisms::<Gf16>();
}

fn assert_generated_root_sets<F: FieldKernels>() {
    for seed in 1_u64..=48 {
        let roots = [
            element::<F>(seed),
            element::<F>(seed.wrapping_mul(17)),
            element::<F>(seed.wrapping_mul(29)),
            element::<F>(seed),
        ];
        let mut polynomial = Polynomial::<F>::one().unwrap();
        for root in roots {
            let factor = Polynomial::<F>::from_coefficients(&[root, F::Elem::ONE]).unwrap();
            polynomial = polynomial.multiply(&factor).unwrap();
        }
        let BaseFieldRoots::Finite(actual) = base_field_roots(&polynomial).unwrap() else {
            panic!("a nonzero generated polynomial has a finite root set");
        };

        assert!(
            actual
                .iter()
                .all(|root| polynomial.evaluate(*root).is_zero())
        );
        let mut expected = Vec::new();
        for root in &roots[..3] {
            if !expected.contains(root) {
                expected.push(*root);
            }
        }
        assert_eq!(actual.len(), expected.len());
        assert!(expected.iter().all(|root| actual.contains(root)));
    }
}

#[test]
fn generated_split_factors_return_exact_distinct_roots() {
    assert_generated_root_sets::<Gf8>();
    assert_generated_root_sets::<Gf16>();
}
