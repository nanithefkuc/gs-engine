use fff::field::{Elem, Field};
use fff::kernel::FieldKernels;
use fff::{Gf8, Gf16};
use gs_engine::{BaseFieldRoots, Polynomial, base_field_roots};

fn gf8(value: u8) -> <Gf8 as Field>::Elem {
    Gf8::read(&[value])
}

fn gf16(value: u16) -> <Gf16 as Field>::Elem {
    Gf16::read(&value.to_le_bytes())
}

fn linear<F: FieldKernels>(root: F::Elem) -> Polynomial<F> {
    Polynomial::from_coefficients(&[root, F::Elem::ONE]).unwrap()
}

fn finite<F: FieldKernels>(polynomial: &Polynomial<F>) -> Vec<F::Elem> {
    match base_field_roots(polynomial).unwrap() {
        BaseFieldRoots::All => panic!("expected a nonzero polynomial"),
        BaseFieldRoots::Finite(roots) => roots,
    }
}

fn exhaustive_gf8(polynomial: &Polynomial<Gf8>) -> Vec<<Gf8 as Field>::Elem> {
    (u8::MIN..=u8::MAX)
        .map(gf8)
        .filter(|root| polynomial.evaluate(*root).is_zero())
        .collect()
}

fn exhaustive_gf16(polynomial: &Polynomial<Gf16>) -> Vec<<Gf16 as Field>::Elem> {
    (u16::MIN..=u16::MAX)
        .map(gf16)
        .filter(|root| polynomial.evaluate(*root).is_zero())
        .collect()
}

#[test]
fn zero_constant_linear_and_rootless_cases_are_explicit() {
    assert_eq!(
        base_field_roots(&Polynomial::<Gf8>::zero()).unwrap(),
        BaseFieldRoots::All
    );
    assert_eq!(
        finite(&Polynomial::<Gf8>::constant(gf8(17)).unwrap()),
        Vec::new()
    );
    assert_eq!(finite(&linear::<Gf8>(gf8(0))), vec![gf8(0)]);
    assert_eq!(finite(&linear::<Gf8>(gf8(91))), vec![gf8(91)]);

    let rootless = (u8::MIN..=u8::MAX)
        .map(gf8)
        .find_map(|constant| {
            let candidate =
                Polynomial::<Gf8>::from_coefficients(&[constant, gf8(1), gf8(1)]).unwrap();
            exhaustive_gf8(&candidate).is_empty().then_some(candidate)
        })
        .expect("GF8 contains a trace-one quadratic constant");
    assert!(finite(&rootless).is_empty());
}

#[test]
fn all_linear_repeated_and_inseparable_factors_are_deduplicated() {
    let expected: Vec<_> = [0_u8, 1, 2, 7, 19, 31, 63, 127, 128, 201, 254, 255]
        .into_iter()
        .map(gf8)
        .collect();
    let mut all_linear = Polynomial::<Gf8>::one().unwrap();
    for root in &expected {
        all_linear = all_linear.multiply(&linear::<Gf8>(*root)).unwrap();
    }
    assert_eq!(finite(&all_linear), expected);

    let repeated_root = gf8(0xa7);
    let repeated_linear = linear::<Gf8>(repeated_root);
    let squared = repeated_linear.multiply(&repeated_linear).unwrap();
    let fourth_power = squared.multiply(&squared).unwrap();
    assert!(fourth_power.formal_derivative().unwrap().is_zero());
    assert_eq!(finite(&fourth_power), vec![repeated_root]);

    let mixed = fourth_power
        .multiply(&linear::<Gf8>(gf8(3)))
        .unwrap()
        .multiply(&linear::<Gf8>(gf8(240)))
        .unwrap();
    assert_eq!(finite(&mixed), vec![gf8(3), repeated_root, gf8(240)]);
}

#[test]
fn generated_gf8_polynomials_match_exhaustive_enumeration() {
    let mut state = 0x6d2b_79f5_u32;
    for case in 0..96 {
        let degree = 1 + case % 12;
        let mut coefficients = Vec::with_capacity(degree + 1);
        for _ in 0..=degree {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            coefficients.push(gf8((state >> 24) as u8));
        }
        if coefficients[degree].is_zero() {
            coefficients[degree] = gf8(1);
        }
        let polynomial = Polynomial::<Gf8>::from_coefficients(&coefficients).unwrap();
        let expected = exhaustive_gf8(&polynomial);
        let actual = finite(&polynomial);
        assert_eq!(actual, expected, "generated GF8 case {case}");
        assert!(
            actual
                .iter()
                .all(|root| polynomial.evaluate(*root).is_zero())
        );
    }
}

#[test]
fn sampled_gf16_polynomials_match_exhaustive_enumeration() {
    let mut state = 0x9e37_79b9_u32;
    for case in 0..8 {
        let degree = 1 + case % 7;
        let mut coefficients = Vec::with_capacity(degree + 1);
        for _ in 0..=degree {
            state = state.wrapping_mul(22_695_477).wrapping_add(1);
            coefficients.push(gf16((state >> 8) as u16));
        }
        if coefficients[degree].is_zero() {
            coefficients[degree] = gf16(1);
        }
        let polynomial = Polynomial::<Gf16>::from_coefficients(&coefficients).unwrap();
        let expected = exhaustive_gf16(&polynomial);
        let actual = finite(&polynomial);
        assert_eq!(actual, expected, "sampled GF16 case {case}");
        assert!(
            actual
                .iter()
                .all(|root| polynomial.evaluate(*root).is_zero())
        );
    }
}
