use fgf::field::Field;
use fgf::kernel::FieldKernels;
use fgf::{Gf8, Gf16};
use gs_engine::{Polynomial, PolynomialProductScratch, ProductStrategy, multiply_batch_truncated};

fn generated<F: FieldKernels>(count: usize, mut state: u64) -> Polynomial<F> {
    let coefficients: Vec<_> = (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bytes = state.to_le_bytes();
            F::read(&bytes[..F::BYTES])
        })
        .collect();
    Polynomial::from_coefficients(&coefficients).unwrap()
}

fn compare_forced<F: butterfly_fft::core::kernel::ButterflyKernels>(
    left: &Polynomial<F>,
    right: &Polynomial<F>,
    coefficient_count: usize,
) {
    let mut scratch = PolynomialProductScratch::new();
    let mut schoolbook = Vec::new();
    let mut afft = Vec::new();
    multiply_batch_truncated(
        &[(left, right)],
        coefficient_count,
        ProductStrategy::Schoolbook,
        &mut scratch,
        &mut schoolbook,
    )
    .unwrap();
    multiply_batch_truncated(
        &[(left, right)],
        coefficient_count,
        ProductStrategy::Afft,
        &mut scratch,
        &mut afft,
    )
    .unwrap();
    assert_eq!(afft.len(), schoolbook.len());
    for (afft, schoolbook) in afft.iter().zip(&schoolbook) {
        assert_eq!(afft.as_packed(), schoolbook.as_packed());
    }
}

#[test]
fn forced_afft_matches_schoolbook_over_gf8_and_gf16() {
    let left8 = generated::<Gf8>(100, 1);
    let right8 = generated::<Gf8>(89, 2);
    compare_forced(&left8, &right8, 188);
    compare_forced(&left8, &right8, 117);

    let left16 = generated::<Gf16>(301, 3);
    let right16 = generated::<Gf16>(270, 4);
    compare_forced(&left16, &right16, 570);
    compare_forced(&left16, &right16, 333);
}

#[test]
fn forced_products_cover_zero_unbalanced_truncation_and_cancellation() {
    let zero = Polynomial::<Gf8>::zero();
    let one = Polynomial::<Gf8>::one().unwrap();
    compare_forced(&zero, &one, 1);

    let unbalanced_left = generated::<Gf16>(127, 11);
    let unbalanced_right = generated::<Gf16>(3, 12);
    compare_forced(&unbalanced_left, &unbalanced_right, 129);
    compare_forced(&unbalanced_left, &unbalanced_right, 1);

    let one16 = <Gf16 as Field>::Elem::ONE;
    let cancelling = Polynomial::<Gf16>::from_coefficients(&[one16, one16]).unwrap();
    let truncated = cancelling.multiply_truncated(&cancelling, 2).unwrap();
    assert_eq!(truncated, Polynomial::one().unwrap());
    compare_forced(&cancelling, &cancelling, 2);
}

#[test]
fn batched_rows_match_independent_products() {
    let left: Vec<_> = (0..8)
        .map(|index| generated::<Gf16>(35 + index, index as u64 + 10))
        .collect();
    let right: Vec<_> = (0..8)
        .map(|index| generated::<Gf16>(29 + index, index as u64 + 100))
        .collect();
    let pairs: Vec<_> = left.iter().zip(&right).collect();
    let mut scratch = PolynomialProductScratch::new();
    let mut actual = Vec::new();
    multiply_batch_truncated(&pairs, 61, ProductStrategy::Afft, &mut scratch, &mut actual).unwrap();

    let expected: Vec<_> = pairs
        .iter()
        .map(|&(left, right)| left.multiply_truncated(right, 61).unwrap())
        .collect();
    assert_eq!(actual, expected);
    assert!(scratch.operand_capacity_bytes() >= 128 * 8 * 2 * Gf16::BYTES);
    assert!(scratch.product_capacity_bytes() >= 128 * 8 * Gf16::BYTES);
}

#[test]
fn zero_truncated_and_auto_products_are_canonical() {
    let zero = Polynomial::<Gf8>::zero();
    let value = generated::<Gf8>(31, 9);
    let mut scratch = PolynomialProductScratch::new();
    let mut output = Vec::new();
    multiply_batch_truncated(
        &[(&zero, &value), (&value, &zero), (&value, &value)],
        0,
        ProductStrategy::Auto,
        &mut scratch,
        &mut output,
    )
    .unwrap();
    assert_eq!(output, vec![zero.clone(), zero.clone(), zero.clone()]);

    multiply_batch_truncated(
        &[(&value, &value), (&value, &value), (&value, &value)],
        61,
        ProductStrategy::Auto,
        &mut scratch,
        &mut output,
    )
    .unwrap();
    let expected = value.multiply(&value).unwrap();
    assert_eq!(output, vec![expected.clone(), expected.clone(), expected]);
}
