use fgf::field::Field;
use fgf::{Gf8, Gf16};
use gs_engine::{
    AlekhnovichLimits, DecodeError, DecodeScratch, DomainError, EvaluationBackend,
    EvaluationDomain, GsParameters, GsPlan, ParameterLimits, Polynomial,
};

const ROOT_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256);

fn gf8(value: u8) -> <Gf8 as Field>::Elem {
    Gf8::read(&[value])
}

fn gf16(value: u16) -> <Gf16 as Field>::Elem {
    Gf16::read(&value.to_le_bytes())
}

fn polynomial<F: fgf::kernel::FieldKernels>(coefficients: &[F::Elem]) -> Polynomial<F> {
    Polynomial::from_coefficients(coefficients).unwrap()
}

fn distance<F: fgf::kernel::FieldKernels>(
    candidate: &Polynomial<F>,
    points: &[F::Elem],
    received: &[F::Elem],
) -> usize {
    points
        .iter()
        .zip(received)
        .filter(|&(point, symbol)| candidate.evaluate(*point) != *symbol)
        .count()
}

#[test]
fn decodes_six_errors_beyond_unique_radius_for_15_5_rs() {
    let parameter_limits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);
    let parameters = GsParameters::new::<Gf16>(15, 4, 6, 2, 4, 17, parameter_limits).unwrap();
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

    let domain = EvaluationDomain::arbitrary(points.clone()).unwrap();
    let plan = GsPlan::new(parameters, domain, ROOT_LIMITS).unwrap();
    let mut scratch = DecodeScratch::new();
    let mut candidates = Vec::new();
    let count = plan
        .decode_into(&received, &mut scratch, &mut candidates)
        .unwrap();

    assert_eq!(count, candidates.len());
    assert!(candidates.contains(&message));
    assert!(
        candidates
            .iter()
            .all(|candidate| distance(candidate, &points, &received) <= 6)
    );

    let second_message = polynomial::<Gf16>(&[
        gf16(0x4444),
        gf16(0x3333),
        gf16(0x0201),
        gf16(0x7777),
        gf16(0x9999),
    ]);
    let mut second_received = second_message.evaluate_many(&points).unwrap();
    for (offset, value) in second_received[9..].iter_mut().enumerate() {
        *value = value.add(gf16((offset + 19) as u16));
    }
    let mut fresh = Vec::new();
    plan.decode_into(&second_received, &mut scratch, &mut candidates)
        .unwrap();
    plan.decode_into(&second_received, &mut DecodeScratch::new(), &mut fresh)
        .unwrap();
    assert_eq!(candidates, fresh);
    assert!(candidates.contains(&second_message));
}

#[test]
fn small_gf8_decode_matches_every_bounded_polynomial_and_cafft() {
    let parameter_limits = ParameterLimits::new(4, 8, usize::MAX, usize::MAX);
    let parameters = GsParameters::new::<Gf8>(4, 0, 2, 1, 2, 1, parameter_limits).unwrap();
    let cafft_domain = EvaluationDomain::<Gf8>::additive_subspace(4).unwrap();
    let points = cafft_domain.points().to_vec();
    let received = [gf8(7), gf8(7), gf8(9), gf8(9)];

    let arbitrary_plan = GsPlan::new(
        parameters,
        EvaluationDomain::arbitrary(points.clone()).unwrap(),
        ROOT_LIMITS,
    )
    .unwrap();
    let cafft_plan = GsPlan::new(parameters, cafft_domain, ROOT_LIMITS).unwrap();
    assert_eq!(arbitrary_plan.domain().backend(), EvaluationBackend::Horner);
    assert_eq!(
        cafft_plan.domain().backend(),
        EvaluationBackend::CafftAdditive
    );

    let mut exhaustive = Vec::new();
    for value in u8::MIN..=u8::MAX {
        let candidate = polynomial::<Gf8>(&[gf8(value)]);
        if distance(&candidate, &points, &received) <= parameters.target_radius() {
            exhaustive.push(candidate);
        }
    }
    let mut arbitrary = Vec::new();
    let mut accelerated = Vec::new();
    let mut arbitrary_scratch = DecodeScratch::new();
    let mut accelerated_scratch = DecodeScratch::new();
    arbitrary_plan
        .decode_into(&received, &mut arbitrary_scratch, &mut arbitrary)
        .unwrap();
    cafft_plan
        .decode_into(&received, &mut accelerated_scratch, &mut accelerated)
        .unwrap();

    assert_eq!(arbitrary, exhaustive);
    assert_eq!(accelerated, exhaustive);
    assert_eq!(accelerated.len(), 2);
}

#[test]
fn affine_coset_scoring_matches_horner() {
    let parameter_limits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);
    let parameters = GsParameters::search::<Gf8>(8, 2, 3, parameter_limits).unwrap();
    let coset = EvaluationDomain::<Gf8>::affine_coset(8, gf8(0x80)).unwrap();
    let points = coset.points().to_vec();
    let message = polynomial::<Gf8>(&[gf8(11), gf8(29), gf8(47)]);
    let mut received = message.evaluate_many(&points).unwrap();
    for (offset, value) in received[5..].iter_mut().enumerate() {
        *value = value.add(gf8((offset + 1) as u8));
    }

    let arbitrary = GsPlan::new(
        parameters,
        EvaluationDomain::arbitrary(points).unwrap(),
        ROOT_LIMITS,
    )
    .unwrap();
    let accelerated = GsPlan::new(parameters, coset, ROOT_LIMITS).unwrap();
    assert_eq!(
        accelerated.domain().backend(),
        EvaluationBackend::CafftAffineCoset
    );
    let mut arbitrary_roots = Vec::new();
    let mut accelerated_roots = Vec::new();
    arbitrary
        .decode_into(&received, &mut DecodeScratch::new(), &mut arbitrary_roots)
        .unwrap();
    accelerated
        .decode_into(&received, &mut DecodeScratch::new(), &mut accelerated_roots)
        .unwrap();

    assert_eq!(accelerated_roots, arbitrary_roots);
    assert!(accelerated_roots.contains(&message));
}

#[test]
fn domain_and_received_lengths_are_validated() {
    assert!(matches!(
        EvaluationDomain::<Gf8>::arbitrary(vec![gf8(1), gf8(2), gf8(1)]),
        Err(DomainError::DuplicatePoint {
            first: 0,
            second: 2,
        })
    ));

    let parameter_limits = ParameterLimits::new(4, 8, usize::MAX, usize::MAX);
    let parameters = GsParameters::new::<Gf8>(4, 0, 2, 1, 2, 1, parameter_limits).unwrap();
    let short_domain = EvaluationDomain::<Gf8>::arbitrary(vec![gf8(0), gf8(1)]).unwrap();
    assert!(matches!(
        GsPlan::new(parameters, short_domain, ROOT_LIMITS),
        Err(DecodeError::Domain(DomainError::LengthMismatch {
            expected: 4,
            got: 2,
        }))
    ));

    let plan = GsPlan::new(
        parameters,
        EvaluationDomain::<Gf8>::additive_subspace(4).unwrap(),
        ROOT_LIMITS,
    )
    .unwrap();
    let mut output = vec![polynomial::<Gf8>(&[gf8(99)])];
    assert_eq!(
        plan.decode_into(&[gf8(0); 3], &mut DecodeScratch::new(), &mut output),
        Err(DecodeError::ReceivedLength {
            expected: 4,
            got: 3,
        })
    );
    assert!(output.is_empty());
}
