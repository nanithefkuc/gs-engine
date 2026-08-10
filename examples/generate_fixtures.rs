use std::fmt::Write as _;

use fgf::field::Field;
use fgf::{Gf8, Gf16};
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
    Polynomial,
};

const ROOT_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256);
const PARAMETER_LIMITS: ParameterLimits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);

fn element<F: Field>(value: u64) -> F::Elem {
    F::read(&value.to_le_bytes()[..F::BYTES])
}

fn encode_element<F: Field>(value: F::Elem) -> String {
    let mut bytes = vec![0_u8; F::BYTES];
    F::write(&mut bytes, value);
    let mut encoded = String::with_capacity(F::BYTES * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").unwrap();
    }
    encoded
}

fn encode_elements<F: Field>(values: &[F::Elem]) -> String {
    values
        .iter()
        .map(|&value| encode_element::<F>(value))
        .collect::<Vec<_>>()
        .join(",")
}

struct FixtureView<'a, F: fgf::kernel::FieldKernels> {
    name: &'a str,
    field: &'a str,
    field_definition: &'a str,
    domain: &'a str,
    parameters: GsParameters,
    points: &'a [F::Elem],
    received: &'a [F::Elem],
    candidates: &'a [Polynomial<F>],
}

fn render<F: fgf::kernel::FieldKernels>(fixture: FixtureView<'_, F>) {
    println!("gs-engine-fixture-v1");
    println!("name={}", fixture.name);
    println!("field={}", fixture.field);
    println!("field-definition={}", fixture.field_definition);
    println!("domain={}", fixture.domain);
    println!("n={}", fixture.parameters.code_length());
    println!("k={}", fixture.parameters.max_degree() + 1);
    println!("target-radius={}", fixture.parameters.target_radius());
    println!("multiplicity={}", fixture.parameters.multiplicity());
    println!("y-degree={}", fixture.parameters.y_degree());
    println!("weighted-degree={}", fixture.parameters.weighted_degree());
    println!("support={}", encode_elements::<F>(fixture.points));
    println!("received={}", encode_elements::<F>(fixture.received));
    for candidate in fixture.candidates {
        let coefficients: Vec<_> = candidate.coefficients().collect();
        println!("expected-candidate={}", encode_elements::<F>(&coefficients));
    }
}

fn gf8_fixture() {
    let parameters = GsParameters::new::<Gf8>(4, 0, 2, 1, 2, 1, PARAMETER_LIMITS).unwrap();
    let domain = EvaluationDomain::<Gf8>::additive_subspace(4).unwrap();
    let received = [
        element::<Gf8>(7),
        element::<Gf8>(7),
        element::<Gf8>(9),
        element::<Gf8>(9),
    ];
    let plan = GsPlan::new(parameters, domain.clone(), ROOT_LIMITS).unwrap();
    let mut candidates = Vec::new();
    plan.decode_into(&received, &mut DecodeScratch::new(), &mut candidates)
        .unwrap();
    candidates.sort_by(|left, right| {
        left.degree()
            .cmp(&right.degree())
            .then_with(|| left.as_packed().cmp(right.as_packed()))
    });
    render::<Gf8>(FixtureView {
        name: "gf8-additive-4-1-radius-2",
        field: "gf8",
        field_definition: "gf2[x]/(x^8+x^4+x^3+x+1);le-polynomial-basis",
        domain: "additive",
        parameters,
        points: domain.points(),
        received: &received,
        candidates: &candidates,
    });
}

fn gf16_fixture() {
    let parameters = GsParameters::new::<Gf16>(15, 4, 6, 2, 4, 17, PARAMETER_LIMITS).unwrap();
    let points: Vec<_> = (0..15).map(element::<Gf16>).collect();
    let message = Polynomial::<Gf16>::from_coefficients(&[
        element::<Gf16>(0x1234),
        element::<Gf16>(0xabcd),
        element::<Gf16>(0x0108),
        element::<Gf16>(0xbeef),
        element::<Gf16>(0x2222),
    ])
    .unwrap();
    let mut received = message.evaluate_many(&points).unwrap();
    for (offset, value) in received[9..].iter_mut().enumerate() {
        *value = value.add(element::<Gf16>((offset + 1) as u64));
    }
    let domain = EvaluationDomain::arbitrary(points).unwrap();
    let plan = GsPlan::new(parameters, domain.clone(), ROOT_LIMITS).unwrap();
    let mut candidates = Vec::new();
    plan.decode_into(&received, &mut DecodeScratch::new(), &mut candidates)
        .unwrap();
    candidates.sort_by(|left, right| {
        left.degree()
            .cmp(&right.degree())
            .then_with(|| left.as_packed().cmp(right.as_packed()))
    });
    render::<Gf16>(FixtureView {
        name: "gf16-arbitrary-15-5-radius-6",
        field: "gf16",
        field_definition: "gf8[u]/(u^2+u+0x20);gf8=aes-0x11b;le-components",
        domain: "arbitrary",
        parameters,
        points: domain.points(),
        received: &received,
        candidates: &candidates,
    });
}

fn main() {
    gf8_fixture();
    println!();
    gf16_fixture();
}
