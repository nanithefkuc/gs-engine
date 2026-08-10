use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Field;
use fgf::{Gf8, Gf16};
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
    Polynomial,
};

const ROOT_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256);
const PARAMETER_LIMITS: ParameterLimits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);

#[derive(Debug)]
struct Fixture<'a> {
    name: &'a str,
    field: &'a str,
    field_definition: &'a str,
    domain: &'a str,
    n: usize,
    k: usize,
    target_radius: usize,
    multiplicity: usize,
    y_degree: usize,
    weighted_degree: usize,
    support: &'a str,
    received: &'a str,
    expected_candidates: Vec<&'a str>,
}

fn value<'a>(line: &'a str, key: &str) -> &'a str {
    line.strip_prefix(key)
        .unwrap_or_else(|| panic!("expected fixture key {key}, got {line}"))
}

fn decimal(line: &str, key: &str) -> usize {
    value(line, key).parse().expect("canonical decimal integer")
}

fn parse_fixture(input: &str) -> Fixture<'_> {
    assert!(input.ends_with('\n'));
    let mut lines = input.lines();
    assert_eq!(lines.next(), Some("gs-engine-fixture-v1"));
    let name = value(lines.next().unwrap(), "name=");
    let field = value(lines.next().unwrap(), "field=");
    let field_definition = value(lines.next().unwrap(), "field-definition=");
    let domain = value(lines.next().unwrap(), "domain=");
    let n = decimal(lines.next().unwrap(), "n=");
    let k = decimal(lines.next().unwrap(), "k=");
    let target_radius = decimal(lines.next().unwrap(), "target-radius=");
    let multiplicity = decimal(lines.next().unwrap(), "multiplicity=");
    let y_degree = decimal(lines.next().unwrap(), "y-degree=");
    let weighted_degree = decimal(lines.next().unwrap(), "weighted-degree=");
    let support = value(lines.next().unwrap(), "support=");
    let received = value(lines.next().unwrap(), "received=");
    let expected_candidates = lines
        .map(|line| value(line, "expected-candidate="))
        .collect();
    Fixture {
        name,
        field,
        field_definition,
        domain,
        n,
        k,
        target_radius,
        multiplicity,
        y_degree,
        weighted_degree,
        support,
        received,
        expected_candidates,
    }
}

fn decode_element<F: Field>(encoded: &str) -> F::Elem {
    assert_eq!(encoded.len(), F::BYTES * 2);
    assert!(
        encoded
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    );
    let bytes: Vec<_> = encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let pair = std::str::from_utf8(pair).unwrap();
            u8::from_str_radix(pair, 16).unwrap()
        })
        .collect();
    F::read(&bytes)
}

fn decode_elements<F: Field>(encoded: &str) -> Vec<F::Elem> {
    encoded.split(',').map(decode_element::<F>).collect()
}

fn validate<F: ButterflyKernels>(fixture: &Fixture<'_>, field: &str, definition: &str) {
    assert_eq!(fixture.field, field);
    assert_eq!(fixture.field_definition, definition);
    assert!(!fixture.name.is_empty());
    let support = decode_elements::<F>(fixture.support);
    let received = decode_elements::<F>(fixture.received);
    assert_eq!(support.len(), fixture.n);
    assert_eq!(received.len(), fixture.n);
    let domain = match fixture.domain {
        "arbitrary" => EvaluationDomain::arbitrary(support.clone()).unwrap(),
        "additive" => {
            let domain = EvaluationDomain::additive_subspace(fixture.n).unwrap();
            assert_eq!(domain.points(), support);
            domain
        }
        "affine" => {
            let domain = EvaluationDomain::affine_coset(fixture.n, support[0]).unwrap();
            assert_eq!(domain.points(), support);
            domain
        }
        other => panic!("unsupported fixture domain {other}"),
    };
    let parameters = GsParameters::new::<F>(
        fixture.n,
        fixture.k - 1,
        fixture.target_radius,
        fixture.multiplicity,
        fixture.y_degree,
        fixture.weighted_degree,
        PARAMETER_LIMITS,
    )
    .unwrap();
    let expected: Vec<_> = fixture
        .expected_candidates
        .iter()
        .map(|encoded| Polynomial::<F>::from_coefficients(&decode_elements::<F>(encoded)).unwrap())
        .collect();
    assert!(expected.windows(2).all(|pair| {
        pair[0].degree() < pair[1].degree()
            || (pair[0].degree() == pair[1].degree() && pair[0].as_packed() < pair[1].as_packed())
    }));

    let plan = GsPlan::new(parameters, domain, ROOT_LIMITS).unwrap();
    let mut actual = Vec::new();
    plan.decode_into(&received, &mut DecodeScratch::new(), &mut actual)
        .unwrap();
    actual.sort_by(|left, right| {
        left.degree()
            .cmp(&right.degree())
            .then_with(|| left.as_packed().cmp(right.as_packed()))
    });
    assert_eq!(actual, expected);
}

#[test]
fn canonical_gf8_candidate_set_is_frozen() {
    let fixture = parse_fixture(include_str!(
        "../external-bench/fixtures/gf8-additive-4-1-radius-2.gsf"
    ));
    validate::<Gf8>(
        &fixture,
        "gf8",
        "gf2[x]/(x^8+x^4+x^3+x+1);le-polynomial-basis",
    );
}

#[test]
fn canonical_gf16_candidate_set_is_frozen() {
    let fixture = parse_fixture(include_str!(
        "../external-bench/fixtures/gf16-arbitrary-15-5-radius-6.gsf"
    ));
    validate::<Gf16>(
        &fixture,
        "gf16",
        "gf8[u]/(u^2+u+0x20);gf8=aes-0x11b;le-components",
    );
}
