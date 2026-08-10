//! Generate and freeze the canonical `.gsf` corpus.
//!
//! Every fixture's expected candidate set is produced by the `gs-engine`
//! decoder itself, then sorted into the canonical order defined in
//! `external-bench/fixtures/FORMAT.md`. Run as:
//!
//! ```text
//! cargo run --example generate_fixtures -- external-bench/fixtures
//! ```

use std::fmt::Write as _;
use std::path::Path;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;

use fgf::{Gf8, Gf16};
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
    Polynomial,
};

const ROOT_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256);
const PARAMETER_LIMITS: ParameterLimits = ParameterLimits::new(16, 32, usize::MAX, usize::MAX);

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

struct Case<'a, F: fgf::kernel::FieldKernels> {
    name: &'a str,
    field: &'a str,
    field_definition: &'a str,
    domain: &'a str,
    parameters: GsParameters,
    points: Vec<F::Elem>,
    received: Vec<F::Elem>,
}

fn render<F: fgf::kernel::FieldKernels>(
    case: &Case<'_, F>,
    candidates: &[Polynomial<F>],
) -> String {
    let mut out = String::new();
    writeln!(out, "gs-engine-fixture-v1").unwrap();
    writeln!(out, "name={}", case.name).unwrap();
    writeln!(out, "field={}", case.field).unwrap();
    writeln!(out, "field-definition={}", case.field_definition).unwrap();
    writeln!(out, "domain={}", case.domain).unwrap();
    writeln!(out, "n={}", case.parameters.code_length()).unwrap();
    writeln!(out, "k={}", case.parameters.max_degree() + 1).unwrap();
    writeln!(out, "target-radius={}", case.parameters.target_radius()).unwrap();
    writeln!(out, "multiplicity={}", case.parameters.multiplicity()).unwrap();
    writeln!(out, "y-degree={}", case.parameters.y_degree()).unwrap();
    writeln!(out, "weighted-degree={}", case.parameters.weighted_degree()).unwrap();
    writeln!(out, "support={}", encode_elements::<F>(&case.points)).unwrap();
    writeln!(out, "received={}", encode_elements::<F>(&case.received)).unwrap();
    for candidate in candidates {
        let coefficients: Vec<_> = candidate.coefficients().collect();
        writeln!(
            out,
            "expected-candidate={}",
            encode_elements::<F>(&coefficients)
        )
        .unwrap();
    }
    out
}

fn decode_sorted<F: FieldKernels + ButterflyKernels>(
    parameters: GsParameters,
    domain: &EvaluationDomain<F>,
    received: &[F::Elem],
) -> Vec<Polynomial<F>> {
    let plan = GsPlan::new(parameters, domain.clone(), ROOT_LIMITS).unwrap();
    let mut candidates = Vec::new();
    plan.decode_into(received, &mut DecodeScratch::new(), &mut candidates)
        .unwrap();
    candidates.sort_by(|left, right| {
        left.degree()
            .cmp(&right.degree())
            .then_with(|| left.as_packed().cmp(right.as_packed()))
    });
    candidates
}

fn emit<F: FieldKernels + ButterflyKernels>(
    dir: &Path,
    case: Case<'_, F>,
    domain: &EvaluationDomain<F>,
) {
    let candidates = decode_sorted::<F>(case.parameters, domain, &case.received);
    let rendered = render::<F>(&case, &candidates);
    let path = dir.join(format!("{}.gsf", case.name));
    std::fs::write(&path, rendered).unwrap();
    println!("wrote {} ({} candidates)", path.display(), candidates.len());
}

/// Evaluate `coefficients`, then add `error` to the received value at each
/// listed position, producing a deterministic corrupted word.
fn corrupt<F: FieldKernels>(
    message: &Polynomial<F>,
    points: &[F::Elem],
    errors: &[(usize, u64)],
) -> Vec<F::Elem> {
    let mut received = message.evaluate_many(points).unwrap();
    for &(position, delta) in errors {
        received[position] = received[position].add(element::<F>(delta));
    }
    received
}

fn gf8_additive(dir: &Path) {
    let parameters = GsParameters::new::<Gf8>(4, 0, 2, 1, 2, 1, PARAMETER_LIMITS).unwrap();
    let domain = EvaluationDomain::<Gf8>::additive_subspace(4).unwrap();
    let received = [
        element::<Gf8>(7),
        element::<Gf8>(7),
        element::<Gf8>(9),
        element::<Gf8>(9),
    ];
    emit::<Gf8>(
        dir,
        Case {
            name: "gf8-additive-4-1-radius-2",
            field: "gf8",
            field_definition: "gf2[x]/(x^8+x^4+x^3+x+1);le-polynomial-basis",
            domain: "additive",
            parameters,
            points: domain.points().to_vec(),
            received: received.to_vec(),
        },
        &domain,
    );
}

fn gf8_arbitrary(dir: &Path) {
    let parameters = GsParameters::search::<Gf8>(7, 2, 1, PARAMETER_LIMITS).unwrap();
    let points: Vec<_> = (0..7).map(element::<Gf8>).collect();
    let message = Polynomial::<Gf8>::from_coefficients(&[
        element::<Gf8>(0x1f),
        element::<Gf8>(0x2a),
        element::<Gf8>(0x33),
    ])
    .unwrap();
    let received = corrupt::<Gf8>(&message, &points, &[(4, 0x11)]);
    let domain = EvaluationDomain::arbitrary(points).unwrap();
    emit::<Gf8>(
        dir,
        Case {
            name: "gf8-arbitrary-7-3-radius-1",
            field: "gf8",
            field_definition: "gf2[x]/(x^8+x^4+x^3+x+1);le-polynomial-basis",
            domain: "arbitrary",
            parameters,
            points: domain.points().to_vec(),
            received,
        },
        &domain,
    );
}

fn gf16_arbitrary_radius6(dir: &Path) {
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
    emit::<Gf16>(
        dir,
        Case {
            name: "gf16-arbitrary-15-5-radius-6",
            field: "gf16",
            field_definition: "gf8[u]/(u^2+u+0x20);gf8=aes-0x11b;le-components",
            domain: "arbitrary",
            parameters,
            points: domain.points().to_vec(),
            received,
        },
        &domain,
    );
}

fn gf16_arbitrary_radius4(dir: &Path) {
    let parameters = GsParameters::search::<Gf16>(15, 4, 4, PARAMETER_LIMITS).unwrap();
    let points: Vec<_> = (1..=15).map(element::<Gf16>).collect();
    let message = Polynomial::<Gf16>::from_coefficients(&[
        element::<Gf16>(0x0001),
        element::<Gf16>(0x8000),
        element::<Gf16>(0x1357),
        element::<Gf16>(0x2468),
        element::<Gf16>(0xace0),
    ])
    .unwrap();
    let received = corrupt::<Gf16>(
        &message,
        &points,
        &[(2, 0x3), (6, 0x5), (11, 0x9), (14, 0x2)],
    );
    let domain = EvaluationDomain::arbitrary(points).unwrap();
    emit::<Gf16>(
        dir,
        Case {
            name: "gf16-arbitrary-15-5-radius-4",
            field: "gf16",
            field_definition: "gf8[u]/(u^2+u+0x20);gf8=aes-0x11b;le-components",
            domain: "arbitrary",
            parameters,
            points: domain.points().to_vec(),
            received,
        },
        &domain,
    );
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: generate_fixtures <output-dir>");
        std::process::exit(2);
    });
    let dir = Path::new(&dir);
    std::fs::create_dir_all(dir).unwrap();
    gf8_additive(dir);
    gf8_arbitrary(dir);
    gf16_arbitrary_radius6(dir);
    gf16_arbitrary_radius4(dir);
}
