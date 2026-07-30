use std::hint::black_box;
use std::time::{Duration, Instant};

use cafft::core::kernel::ButterflyKernels;
use fff::field::{Elem, Field};
use fff::kernel::{FieldKernels, backend_for};
use fff::{Gf8, Gf16};
use gs_engine::{
    AlekhnovichLimits, AlekhnovichScratch, BivariatePolynomial, DecodeScratch, EvaluationDomain,
    GsParameters, GsPlan, KoetterScratch, ParameterLimits, Polynomial, alekhnovich_roots,
    interpolate_koetter, interpolate_koetter_into, interpolate_module,
};

const ROOT_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256);

fn element<F: Field>(value: u64) -> F::Elem {
    let bytes = value.to_le_bytes();
    F::read(&bytes[..F::BYTES])
}

fn polynomial<F: FieldKernels>(coefficients: &[F::Elem]) -> Polynomial<F> {
    Polynomial::from_coefficients(coefficients).unwrap()
}

fn elapsed(mut operation: impl FnMut(), iterations: usize) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    start.elapsed()
}

fn run<F: ButterflyKernels>(field: &str) {
    let backend = backend_for::<F>().name();
    let limits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);
    let parameters = GsParameters::new::<F>(15, 4, 6, 2, 4, 17, limits).unwrap();
    let points: Vec<_> = (0..15).map(|value| element::<F>(value)).collect();
    let message = polynomial::<F>(&[
        element::<F>(0x1234),
        element::<F>(0xabcd),
        element::<F>(0x0108),
        element::<F>(0xbeef),
        element::<F>(0x2222),
    ]);
    let mut received = message.evaluate_many(&points).unwrap();
    for (offset, value) in received[9..].iter_mut().enumerate() {
        *value = value.add(element::<F>((offset + 1) as u64));
    }
    let alternate_message = polynomial::<F>(&[
        element::<F>(0x4321),
        element::<F>(0xdcba),
        element::<F>(0x0801),
        element::<F>(0xfeeb),
        element::<F>(0x1111),
    ]);
    let mut alternate_received = alternate_message.evaluate_many(&points).unwrap();
    for (offset, value) in alternate_received[9..].iter_mut().enumerate() {
        *value = value.add(element::<F>((offset + 17) as u64));
    }
    let interpolation = interpolate_koetter::<F>(parameters, &points, &received).unwrap();
    let mut interpolation_scratch = KoetterScratch::new();
    let mut interpolation_output = BivariatePolynomial::zero();

    let interpolation_time = elapsed(
        || {
            black_box(interpolate_koetter_into::<F>(
                parameters,
                black_box(&points),
                black_box(&received),
                &mut interpolation_scratch,
                &mut interpolation_output,
            ))
            .unwrap();
        },
        200,
    );
    let module_time = elapsed(
        || {
            black_box(interpolate_module::<F>(
                parameters,
                black_box(&points),
                black_box(&received),
            ))
            .unwrap();
        },
        200,
    );
    let mut root_scratch = AlekhnovichScratch::new();
    let roots_time = elapsed(
        || {
            black_box(alekhnovich_roots(
                black_box(&interpolation),
                parameters.max_degree(),
                ROOT_LIMITS,
                &mut root_scratch,
            ))
            .unwrap();
        },
        200,
    );

    let domain = EvaluationDomain::<F>::arbitrary(points).unwrap();
    let plan = GsPlan::new(parameters, domain, ROOT_LIMITS).unwrap();
    let mut decode_scratch = DecodeScratch::new();
    let mut output = Vec::new();
    let mut decode_iteration = 0_usize;
    let decode_time = elapsed(
        || {
            let word = if decode_iteration.is_multiple_of(2) {
                &received
            } else {
                &alternate_received
            };
            decode_iteration += 1;
            black_box(plan.decode_into(black_box(word), &mut decode_scratch, &mut output)).unwrap();
        },
        200,
    );

    for (name, duration) in [
        ("koetter", interpolation_time),
        ("module", module_time),
        ("roots", roots_time),
        ("decode", decode_time),
    ] {
        println!(
            "{field},{backend},{name},200,{},{}",
            duration.as_nanos(),
            duration.as_nanos() / 200
        );
    }
}

fn main() {
    println!("field,backend,stage,iterations,nanoseconds,ns_per_iteration");
    run::<Gf8>("gf8");
    run::<Gf16>("gf16");
}
