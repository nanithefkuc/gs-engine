use std::hint::black_box;
use std::time::{Duration, Instant};

use fff::field::{Elem, Field};
use fff::kernel::{FieldKernels, backend_for};
use fff::{Gf8, Gf16};
use gs_engine::{
    BivariatePolynomial, GsParameters, KoetterScratch, ParameterLimits, Polynomial,
    interpolate_koetter_into, interpolate_module,
};

fn element<F: Field>(value: u64) -> F::Elem {
    let bytes = value.to_le_bytes();
    F::read(&bytes[..F::BYTES])
}

fn elapsed(mut operation: impl FnMut(), iterations: usize) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    start.elapsed()
}

fn run<F: FieldKernels>(field: &str) {
    let backend = backend_for::<F>().name();
    for size in [4, 8, 15, 31, 63, 127, 255] {
        let max_degree = size / 3;
        let radius = size * 2 / 5;
        let parameters = GsParameters::search::<F>(
            size,
            max_degree,
            radius,
            ParameterLimits::new(8, 16, usize::MAX, usize::MAX),
        )
        .unwrap();
        let points: Vec<_> = (0..size).map(|value| element::<F>(value as u64)).collect();
        let coefficients: Vec<_> = (0..=max_degree)
            .map(|index| element::<F>((index * 257 + 1) as u64))
            .collect();
        let message = Polynomial::<F>::from_coefficients(&coefficients).unwrap();
        let mut received = message.evaluate_many(&points).unwrap();
        for (offset, value) in received[size - radius..].iter_mut().enumerate() {
            *value = value.add(element::<F>((offset + 1) as u64));
        }
        let iterations = (256 / size).max(1);
        let mut scratch = KoetterScratch::new();
        let mut output = BivariatePolynomial::zero();
        let koetter = elapsed(
            || {
                black_box(interpolate_koetter_into::<F>(
                    parameters,
                    black_box(&points),
                    black_box(&received),
                    &mut scratch,
                    &mut output,
                ))
                .unwrap();
            },
            iterations,
        );
        let module = elapsed(
            || {
                black_box(interpolate_module::<F>(
                    parameters,
                    black_box(&points),
                    black_box(&received),
                ))
                .unwrap();
            },
            iterations,
        );
        for (algorithm, duration) in [("koetter", koetter), ("module", module)] {
            println!(
                "{field},{backend},{algorithm},{size},{max_degree},{radius},{},{},{iterations},{}",
                parameters.multiplicity(),
                parameters.y_degree(),
                duration.as_nanos()
            );
        }
    }
}

fn main() {
    println!(
        "field,backend,algorithm,points,max_degree,radius,multiplicity,y_degree,iterations,nanoseconds"
    );
    run::<Gf8>("gf8");
    run::<Gf16>("gf16");
}
