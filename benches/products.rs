use std::hint::black_box;
use std::time::{Duration, Instant};

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::kernel::{FieldKernels, backend_for};
use fgf::{Gf8, Gf16};
use gs_engine::{Polynomial, PolynomialProductScratch, ProductStrategy, multiply_batch_truncated};

fn generated<F: FieldKernels>(count: usize, mut state: u64) -> Polynomial<F> {
    let coefficients: Vec<_> = (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            F::read(&state.to_le_bytes()[..F::BYTES])
        })
        .collect();
    Polynomial::from_coefficients(&coefficients).unwrap()
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
    for coefficients in [
        8_usize, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768,
    ] {
        if (coefficients * 2 - 1).next_power_of_two() as u128 > F::ORDER {
            break;
        }
        let batches: &[usize] = if coefficients >= 16_384 {
            &[1, 4]
        } else {
            &[1, 4, 8, 16]
        };
        for &batch in batches {
            let left: Vec<_> = (0..batch)
                .map(|lane| generated::<F>(coefficients, lane as u64 + 1))
                .collect();
            let right: Vec<_> = (0..batch)
                .map(|lane| generated::<F>(coefficients, lane as u64 + 10_000))
                .collect();
            let pairs: Vec<_> = left.iter().zip(&right).collect();
            let iterations = (4_096 / coefficients / batch).max(1);
            let mut scratch = PolynomialProductScratch::new();
            let mut output = Vec::new();
            let schoolbook = elapsed(
                || {
                    multiply_batch_truncated(
                        black_box(&pairs),
                        coefficients * 2 - 1,
                        ProductStrategy::Schoolbook,
                        &mut scratch,
                        &mut output,
                    )
                    .unwrap();
                    black_box(&output);
                },
                iterations,
            );
            let afft = elapsed(
                || {
                    multiply_batch_truncated(
                        black_box(&pairs),
                        coefficients * 2 - 1,
                        ProductStrategy::Afft,
                        &mut scratch,
                        &mut output,
                    )
                    .unwrap();
                    black_box(&output);
                },
                iterations,
            );
            println!(
                "{field},{backend},schoolbook,{coefficients},{batch},{iterations},{}",
                schoolbook.as_nanos()
            );
            println!(
                "{field},{backend},afft,{coefficients},{batch},{iterations},{}",
                afft.as_nanos()
            );
        }
    }
}

fn main() {
    println!("field,backend,algorithm,coefficients,batch,iterations,nanoseconds");
    run::<Gf8>("gf8");
    run::<Gf16>("gf16");
}
