use std::hint::black_box;
use std::time::{Duration, Instant};

use cafft::basis::{conversion_scratch_elements, monomial_to_novel_bytes};
use cafft::core::kernel::ButterflyKernels;
use cafft::core::transform::TransformPlan;
use fgf::field::{Elem, Field};
use fgf::kernel::backend_for;
use fgf::{Gf8, Gf16};
use gs_engine::Polynomial;

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

fn run<F: ButterflyKernels>(field: &str) {
    let backend = backend_for::<F>().name();
    for size in [4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048] {
        if size as u128 > F::ORDER {
            break;
        }
        let plan = TransformPlan::<F>::new(size).unwrap();
        let points: Vec<_> = (0..size).map(|index| plan.point_element(index)).collect();
        let degree = (size / 4).max(1);
        for candidate_count in [1, 2, 4, 8, 16] {
            let candidates: Vec<_> = (0..candidate_count)
                .map(|lane| {
                    let coefficients: Vec<_> = (0..=degree)
                        .map(|index| element::<F>((lane * 251 + index * 17 + 1) as u64))
                        .collect();
                    Polynomial::<F>::from_coefficients(&coefficients).unwrap()
                })
                .collect();
            let row_len = candidate_count * F::BYTES;
            let mut rows = vec![0_u8; size * row_len];
            let mut conversion = vec![0_u8; conversion_scratch_elements(size) * row_len];
            let iterations = (8_192 / size / candidate_count).max(4);
            let horner = elapsed(
                || {
                    let mut checksum = element::<F>(0);
                    for candidate in black_box(&candidates) {
                        for &point in &points {
                            checksum = checksum.add(candidate.evaluate(point));
                        }
                    }
                    black_box(checksum);
                },
                iterations,
            );
            let cafft = elapsed(
                || {
                    rows.fill(0);
                    for (lane, candidate) in candidates.iter().enumerate() {
                        for (coefficient, value) in candidate.coefficients().enumerate() {
                            let offset = coefficient * row_len + lane * F::BYTES;
                            F::write(&mut rows[offset..offset + F::BYTES], value);
                        }
                    }
                    monomial_to_novel_bytes::<F>(&mut rows, row_len, &plan, &mut conversion)
                        .unwrap();
                    plan.forward_bytes(&mut rows, row_len).unwrap();
                    black_box(&rows);
                },
                iterations,
            );
            println!(
                "{field},{backend},horner,{size},{candidate_count},{degree},{iterations},{}",
                horner.as_nanos()
            );
            println!(
                "{field},{backend},cafft,{size},{candidate_count},{degree},{iterations},{}",
                cafft.as_nanos()
            );
        }
    }
}

fn main() {
    println!("field,backend,algorithm,points,candidates,degree,iterations,nanoseconds");
    run::<Gf8>("gf8");
    run::<Gf16>("gf16");
}
