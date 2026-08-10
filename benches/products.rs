mod common;

use std::hint::black_box;
use std::time::Duration;

use butterfly_fft::core::kernel::ButterflyKernels;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fgf::field::Elem;
use fgf::{Gf8, Gf16};
use gs_engine::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated};

use common::{backend_name, generated_polynomial, measure_allocations, report_allocations};

#[derive(Clone, Copy)]
struct ProductCase {
    left: usize,
    right: usize,
    precision: usize,
    batch: usize,
    shape: &'static str,
    nonzero_stride: usize,
}

const PRODUCT_CASES: &[ProductCase] = &[
    ProductCase {
        left: 8,
        right: 8,
        precision: 15,
        batch: 1,
        shape: "balanced-full",
        nonzero_stride: 1,
    },
    ProductCase {
        left: 32,
        right: 4,
        precision: 35,
        batch: 4,
        shape: "unbalanced-full",
        nonzero_stride: 1,
    },
    ProductCase {
        left: 128,
        right: 8,
        precision: 32,
        batch: 8,
        shape: "unbalanced-truncated",
        nonzero_stride: 1,
    },
    ProductCase {
        left: 96,
        right: 64,
        precision: 40,
        batch: 16,
        shape: "balanced-truncated",
        nonzero_stride: 1,
    },
    ProductCase {
        left: 512,
        right: 32,
        precision: 128,
        batch: 4,
        shape: "unbalanced-truncated-large",
        nonzero_stride: 1,
    },
    ProductCase {
        left: 1_024,
        right: 256,
        precision: 640,
        batch: 1,
        shape: "unbalanced-half-large",
        nonzero_stride: 1,
    },
    ProductCase {
        left: 128,
        right: 128,
        precision: 32,
        batch: 1,
        shape: "balanced-sparse-truncated",
        nonzero_stride: 8,
    },
    ProductCase {
        left: 256,
        right: 32,
        precision: 64,
        batch: 16,
        shape: "unbalanced-sparse-batched",
        nonzero_stride: 16,
    },
];

fn run_field<F: ButterflyKernels>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    field: &str,
) {
    for &case in PRODUCT_CASES {
        let full_count = case.left + case.right - 1;
        if full_count.next_power_of_two() as u128 > F::ORDER {
            continue;
        }
        let left: Vec<_> = (0..case.batch)
            .map(|lane| {
                generated_sparse_polynomial::<F>(case.left, lane as u64 + 1, case.nonzero_stride)
            })
            .collect();
        let right: Vec<_> = (0..case.batch)
            .map(|lane| {
                generated_sparse_polynomial::<F>(
                    case.right,
                    lane as u64 + 10_000,
                    case.nonzero_stride,
                )
            })
            .collect();
        let pairs: Vec<_> = left.iter().zip(&right).collect();
        let geometry = if case.nonzero_stride == 1 {
            format!(
                "{field}/{}/left{}/right{}/precision{}/batch{}/{}",
                backend_name::<F>(),
                case.left,
                case.right,
                case.precision,
                case.batch,
                case.shape
            )
        } else {
            format!(
                "{field}/{}/left{}/right{}/precision{}/batch{}/stride{}/{}",
                backend_name::<F>(),
                case.left,
                case.right,
                case.precision,
                case.batch,
                case.nonzero_stride,
                case.shape
            )
        };
        group.throughput(Throughput::Elements((case.precision * case.batch) as u64));

        for (name, strategy) in [
            ("forced-schoolbook", ProductStrategy::Schoolbook),
            ("forced-afft", ProductStrategy::Afft),
            ("auto", ProductStrategy::Auto),
        ] {
            let ((scratch, output), allocations) = measure_allocations(|| {
                let mut scratch = PolynomialProductScratch::new();
                let mut output = Vec::new();
                multiply_batch_truncated(
                    &pairs,
                    case.precision,
                    strategy,
                    &mut scratch,
                    &mut output,
                )
                .unwrap();
                (scratch, output)
            });
            black_box((&scratch, &output));
            report_allocations(
                &format!("products/{name}/{geometry}/prepared-false"),
                allocations,
            );

            let mut scratch = PolynomialProductScratch::new();
            let mut output = Vec::new();
            multiply_batch_truncated(&pairs, case.precision, strategy, &mut scratch, &mut output)
                .unwrap();
            group.bench_function(
                BenchmarkId::new(name, format!("{geometry}/prepared-true")),
                |bencher| {
                    bencher.iter(|| {
                        multiply_batch_truncated(
                            black_box(&pairs),
                            case.precision,
                            strategy,
                            &mut scratch,
                            &mut output,
                        )
                        .unwrap();
                        black_box(&output);
                    });
                },
            );
        }
    }
}

fn generated_sparse_polynomial<F: ButterflyKernels>(
    count: usize,
    seed: u64,
    nonzero_stride: usize,
) -> gs_engine::Polynomial<F> {
    let mut polynomial = generated_polynomial::<F>(count, seed);
    if nonzero_stride > 1 {
        for degree in 0..count.saturating_sub(1) {
            if !degree.is_multiple_of(nonzero_stride) {
                polynomial.set_coefficient(degree, F::Elem::ZERO).unwrap();
            }
        }
    }
    polynomial
}

fn products(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("products");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);
    run_field::<Gf8>(&mut group, "gf8");
    run_field::<Gf16>(&mut group, "gf16");
    group.finish();
}

criterion_group!(benches, products);
criterion_main!(benches);
