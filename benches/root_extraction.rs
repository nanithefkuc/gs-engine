mod common;

use std::hint::black_box;
use std::time::Duration;

use butterfly_fft::core::kernel::ButterflyKernels;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fgf::field::Elem;
use fgf::{Gf8, Gf16};
use gs_engine::{
    AlekhnovichLimits, AlekhnovichScratch, BivariatePolynomial, GsParameters, Polynomial,
    RothRuckensteinLimits, alekhnovich_roots, interpolate_module, roth_ruckenstein_roots,
};

use common::{
    PARAMETER_LIMITS, backend_name, element, generated_polynomial, measure_allocations,
    report_allocations,
};

const ROTH_LIMITS: RothRuckensteinLimits = RothRuckensteinLimits::new(10_000_000, 256);
const FORCED_ALEKHNOVICH_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256)
        .with_roth_ruckenstein_crossover(0);
const AUTO_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256);

fn product_of_y_plus<F: ButterflyKernels>(roots: &[Polynomial<F>]) -> BivariatePolynomial<F> {
    let mut rows = vec![Polynomial::<F>::one().unwrap()];
    for root in roots {
        let mut product = vec![Polynomial::<F>::zero(); rows.len() + 1];
        for (y_degree, row) in rows.iter().enumerate() {
            product[y_degree]
                .add_assign(&row.multiply(root).unwrap())
                .unwrap();
            product[y_degree + 1].add_assign(row).unwrap();
        }
        rows = product;
    }
    BivariatePolynomial::from_y_coefficients(rows)
}

fn synthetic_fixture<F: ButterflyKernels>(max_degree: usize) -> BivariatePolynomial<F> {
    let roots: Vec<_> = (0..4)
        .map(|root| {
            let coefficients: Vec<_> = (0..=max_degree)
                .map(|degree| element::<F>((root * 61 + degree * 29 + 1) as u64))
                .collect();
            Polynomial::from_coefficients(&coefficients).unwrap()
        })
        .collect();
    product_of_y_plus(&roots)
}

fn interpolation_fixture<F: ButterflyKernels>(
    n: usize,
    max_degree: usize,
    radius: usize,
) -> BivariatePolynomial<F> {
    let parameters = GsParameters::search::<F>(n, max_degree, radius, PARAMETER_LIMITS).unwrap();
    let points: Vec<_> = (0..n).map(|index| element::<F>(index as u64)).collect();
    let message = generated_polynomial::<F>(max_degree + 1, 0x7007_0000 + n as u64);
    let mut received = message.evaluate_many(&points).unwrap();
    for (offset, symbol) in received[n - radius..].iter_mut().enumerate() {
        *symbol = symbol.add(element::<F>((offset + 1) as u64));
    }
    interpolate_module::<F>(parameters, &points, &received).unwrap()
}

fn benchmark_fixture<F: ButterflyKernels>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    field: &str,
    fixture_name: &str,
    max_degree: usize,
    polynomial: &BivariatePolynomial<F>,
) {
    let weighted_degree = polynomial.weighted_degree(max_degree).unwrap().unwrap();
    let weighted_size = (weighted_degree + 1) * polynomial.y_coefficient_count();
    let geometry = format!(
        "{field}/{}/{fixture_name}/max-degree{max_degree}/weighted-size{weighted_size}/y-rows{}",
        backend_name::<F>(),
        polynomial.y_coefficient_count()
    );
    group.throughput(Throughput::Elements(weighted_size as u64));

    let (roots, roth_allocations) = measure_allocations(|| {
        roth_ruckenstein_roots(polynomial, max_degree, ROTH_LIMITS).unwrap()
    });
    black_box(&roots);
    report_allocations(
        &format!("root-extraction/forced-roth-ruckenstein/{geometry}/prepared-false"),
        roth_allocations,
    );
    let ((scratch, roots), alekhnovich_allocations) = measure_allocations(|| {
        let mut scratch = AlekhnovichScratch::new();
        let roots = alekhnovich_roots(
            polynomial,
            max_degree,
            FORCED_ALEKHNOVICH_LIMITS,
            &mut scratch,
        )
        .unwrap();
        (scratch, roots)
    });
    black_box((&scratch, &roots));
    report_allocations(
        &format!("root-extraction/forced-alekhnovich/{geometry}/prepared-false"),
        alekhnovich_allocations,
    );

    group.bench_function(
        BenchmarkId::new(
            "forced-roth-ruckenstein",
            format!("{geometry}/prepared-false"),
        ),
        |bencher| {
            bencher.iter(|| {
                black_box(
                    roth_ruckenstein_roots(black_box(polynomial), max_degree, ROTH_LIMITS).unwrap(),
                );
            });
        },
    );
    let mut forced_scratch = AlekhnovichScratch::new();
    group.bench_function(
        BenchmarkId::new("forced-alekhnovich", format!("{geometry}/prepared-true")),
        |bencher| {
            bencher.iter(|| {
                black_box(
                    alekhnovich_roots(
                        black_box(polynomial),
                        max_degree,
                        FORCED_ALEKHNOVICH_LIMITS,
                        &mut forced_scratch,
                    )
                    .unwrap(),
                );
            });
        },
    );
    let mut auto_scratch = AlekhnovichScratch::new();
    group.bench_function(
        BenchmarkId::new("auto", format!("{geometry}/prepared-true")),
        |bencher| {
            bencher.iter(|| {
                black_box(
                    alekhnovich_roots(
                        black_box(polynomial),
                        max_degree,
                        AUTO_LIMITS,
                        &mut auto_scratch,
                    )
                    .unwrap(),
                );
            });
        },
    );
}

fn run_field<F: ButterflyKernels>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    field: &str,
) {
    for max_degree in [4_usize, 32, 128] {
        let polynomial = synthetic_fixture::<F>(max_degree);
        benchmark_fixture(group, field, "synthetic-four-root", max_degree, &polynomial);
    }
    for (n, max_degree, radius) in [(8_usize, 2_usize, 3_usize), (16, 5, 5), (32, 10, 11)] {
        let polynomial = interpolation_fixture::<F>(n, max_degree, radius);
        benchmark_fixture(
            group,
            field,
            &format!("interpolation-n{n}-tau{radius}"),
            max_degree,
            &polynomial,
        );
    }
}

fn root_extraction(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("root-extraction");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);
    run_field::<Gf8>(&mut group, "gf8");
    run_field::<Gf16>(&mut group, "gf16");
    group.finish();
}

criterion_group!(benches, root_extraction);
criterion_main!(benches);
