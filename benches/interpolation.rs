mod common;

use std::hint::black_box;
use std::time::Duration;

use butterfly_fft::core::kernel::ButterflyKernels;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fgf::field::Elem;
use fgf::{Gf8, Gf16};
use gs_engine::{
    BivariatePolynomial, GsParameters, InterpolationPlan, KoetterScratch, ModuleScratch,
    interpolate_koetter_into, interpolate_module, interpolate_module_into,
};

use common::{
    PARAMETER_LIMITS, backend_name, element, generated_polynomial, measure_allocations,
    report_allocations,
};

fn run_field<F: ButterflyKernels>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    field: &str,
) {
    for n in [4_usize, 8, 16, 32, 64, 128, 255] {
        let max_degree = n / 3;
        let radius = n * 2 / 5;
        let parameters =
            GsParameters::search::<F>(n, max_degree, radius, PARAMETER_LIMITS).unwrap();
        let points: Vec<_> = (0..n).map(|value| element::<F>(value as u64)).collect();
        let message = generated_polynomial::<F>(max_degree + 1, 0x1a2b_0000 + n as u64);
        let mut received = message.evaluate_many(&points).unwrap();
        for (offset, value) in received[n - radius..].iter_mut().enumerate() {
            *value = value.add(element::<F>((offset + 1) as u64));
        }
        let geometry = format!(
            "{field}/{}/arbitrary/n{n}/k{}/tau{radius}/s{}/ell{}/D{}",
            backend_name::<F>(),
            max_degree + 1,
            parameters.multiplicity(),
            parameters.y_degree(),
            parameters.weighted_degree()
        );
        group.throughput(Throughput::Elements(
            parameters.resources().constraints() as u64
        ));

        let ((scratch, output), koetter_allocations) = measure_allocations(|| {
            let mut scratch = KoetterScratch::new();
            let mut output = BivariatePolynomial::zero();
            interpolate_koetter_into::<F>(
                parameters,
                &points,
                &received,
                &mut scratch,
                &mut output,
            )
            .unwrap();
            (scratch, output)
        });
        black_box((&scratch, &output));
        report_allocations(
            &format!("interpolation/forced-koetter/{geometry}/prepared-false"),
            koetter_allocations,
        );
        let (module_output, module_allocations) = measure_allocations(|| {
            interpolate_module::<F>(parameters, &points, &received).unwrap()
        });
        black_box(&module_output);
        report_allocations(
            &format!("interpolation/forced-module/{geometry}/prepared-false"),
            module_allocations,
        );

        let mut scratch = KoetterScratch::new();
        let mut output = BivariatePolynomial::zero();
        interpolate_koetter_into::<F>(parameters, &points, &received, &mut scratch, &mut output)
            .unwrap();
        let plan: InterpolationPlan<F> = InterpolationPlan::new(parameters, &points).unwrap();
        let mut module_scratch = ModuleScratch::new();
        let mut module_output = BivariatePolynomial::zero();
        interpolate_module_into(
            parameters,
            &points,
            &received,
            &plan,
            &mut module_scratch,
            &mut module_output,
        )
        .unwrap();
        group.bench_function(
            BenchmarkId::new("forced-koetter", format!("{geometry}/prepared-true")),
            |bencher| {
                bencher.iter(|| {
                    interpolate_koetter_into::<F>(
                        parameters,
                        black_box(&points),
                        black_box(&received),
                        &mut scratch,
                        &mut output,
                    )
                    .unwrap();
                    black_box(&output);
                });
            },
        );
        group.bench_function(
            BenchmarkId::new("forced-module", format!("{geometry}/prepared-false")),
            |bencher| {
                bencher.iter(|| {
                    black_box(
                        interpolate_module::<F>(
                            parameters,
                            black_box(&points),
                            black_box(&received),
                        )
                        .unwrap(),
                    );
                });
            },
        );
        group.bench_function(
            BenchmarkId::new("forced-module", format!("{geometry}/prepared-true")),
            |bencher| {
                bencher.iter(|| {
                    interpolate_module_into(
                        parameters,
                        black_box(&points),
                        black_box(&received),
                        &plan,
                        &mut module_scratch,
                        &mut module_output,
                    )
                    .unwrap();
                    black_box(&module_output);
                });
            },
        );
    }
}

fn interpolation(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("interpolation");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);
    run_field::<Gf8>(&mut group, "gf8");
    run_field::<Gf16>(&mut group, "gf16");
    group.finish();
}

criterion_group!(benches, interpolation);
criterion_main!(benches);
