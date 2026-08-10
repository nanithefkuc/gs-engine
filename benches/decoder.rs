mod common;

use std::hint::black_box;
use std::time::Duration;

use butterfly_fft::core::kernel::ButterflyKernels;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fgf::{Gf8, Gf16};
use gs_engine::DecodeScratch;

use common::{DECODE_SPECS, DecodeCase};

fn run_field<F: ButterflyKernels>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    field: &str,
) {
    for &spec in DECODE_SPECS {
        let case = DecodeCase::<F>::new(spec);
        case.report_allocation_records(field);
        group.throughput(Throughput::Elements(spec.n as u64));

        let construction_id = spec.id::<F>("construction", field);
        group.bench_function(BenchmarkId::from_parameter(construction_id), |bencher| {
            bencher.iter(|| black_box(case.build_plan()));
        });

        let cold_id = spec.id::<F>("cold-decode", field);
        let cold_plan = case.build_plan();
        group.bench_function(BenchmarkId::from_parameter(cold_id), |bencher| {
            bencher.iter_batched(
                || (DecodeScratch::new(), Vec::new()),
                |(mut scratch, mut output)| {
                    black_box(
                        cold_plan
                            .decode_into(black_box(&case.received), &mut scratch, &mut output)
                            .expect("cold benchmark decode"),
                    );
                    black_box(output);
                },
                BatchSize::SmallInput,
            );
        });

        let changed_id = spec.id::<F>("warmed-changed-word", field);
        let (changed_plan, mut changed_scratch, mut changed_output) = case.prepared_state();
        changed_plan
            .decode_into(&case.received, &mut changed_scratch, &mut changed_output)
            .expect("warm changed-word benchmark");
        let mut iteration = 0_usize;
        group.bench_function(BenchmarkId::from_parameter(changed_id), |bencher| {
            bencher.iter(|| {
                let received = if iteration.is_multiple_of(2) {
                    &case.alternate_received
                } else {
                    &case.received
                };
                iteration += 1;
                black_box(
                    changed_plan
                        .decode_into(
                            black_box(received),
                            &mut changed_scratch,
                            &mut changed_output,
                        )
                        .expect("changed-word benchmark decode"),
                );
                black_box(&changed_output);
            });
        });

        let repeat_id = spec.id::<F>("exact-repeat", field);
        let (repeat_plan, mut repeat_scratch, mut repeat_output) = case.prepared_state();
        repeat_plan
            .decode_into(&case.received, &mut repeat_scratch, &mut repeat_output)
            .expect("warm exact-repeat benchmark");
        group.bench_function(BenchmarkId::from_parameter(repeat_id), |bencher| {
            bencher.iter(|| {
                black_box(
                    repeat_plan
                        .decode_into(
                            black_box(&case.received),
                            &mut repeat_scratch,
                            &mut repeat_output,
                        )
                        .expect("exact-repeat benchmark decode"),
                );
                black_box(&repeat_output);
            });
        });
    }
}

fn decoder_matrix(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("end-to-end");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);
    run_field::<Gf8>(&mut group, "gf8");
    run_field::<Gf16>(&mut group, "gf16");
    group.finish();
}

criterion_group!(benches, decoder_matrix);
criterion_main!(benches);
