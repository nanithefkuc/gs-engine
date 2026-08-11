//! Parallel batch decode benchmark.
//!
//! Requires the `parallel` feature. Measures shared-plan batch decode against
//! single-thread per-word decode for batches above the parallel crossover, so
//! the thread-count and crossover policy carry a measurement record.

#![cfg(feature = "parallel")]

mod common;

use std::hint::black_box;
use std::time::Duration;

use butterfly_fft::core::kernel::ButterflyKernels;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fgf::{Gf8, Gf16};
use gs_engine::DecodeScratch;

use common::{DECODE_SPECS, DecodeCase};

/// Number of received words per batch — above `PARALLEL_BATCH_CROSSOVER` so the
/// Rayon pool is exercised, and large enough that per-job setup is amortized.
const BATCH_SIZE: usize = gs_engine::PARALLEL_BATCH_CROSSOVER * 4;

fn run_field<F: ButterflyKernels + gs_engine::ParallelField>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    field: &str,
) where
    F::Elem: gs_engine::ParallelElem,
{
    for &spec in DECODE_SPECS {
        let case = DecodeCase::<F>::new(spec);
        let plan = case.build_plan();
        // Build a batch of distinct corrupted words by cycling the two
        // prepared received words and shifting their tail symbols.
        let words: Vec<Vec<F::Elem>> = (0..BATCH_SIZE)
            .map(|i| {
                if i % 2 == 0 {
                    case.received.clone()
                } else {
                    case.alternate_received.clone()
                }
            })
            .collect();
        let received_refs: Vec<&[F::Elem]> = words.iter().map(|w| w.as_slice()).collect();

        group.throughput(Throughput::Elements((spec.n * BATCH_SIZE) as u64));

        let seq_id = spec.id::<F>("batch-sequential", field);
        group.bench_function(BenchmarkId::from_parameter(seq_id), |bencher| {
            bencher.iter(|| {
                let mut scratches: Vec<DecodeScratch<F>> =
                    (0..BATCH_SIZE).map(|_| DecodeScratch::new()).collect();
                let mut outputs: Vec<Vec<gs_engine::Polynomial<F>>> =
                    (0..BATCH_SIZE).map(|_| Vec::new()).collect();
                for (i, word) in words.iter().enumerate() {
                    plan.decode_into(word, &mut scratches[i], &mut outputs[i])
                        .expect("sequential batch decode");
                }
                black_box(outputs);
            });
        });

        let par_id = spec.id::<F>("batch-parallel", field);
        group.bench_function(BenchmarkId::from_parameter(par_id), |bencher| {
            bencher.iter(|| {
                let mut scratches: Vec<DecodeScratch<F>> =
                    (0..BATCH_SIZE).map(|_| DecodeScratch::new()).collect();
                let mut outputs: Vec<Vec<gs_engine::Polynomial<F>>> =
                    (0..BATCH_SIZE).map(|_| Vec::new()).collect();
                for (i, word) in words.iter().enumerate() {
                    plan.prepare_scratch(&mut scratches[i], &mut outputs[i])
                        .expect("prepare batch scratch");
                    let _ = word;
                }
                plan.decode_batch_into(&received_refs, &mut scratches, &mut outputs)
                    .expect("parallel batch decode");
                black_box(outputs);
            });
        });
    }
}

fn parallel_batch(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("parallel-batch");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);
    run_field::<Gf8>(&mut group, "gf8");
    run_field::<Gf16>(&mut group, "gf16");
    group.finish();
}

criterion_group!(benches, parallel_batch);
criterion_main!(benches);
