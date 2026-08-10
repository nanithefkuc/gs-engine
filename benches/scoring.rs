mod common;

use std::hint::black_box;
use std::time::Duration;

use butterfly_fft::core::kernel::ButterflyKernels;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fgf::{Gf8, Gf16};
use gs_engine::{DecodeScratch, ScoringStrategy, score_candidates_with_strategy};

use common::{
    DomainSpec, backend_name, generated_polynomial, measure_allocations, report_allocations,
};

#[derive(Clone, Copy)]
struct ScoringCase {
    points: usize,
    candidates: usize,
    degree: usize,
    domain: DomainSpec,
}

const SCORING_CASES: &[ScoringCase] = &[
    ScoringCase {
        points: 16,
        candidates: 1,
        degree: 1,
        domain: DomainSpec::Affine,
    },
    ScoringCase {
        points: 16,
        candidates: 16,
        degree: 8,
        domain: DomainSpec::Affine,
    },
    ScoringCase {
        points: 64,
        candidates: 2,
        degree: 4,
        domain: DomainSpec::Additive,
    },
    ScoringCase {
        points: 64,
        candidates: 8,
        degree: 32,
        domain: DomainSpec::Additive,
    },
    ScoringCase {
        points: 256,
        candidates: 4,
        degree: 16,
        domain: DomainSpec::Additive,
    },
    ScoringCase {
        points: 512,
        candidates: 16,
        degree: 256,
        domain: DomainSpec::Additive,
    },
];

fn run_field<F: ButterflyKernels>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    field: &str,
) {
    for &case in SCORING_CASES {
        if case.points as u128 > F::ORDER {
            continue;
        }
        let domain = case.domain.build::<F>(case.points);
        let candidates: Vec<_> = (0..case.candidates)
            .map(|lane| {
                generated_polynomial::<F>(
                    case.degree + 1,
                    0x5c0a_0000 + (lane * 257 + case.points) as u64,
                )
            })
            .collect();
        let received = candidates[0].evaluate_many(domain.points()).unwrap();
        let total_coefficients: usize = candidates
            .iter()
            .map(gs_engine::Polynomial::coefficient_count)
            .sum();
        let geometry = format!(
            "{field}/{}/{}/points{}/candidates{}/degree{}/coefficients{}",
            backend_name::<F>(),
            case.domain.name(),
            case.points,
            case.candidates,
            case.degree,
            total_coefficients
        );
        group.throughput(Throughput::Elements(
            (case.points * total_coefficients) as u64,
        ));

        for (name, strategy) in [
            ("forced-horner", ScoringStrategy::Horner),
            ("forced-butterfly-fft", ScoringStrategy::ButterflyFft),
            ("auto", ScoringStrategy::Auto),
        ] {
            let ((scratch, output), allocations) = measure_allocations(|| {
                let mut scratch = DecodeScratch::new();
                let mut output = Vec::new();
                score_candidates_with_strategy(
                    &domain,
                    &received,
                    &candidates,
                    case.points,
                    strategy,
                    &mut scratch,
                    &mut output,
                )
                .unwrap();
                (scratch, output)
            });
            black_box((&scratch, &output));
            report_allocations(
                &format!("scoring/{name}/{geometry}/prepared-false"),
                allocations,
            );

            let mut scratch = DecodeScratch::new();
            let mut output = Vec::new();
            score_candidates_with_strategy(
                &domain,
                &received,
                &candidates,
                case.points,
                strategy,
                &mut scratch,
                &mut output,
            )
            .unwrap();
            group.bench_function(
                BenchmarkId::new(name, format!("{geometry}/prepared-true")),
                |bencher| {
                    bencher.iter(|| {
                        score_candidates_with_strategy(
                            black_box(&domain),
                            black_box(&received),
                            black_box(&candidates),
                            case.points,
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

fn scoring(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("scoring");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);
    run_field::<Gf8>(&mut group, "gf8");
    run_field::<Gf16>(&mut group, "gf16");
    group.finish();
}

criterion_group!(benches, scoring);
criterion_main!(benches);
