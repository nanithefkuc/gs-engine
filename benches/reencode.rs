mod common;

use std::hint::black_box;
use std::time::Duration;

use butterfly_fft::core::kernel::ButterflyKernels;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fgf::Gf8;
use fgf::field::Elem;
use gs_engine::{DecodeScratch, EvaluationDomain, GsParameters, GsPlan};

use common::{PARAMETER_LIMITS, ROOT_LIMITS, backend_name, element, generated_polynomial};

/// High-rate geometry where re-encoding zeroes many coordinates.
#[derive(Clone, Copy)]
struct RateSpec {
    n: usize,
    k: usize,
    tau: usize,
    rate: &'static str,
}

const RATE_SPECS: &[RateSpec] = &[
    RateSpec {
        n: 64,
        k: 48,
        tau: 8,
        rate: "3-4",
    },
    RateSpec {
        n: 64,
        k: 58,
        tau: 3,
        rate: "9-10",
    },
    RateSpec {
        n: 128,
        k: 96,
        tau: 16,
        rate: "3-4",
    },
    RateSpec {
        n: 128,
        k: 115,
        tau: 6,
        rate: "9-10",
    },
];

struct Case<F: ButterflyKernels> {
    parameters: GsParameters,
    domain: EvaluationDomain<F>,
    received: Vec<F::Elem>,
    alternate: Vec<F::Elem>,
}

fn build_case<F: ButterflyKernels>(spec: RateSpec) -> Case<F> {
    let parameters = GsParameters::search::<F>(spec.n, spec.k - 1, spec.tau, PARAMETER_LIMITS)
        .expect("feasible high-rate geometry");
    let domain = EvaluationDomain::arbitrary((0..spec.n).map(|i| element::<F>(i as u64)).collect())
        .expect("valid benchmark domain");
    let message = generated_polynomial::<F>(spec.k, 0x5e00_0000 + spec.n as u64);
    let alternate = generated_polynomial::<F>(spec.k, 0x5e80_0000 + spec.n as u64);
    let mut received = message
        .evaluate_many(domain.points())
        .expect("valid benchmark evaluation");
    let mut alternate_received = alternate
        .evaluate_many(domain.points())
        .expect("valid alternate evaluation");
    for (offset, symbol) in received[spec.n - spec.tau..].iter_mut().enumerate() {
        *symbol = symbol.add(element::<F>((offset + 1) as u64));
    }
    for (offset, symbol) in alternate_received[spec.n - spec.tau..]
        .iter_mut()
        .enumerate()
    {
        *symbol = symbol.add(element::<F>((offset + 0x41) as u64));
    }
    Case {
        parameters,
        domain,
        received,
        alternate: alternate_received,
    }
}

fn bench_field<F: ButterflyKernels>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    field: &str,
) {
    for &spec in RATE_SPECS {
        let case = build_case::<F>(spec);
        group.throughput(Throughput::Elements(spec.n as u64));

        for (label, enabled) in [("direct", false), ("reencode", true)] {
            let plan = GsPlan::new(case.parameters, case.domain.clone(), ROOT_LIMITS)
                .expect("valid plan")
                .with_reencode(enabled)
                .expect("re-encoding override");
            assert_eq!(plan.uses_reencode(), enabled);

            let mut scratch = DecodeScratch::new();
            let mut output = Vec::new();
            plan.prepare_scratch(&mut scratch, &mut output)
                .expect("prepare scratch");
            // Warm both received words so all data-dependent capacity is reached.
            plan.decode_into(&case.received, &mut scratch, &mut output)
                .expect("warm received decode");
            plan.decode_into(&case.alternate, &mut scratch, &mut output)
                .expect("warm alternate decode");

            let id = format!(
                "{label}/{}/{}/n{}/k{}/tau{}/s{}/ell{}/rate-{}",
                field,
                backend_name::<F>(),
                spec.n,
                spec.k,
                spec.tau,
                plan.parameters().multiplicity(),
                plan.parameters().y_degree(),
                spec.rate,
            );
            let mut toggle = false;
            group.bench_function(BenchmarkId::from_parameter(id), |bencher| {
                bencher.iter(|| {
                    let word = if toggle {
                        &case.alternate
                    } else {
                        &case.received
                    };
                    toggle = !toggle;
                    plan.decode_into(black_box(word), &mut scratch, &mut output)
                        .expect("changed-word decode");
                    black_box(output.len())
                });
            });
        }
    }
}

fn reencode_matrix(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("reencode-high-rate");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);
    bench_field::<Gf8>(&mut group, "gf8");
    group.finish();
}

criterion_group!(benches, reencode_matrix);
criterion_main!(benches);
