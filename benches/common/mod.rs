#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::{Elem, Field};
use fgf::kernel::backend_for;
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
    Polynomial,
};

pub const ROOT_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256);
pub const PARAMETER_LIMITS: ParameterLimits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);

pub struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every operation delegates to `System` with the original pointer and
// layout. The atomic bookkeeping does not affect allocation validity.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is passed through unchanged to the system allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        DEALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: `pointer` and `layout` came from the corresponding system allocation.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `pointer` and `old` came from the system allocator, and
        // `new_size` is forwarded unchanged.
        let replacement = unsafe { System.realloc(pointer, old, new_size) };
        if !replacement.is_null() {
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            DEALLOCATED_BYTES.fetch_add(old.size(), Ordering::Relaxed);
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
            if new_size >= old.size() {
                LIVE_BYTES.fetch_add(new_size - old.size(), Ordering::Relaxed);
            } else {
                LIVE_BYTES.fetch_sub(old.size() - new_size, Ordering::Relaxed);
            }
        }
        replacement
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Debug)]
struct AllocationSnapshot {
    allocations: usize,
    allocated_bytes: usize,
    deallocations: usize,
    deallocated_bytes: usize,
    live_bytes: usize,
}

impl AllocationSnapshot {
    fn capture() -> Self {
        Self {
            allocations: ALLOCATIONS.load(Ordering::Relaxed),
            allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
            deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
            deallocated_bytes: DEALLOCATED_BYTES.load(Ordering::Relaxed),
            live_bytes: LIVE_BYTES.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AllocationStats {
    pub allocations: usize,
    pub allocated_bytes: usize,
    pub deallocations: usize,
    pub deallocated_bytes: usize,
    pub retained_bytes: i128,
}

pub fn measure_allocations<T>(operation: impl FnOnce() -> T) -> (T, AllocationStats) {
    let before = AllocationSnapshot::capture();
    let value = operation();
    let after = AllocationSnapshot::capture();
    let stats = AllocationStats {
        allocations: after.allocations - before.allocations,
        allocated_bytes: after.allocated_bytes - before.allocated_bytes,
        deallocations: after.deallocations - before.deallocations,
        deallocated_bytes: after.deallocated_bytes - before.deallocated_bytes,
        retained_bytes: after.live_bytes as i128 - before.live_bytes as i128,
    };
    (value, stats)
}

pub fn report_allocations(id: &str, stats: AllocationStats) {
    eprintln!(
        "allocation,{id},{},{},{},{},{}",
        stats.allocations,
        stats.allocated_bytes,
        stats.deallocations,
        stats.deallocated_bytes,
        stats.retained_bytes
    );
    let Some(path) = std::env::var_os("GS_BENCH_ALLOC_CSV") else {
        return;
    };
    let needs_header = std::fs::metadata(&path).map_or(true, |metadata| metadata.len() == 0);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open allocation CSV");
    if needs_header {
        writeln!(
            file,
            "kind,id,allocations,allocated_bytes,deallocations,deallocated_bytes,retained_bytes"
        )
        .expect("write allocation CSV header");
    }
    writeln!(
        file,
        "allocation,{id},{},{},{},{},{}",
        stats.allocations,
        stats.allocated_bytes,
        stats.deallocations,
        stats.deallocated_bytes,
        stats.retained_bytes
    )
    .expect("write allocation CSV row");
}

pub fn element<F: Field>(value: u64) -> F::Elem {
    F::read(&value.to_le_bytes()[..F::BYTES])
}

pub fn polynomial<F: ButterflyKernels>(coefficients: &[F::Elem]) -> Polynomial<F> {
    Polynomial::from_coefficients(coefficients).expect("valid benchmark polynomial")
}

pub fn generated_polynomial<F: ButterflyKernels>(count: usize, mut state: u64) -> Polynomial<F> {
    let coefficients: Vec<_> = (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            F::read(&state.to_le_bytes()[..F::BYTES])
        })
        .collect();
    polynomial::<F>(&coefficients)
}

pub fn backend_name<F: ButterflyKernels>() -> &'static str {
    backend_for::<F>().name()
}

#[derive(Clone, Copy, Debug)]
pub enum DomainSpec {
    Arbitrary,
    Additive,
    Affine,
}

impl DomainSpec {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Arbitrary => "arbitrary",
            Self::Additive => "additive",
            Self::Affine => "affine",
        }
    }

    pub fn build<F: ButterflyKernels>(self, n: usize) -> EvaluationDomain<F> {
        match self {
            Self::Arbitrary => EvaluationDomain::arbitrary(
                (0..n).map(|index| element::<F>(index as u64)).collect(),
            ),
            Self::Additive => EvaluationDomain::additive_subspace(n),
            Self::Affine => EvaluationDomain::affine_coset(n, element::<F>(1 << (F::BITS - 1))),
        }
        .expect("valid benchmark domain")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DecodeSpec {
    pub n: usize,
    pub k: usize,
    pub tau: usize,
    pub rate: &'static str,
    pub radius: &'static str,
    pub domain: DomainSpec,
}

impl DecodeSpec {
    pub fn id<F: ButterflyKernels>(self, workload: &str, field: &str) -> String {
        let parameters = GsParameters::search::<F>(self.n, self.k - 1, self.tau, PARAMETER_LIMITS)
            .expect("feasible benchmark geometry");
        format!(
            "{workload}/{field}/{}/{}/n{}/k{}/tau{}/s{}/ell{}/D{}/rate-{}/radius-{}",
            backend_name::<F>(),
            self.domain.name(),
            self.n,
            self.k,
            self.tau,
            parameters.multiplicity(),
            parameters.y_degree(),
            parameters.weighted_degree(),
            self.rate,
            self.radius
        )
    }
}

pub const DECODE_SPECS: &[DecodeSpec] = &[
    DecodeSpec {
        n: 4,
        k: 1,
        tau: 2,
        rate: "1-4",
        radius: "gs-near",
        domain: DomainSpec::Additive,
    },
    DecodeSpec {
        n: 8,
        k: 3,
        tau: 3,
        rate: "3-8",
        radius: "mid",
        domain: DomainSpec::Affine,
    },
    DecodeSpec {
        n: 8,
        k: 4,
        tau: 2,
        rate: "1-2",
        radius: "unique",
        domain: DomainSpec::Arbitrary,
    },
    DecodeSpec {
        n: 16,
        k: 4,
        tau: 6,
        rate: "1-4",
        radius: "gs-near",
        domain: DomainSpec::Additive,
    },
    DecodeSpec {
        n: 16,
        k: 12,
        tau: 1,
        rate: "3-4",
        radius: "unique",
        domain: DomainSpec::Affine,
    },
    DecodeSpec {
        n: 32,
        k: 29,
        tau: 1,
        rate: "9-10",
        radius: "unique",
        domain: DomainSpec::Arbitrary,
    },
    DecodeSpec {
        n: 64,
        k: 16,
        tau: 24,
        rate: "1-4",
        radius: "gs-near",
        domain: DomainSpec::Additive,
    },
];

pub struct DecodeCase<F: ButterflyKernels> {
    pub spec: DecodeSpec,
    pub parameters: GsParameters,
    pub domain: EvaluationDomain<F>,
    pub received: Vec<F::Elem>,
    pub alternate_received: Vec<F::Elem>,
}

impl<F: ButterflyKernels> DecodeCase<F> {
    pub fn new(spec: DecodeSpec) -> Self {
        let parameters = GsParameters::search::<F>(spec.n, spec.k - 1, spec.tau, PARAMETER_LIMITS)
            .expect("feasible benchmark geometry");
        let domain = spec.domain.build::<F>(spec.n);
        let message = generated_polynomial::<F>(spec.k, 0x4753_0000 + spec.n as u64);
        let alternate = generated_polynomial::<F>(spec.k, 0x4753_8000 + spec.n as u64);
        let mut received = message
            .evaluate_many(domain.points())
            .expect("valid benchmark evaluation");
        let mut alternate_received = alternate
            .evaluate_many(domain.points())
            .expect("valid alternate benchmark evaluation");
        for (offset, symbol) in received[spec.n - spec.tau..].iter_mut().enumerate() {
            *symbol = symbol.add(element::<F>((offset + 1) as u64));
        }
        for (offset, symbol) in alternate_received[spec.n - spec.tau..]
            .iter_mut()
            .enumerate()
        {
            *symbol = symbol.add(element::<F>((offset + 0x41) as u64));
        }
        Self {
            spec,
            parameters,
            domain,
            received,
            alternate_received,
        }
    }

    pub fn build_plan(&self) -> GsPlan<F> {
        let parameters = GsParameters::search::<F>(
            self.spec.n,
            self.spec.k - 1,
            self.spec.tau,
            PARAMETER_LIMITS,
        )
        .expect("feasible benchmark geometry");
        GsPlan::new(
            parameters,
            self.spec.domain.build::<F>(self.spec.n),
            ROOT_LIMITS,
        )
        .expect("valid benchmark plan")
    }

    pub fn prepared_state(&self) -> (GsPlan<F>, DecodeScratch<F>, Vec<Polynomial<F>>) {
        let plan = GsPlan::new(self.parameters, self.domain.clone(), ROOT_LIMITS)
            .expect("valid benchmark plan");
        let mut scratch = DecodeScratch::new();
        let mut output = Vec::new();
        plan.prepare_scratch(&mut scratch, &mut output)
            .expect("prepare benchmark scratch");
        (plan, scratch, output)
    }

    pub fn report_allocation_records(&self, field: &str) {
        let (plan, construction) = measure_allocations(|| self.build_plan());
        report_allocations(&self.spec.id::<F>("construction", field), construction);

        let mut cold_scratch = DecodeScratch::new();
        let mut cold_output = Vec::new();
        let (_, cold) = measure_allocations(|| {
            plan.decode_into(&self.received, &mut cold_scratch, &mut cold_output)
                .expect("cold decode")
        });
        report_allocations(&self.spec.id::<F>("cold-decode", field), cold);

        let (plan, mut changed_scratch, mut changed_output) = self.prepared_state();
        plan.decode_into(&self.received, &mut changed_scratch, &mut changed_output)
            .expect("warm benchmark decode");
        let (_, changed) = measure_allocations(|| {
            plan.decode_into(
                &self.alternate_received,
                &mut changed_scratch,
                &mut changed_output,
            )
            .expect("changed-word decode")
        });
        report_allocations(&self.spec.id::<F>("warmed-changed-word", field), changed);

        let (plan, mut repeat_scratch, mut repeat_output) = self.prepared_state();
        plan.decode_into(&self.received, &mut repeat_scratch, &mut repeat_output)
            .expect("warm repeated benchmark decode");
        let (_, repeated) = measure_allocations(|| {
            plan.decode_into(&self.received, &mut repeat_scratch, &mut repeat_output)
                .expect("identical-word decode")
        });
        report_allocations(&self.spec.id::<F>("exact-repeat", field), repeated);

        let ((scratch, output), retained) = measure_allocations(|| {
            let (plan, mut scratch, mut output) = self.prepared_state();
            plan.decode_into(&self.received, &mut scratch, &mut output)
                .expect("retained scratch decode");
            (scratch, output)
        });
        black_box((&scratch, &output));
        report_allocations(&self.spec.id::<F>("retained-scratch", field), retained);
    }
}
