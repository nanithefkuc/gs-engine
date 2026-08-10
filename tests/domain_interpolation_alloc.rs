//! Zero-allocation contract for the additive transform received-word path.
//!
//! Isolated from `domain_interpolation.rs` because the global counting
//! allocator must not be polluted by concurrent test allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use fgf::Gf16;
use fgf::field::Field;
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
    Polynomial,
};

const ROOT_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256);
const PARAMETER_LIMITS: ParameterLimits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);

fn elem_u64(value: u64) -> <Gf16 as Field>::Elem {
    Gf16::read(&value.to_le_bytes()[..Gf16::BYTES])
}

struct CountingAllocator;

static TRACKING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn warmed_changed_word_additive_decode_uses_no_heap_allocations() {
    let parameters = GsParameters::search::<Gf16>(16, 4, 6, PARAMETER_LIMITS).unwrap();
    let domain = EvaluationDomain::<Gf16>::additive_subspace(16).unwrap();
    let points = domain.points().to_vec();

    let message = Polynomial::<Gf16>::from_coefficients(&[
        elem_u64(0x1234),
        elem_u64(0xabcd),
        elem_u64(0x0108),
        elem_u64(0xbeef),
        elem_u64(0x2222),
    ])
    .unwrap();
    let clean = message.evaluate_many(&points).unwrap();

    let corrupt = |errors: &[(usize, u64)]| -> Vec<<Gf16 as Field>::Elem> {
        let mut word = clean.clone();
        for &(position, value) in errors {
            word[position] = word[position].add(elem_u64(value));
        }
        word
    };

    // Three distinct received words, each within the target radius of the
    // same codeword but corrupted in different positions.
    let a = corrupt(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]);
    let b = corrupt(&[(6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)]);
    let c = corrupt(&[(12, 13), (13, 14), (14, 15), (15, 16), (0, 17), (1, 18)]);

    let plan = GsPlan::new(parameters, domain, ROOT_LIMITS).unwrap();
    let mut scratch = DecodeScratch::new();
    let mut output = Vec::new();
    plan.prepare_scratch(&mut scratch, &mut output).unwrap();
    assert!(plan.prepared_bytes() > 0);

    // Warm every distinct word so all data-dependent capacity is reached.
    for word in [&b, &a, &c] {
        plan.decode_into(word, &mut scratch, &mut output).unwrap();
    }

    ALLOCATIONS.store(0, Ordering::Relaxed);
    TRACKING.store(true, Ordering::SeqCst);
    for word in [&b, &a, &c, &b] {
        plan.decode_into(word, &mut scratch, &mut output).unwrap();
    }
    TRACKING.store(false, Ordering::SeqCst);

    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert!(output.contains(&message));
}
