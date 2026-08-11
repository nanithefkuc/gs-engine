#![cfg(feature = "std")]

//! The factor-reduced re-encoding path allocates nothing in steady state.
//!
//! Mirrors `decode_allocations.rs` but forces the re-encoding path so its
//! prepared plan and reused scratch are exercised. After warm-up, alternating
//! received words must perform no internal heap allocation.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use fgf::Gf16;
use fgf::field::Field;
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
    Polynomial,
};

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

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn gf16(value: u16) -> <Gf16 as Field>::Elem {
    Gf16::read(&value.to_le_bytes())
}

#[test]
fn warmed_reencoded_decode_uses_no_internal_heap_allocations() {
    let parameters = GsParameters::search::<Gf16>(
        16,
        11,
        2,
        ParameterLimits::new(8, 16, usize::MAX, usize::MAX),
    )
    .unwrap();
    let points: Vec<_> = (0..16).map(gf16).collect();
    let message = Polynomial::<Gf16>::from_coefficients(
        &(0..12)
            .map(|i| gf16((3 * i + 1) as u16))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let clean = message.evaluate_many(&points).unwrap();
    let corrupt = |errors: &[(usize, u16)]| {
        let mut word = clean.clone();
        for &(position, value) in errors {
            word[position] = word[position].add(gf16(value));
        }
        word
    };
    let a = corrupt(&[(1, 7), (10, 3)]);
    let b = corrupt(&[(4, 5), (13, 9)]);
    let c = corrupt(&[(0, 2), (8, 6)]);

    let plan = GsPlan::new(
        parameters,
        EvaluationDomain::arbitrary(points).unwrap(),
        AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256),
    )
    .unwrap()
    .with_reencode(true)
    .unwrap();
    assert!(plan.uses_reencode());
    assert!(plan.prepared_bytes() > 0);

    let mut scratch = DecodeScratch::new();
    let mut output = Vec::new();
    plan.prepare_scratch(&mut scratch, &mut output).unwrap();
    for word in [&b, &a, &c, &b, &a, &c] {
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
