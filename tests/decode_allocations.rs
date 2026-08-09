#![cfg(feature = "std")]

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
        // SAFETY: the request is forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout came from the system allocator.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the pointer and layout came from the system allocator and
        // the requested size is forwarded unchanged.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn gf16(value: u16) -> <Gf16 as Field>::Elem {
    Gf16::read(&value.to_le_bytes())
}

#[test]
fn warmed_decode_uses_no_internal_heap_allocations() {
    let parameters = GsParameters::new::<Gf16>(
        15,
        4,
        6,
        2,
        4,
        17,
        ParameterLimits::new(8, 16, usize::MAX, usize::MAX),
    )
    .unwrap();
    let points: Vec<_> = (0..15).map(gf16).collect();
    let message = Polynomial::<Gf16>::from_coefficients(&[
        gf16(0x1234),
        gf16(0xabcd),
        gf16(0x0108),
        gf16(0xbeef),
        gf16(0x2222),
    ])
    .unwrap();
    let mut received = message.evaluate_many(&points).unwrap();
    for (offset, value) in received[9..].iter_mut().enumerate() {
        *value = value.add(gf16((offset + 1) as u16));
    }
    let plan = GsPlan::new(
        parameters,
        EvaluationDomain::arbitrary(points).unwrap(),
        AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256),
    )
    .unwrap();
    let mut scratch = DecodeScratch::new();
    let mut output = Vec::new();
    plan.prepare_scratch(&mut scratch, &mut output).unwrap();
    plan.decode_into(&received, &mut scratch, &mut output)
        .unwrap();

    ALLOCATIONS.store(0, Ordering::Relaxed);
    TRACKING.store(true, Ordering::SeqCst);
    plan.decode_into(&received, &mut scratch, &mut output)
        .unwrap();
    TRACKING.store(false, Ordering::SeqCst);

    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert!(output.contains(&message));
}
