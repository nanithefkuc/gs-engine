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
fn warmed_changed_word_decode_uses_no_internal_heap_allocations() {
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
    let clean = message.evaluate_many(&points).unwrap();
    let corrupt = |errors: &[(usize, u16)]| {
        let mut received = clean.clone();
        for &(index, delta) in errors {
            received[index] = received[index].add(gf16(delta));
        }
        received
    };
    // Three distinct received words, each within the target radius of the same
    // codeword but corrupted in different positions.
    let a = corrupt(&[
        (0, 0x0007),
        (2, 0x1000),
        (4, 0x00a1),
        (6, 0x0f0f),
        (8, 0x2020),
        (10, 0x0003),
    ]);
    let b = corrupt(&[
        (1, 0x0003),
        (3, 0x2020),
        (5, 0x0f0f),
        (7, 0x00a1),
        (9, 0x1000),
        (11, 0x0007),
    ]);
    let c = corrupt(&[
        (2, 0x0011),
        (4, 0x0101),
        (6, 0x1010),
        (8, 0x0110),
        (10, 0x1100),
        (12, 0x1111),
    ]);

    let plan = GsPlan::new(
        parameters,
        EvaluationDomain::arbitrary(points).unwrap(),
        AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256),
    )
    .unwrap();
    let mut scratch = DecodeScratch::new();
    let mut output = Vec::new();
    plan.prepare_scratch(&mut scratch, &mut output).unwrap();
    // Prepared-plan memory is bounded and reportable before any decode.
    assert!(plan.prepared_bytes() > 0);
    // Warm every distinct word so all data-dependent capacity is reached.
    for word in [&b, &a, &c] {
        plan.decode_into(word, &mut scratch, &mut output).unwrap();
        assert!(output.contains(&message));
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
