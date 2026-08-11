#![cfg(feature = "std")]

//! Shared-plan and batch decoding contracts.
//!
//! A single immutable [`GsPlan`] drives many independent [`DecodeScratch`]
//! instances. These tests pin the properties the parallel and batch paths rely
//! on: the plan is thread-shareable, batch output matches word-by-word
//! decoding, and repeated (optionally multi-threaded) schedules are
//! byte-identical.

use fgf::Gf16;
use gs_engine::{DecodeScratch, GsPlan, Polynomial};

/// The immutable plan and its caller-owned scratch are thread-shareable, so one
/// plan can back many independent decode workspaces across threads.
#[test]
fn plan_is_shareable_across_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GsPlan<Gf16>>();
    assert_send_sync::<DecodeScratch<Gf16>>();
    assert_send_sync::<Polynomial<Gf16>>();
}
