#![cfg(feature = "parallel")]

//! Shared-plan and batch decoding contracts.
//!
//! A single immutable [`GsPlan`] drives many independent [`DecodeScratch`]
//! instances. These tests pin the properties the parallel and batch paths rely
//! on: the plan is thread-shareable, batch output matches word-by-word
//! decoding, and repeated (optionally multi-threaded) schedules are
//! byte-identical.

use fgf::Gf16;
use fgf::field::Field;
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
    Polynomial,
};

fn gf16(value: u16) -> <Gf16 as Field>::Elem {
    Gf16::read(&value.to_le_bytes())
}

/// The immutable plan and its caller-owned scratch are thread-shareable, so one
/// plan can back many independent decode workspaces across threads.
#[test]
fn plan_is_shareable_across_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GsPlan<Gf16>>();
    assert_send_sync::<DecodeScratch<Gf16>>();
    assert_send_sync::<Polynomial<Gf16>>();
}

/// A small but real decode geometry: `n = 15`, `k = 4`, two errors, four words.
fn sample_plan() -> (GsPlan<Gf16>, Vec<[<Gf16 as Field>::Elem; 15]>, Polynomial<Gf16>) {
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
    let message =
        Polynomial::<Gf16>::from_coefficients(&[gf16(3), gf16(1), gf16(4), gf16(1)]).unwrap();
    let clean = message.evaluate_many(&points).unwrap();
    let corrupt = |errors: &[(usize, u16)]| {
        let mut word = [gf16(0); 15];
        for (index, value) in clean.iter().copied().enumerate() {
            word[index] = value;
        }
        for &(index, value) in errors {
            word[index] = gf16(value);
        }
        word
    };
    let words = vec![
        corrupt(&[(0, 9), (7, 2)]),
        corrupt(&[(3, 5), (11, 8)]),
        corrupt(&[(1, 6), (14, 1)]),
        corrupt(&[(2, 7), (9, 3)]),
    ];
    let plan = GsPlan::new(
        parameters,
        EvaluationDomain::arbitrary(points).unwrap(),
        AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256),
    )
    .unwrap();
    (plan, words, message)
}

fn decode_batch(
    plan: &GsPlan<Gf16>,
    words: &[[<Gf16 as Field>::Elem; 15]],
) -> Vec<Vec<Polynomial<Gf16>>> {
    let received: Vec<&[<Gf16 as Field>::Elem]> = words.iter().map(|w| w.as_slice()).collect();
    let mut scratches: Vec<DecodeScratch<Gf16>> = (0..words.len()).map(|_| DecodeScratch::new()).collect();
    let mut outputs: Vec<Vec<Polynomial<Gf16>>> = (0..words.len()).map(|_| Vec::new()).collect();
    for (i, word) in words.iter().enumerate() {
        plan.prepare_scratch(&mut scratches[i], &mut outputs[i]).unwrap();
        let _ = word;
    }
    plan.decode_batch_into(&received, &mut scratches, &mut outputs).unwrap();
    outputs
}

/// Batch decoding produces exactly the candidates that per-word `decode_into`
/// produces, in the same order.
#[test]
fn batch_matches_word_by_word_decode() {
    let (plan, words, message) = sample_plan();
    let mut per_word: Vec<Vec<Polynomial<Gf16>>> = Vec::new();
    for word in &words {
        let mut scratch = DecodeScratch::new();
        let mut output = Vec::new();
        plan.prepare_scratch(&mut scratch, &mut output).unwrap();
        plan.decode_into(word, &mut scratch, &mut output).unwrap();
        per_word.push(output);
    }

    let outputs = decode_batch(&plan, &words);
    assert_eq!(outputs.len(), per_word.len());
    for (batch, single) in outputs.iter().zip(&per_word) {
        assert_eq!(batch, single, "batch and single-thread output differ");
        assert!(batch.contains(&message), "message missing from batch output");
    }
}

/// Repeated batch decodes — including the multi-threaded path when the
/// `parallel` feature is on — return byte-identical output across runs.
#[test]
fn batch_is_deterministic_across_schedules() {
    let (plan, words, _message) = sample_plan();
    // Enough words to cross the default parallel batch crossover when the
    // feature is enabled; below it the path is sequential and still
    // deterministic. Duplicate the four words to reach the threshold.
    let words: Vec<[<Gf16 as Field>::Elem; 15]> = words
        .iter()
        .cycle()
        .take(gs_engine::PARALLEL_BATCH_CROSSOVER + 2)
        .copied()
        .collect();

    let first = decode_batch(&plan, &words);
    for _ in 0..3 {
        let again = decode_batch(&plan, &words);
        assert_eq!(again, first, "batch output changed across schedules");
    }
}

/// A length mismatch is rejected before any work runs.
#[test]
fn batch_rejects_mismatched_slices() {
    let (plan, words, _message) = sample_plan();
    let received: Vec<&[<Gf16 as Field>::Elem]> = words.iter().map(|w| w.as_slice()).collect();
    let mut scratches: Vec<DecodeScratch<Gf16>> = (0..words.len()).map(|_| DecodeScratch::new()).collect();
    let mut outputs: Vec<Vec<Polynomial<Gf16>>> = (0..words.len()).map(|_| Vec::new()).collect();
    // One too few scratch buffers.
    let err = plan
        .decode_batch_into(&received, &mut scratches[..words.len() - 1], &mut outputs)
        .unwrap_err();
    assert!(matches!(
        err,
        gs_engine::DecodeError::BatchLengthMismatch { received: 4, scratches: 3, outputs: 4 }
    ));
}
