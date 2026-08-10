#![cfg(feature = "internals")]

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Field;
use fgf::{Gf8, Gf16};
use gs_engine::{
    DecodeScratch, EvaluationDomain, Polynomial, ScoringStrategy, score_candidates_with_strategy,
};

fn element<F: Field>(value: u64) -> F::Elem {
    F::read(&value.to_le_bytes()[..F::BYTES])
}

fn candidate<F: ButterflyKernels>(degree: usize, seed: usize) -> Polynomial<F> {
    let coefficients: Vec<_> = (0..=degree)
        .map(|index| element::<F>((seed * 251 + index * 17 + 1) as u64))
        .collect();
    Polynomial::from_coefficients(&coefficients).unwrap()
}

fn forced_scoring_matches<F: ButterflyKernels>() {
    let domain = EvaluationDomain::<F>::additive_subspace(64).unwrap();
    let candidates = vec![
        candidate::<F>(3, 1),
        candidate::<F>(7, 2),
        candidate::<F>(31, 3),
        candidate::<F>(47, 4),
    ];
    let received = candidates[0].evaluate_many(domain.points()).unwrap();
    let mut horner = Vec::new();
    let mut butterfly_fft = Vec::new();
    let mut automatic = Vec::new();
    score_candidates_with_strategy(
        &domain,
        &received,
        &candidates,
        0,
        ScoringStrategy::Horner,
        &mut DecodeScratch::new(),
        &mut horner,
    )
    .unwrap();
    score_candidates_with_strategy(
        &domain,
        &received,
        &candidates,
        0,
        ScoringStrategy::ButterflyFft,
        &mut DecodeScratch::new(),
        &mut butterfly_fft,
    )
    .unwrap();
    score_candidates_with_strategy(
        &domain,
        &received,
        &candidates,
        0,
        ScoringStrategy::Auto,
        &mut DecodeScratch::new(),
        &mut automatic,
    )
    .unwrap();
    assert_eq!(horner, vec![candidates[0].clone()]);
    assert_eq!(butterfly_fft, horner);
    assert_eq!(automatic, horner);
}

#[test]
fn forced_scoring_strategies_match_in_gf8() {
    forced_scoring_matches::<Gf8>();
}

#[test]
fn forced_scoring_strategies_match_in_gf16() {
    forced_scoring_matches::<Gf16>();
}
