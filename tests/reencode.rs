#![cfg(feature = "std")]

//! Re-encoding decodes identically to the direct module path.
//!
//! The factor-reduced re-encoding path and the direct interpolation path must
//! return the same Guruswami–Sudan list for every received word. Scoring
//! re-filters candidates by Hamming distance against the original received
//! word, so any valid interpolation polynomial yields exactly the list; these
//! tests exercise the full re-encoding pipeline (helper interpolation, shifted
//! module construction, factor reconstruction, candidate unshift) against the
//! direct path as the differential oracle.

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8, Gf16, Gf32};
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
    Polynomial,
};

const ROOT_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256);
const PARAMETER_LIMITS: ParameterLimits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);

/// Deterministic xorshift-style stream, avoiding a runtime RNG dependency.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 ^ (self.0 >> 31)
    }
}

fn element<F: FieldKernels>(value: u64) -> F::Elem {
    F::read(&value.to_le_bytes()[..F::BYTES])
}

fn nonzero_delta<F: FieldKernels>(value: u64) -> F::Elem {
    element::<F>((value & 0x7f) | 1)
}

fn sorted<F: FieldKernels>(mut candidates: Vec<Polynomial<F>>) -> Vec<Polynomial<F>> {
    candidates.sort_by(|left, right| {
        left.degree()
            .cmp(&right.degree())
            .then_with(|| left.as_packed().cmp(right.as_packed()))
    });
    candidates
}

fn decode<F: ButterflyKernels>(plan: &GsPlan<F>, received: &[F::Elem]) -> Vec<Polynomial<F>> {
    let mut output = Vec::new();
    plan.decode_into(received, &mut DecodeScratch::new(), &mut output)
        .unwrap();
    output
}

/// Decode many received words through both paths and require identical lists.
fn differential<F: ButterflyKernels>(n: usize, k: usize, target_radius: usize, seed: u64) {
    let parameters = GsParameters::search::<F>(n, k - 1, target_radius, PARAMETER_LIMITS).unwrap();
    let points: Vec<F::Elem> = (0..n as u64).map(element::<F>).collect();
    let domain = EvaluationDomain::arbitrary(points.clone()).unwrap();

    let plan_off = GsPlan::new(parameters, domain.clone(), ROOT_LIMITS)
        .unwrap()
        .with_reencode(false)
        .unwrap();
    let plan_on = GsPlan::new(parameters, domain, ROOT_LIMITS)
        .unwrap()
        .with_reencode(true)
        .unwrap();
    assert!(plan_on.uses_reencode());
    assert!(!plan_off.uses_reencode());

    let mut rng = Lcg(seed);
    let mut recovered = 0_usize;
    for _ in 0..12 {
        let message: Vec<F::Elem> = (0..k).map(|_| element::<F>(rng.next())).collect();
        let message = Polynomial::<F>::from_coefficients(&message).unwrap();
        let mut received = message.evaluate_many(&points).unwrap();
        let base = (rng.next() as usize) % n;
        for offset in 0..target_radius {
            let position = (base + offset) % n;
            received[position] = received[position].add(nonzero_delta::<F>(rng.next()));
        }

        let direct = sorted(decode(&plan_off, &received));
        let reencoded = sorted(decode(&plan_on, &received));
        assert_eq!(
            direct, reencoded,
            "re-encoding diverged from the direct path"
        );
        if reencoded.contains(&message) {
            recovered += 1;
        }
    }
    // The corrupted words sit within the target radius, so the true message is
    // in the list on both paths.
    assert_eq!(recovered, 12);
}

#[test]
fn gf8_high_rate_reencoding_matches_direct_path() {
    differential::<Gf8>(8, 6, 1, 0x1234);
}

#[test]
fn gf16_high_rate_reencoding_matches_direct_path() {
    differential::<Gf16>(16, 12, 2, 0xabcd);
}

#[test]
fn gf32_large_high_rate_reencoding_matches_direct_path() {
    differential::<Gf32>(32, 24, 4, 0x9e37);
}

#[test]
fn conservative_selector_gates_on_rate_and_length() {
    // Large and high-rate: auto-selects re-encoding.
    let large = GsParameters::search::<Gf32>(32, 23, 4, PARAMETER_LIMITS).unwrap();
    let points: Vec<<Gf32 as Field>::Elem> = (0..32u64).map(element::<Gf32>).collect();
    let domain = EvaluationDomain::<Gf32>::arbitrary(points).unwrap();
    let plan = GsPlan::new(large, domain, ROOT_LIMITS).unwrap();
    assert!(plan.uses_reencode());
    assert!(plan.prepared_bytes() > 0);

    // Tiny geometry: stays on the direct path even at high rate.
    let tiny = GsParameters::search::<Gf16>(15, 11, 2, PARAMETER_LIMITS).unwrap();
    let tiny_points: Vec<<Gf16 as Field>::Elem> = (0..15u64).map(element::<Gf16>).collect();
    let tiny_domain = EvaluationDomain::<Gf16>::arbitrary(tiny_points).unwrap();
    let tiny_plan = GsPlan::new(tiny, tiny_domain, ROOT_LIMITS).unwrap();
    assert!(!tiny_plan.uses_reencode());

    // Low rate at sufficient length: stays on the direct path.
    let low = GsParameters::search::<Gf32>(32, 8, 8, PARAMETER_LIMITS).unwrap();
    let low_points: Vec<<Gf32 as Field>::Elem> = (0..32u64).map(element::<Gf32>).collect();
    let low_domain = EvaluationDomain::<Gf32>::arbitrary(low_points).unwrap();
    let low_plan = GsPlan::new(low, low_domain, ROOT_LIMITS).unwrap();
    assert!(!low_plan.uses_reencode());
}
