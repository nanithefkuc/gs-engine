//! Domain-specialized interpolation tests (roadmap work package 3).
//!
//! Verify that additive-subspace and affine-coset domains produce the same
//! received-word interpolant `R` and vanishing polynomial `G` as the arbitrary
//! Newton path, and that end-to-end decode results match across domain kinds.
//! The zero-allocation contract for the additive transform path lives in
//! `domain_interpolation_alloc.rs`.

use fgf::field::Elem;
use fgf::field::Field;
use fgf::kernel::FieldKernels;
use fgf::{Gf8, Gf16};
use gs_engine::{
    AlekhnovichLimits, BivariatePolynomial, DecodeScratch, EvaluationDomain, GsParameters, GsPlan,
    InterpolationPlan, ModuleScratch, ParameterLimits, Polynomial, interpolate_module_into,
};

const ROOT_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256);
const PARAMETER_LIMITS: ParameterLimits = ParameterLimits::new(8, 16, usize::MAX, usize::MAX);

fn elem_u64<F: Field>(value: u64) -> F::Elem {
    F::read(&value.to_le_bytes()[..F::BYTES])
}

/// Deterministic pseudo-random element generator (xorshift64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn elem<F: FieldKernels>(&mut self) -> F::Elem {
        elem_u64::<F>(self.next())
    }

    fn polynomial<F: FieldKernels>(&mut self, degree: usize) -> Polynomial<F> {
        let coefficients: Vec<F::Elem> = (0..=degree).map(|_| self.elem::<F>()).collect();
        Polynomial::from_coefficients(&coefficients).unwrap()
    }
}

/// Build a corrupted received word within the target radius of `message`.
fn corrupted<F: FieldKernels>(
    parameters: GsParameters,
    message: &Polynomial<F>,
    points: &[F::Elem],
    rng: &mut Rng,
) -> Vec<F::Elem> {
    let n = parameters.code_length();
    let radius = parameters.target_radius();
    let mut received = message.evaluate_many(points).unwrap();
    // Corrupt the last `radius` positions with distinct nonzero deltas.
    for (offset, value) in received[n - radius..].iter_mut().enumerate() {
        *value = value.add(elem_u64::<F>((offset + 1) as u64));
    }
    let _ = rng;
    received
}

// ---------------------------------------------------------------------------
// 1. R(alpha_i) == received_i and G(alpha_i) == 0
//
// The transform received-word path is exercised through the module
// interpolation backend. validate_result inside interpolate_module_into
// checks all Hasse constraints, which include R(alpha_i) = received_i for
// multiplicity >= 1. A successful return is the proof.
// ---------------------------------------------------------------------------

#[test]
fn transform_received_interpolant_satisfies_constraints_on_additive_domain() {
    fn check<F: butterfly_fft::core::kernel::ButterflyKernels>(size: usize, rng: &mut Rng) {
        let max_degree = size / 3;
        let radius = size * 2 / 5;
        let parameters =
            GsParameters::search::<F>(size, max_degree, radius, PARAMETER_LIMITS).unwrap();
        let domain = EvaluationDomain::<F>::additive_subspace(size).unwrap();
        let points = domain.points();
        let message = rng.polynomial::<F>(max_degree);
        let received = corrupted::<F>(parameters, &message, points, rng);

        let plan = InterpolationPlan::new_with_domain(parameters, &domain).unwrap();
        let mut scratch = ModuleScratch::new();
        let mut output = BivariatePolynomial::zero();
        interpolate_module_into(
            parameters,
            points,
            &received,
            &plan,
            Some(&domain),
            &mut scratch,
            &mut output,
        )
        .unwrap();
        assert!(!output.is_zero());
    }
    let mut rng = Rng(0x9e37_79b9);
    check::<Gf8>(16, &mut rng);
    check::<Gf16>(16, &mut rng);
}

// ---------------------------------------------------------------------------
// 2. Subspace G matches incremental G
//
// Both plans share the same column shifts and Newton basis; only G differs in
// its construction. Since G is the product of (X + alpha_i), both paths must
// produce the identical polynomial. We compare through the public module
// interpolation output: the final bivariate Q depends on G, so equal Q implies
// equal G, equal R, and equal reduction.
// ---------------------------------------------------------------------------

#[test]
fn transform_module_matches_newton_module_on_additive_domain() {
    fn check<F: butterfly_fft::core::kernel::ButterflyKernels>(size: usize, rng: &mut Rng) {
        let max_degree = size / 3;
        let radius = size * 2 / 5;
        let parameters =
            GsParameters::search::<F>(size, max_degree, radius, PARAMETER_LIMITS).unwrap();
        let domain = EvaluationDomain::<F>::additive_subspace(size).unwrap();
        let points = domain.points();
        let message = rng.polynomial::<F>(max_degree);
        let received = corrupted::<F>(parameters, &message, points, rng);

        // Transform path (domain-aware plan).
        let domain_plan = InterpolationPlan::new_with_domain(parameters, &domain).unwrap();
        let mut t_scratch = ModuleScratch::new();
        let mut t_output = BivariatePolynomial::zero();
        interpolate_module_into(
            parameters,
            points,
            &received,
            &domain_plan,
            Some(&domain),
            &mut t_scratch,
            &mut t_output,
        )
        .unwrap();

        // Newton path (arbitrary-domain plan, no domain parameter).
        let arb_plan = InterpolationPlan::new(parameters, points).unwrap();
        let mut n_scratch = ModuleScratch::new();
        let mut n_output = BivariatePolynomial::zero();
        interpolate_module_into(
            parameters,
            points,
            &received,
            &arb_plan,
            None,
            &mut n_scratch,
            &mut n_output,
        )
        .unwrap();

        assert_eq!(
            t_output, n_output,
            "transform and Newton module interpolation disagree at size {size}"
        );
    }
    let mut rng = Rng(0x1234_5678_9abc_def0);
    for &size in &[8usize, 16, 32] {
        check::<Gf8>(size, &mut rng);
        check::<Gf16>(size, &mut rng);
    }
}

// ---------------------------------------------------------------------------
// 3. End-to-end: additive decode matches arbitrary decode
// ---------------------------------------------------------------------------

#[test]
fn additive_decode_matches_arbitrary_decode() {
    fn check<F: butterfly_fft::core::kernel::ButterflyKernels>(size: usize, rng: &mut Rng) {
        let max_degree = size / 3;
        let radius = size * 2 / 5;
        let parameters =
            GsParameters::search::<F>(size, max_degree, radius, PARAMETER_LIMITS).unwrap();
        let domain = EvaluationDomain::<F>::additive_subspace(size).unwrap();
        let points = domain.points();
        let message = rng.polynomial::<F>(max_degree);
        let received = corrupted::<F>(parameters, &message, points, rng);

        let arb_plan = GsPlan::new(
            parameters,
            EvaluationDomain::arbitrary(points.to_vec()).unwrap(),
            ROOT_LIMITS,
        )
        .unwrap();
        let add_plan = GsPlan::new(parameters, domain, ROOT_LIMITS).unwrap();

        let mut arb_out = Vec::new();
        let mut add_out = Vec::new();
        let mut arb_scr = DecodeScratch::new();
        let mut add_scr = DecodeScratch::new();
        arb_plan
            .prepare_scratch(&mut arb_scr, &mut arb_out)
            .unwrap();
        add_plan
            .prepare_scratch(&mut add_scr, &mut add_out)
            .unwrap();
        arb_plan
            .decode_into(&received, &mut arb_scr, &mut arb_out)
            .unwrap();
        add_plan
            .decode_into(&received, &mut add_scr, &mut add_out)
            .unwrap();

        assert_eq!(arb_out, add_out, "additive decode disagrees at size {size}");
        assert!(add_out.contains(&message));
    }
    let mut rng = Rng(0xfeed_face_dead_beef);
    for &size in &[8usize, 16, 32, 64] {
        check::<Gf8>(size, &mut rng);
        check::<Gf16>(size, &mut rng);
    }
}

// ---------------------------------------------------------------------------
// 4. End-to-end: affine decode matches arbitrary decode
// ---------------------------------------------------------------------------

#[test]
fn affine_decode_matches_arbitrary_decode() {
    fn check<F: butterfly_fft::core::kernel::ButterflyKernels>(size: usize, rng: &mut Rng) {
        let max_degree = size / 3;
        let radius = size * 2 / 5;
        let parameters =
            GsParameters::search::<F>(size, max_degree, radius, PARAMETER_LIMITS).unwrap();

        // Shift outside the subspace basis so the coset is distinct.
        let shift = elem_u64::<F>(1u64 << size.trailing_zeros());
        let domain = EvaluationDomain::<F>::affine_coset(size, shift).unwrap();
        let points = domain.points();
        let message = rng.polynomial::<F>(max_degree);
        let received = corrupted::<F>(parameters, &message, points, rng);

        let arb_plan = GsPlan::new(
            parameters,
            EvaluationDomain::arbitrary(points.to_vec()).unwrap(),
            ROOT_LIMITS,
        )
        .unwrap();
        let aff_plan = GsPlan::new(parameters, domain, ROOT_LIMITS).unwrap();

        let mut arb_out = Vec::new();
        let mut aff_out = Vec::new();
        let mut arb_scr = DecodeScratch::new();
        let mut aff_scr = DecodeScratch::new();
        arb_plan
            .prepare_scratch(&mut arb_scr, &mut arb_out)
            .unwrap();
        aff_plan
            .prepare_scratch(&mut aff_scr, &mut aff_out)
            .unwrap();
        arb_plan
            .decode_into(&received, &mut arb_scr, &mut arb_out)
            .unwrap();
        aff_plan
            .decode_into(&received, &mut aff_scr, &mut aff_out)
            .unwrap();

        assert_eq!(arb_out, aff_out, "affine decode disagrees at size {size}");
        assert!(aff_out.contains(&message));
    }
    let mut rng = Rng(0xc0de_dead);
    for &size in &[8usize, 16, 32] {
        check::<Gf8>(size, &mut rng);
        check::<Gf16>(size, &mut rng);
    }
}
