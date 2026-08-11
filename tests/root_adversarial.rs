#![cfg(feature = "std")]

//! Adversarial root-extraction fixtures.
//!
//! List-rich polynomials (many distinct bounded roots) and no-root polynomials
//! (roots only beyond the degree bound) exercise the divide-and-conquer path,
//! its pooled base-field factor stack, and the degree/candidate-count bounds.
//! Roth–Ruckenstein is the differential oracle: both backends must return the
//! same complete root set, every output must satisfy `Q(X,f(X)) == 0`, and the
//! candidate count must respect `deg_Y Q`.

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::kernel::FieldKernels;
use fgf::{Gf8, Gf16};
use gs_engine::{
    AlekhnovichLimits, AlekhnovichScratch, BivariatePolynomial, Polynomial, RothRuckensteinLimits,
    alekhnovich_roots, roth_ruckenstein_roots,
};

const ROTH: RothRuckensteinLimits = RothRuckensteinLimits::new(10_000_000, 4096);
// A zero crossover forces the divide-and-conquer path for every input.
const FORCE_ALEKHNOVICH: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 4096)
        .with_roth_ruckenstein_crossover(0);

fn element<F: FieldKernels>(value: u64) -> F::Elem {
    F::read(&value.to_le_bytes()[..F::BYTES])
}

/// Build `Q(X,Y) = prod_i (Y - f_i(X))` from the given roots.
fn product_of_y_minus<F: ButterflyKernels>(roots: &[Polynomial<F>]) -> BivariatePolynomial<F> {
    let mut rows = vec![Polynomial::<F>::one().unwrap()];
    for root in roots {
        let mut next = vec![Polynomial::<F>::zero(); rows.len() + 1];
        for (degree, row) in rows.iter().enumerate() {
            // Y - root = Y + root in characteristic two.
            next[degree]
                .add_assign(&row.multiply(root).unwrap())
                .unwrap();
            next[degree + 1].add_assign(row).unwrap();
        }
        rows = next;
    }
    BivariatePolynomial::from_y_coefficients(rows)
}

fn sorted<F: FieldKernels>(mut roots: Vec<Polynomial<F>>) -> Vec<Polynomial<F>> {
    roots.sort_by(|left, right| {
        left.degree()
            .cmp(&right.degree())
            .then_with(|| left.as_packed().cmp(right.as_packed()))
    });
    roots
}

/// A distinct degree-`degree` polynomial keyed by `index`.
fn keyed<F: FieldKernels>(index: u64, degree: usize) -> Polynomial<F> {
    let mut coefficients: Vec<F::Elem> = (0..degree)
        .map(|position| element::<F>(index.wrapping_mul(7) + position as u64 + 3))
        .collect();
    coefficients.push(element::<F>(index * 5 + 1)); // nonzero leading term
    Polynomial::from_coefficients(&coefficients).unwrap()
}

fn extract<F: ButterflyKernels>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
) -> (Vec<Polynomial<F>>, Vec<Polynomial<F>>) {
    let rr = sorted(roth_ruckenstein_roots(q, max_degree, ROTH).unwrap());
    let mut scratch = AlekhnovichScratch::new();
    let alekhnovich =
        sorted(alekhnovich_roots(q, max_degree, FORCE_ALEKHNOVICH, &mut scratch).unwrap());
    (rr, alekhnovich)
}

fn list_rich<F: ButterflyKernels>(count: u64, root_degree: usize, max_degree: usize) {
    let roots: Vec<Polynomial<F>> = (0..count)
        .map(|index| keyed::<F>(index, root_degree))
        .collect();
    let q = product_of_y_minus(&roots);
    let deg_y = q.y_degree().unwrap();
    assert_eq!(deg_y, count as usize);

    let (rr, alekhnovich) = extract(&q, max_degree);
    let expected = sorted(roots);
    assert_eq!(rr, expected, "Roth–Ruckenstein missed a bounded root");
    assert_eq!(
        alekhnovich, expected,
        "Alekhnovich diverged from Roth–Ruckenstein"
    );
    assert!(rr.len() <= deg_y, "candidate count exceeds deg_Y Q");
    for root in &rr {
        assert!(q.has_root(root).unwrap(), "Q(X,f(X)) != 0");
    }
}

fn no_bounded_root<F: ButterflyKernels>(count: u64, max_degree: usize) {
    // Every factor's root has degree `max_degree + 1`, so none is admissible.
    let roots: Vec<Polynomial<F>> = (0..count)
        .map(|index| keyed::<F>(index + 1, max_degree + 1))
        .collect();
    let q = product_of_y_minus(&roots);
    let (rr, alekhnovich) = extract(&q, max_degree);
    assert!(
        rr.is_empty(),
        "Roth–Ruckenstein returned an over-degree root"
    );
    assert!(
        alekhnovich.is_empty(),
        "Alekhnovich returned an over-degree root"
    );
}

fn mixed_bound<F: ButterflyKernels>(max_degree: usize) {
    let bounded: Vec<Polynomial<F>> = (0..3).map(|index| keyed::<F>(index, max_degree)).collect();
    let mut roots = bounded.clone();
    // Two extra roots strictly beyond the degree bound.
    roots.push(keyed::<F>(50, max_degree + 1));
    roots.push(keyed::<F>(51, max_degree + 2));
    let q = product_of_y_minus(&roots);

    let (rr, alekhnovich) = extract(&q, max_degree);
    let expected = sorted(bounded);
    assert_eq!(rr, expected, "Roth–Ruckenstein kept an over-degree root");
    assert_eq!(
        alekhnovich, expected,
        "Alekhnovich kept an over-degree root"
    );
    assert!(rr.len() <= q.y_degree().unwrap());
}

#[test]
fn gf8_list_rich_matches_roth_ruckenstein() {
    list_rich::<Gf8>(5, 2, 3);
    list_rich::<Gf8>(7, 1, 4);
}

#[test]
fn gf16_list_rich_matches_roth_ruckenstein() {
    list_rich::<Gf16>(6, 3, 4);
    list_rich::<Gf16>(9, 2, 5);
}

#[test]
fn no_bounded_root_yields_empty_lists() {
    no_bounded_root::<Gf8>(3, 2);
    no_bounded_root::<Gf16>(4, 3);
}

#[test]
fn over_degree_roots_are_filtered() {
    mixed_bound::<Gf8>(3);
    mixed_bound::<Gf16>(4);
}
