use std::hint::black_box;
use std::time::{Duration, Instant};

use cafft::core::kernel::ButterflyKernels;
use fgf::field::Field;
use fgf::kernel::backend_for;
use fgf::{Gf8, Gf16};
use gs_engine::{
    AlekhnovichLimits, AlekhnovichScratch, BivariatePolynomial, Polynomial, RothRuckensteinLimits,
    alekhnovich_roots, roth_ruckenstein_roots,
};

const ROTH_LIMITS: RothRuckensteinLimits = RothRuckensteinLimits::new(10_000_000, 256);
const ALEKHNOVICH_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256)
        .with_roth_ruckenstein_crossover(0);

fn element<F: Field>(value: u64) -> F::Elem {
    let bytes = value.to_le_bytes();
    F::read(&bytes[..F::BYTES])
}

fn polynomial<F: ButterflyKernels>(coefficients: &[F::Elem]) -> Polynomial<F> {
    Polynomial::from_coefficients(coefficients).unwrap()
}

fn product_of_y_plus<F: ButterflyKernels>(roots: &[Polynomial<F>]) -> BivariatePolynomial<F> {
    let mut rows = vec![Polynomial::<F>::one().unwrap()];
    for root in roots {
        let mut product = vec![Polynomial::<F>::zero(); rows.len() + 1];
        for (y_degree, row) in rows.iter().enumerate() {
            product[y_degree]
                .add_assign(&row.multiply(root).unwrap())
                .unwrap();
            product[y_degree + 1].add_assign(row).unwrap();
        }
        rows = product;
    }
    BivariatePolynomial::from_y_coefficients(rows)
}

fn fixture<F: ButterflyKernels>(max_degree: usize) -> BivariatePolynomial<F> {
    let roots: Vec<_> = (0..4)
        .map(|root| {
            let coefficients: Vec<_> = (0..=max_degree)
                .map(|degree| element::<F>((root * 61 + degree * 29 + 1) as u64))
                .collect();
            polynomial(&coefficients)
        })
        .collect();
    product_of_y_plus(&roots)
}

fn elapsed(mut operation: impl FnMut(), iterations: usize) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    start.elapsed()
}

fn run<F: ButterflyKernels>(field: &str) {
    let backend = backend_for::<F>().name();
    for max_degree in [1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 384, 448, 512, 1_024] {
        let q = fixture::<F>(max_degree);
        let precision = q.weighted_degree(max_degree).unwrap().unwrap() + 1;
        let weighted_size = precision * q.y_coefficient_count();
        let iterations = (2_000 / max_degree).max(5);
        let roth = elapsed(
            || {
                black_box(roth_ruckenstein_roots(
                    black_box(&q),
                    max_degree,
                    ROTH_LIMITS,
                ))
                .unwrap();
            },
            iterations,
        );
        let mut scratch = AlekhnovichScratch::new();
        let alekhnovich = elapsed(
            || {
                black_box(alekhnovich_roots(
                    black_box(&q),
                    max_degree,
                    ALEKHNOVICH_LIMITS,
                    &mut scratch,
                ))
                .unwrap();
            },
            iterations,
        );
        println!(
            "{field},{backend},roth_ruckenstein,{max_degree},{weighted_size},{iterations},{}",
            roth.as_nanos()
        );
        println!(
            "{field},{backend},alekhnovich,{max_degree},{weighted_size},{iterations},{}",
            alekhnovich.as_nanos()
        );
    }
}

fn main() {
    println!("field,backend,algorithm,max_degree,weighted_size,iterations,nanoseconds");
    run::<Gf8>("gf8");
    run::<Gf16>("gf16");
}
