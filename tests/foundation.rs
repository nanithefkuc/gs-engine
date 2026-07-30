use cafft::basis::{
    conversion_scratch_elements, monomial_to_novel_bytes, monomial_to_novel_with_scratch,
    novel_to_monomial_bytes, novel_to_monomial_with_scratch,
};
use cafft::core::transform::TransformPlan;
use fff::field::{Elem, Field};
use fff::{Gf8, Gf16};
use gs_engine::ConfigError;
use gs_engine::geometry::{checked_product, checked_sum, try_zeroed};

fn element_round_trip<F: cafft::core::kernel::ButterflyKernels>() {
    let plan = TransformPlan::<F>::new(8).expect("valid transform plan");
    let original: Vec<_> = (0..plan.size())
        .map(|index| F::GENERATOR.pow(index as u64))
        .collect();
    let mut values = original.clone();
    let mut scratch = vec![F::Elem::ZERO; conversion_scratch_elements(plan.size())];

    monomial_to_novel_with_scratch(&mut values, &plan, &mut scratch)
        .expect("matching coefficient and scratch geometry");
    novel_to_monomial_with_scratch(&mut values, &plan, &mut scratch)
        .expect("matching coefficient and scratch geometry");

    assert_eq!(values, original);
}

#[test]
fn gf8_and_gf16_plans_use_the_direct_fff_types() {
    element_round_trip::<Gf8>();
    element_round_trip::<Gf16>();
}

#[test]
fn byte_row_basis_conversion_round_trips() {
    let plan = TransformPlan::<Gf16>::new(8).expect("valid transform plan");
    let row_len = 4;
    let mut rows = vec![0u8; plan.size() * row_len];
    for (index, row) in rows.chunks_exact_mut(row_len).enumerate() {
        Gf16::write(&mut row[..Gf16::BYTES], Gf16::GENERATOR.pow(index as u64));
        Gf16::write(
            &mut row[Gf16::BYTES..],
            Gf16::GENERATOR.pow((index + 17) as u64),
        );
    }
    let original = rows.clone();
    let mut scratch = vec![0u8; conversion_scratch_elements(plan.size()) * row_len];

    monomial_to_novel_bytes(&mut rows, row_len, &plan, &mut scratch)
        .expect("matching byte-row and scratch geometry");
    novel_to_monomial_bytes(&mut rows, row_len, &plan, &mut scratch)
        .expect("matching byte-row and scratch geometry");

    assert_eq!(rows, original);
}

#[test]
fn checked_geometry_rejects_overflow() {
    assert_eq!(checked_product("rows", 7, 9), Ok(63));
    assert_eq!(checked_sum("coefficients", 7, 9), Ok(16));
    assert_eq!(
        checked_product("rows", usize::MAX, 2),
        Err(ConfigError::GeometryOverflow { context: "rows" })
    );
    assert_eq!(
        checked_sum("coefficients", usize::MAX, 1),
        Err(ConfigError::GeometryOverflow {
            context: "coefficients"
        })
    );

    let values = try_zeroed::<u16>("scratch", 8).expect("small allocation");
    assert_eq!(values, vec![0; 8]);
}
