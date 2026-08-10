//! `gs-engine` reference oracle.
//!
//! Decodes a fixture with the in-repo decoder and returns its normalized
//! candidate set. This is the authority the frozen corpus is validated against
//! and the tie-breaker every external adapter is compared to. A macro
//! specializes the body per concrete field so the engine's `ButterflyKernels`
//! bound never has to be named here.

use fgf::field::Field;
use fgf::{Gf8, Gf16};
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
};

use crate::field::normalize_set;
use crate::fixture::{FieldTag, Fixture};

const ROOT_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(10_000_000, 1_000_000, usize::MAX, usize::MAX, 256);
const PARAMETER_LIMITS: ParameterLimits = ParameterLimits::new(64, 64, usize::MAX, usize::MAX);

/// Decode `fixture` with `gs-engine`; return the normalized candidate set.
pub fn decode(fixture: &Fixture) -> Result<Vec<Vec<Vec<u8>>>, String> {
    match fixture.field {
        FieldTag::Gf8 => decode_gf8(fixture),
        FieldTag::Gf16 => decode_gf16(fixture),
    }
}

macro_rules! decode_impl {
    ($name:ident, $F:ty) => {
        fn $name(fixture: &Fixture) -> Result<Vec<Vec<Vec<u8>>>, String> {
            let max_degree = fixture
                .k
                .checked_sub(1)
                .ok_or_else(|| "k must be at least 1".to_string())?;
            let points: Vec<_> = fixture
                .support
                .iter()
                .map(|bytes| <$F as Field>::read(bytes))
                .collect();
            let received: Vec<_> = fixture
                .received
                .iter()
                .map(|bytes| <$F as Field>::read(bytes))
                .collect();
            let parameters = GsParameters::new::<$F>(
                fixture.n,
                max_degree,
                fixture.target_radius,
                fixture.multiplicity,
                fixture.y_degree,
                fixture.weighted_degree,
                PARAMETER_LIMITS,
            )
            .map_err(|error| format!("parameters: {error:?}"))?;
            let domain = EvaluationDomain::<$F>::arbitrary(points)
                .map_err(|error| format!("domain: {error:?}"))?;
            let plan = GsPlan::new(parameters, domain, ROOT_LIMITS)
                .map_err(|error| format!("plan: {error:?}"))?;
            let mut scratch = DecodeScratch::new();
            let mut candidates = Vec::new();
            plan.decode_into(&received, &mut scratch, &mut candidates)
                .map_err(|error| format!("decode: {error:?}"))?;

            let set: Vec<Vec<Vec<u8>>> = candidates
                .iter()
                .map(|candidate| {
                    candidate
                        .coefficients()
                        .map(|coefficient| {
                            let mut bytes = vec![0_u8; <$F as Field>::BYTES];
                            <$F as Field>::write(&mut bytes, coefficient);
                            bytes
                        })
                        .collect()
                })
                .collect();
            Ok(normalize_set(&set, fixture.field))
        }
    };
}

decode_impl!(decode_gf8, Gf8);
decode_impl!(decode_gf16, Gf16);
