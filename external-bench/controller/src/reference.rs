use fgf::field::Field;
use fgf::{Gf8, Gf16};
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
};

use crate::field::normalize_set;
use crate::fixture::{BatchFixture, FieldTag, Fixture};


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

pub fn decode_warm(fixture: &Fixture, reps: usize) -> u128 {
    match fixture.field {
        FieldTag::Gf8 => decode_warm_impl_gf8(fixture, reps),
        FieldTag::Gf16 => decode_warm_impl_gf16(fixture, reps),
    }
}

macro_rules! warm_impl {
    ($name:ident, $F:ty) => {
        fn $name(fixture: &Fixture, reps: usize) -> u128 {
            let max_degree = match fixture.k.checked_sub(1) {
                Some(d) => d,
                None => return 0,
            };
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
            let parameters = match GsParameters::new::<$F>(
                fixture.n,
                max_degree,
                fixture.target_radius,
                fixture.multiplicity,
                fixture.y_degree,
                fixture.weighted_degree,
                PARAMETER_LIMITS,
            ) {
                Ok(p) => p,
                Err(_) => return 0,
            };
            let domain = match EvaluationDomain::<$F>::arbitrary(points) {
                Ok(d) => d,
                Err(_) => return 0,
            };
            let plan = match GsPlan::new(parameters, domain, ROOT_LIMITS) {
                Ok(p) => p,
                Err(_) => return 0,
            };
            let mut scratch = DecodeScratch::new();
            let mut candidates = Vec::new();
            // Warm up.
            let _ = plan.decode_into(&received, &mut scratch, &mut candidates);
            candidates.clear();
            let mut samples: Vec<u128> = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = std::time::Instant::now();
                let _ = plan.decode_into(&received, &mut scratch, &mut candidates);
                samples.push(start.elapsed().as_nanos());
                candidates.clear();
            }
            samples.sort();
            samples[samples.len() / 2]
        }
    };
}

warm_impl!(decode_warm_impl_gf8, Gf8);
warm_impl!(decode_warm_impl_gf16, Gf16);

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


/// Time a warm batch decode of all words in `fixture`, returning the median
/// nanoseconds over `reps` runs. Builds one shared plan and one scratch per
/// word, warms once, then times `reps` full batch decodes. `parallel` selects
/// `decode_batch_into` (Rayon pool when the `parallel` feature is enabled and
/// the word count reaches the crossover) versus the in-order sequential loop.
pub fn decode_batch_warm(fixture: &BatchFixture, reps: usize, parallel: bool) -> u128 {
    match fixture.inner.field {
        FieldTag::Gf8 => decode_batch_warm_impl_gf8(fixture, reps, parallel),
        FieldTag::Gf16 => decode_batch_warm_impl_gf16(fixture, reps, parallel),
    }
}

macro_rules! batch_warm_impl {
    ($name:ident, $F:ty) => {
        fn $name(fixture: &BatchFixture, reps: usize, parallel: bool) -> u128 {
            let inner = &fixture.inner;
            let max_degree = match inner.k.checked_sub(1) {
                Some(d) => d,
                None => return 0,
            };
            let points: Vec<_> = inner
                .support
                .iter()
                .map(|bytes| <$F as Field>::read(bytes))
                .collect();
            let words: Vec<Vec<<$F as Field>::Elem>> = fixture
                .received_words
 .iter()
                .map(|word| {
                    word.iter().map(|bytes| <$F as Field>::read(bytes)).collect()
                })
                .collect();
            let parameters = match GsParameters::new::<$F>(
                inner.n,
                max_degree,
                inner.target_radius,
                inner.multiplicity,
                inner.y_degree,
                inner.weighted_degree,
                PARAMETER_LIMITS,
            ) {
                Ok(p) => p,
                Err(_) => return 0,
            };
            let domain = match EvaluationDomain::<$F>::arbitrary(points) {
                Ok(d) => d,
                Err(_) => return 0,
            };
            let plan = match GsPlan::new(parameters, domain, ROOT_LIMITS) {
                Ok(p) => p,
                Err(_) => return 0,
            };
            let count = words.len();
            let received_refs: Vec<&[<$F as Field>::Elem]> =
                words.iter().map(|w| w.as_slice()).collect();
            // Warm: one round of scratch/output allocation and a warm-up decode.
            let mut scratches: Vec<DecodeScratch<$F>> =
                (0..count).map(|_| DecodeScratch::new()).collect();
            let mut outputs: Vec<Vec<gs_engine::Polynomial<$F>>> =
                (0..count).map(|_| Vec::new()).collect();
            for (i, word) in words.iter().enumerate() {
                plan.prepare_scratch(&mut scratches[i], &mut outputs[i]).ok();
                let _ = word;
            }
            let _ = plan.decode_batch_into(&received_refs, &mut scratches, &mut outputs);

            let mut samples: Vec<u128> = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = std::time::Instant::now();
                if parallel {
                    let _ = plan.decode_batch_into(&received_refs, &mut scratches, &mut outputs);
                } else {
                    for (i, word) in words.iter().enumerate() {
                        let _ = plan.decode_into(word, &mut scratches[i], &mut outputs[i]);
                    }
                }
                samples.push(start.elapsed().as_nanos());
            }
            samples.sort();
            samples[samples.len() / 2]
        }
    };
}

batch_warm_impl!(decode_batch_warm_impl_gf8, Gf8);
batch_warm_impl!(decode_batch_warm_impl_gf16, Gf16);