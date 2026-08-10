use alloc::vec::Vec;

use butterfly_fft::basis::{conversion_scratch_elements, monomial_to_novel_bytes};
use butterfly_fft::core::kernel::ButterflyKernels;

use crate::decoder::DecodeError;
use crate::{ConfigError, DecodeScratch, EvaluationDomain, Polynomial};

/// Butterfly-FFT scoring crossover in points, one candidate. See `BENCHMARKS.md`.
pub const BUTTERFLY_FFT_SINGLE_SCORING_CROSSOVER: usize = 256;
/// Butterfly-FFT scoring crossover in points, two or three candidates. See `BENCHMARKS.md`.
pub const BUTTERFLY_FFT_BATCH2_SCORING_CROSSOVER: usize = 64;
/// Butterfly-FFT scoring crossover in points, four to seven candidates. See `BENCHMARKS.md`.
pub const BUTTERFLY_FFT_BATCH4_SCORING_CROSSOVER: usize = 64;
/// Butterfly-FFT scoring crossover in points, eight to fifteen candidates. See `BENCHMARKS.md`.
pub const BUTTERFLY_FFT_BATCH8_SCORING_CROSSOVER: usize = 32;
/// Butterfly-FFT scoring crossover in points, sixteen or more candidates. See `BENCHMARKS.md`.
pub const BUTTERFLY_FFT_BATCH16_SCORING_CROSSOVER: usize = 16;
/// Candidate-scoring implementation override.
///
/// This is exposed only by the crate's `internals` feature so benchmarks can
/// force both sides of the automatic selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoringStrategy {
    /// Use the measured point/candidate crossover.
    Auto,
    #[cfg(feature = "internals")]
    /// Evaluate each candidate with Horner's method.
    Horner,
    #[cfg(feature = "internals")]
    /// Require packed butterfly-FFT evaluation.
    ButterflyFft,
}

pub(crate) fn score_candidates<F: ButterflyKernels>(
    domain: &EvaluationDomain<F>,
    received: &[F::Elem],
    candidates: &[Polynomial<F>],
    radius: usize,
    scratch: &mut DecodeScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), DecodeError> {
    score_candidates_with_strategy(
        domain,
        received,
        candidates,
        radius,
        ScoringStrategy::Auto,
        scratch,
        output,
    )
}

/// Score candidates with an explicit implementation strategy.
///
/// The forced butterfly-FFT strategy requires an additive or affine domain.
pub fn score_candidates_with_strategy<F: ButterflyKernels>(
    domain: &EvaluationDomain<F>,
    received: &[F::Elem],
    candidates: &[Polynomial<F>],
    radius: usize,
    strategy: ScoringStrategy,
    scratch: &mut DecodeScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), DecodeError> {
    if candidates.is_empty() {
        output.clear();
        return Ok(());
    }
    if output.capacity() < candidates.len() {
        output
            .try_reserve(candidates.len().saturating_sub(output.len()))
            .map_err(|_| ConfigError::AllocationFailed {
                context: "decoded candidates",
                elements: candidates.len(),
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })?;
    }

    let use_fft = match strategy {
        ScoringStrategy::Auto => {
            domain.transform_plan().is_some()
                && crate::cost::select_scoring(crate::cost::ScoringCostKey {
                    points: domain.len(),
                    candidates: candidates.len(),
                    total_coefficients: candidates.iter().map(Polynomial::coefficient_count).sum(),
                    backend: crate::cost::BackendClass::detect::<F>(),
                }) == crate::cost::ScoringBackend::ButterflyFft
        }
        #[cfg(feature = "internals")]
        ScoringStrategy::Horner => false,
        #[cfg(feature = "internals")]
        ScoringStrategy::ButterflyFft => true,
    };
    if use_fft {
        score_butterfly_fft(domain, received, candidates, radius, scratch, output)
    } else {
        score_horner(
            domain.points(),
            received,
            candidates,
            radius,
            scratch,
            output,
        )
    }
}

fn score_horner<F: ButterflyKernels>(
    points: &[F::Elem],
    received: &[F::Elem],
    candidates: &[Polynomial<F>],
    radius: usize,
    scratch: &mut DecodeScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), DecodeError> {
    let candidate_count = candidates.len();
    let mut output_count = 0_usize;
    if candidates.len() <= points.len() {
        for candidate in candidates {
            let mut distance = 0_usize;
            for (&point, &symbol) in points.iter().zip(received) {
                if candidate.evaluate(point) != symbol {
                    distance += 1;
                    if distance > radius {
                        break;
                    }
                }
            }
            if distance <= radius {
                write_candidate(output, output_count, candidate)?;
                output_count += 1;
            }
        }
        output.truncate(output_count);
        return Ok(());
    }

    ensure_len(
        &mut scratch.distances,
        candidate_count,
        "candidate distances",
    )?;
    scratch.distances[..candidate_count].fill(0);
    for (&point, &symbol) in points.iter().zip(received) {
        for (candidate, distance) in candidates
            .iter()
            .zip(&mut scratch.distances[..candidate_count])
        {
            if *distance <= radius && candidate.evaluate(point) != symbol {
                *distance += 1;
            }
        }
    }
    for (candidate, &distance) in candidates.iter().zip(&scratch.distances[..candidate_count]) {
        if distance <= radius {
            write_candidate(output, output_count, candidate)?;
            output_count += 1;
        }
    }
    output.truncate(output_count);
    Ok(())
}

fn score_butterfly_fft<F: ButterflyKernels>(
    domain: &EvaluationDomain<F>,
    received: &[F::Elem],
    candidates: &[Polynomial<F>],
    radius: usize,
    scratch: &mut DecodeScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), DecodeError> {
    let candidate_count = candidates.len();
    let plan = domain
        .transform_plan()
        .ok_or(DecodeError::InternalInvariant {
            reason: "butterfly-fft scoring selected without a transform plan",
        })?;
    let row_len = candidates
        .len()
        .checked_mul(F::BYTES)
        .ok_or(ConfigError::GeometryOverflow {
            context: "batched evaluation row bytes",
        })?;
    let evaluation_bytes =
        plan.size()
            .checked_mul(row_len)
            .ok_or(ConfigError::GeometryOverflow {
                context: "batched evaluation bytes",
            })?;
    let conversion_bytes = conversion_scratch_elements(plan.size())
        .checked_mul(row_len)
        .ok_or(ConfigError::GeometryOverflow {
            context: "batched conversion scratch bytes",
        })?;
    ensure_len(
        &mut scratch.packed_evaluations,
        evaluation_bytes,
        "batched candidate evaluations",
    )?;
    ensure_len(
        &mut scratch.conversion,
        conversion_bytes,
        "batched conversion scratch",
    )?;
    scratch.packed_evaluations[..evaluation_bytes].fill(0);

    for (lane, candidate) in candidates.iter().enumerate() {
        if candidate.coefficient_count() > plan.size() {
            return Err(DecodeError::InternalInvariant {
                reason: "an extracted candidate exceeds the transform domain",
            });
        }
        let lane_offset = lane * F::BYTES;
        for (degree, coefficient) in candidate.coefficients().enumerate() {
            let row_offset = degree * row_len + lane_offset;
            F::write(
                &mut scratch.packed_evaluations[row_offset..row_offset + F::BYTES],
                coefficient,
            );
        }
    }

    monomial_to_novel_bytes::<F>(
        &mut scratch.packed_evaluations[..evaluation_bytes],
        row_len,
        plan,
        &mut scratch.conversion[..conversion_bytes],
    )?;
    // Hamming scoring needs every coordinate, so selected/ranged transforms
    // cannot reduce work here. Execute one full transform with candidates
    // packed across each byte row.
    domain.forward_bytes(&mut scratch.packed_evaluations[..evaluation_bytes], row_len)?;

    ensure_len(
        &mut scratch.distances,
        candidate_count,
        "candidate distances",
    )?;
    scratch.distances[..candidate_count].fill(0);
    for (point, &symbol) in received.iter().enumerate() {
        let row_offset = point * row_len;
        for (lane, distance) in scratch.distances[..candidate_count].iter_mut().enumerate() {
            let offset = row_offset + lane * F::BYTES;
            if F::read(&scratch.packed_evaluations[offset..offset + F::BYTES]) != symbol {
                *distance += 1;
            }
        }
    }

    let mut output_count = 0_usize;
    for (candidate, &distance) in candidates.iter().zip(&scratch.distances[..candidate_count]) {
        if distance <= radius {
            write_candidate(output, output_count, candidate)?;
            output_count += 1;
        }
    }
    output.truncate(output_count);
    Ok(())
}

fn write_candidate<F: ButterflyKernels>(
    output: &mut Vec<Polynomial<F>>,
    index: usize,
    candidate: &Polynomial<F>,
) -> Result<(), DecodeError> {
    if let Some(existing) = output.get_mut(index) {
        existing.assign_packed(candidate.as_packed())?;
    } else {
        output
            .try_reserve(1)
            .map_err(|_| ConfigError::AllocationFailed {
                context: "decoded candidate",
                elements: 1,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })?;
        output.push(candidate.clone());
    }
    Ok(())
}

fn ensure_len<T: Default + Clone>(
    values: &mut Vec<T>,
    required: usize,
    context: &'static str,
) -> Result<(), ConfigError> {
    if required > values.len() {
        values
            .try_reserve_exact(required - values.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context,
                elements: required,
                element_size: core::mem::size_of::<T>(),
            })?;
        values.resize(required, T::default());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use fgf::Gf16;
    use fgf::field::Field;

    use super::score_candidates;
    use crate::{DecodeScratch, EvaluationDomain, Polynomial};

    fn gf16(value: u16) -> <Gf16 as Field>::Elem {
        Gf16::read(&value.to_le_bytes())
    }

    #[test]
    fn butterfly_fft_scores_four_candidates_in_packed_lanes() {
        let domain = EvaluationDomain::<Gf16>::additive_subspace(64).unwrap();
        let candidates: Vec<_> = (1..=4)
            .map(|value| Polynomial::<Gf16>::constant(gf16(value)).unwrap())
            .collect();
        let received = vec![gf16(1); domain.len()];
        let mut scratch = DecodeScratch::new();
        let mut output = Vec::new();

        score_candidates(
            &domain,
            &received,
            &candidates,
            0,
            &mut scratch,
            &mut output,
        )
        .unwrap();

        assert_eq!(output, &candidates[..1]);
        assert!(scratch.evaluation_capacity_bytes() >= domain.len() * 4 * Gf16::BYTES);
    }
}
