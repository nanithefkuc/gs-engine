//! Canonical element encoding and candidate-set normalization.
//!
//! Every comparator, whatever its native field representation, is normalized to
//! this canonical form outside any timed region: sorted, distinct message
//! polynomials, coefficients low-degree first, in fixed-width little-endian
//! field bytes. A candidate is `Vec<element>`; a candidate set is `Vec<candidate>`.

use crate::fixture::FieldTag;

/// Lowercase hex encoding of one element.
pub fn encode_element(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push_str(&format!("{byte:02x}"));
    }
    encoded
}

/// Strip trailing zero coefficients, collapsing the zero polynomial to a single
/// zero element, so equal polynomials compare equal regardless of padding.
pub fn normalize_candidate(candidate: &[Vec<u8>], field: FieldTag) -> Vec<Vec<u8>> {
    let zero = vec![0_u8; field.width()];
    let mut trimmed = candidate.to_vec();
    while trimmed.len() > 1 && trimmed.last() == Some(&zero) {
        trimmed.pop();
    }
    if trimmed.is_empty() {
        trimmed.push(zero);
    }
    trimmed
}

/// Normalize a candidate set: trim each candidate, deduplicate, and sort by
/// degree then by packed little-endian coefficient bytes.
pub fn normalize_set(candidates: &[Vec<Vec<u8>>], field: FieldTag) -> Vec<Vec<Vec<u8>>> {
    let mut normalized: Vec<Vec<Vec<u8>>> = candidates
        .iter()
        .map(|candidate| normalize_candidate(candidate, field))
        .collect();
    normalized.sort_by(|left, right| {
        left.len()
            .cmp(&right.len())
            .then_with(|| packed(left).cmp(&packed(right)))
    });
    normalized.dedup();
    normalized
}

fn packed(candidate: &[Vec<u8>]) -> Vec<u8> {
    candidate.iter().flatten().copied().collect()
}
