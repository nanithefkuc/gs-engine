//! Separate-process adapter protocol.
//!
//! Every comparator is an independent executable (GPL adapters never link into
//! this MIT controller). The controller invokes it with the fixture path and
//! reads a small line-oriented result from stdout:
//!
//! ```text
//! status=complete|radius|contains|unsupported|error[:message]
//! candidate=<hex>[,<hex>...]
//! [candidate=...]
//! ```
//!
//! All field mapping, output sorting, and process startup happen inside the
//! adapter, outside any region it reports as timed.

use std::path::Path;
use std::process::Command;

use crate::field::normalize_set;
use crate::fixture::{FieldTag, Fixture};

/// Status an adapter declares for a fixture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterStatus {
    /// Ran the same explicit GS geometry.
    Complete,
    /// Ran the same code/radius with decoder-chosen internal parameters.
    Radius,
    /// May return a subset or hint-assisted result.
    Contains,
    /// Fixture cannot be represented by this decoder; carries the reason.
    Unsupported(String),
    /// Adapter failed; carries its message.
    Error(String),
}

/// Parsed adapter output.
#[derive(Clone, Debug)]
pub struct AdapterRun {
    /// Declared status.
    pub status: AdapterStatus,
    /// Normalized candidate set (empty for unsupported/error).
    pub candidates: Vec<Vec<Vec<u8>>>,
}

/// Comparison class of an adapter run against the frozen expected set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Same explicit geometry and complete set.
    CompleteMatched,
    /// Same radius, decoder-chosen parameters, complete set.
    RadiusMatched,
    /// Returned a strict subset of the expected set.
    ContainsOnly,
    /// Fixture unsupported by this decoder.
    Unsupported,
    /// Set disagrees with the expected set (neither equal nor a subset).
    Discrepancy,
    /// Adapter reported an error.
    Error,
}

/// Invoke `executable` on `fixture_path`, capturing and normalizing its output.
pub fn run(
    executable: &Path,
    fixture_path: &Path,
    field: FieldTag,
) -> Result<AdapterRun, String> {
    let output = Command::new(executable)
        .arg(fixture_path)
        .output()
        .map_err(|error| format!("spawning {}: {error}", executable.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Ok(AdapterRun {
            status: AdapterStatus::Error(format!(
                "exit {:?}: {}",
                output.status.code(),
                stderr.trim()
            )),
            candidates: Vec::new(),
        });
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("adapter stdout not UTF-8: {error}"))?;
    parse_output(&stdout, field)
}

fn parse_output(stdout: &str, field: FieldTag) -> Result<AdapterRun, String> {
    let mut status = None;
    let mut candidates = Vec::new();
    for line in stdout.lines() {
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("adapter line without '=': {line:?}"))?;
        match key {
            "status" => {
                let (head, message) = value.split_once(':').unwrap_or((value, ""));
                let parsed = match head {
                    "complete" => AdapterStatus::Complete,
                    "radius" => AdapterStatus::Radius,
                    "contains" => AdapterStatus::Contains,
                    "unsupported" => AdapterStatus::Unsupported(message.to_string()),
                    "error" => AdapterStatus::Error(message.to_string()),
                    other => return Err(format!("unknown adapter status {other:?}")),
                };
                if status.replace(parsed).is_some() {
                    return Err("adapter emitted more than one status".into());
                }
            }
            "candidate" => candidates.push(parse_candidate(value, field)?),
            "field-id" => {}
            other => return Err(format!("unknown adapter key {other:?}")),
        }
    }
    let status = status.ok_or("adapter emitted no status")?;
    Ok(AdapterRun {
        candidates: normalize_set(&candidates, field),
        status,
    })
}

fn parse_candidate(value: &str, field: FieldTag) -> Result<Vec<Vec<u8>>, String> {
    let mut coefficients = Vec::new();
    for token in value.split(',') {
        if token.len() != field.hex_len() {
            return Err(format!("adapter candidate element {token:?} has wrong width"));
        }
        let mut bytes = Vec::with_capacity(field.width());
        for pair in token.as_bytes().chunks_exact(2) {
            let hi = (pair[0] as char)
                .to_digit(16)
                .ok_or_else(|| format!("bad hex {token:?}"))? as u8;
            let lo = (pair[1] as char)
                .to_digit(16)
                .ok_or_else(|| format!("bad hex {token:?}"))? as u8;
            bytes.push((hi << 4) | lo);
        }
        coefficients.push(bytes);
    }
    Ok(coefficients)
}

/// Classify an adapter run against the fixture's frozen expected set.
pub fn classify(run: &AdapterRun, fixture: &Fixture) -> Class {
    let expected = normalize_set(&fixture.expected_candidates, fixture.field);
    match &run.status {
        AdapterStatus::Error(_) => Class::Error,
        AdapterStatus::Unsupported(_) => Class::Unsupported,
        AdapterStatus::Contains => {
            if is_subset(&run.candidates, &expected) {
                Class::ContainsOnly
            } else {
                Class::Discrepancy
            }
        }
        AdapterStatus::Complete | AdapterStatus::Radius => {
            if run.candidates == expected {
                if run.status == AdapterStatus::Complete {
                    Class::CompleteMatched
                } else {
                    Class::RadiusMatched
                }
            } else if is_subset(&run.candidates, &expected) {
                Class::ContainsOnly
            } else {
                Class::Discrepancy
            }
        }
    }
}

fn is_subset(candidates: &[Vec<Vec<u8>>], expected: &[Vec<Vec<u8>>]) -> bool {
    candidates.iter().all(|candidate| expected.contains(candidate))
}
