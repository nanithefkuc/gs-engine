//! `lambdaworks-gs` — external-decoder comparison adapter, native-prime track.
//!
//! lambdaworks' Reed-Solomon Guruswami-Sudan example
//! (`examples/reed-solomon-codes`) decodes over the **BabyBear** prime STARK
//! field, `p = 2^31 - 2^27 + 1 = 2013265921` (`Babybear31PrimeField`). The
//! canonical `.gsf` corpus is defined over the binary fields GF(2^8)/GF(2^16),
//! whose characteristic-2 arithmetic is not representable in that native prime
//! track. This baseline is a native-prime CONTEXTUAL reference,
//! not a matched binary-field comparator, so this adapter emits a loud, honest
//! `unsupported` rejection for every binary fixture and NEVER fabricates
//! candidates.
//!
//! Protocol note: the controller carries an inline `:reason` on the
//! `unsupported` status, so the machine-readable stdout line is
//! `status=unsupported:<reason>`; the same reason is mirrored to stderr as the
//! adapter's loud channel on direct invocation.
//!
//! Standalone executable: it depends on nothing from `gs-engine`, the controller,
//! or lambdaworks, and only implements the `.gso` protocol.

use std::process::ExitCode;

/// The required loud rejection reason for binary fixtures in the native track.
const UNSUPPORTED_REASON: &str = "native lambdaworks GS runs over a prime STARK field; gf8/gf16 binary fixtures are not representable in the native track";

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let Some(arg) = args.next() else {
        eprintln!("usage: lambdaworks-gs <fixture.gsf>");
        println!("status=error:missing fixture path argument");
        return ExitCode::FAILURE;
    };
    let path = std::path::PathBuf::from(arg);

    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("lambdaworks-gs: reading {}: {error}", path.display());
            println!("status=error:cannot read fixture: {error}");
            return ExitCode::FAILURE;
        }
    };

    match fixture_field(&text) {
        Ok(field) => {
            // The BabyBear prime native track cannot represent a binary field.
            eprintln!(
                "lambdaworks-gs: unsupported {} (field={field}): {UNSUPPORTED_REASON}",
                path.display()
            );
            println!("status=unsupported:{UNSUPPORTED_REASON}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("lambdaworks-gs: {message}");
            println!("status=error:{message}");
            ExitCode::FAILURE
        }
    }
}

/// Parse the `.gsf` v1 header and return the declared field tag.
///
/// Deliberately minimal: this adapter only needs the field tag to reject the
/// fixture. It does not decode, so no support/received parsing or field
/// isomorphism is performed (all such work would be outside any timed region
/// anyway). Malformed input is reported as a strict error.
fn fixture_field(text: &str) -> Result<&'static str, String> {
    let mut lines = text.lines();
    match lines.next() {
        Some("gs-engine-fixture-v1") => {}
        Some(other) => return Err(format!("unexpected fixture header {other:?}")),
        None => return Err("empty fixture".into()),
    }
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("fixture line without '=': {line:?}"));
        };
        if key == "field" {
            return match value {
                "gf8" => Ok("gf8"),
                "gf16" => Ok("gf16"),
                other => Err(format!("unknown field tag {other:?}")),
            };
        }
    }
    Err("fixture missing field key".into())
}
