#![cfg(feature = "std")]

use std::fmt::Write as _;
use std::process::Command;

use fgf::Gf8;
use fgf::field::Field;
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
};

const CHILD_ENV: &str = "GS_ENGINE_BACKEND_EQUIVALENCE_CHILD";
const RESULT_PREFIX: &str = "GS_ENGINE_BACKEND_RESULT=";
const ROOT_LIMITS: AlekhnovichLimits =
    AlekhnovichLimits::new(1_000_000, 100_000, usize::MAX, usize::MAX, 128);

fn gf8(value: u8) -> <Gf8 as Field>::Elem {
    Gf8::read(&[value])
}

fn candidate_fingerprint() -> String {
    let parameters = GsParameters::new::<Gf8>(
        4,
        0,
        2,
        1,
        2,
        1,
        ParameterLimits::new(4, 8, usize::MAX, usize::MAX),
    )
    .unwrap();
    let plan = GsPlan::new(
        parameters,
        EvaluationDomain::<Gf8>::additive_subspace(4).unwrap(),
        ROOT_LIMITS,
    )
    .unwrap();
    let received = [gf8(7), gf8(7), gf8(9), gf8(9)];
    let mut candidates = Vec::new();
    plan.decode_into(&received, &mut DecodeScratch::new(), &mut candidates)
        .unwrap();

    let mut fingerprint = String::new();
    for candidate in candidates {
        for byte in candidate.as_packed() {
            write!(&mut fingerprint, "{byte:02x}").unwrap();
        }
        fingerprint.push(';');
    }
    fingerprint
}

fn child_result(backend: Option<&str>) -> String {
    let executable = std::env::current_exe().unwrap();
    let mut command = Command::new(executable);
    command
        .arg("--exact")
        .arg("scalar_and_selected_backends_return_identical_candidates")
        .arg("--nocapture")
        .env(CHILD_ENV, "1");
    match backend {
        Some(backend) => {
            command.env("SIMD_BACKEND", backend);
        }
        None => {
            command.env_remove("SIMD_BACKEND");
        }
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "backend child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix(RESULT_PREFIX).map(str::to_owned))
        .expect("backend child emitted a candidate fingerprint")
}

#[test]
fn scalar_and_selected_backends_return_identical_candidates() {
    if std::env::var_os(CHILD_ENV).is_some() {
        println!("{RESULT_PREFIX}{}", candidate_fingerprint());
        return;
    }

    let scalar = child_result(Some("scalar"));
    let selected = child_result(None);
    assert_eq!(scalar, selected);
}
