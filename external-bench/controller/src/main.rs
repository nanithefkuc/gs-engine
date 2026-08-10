//! Cross-library Guruswami-Sudan comparison controller.
//!
//! Commands:
//! - `validate <fixtures-dir>` — parse every `.gsf` and confirm its frozen
//!   expected set equals the `gs-engine` reference decode. This gates the
//!   corpus before any external comparison or timing.
//! - `run <adapter-exe> <fixtures-dir>` — invoke a separate-process adapter on
//!   every fixture and classify its output against the frozen set.

mod adapter;
mod field;
mod fixture;
mod reference;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("validate") => match args.get(2) {
            Some(dir) => validate(Path::new(dir)),
            None => Err("usage: gs-external-bench validate <fixtures-dir>".into()),
        },
        Some("run") => match (args.get(2), args.get(3)) {
            (Some(exe), Some(dir)) => run(Path::new(exe), Path::new(dir)),
            _ => Err("usage: gs-external-bench run <adapter-exe> <fixtures-dir>".into()),
        },
        Some("bench") => bench(&args[2..]),
        _ => Err("usage: gs-external-bench <validate|run|bench> ...".into()),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn collect_fixtures(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|error| format!("reading {}: {error}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "gsf"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no .gsf fixtures under {}", dir.display()));
    }
    Ok(paths)
}

fn validate(dir: &Path) -> Result<bool, String> {
    let mut all_ok = true;
    for path in collect_fixtures(dir)? {
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("reading {}: {error}", path.display()))?;
        let name = path.file_name().unwrap().to_string_lossy();
        match check_fixture(&text) {
            Ok(count) => println!("PASS {name}  ({count} candidates)"),
            Err(message) => {
                all_ok = false;
                println!("FAIL {name}  {message}");
            }
        }
    }
    Ok(all_ok)
}

fn check_fixture(text: &str) -> Result<usize, String> {
    let parsed = fixture::parse(text)?;
    let expected = field::normalize_set(&parsed.expected_candidates, parsed.field);
    let reference = reference::decode(&parsed)?;
    if reference != expected {
        return Err(format!(
            "reference decode disagrees with frozen set: expected {} candidates, reference produced {}",
            expected.len(),
            reference.len()
        ));
    }
    Ok(expected.len())
}

fn run(executable: &Path, dir: &Path) -> Result<bool, String> {
    let mut all_ok = true;
    println!("{:<32} {:<10} {:<16} status", "fixture", "domain", "class");
    for path in collect_fixtures(dir)? {
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("reading {}: {error}", path.display()))?;
        let parsed = fixture::parse(&text)?;
        let outcome = adapter::run(executable, &path, parsed.field)?;
        let class = adapter::classify(&outcome, &parsed);
        let status = match &outcome.status {
            adapter::AdapterStatus::Error(message) => format!("error: {message}"),
            other => format!("{other:?}"),
        };
        if matches!(class, adapter::Class::Discrepancy | adapter::Class::Error) {
            all_ok = false;
        }
        println!("{:<32} {:<10} {class:<16?} {status}", parsed.name, parsed.domain);
        if class == adapter::Class::Discrepancy {
            let expected = field::normalize_set(&parsed.expected_candidates, parsed.field);
            print_discrepancy(&expected, &outcome.candidates);
        }
    }
    Ok(all_ok)
}

fn print_discrepancy(expected: &[Vec<Vec<u8>>], got: &[Vec<Vec<u8>>]) {
    let render = |set: &[Vec<Vec<u8>>]| {
        set.iter()
            .map(|candidate| {
                candidate
                    .iter()
                    .map(|element| field::encode_element(element))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect::<Vec<_>>()
            .join(" | ")
    };
    println!("    expected: {}", render(expected));
    println!("    adapter:  {}", render(got));
}

fn bench(args: &[String]) -> Result<bool, String> {
    // bench <repetitions> <fixtures-dir> [adapter-exe...]
    let reps: usize = args
        .first()
        .ok_or("usage: gs-external-bench bench <repetitions> <fixtures-dir> [adapter-exe...]")?
        .parse()
        .map_err(|_| "repetitions must be a positive integer")?;
    let dir = Path::new(
        args.get(1)
            .ok_or("usage: gs-external-bench bench <repetitions> <fixtures-dir> [adapter-exe...]")?,
    );
    let adapters: Vec<PathBuf> = args[2..].iter().map(PathBuf::from).collect();

    let fixtures = collect_fixtures(dir)?;
    // Header
    print!("{:<34} {:>6} {:>12} {:>12}", "fixture", "cands", "gs-engine", "gs-decode");
    for adapter in &adapters {
        let label = adapter
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .replace("-gs", "");
        print!(" {:>12}", label);
    }
    println!();

    let mut all_ok = true;
    for path in &fixtures {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("reading {}: {error}", path.display()))?;
        let parsed = fixture::parse(&text)?;
        let expected = field::normalize_set(&parsed.expected_candidates, parsed.field);
        let name = parsed.name.clone();
        // gs-engine cold: construction + decode (new scratch each run).
        let engine_ns = median(reps, || {
            let start = std::time::Instant::now();
            let _ = reference::decode(&parsed);
            start.elapsed().as_nanos()
        });

        // gs-engine warm: decode_into only (plan + scratch pre-built).
        let warm_ns = reference::decode_warm(&parsed, reps);

        // Each adapter: report its self-timed decode-ns (decode-only, excluding
        // process startup, parsing, and field mapping).
        let mut adapter_ns: Vec<u64> = Vec::new();
        for adapter in &adapters {
            let exe = adapter.clone();
            let fixture_path = path.clone();
            let field = parsed.field;
            let mut samples: Vec<u64> = (0..reps)
                .map(|_| {
                    match adapter::run(&exe, &fixture_path, field) {
                        Ok(run) => {
                            let class = adapter::classify(&run, &parsed);
                            if matches!(class, adapter::Class::Discrepancy | adapter::Class::Error)
                            {
                                all_ok = false;
                            }
                            run.decode_ns
                        }
                        Err(_) => {
                            all_ok = false;
                            0
                        }
                    }
                })
                .collect();
            samples.sort();
            adapter_ns.push(samples[samples.len() / 2]);
        }

        print!("{name:<34} {:>6} {:>9.1} us {:>9.1} us", expected.len(), engine_ns as f64 / 1000.0, warm_ns as f64 / 1000.0);
        for &ns in &adapter_ns {
            print!(" {:>9.1} us", ns as f64 / 1000.0);
        }
        println!();
    }
    println!("\n(reps={reps}, median, us = microseconds)");
    println!("gs-engine  = construction + decode (cold, new scratch each run)");
    println!("gs-decode  = decode_into only (plan + scratch pre-built, warmed)");
    println!("adapters   = self-timed decode (excludes startup/parse/field-map)");
    Ok(all_ok)
}

fn median<F: FnMut() -> u128>(reps: usize, mut f: F) -> u128 {
    let mut samples: Vec<u128> = (0..reps).map(|_| f()).collect();
    samples.sort();
    samples[samples.len() / 2]
}
