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
        Some("bench-batch") => bench_batch(&args[2..]),
        _ => Err("usage: gs-external-bench <validate|run|bench|bench-batch> ...".into()),
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
    // Route on the version header so both single-word and batch fixtures
    // validate against the gs-engine reference.
    let header = text.split('\n').next().unwrap_or("");
    if header == "gs-engine-batch-v1" {
        check_batch_fixture(text)
    } else {
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
}

fn check_batch_fixture(text: &str) -> Result<usize, String> {
    let parsed = fixture::parse_batch(text)?;
    let expected = field::normalize_set(&parsed.inner.expected_candidates, parsed.inner.field);
    // Every received word must decode to the shared expected set.
    for (i, word) in parsed.received_words.iter().enumerate() {
        let mut fixture = parsed.inner.clone();
        fixture.received = word.clone();
        let reference = reference::decode(&fixture)?;
        if reference != expected {
            return Err(format!(
                "word {} decodes to {} candidates, expected {}",
                i + 1,
                reference.len(),
                expected.len()
            ));
        }
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


/// Collect only batch fixtures (header `gs-engine-batch-v1`).
fn collect_batch_fixtures(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|error| format!("reading {}: {error}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "gsf"))
        .filter(|path| {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|text| text.split('\n').next().map(|h| h == "gs-engine-batch-v1"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no batch fixtures under {}", dir.display()));
    }
    Ok(paths)
}

/// Render one received word as a version-1 single-word fixture
/// so an external adapter can decode it. Shares the batch geometry.
fn render_word_v1(batch: &crate::fixture::BatchFixture, word_index: usize) -> String {
    let inner = &batch.inner;
    let list = |row: &[Vec<u8>]| -> String {
        row.iter()
            .map(|bytes| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>())
            .collect::<Vec<_>>()
            .join(",")
    };
    let mut lines = vec!["gs-engine-fixture-v1".to_string()];
    lines.push(format!("name={}-word{}", inner.name, word_index));
    lines.push(format!("field={}", inner.field.tag_str()));
    lines.push(format!("field-definition={}", inner.field_definition));
    lines.push(format!("domain={}", inner.domain));
    lines.push(format!("n={}", inner.n));
    lines.push(format!("k={}", inner.k));
    lines.push(format!("target-radius={}", inner.target_radius));
    lines.push(format!("multiplicity={}", inner.multiplicity));
    lines.push(format!("y-degree={}", inner.y_degree));
    lines.push(format!("weighted-degree={}", inner.weighted_degree));
    lines.push(format!("support={}", list(&inner.support)));
    lines.push(format!("received={}", list(&batch.received_words[word_index])));
    for c in &inner.expected_candidates {
        lines.push(format!("expected-candidate={}", list(c)));
    }
    lines.join("\n") + "\n"
}

fn bench_batch(args: &[String]) -> Result<bool, String> {
    // bench-batch <reps> <fixtures-dir> [adapter-exe...]
    let reps: usize = args
        .first()
        .ok_or("usage: gs-external-bench bench-batch <reps> <fixtures-dir> [adapter-exe...]")?
        .parse()
        .map_err(|_| "repetitions must be a positive integer")?;
    let dir = Path::new(
        args.get(1)
            .ok_or("usage: gs-external-bench bench-batch <reps> <fixtures-dir> [adapter-exe...]")?,
    );
    let adapters: Vec<PathBuf> = args[2..].iter().map(PathBuf::from).collect();

    let fixtures = collect_batch_fixtures(dir)?;

    // Header
    print!("{:<34} {:>5} {:>12} {:>12}", "fixture", "words", "gs-parallel", "gs-sequential");
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
        let parsed = fixture::parse_batch(&text)?;
        let name = parsed.inner.name.clone();
        let word_count = parsed.word_count();
        let _expected = field::normalize_set(&parsed.inner.expected_candidates, parsed.inner.field);

        // gs-engine batch: warm shared plan, one scratch per word.
        let par_ns = reference::decode_batch_warm(&parsed, reps, true);
        let seq_ns = reference::decode_batch_warm(&parsed, reps, false);

        // Each adapter: decode every word in a temp v1 fixture, sequential sum
        // of self-timed decode-ns (excludes startup/parse/field-map per word).
        let mut adapter_ns: Vec<u64> = Vec::new();
        for adapter in &adapters {
            let mut total_samples: Vec<u64> = Vec::new();
            for _ in 0..reps {
                let mut sum: u64 = 0;
                let mut ok = true;
                for i in 0..word_count {
                    let v1 = render_word_v1(&parsed, i);
                    let tmp = std::env::temp_dir().join(format!(
                        "gs-batch-word-{}-{}.gsf",
                        std::process::id(),
                        i
                    ));
                    std::fs::write(&tmp, &v1)
                        .map_err(|error| format!("temp fixture: {error}"))?;
                    match adapter::run(adapter, &tmp, parsed.inner.field) {
                        Ok(run) => {
                            let class = adapter::classify(&run, &parsed.inner);
                            if matches!(class, adapter::Class::Discrepancy | adapter::Class::Error)
                            {
                                all_ok = false;
                                ok = false;
                            }
                            sum = sum.saturating_add(run.decode_ns);
                        }
                        Err(_) => {
                            all_ok = false;
                            ok = false;
                        }
                    }
                    let _ = std::fs::remove_file(&tmp);
                }
                total_samples.push(if ok { sum } else { 0 });
            }
            total_samples.sort();
            adapter_ns.push(total_samples[total_samples.len() / 2]);
        }

        print!("{name:<34} {:>5} {:>9.1} us {:>9.1} us", word_count, par_ns as f64 / 1000.0, seq_ns as f64 / 1000.0);
        for &ns in &adapter_ns {
            print!(" {:>9.1} us", ns as f64 / 1000.0);
        }
        println!();
    }
    println!("\n(reps={reps}, median, us = microseconds, batch = all words)");
    println!("gs-parallel   = decode_batch_into (shared plan, Rayon pool, warmed)");
    println!("gs-sequential = per-word decode_into in order (shared plan, warmed)");
    println!("adapters      = sum of self-timed per-word decode (sequential, excludes startup/parse/field-map)");
    Ok(all_ok)
}