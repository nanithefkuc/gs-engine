//! Strict `.gsf` version-1 parser.
//!
//! The format is defined in `external-bench/fixtures/FORMAT.md`. Parsing is
//! deliberately unforgiving: any deviation (unknown key, out-of-order record,
//! duplicate singleton, stray whitespace, wrong element width, trailing comma,
//! sign, blank line, comment, or a missing/absent-final newline) is rejected
//! loudly so a malformed fixture can never silently reach a decoder.

/// Field tag carried by a fixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldTag {
    /// GF(2^8).
    Gf8,
    /// GF(2^16).
    Gf16,
}

impl FieldTag {
    /// Fixed little-endian element width in bytes.
    pub const fn width(self) -> usize {
        match self {
            FieldTag::Gf8 => 1,
            FieldTag::Gf16 => 2,
        }
    }

    /// Number of lowercase hex digits encoding one element.
    pub const fn hex_len(self) -> usize {
        self.width() * 2
    }

    /// Canonical `field-definition` string for version-1 fixtures.
    pub const fn canonical_definition(self) -> &'static str {
        match self {
            FieldTag::Gf8 => "gf2[x]/(x^8+x^4+x^3+x+1);le-polynomial-basis",
            FieldTag::Gf16 => "gf8[u]/(u^2+u+0x20);gf8=aes-0x11b;le-components",
        }
    }

    /// Short fixture tag string (`gf8` / `gf16`).
    pub const fn tag_str(self) -> &'static str {
        match self {
            FieldTag::Gf8 => "gf8",
            FieldTag::Gf16 => "gf16",
        }
    }
}

/// A parsed fixture: decoder input plus its complete expected candidate set.
#[derive(Clone, Debug)]
pub struct Fixture {
    /// ASCII identifier.
    pub name: String,
    /// Field tag.
    pub field: FieldTag,
    /// Declared field definition (semantic, validated against the canonical).
    #[allow(dead_code)]
    pub field_definition: String,
    /// Domain construction descriptor.
    pub domain: String,
    /// Number of evaluation points.
    pub n: usize,
    /// Message dimension; `max_degree = k - 1`.
    pub k: usize,
    /// Target Hamming radius.
    pub target_radius: usize,
    /// Interpolation multiplicity `s`.
    pub multiplicity: usize,
    /// Interpolation `Y`-degree `ell`.
    pub y_degree: usize,
    /// `(1, max_degree)` weighted-degree bound `D`.
    pub weighted_degree: usize,
    /// Support points, each a fixed-width little-endian element.
    pub support: Vec<Vec<u8>>,
    /// Received symbols, one per support point.
    pub received: Vec<Vec<u8>>,
    /// Complete expected candidates, each a coefficient list (constant first).
    pub expected_candidates: Vec<Vec<Vec<u8>>>,
    /// Optional expected codewords, each `n` elements.
    #[allow(dead_code)]
    pub expected_codewords: Vec<Vec<Vec<u8>>>,
}

/// A parsed batch fixture: shared geometry plus multiple received words that
/// all decode to the same expected candidate set.
///
/// Used by the batch timing comparison (`bench-batch`), which decodes every
/// word against one shared `gs-engine` plan and compares against $N$ sequential
/// adapter invocations. The expected set is shared because every word is a
/// corruption of the same codeword; each word is validated to decode to it.
#[derive(Clone, Debug)]
pub struct BatchFixture {
    /// Inherited from the single-word shape.
    pub inner: Fixture,
    /// Received words, each `n` elements. Always non-empty.
    pub received_words: Vec<Vec<Vec<u8>>>,
}

impl BatchFixture {
    /// Number of received words in the batch.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.received_words.len()
    }
}

const SINGLETONS: [&str; 12] = [
    "name",
    "field",
    "field-definition",
    "domain",
    "n",
    "k",
    "target-radius",
    "multiplicity",
    "y-degree",
    "weighted-degree",
    "support",
    "received",
];

/// Parse a `.gsf` document, returning a descriptive error on any deviation.
pub fn parse(text: &str) -> Result<Fixture, String> {
    if !text.ends_with('\n') {
        return Err("file must end with exactly one LF byte".into());
    }
    if text.ends_with("\n\n") {
        return Err("trailing blank line is invalid".into());
    }
    let mut lines = text.split('\n');
    let header = lines.next().ok_or("empty file")?;
    if header != "gs-engine-fixture-v1" {
        return Err(format!("bad version header {header:?}"));
    }

    // Collect the remaining records (the final split element is the empty
    // string after the terminating LF and is dropped).
    let mut records: Vec<(&str, &str)> = Vec::new();
    let mut rest: Vec<&str> = lines.collect();
    let last = rest.pop();
    if last != Some("") {
        return Err("file must end with exactly one LF byte".into());
    }
    for line in rest {
        if line.is_empty() {
            return Err("blank lines are invalid".into());
        }
        if line.starts_with('#') {
            return Err("comments are invalid".into());
        }
        if line != line.trim() {
            return Err(format!("stray whitespace in line {line:?}"));
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("record without '=': {line:?}"))?;
        if value != value.trim() {
            return Err(format!("whitespace around value in {line:?}"));
        }
        records.push((key, value));
    }

    // Required singletons appear exactly once, in order, before candidates.
    let mut fields = std::collections::HashMap::new();
    let mut cursor = 0;
    let mut index = 0;
    while index < records.len() && cursor < SINGLETONS.len() {
        let (key, value) = records[index];
        if key != SINGLETONS[cursor] {
            return Err(format!(
                "expected key {:?} but found {key:?}",
                SINGLETONS[cursor]
            ));
        }
        if fields.insert(key, value).is_some() {
            return Err(format!("duplicate singleton {key:?}"));
        }
        cursor += 1;
        index += 1;
    }
    if cursor != SINGLETONS.len() {
        return Err(format!("missing required key {:?}", SINGLETONS[cursor]));
    }

    let field = match fields["field"] {
        "gf8" => FieldTag::Gf8,
        "gf16" => FieldTag::Gf16,
        other => return Err(format!("unknown field {other:?}")),
    };
    match fields["domain"] {
        "arbitrary" | "additive" | "affine" => {}
        other => return Err(format!("unknown domain {other:?}")),
    }
    let field_definition = fields["field-definition"].to_string();
    if field_definition != field.canonical_definition() {
        return Err(format!(
            "field-definition {field_definition:?} is not the canonical {:?}",
            field.canonical_definition()
        ));
    }

    let n = parse_usize(fields["n"], "n")?;
    let k = parse_usize(fields["k"], "k")?;
    let target_radius = parse_usize(fields["target-radius"], "target-radius")?;
    let multiplicity = parse_usize(fields["multiplicity"], "multiplicity")?;
    let y_degree = parse_usize(fields["y-degree"], "y-degree")?;
    let weighted_degree = parse_usize(fields["weighted-degree"], "weighted-degree")?;

    let support = parse_elements(fields["support"], field, "support")?;
    let received = parse_elements(fields["received"], field, "received")?;
    if support.len() != n {
        return Err(format!("support has {} elements, expected n={n}", support.len()));
    }
    if received.len() != n {
        return Err(format!("received has {} elements, expected n={n}", received.len()));
    }

    // Repeated records: candidates, then codewords.
    let mut expected_candidates = Vec::new();
    let mut expected_codewords = Vec::new();
    let mut seen_codeword = false;
    for &(key, value) in &records[index..] {
        match key {
            "expected-candidate" => {
                if seen_codeword {
                    return Err("expected-candidate after expected-codeword".into());
                }
                let candidate = parse_elements(value, field, "expected-candidate")?;
                validate_candidate(&candidate, field)?;
                expected_candidates.push(candidate);
            }
            "expected-codeword" => {
                seen_codeword = true;
                let codeword = parse_elements(value, field, "expected-codeword")?;
                if codeword.len() != n {
                    return Err(format!(
                        "expected-codeword has {} elements, expected n={n}",
                        codeword.len()
                    ));
                }
                expected_codewords.push(codeword);
            }
            other => return Err(format!("unknown key {other:?}")),
        }
    }
    if expected_candidates.is_empty() {
        return Err("at least one expected-candidate is required".into());
    }

    Ok(Fixture {
        name: fields["name"].to_string(),
        field,
        field_definition,
        domain: fields["domain"].to_string(),
        n,
        k,
        target_radius,
        multiplicity,
        y_degree,
        weighted_degree,
        support,
        received,
        expected_candidates,
        expected_codewords,
    })
}

/// Parse a `.gsf` batch document (header `gs-engine-batch-v1`).
///
/// The geometry singletons are the same as version 1, except `received` is
/// replaced by `received-words=<count>` followed by exactly that many
/// `received-word=` records (each `n` elements). The `expected-candidate` set
/// is shared: every word in the batch is a corruption of the same codeword and
/// decodes to it, so the batch does uniform work across all words.
pub fn parse_batch(text: &str) -> Result<BatchFixture, String> {
    if !text.ends_with('\n') {
        return Err("file must end with exactly one LF byte".into());
    }
    if text.ends_with("\n\n") {
        return Err("trailing blank line is invalid".into());
    }
    let mut lines = text.split('\n');
    let header = lines.next().ok_or("empty file")?;
    if header != "gs-engine-batch-v1" {
        return Err(format!("bad version header {header:?} (expected gs-engine-batch-v1)"));
    }
    let mut rest: Vec<&str> = lines.collect();
    let last = rest.pop();
    if last != Some("") {
        return Err("file must end with exactly one LF byte".into());
    }
    for line in &rest {
        if line.is_empty() {
            return Err("blank lines are invalid".into());
        }
        if line.starts_with('#') {
            return Err("comments are invalid".into());
        }
        if *line != line.trim() {
            return Err(format!("stray whitespace in line {line:?}"));
        }
    }
    let records: Vec<(&str, &str)> = rest
        .iter()
        .map(|line| {
            line.split_once('=')
                .ok_or_else(|| format!("record without '=': {line:?}"))
        })
        .collect::<Result<_, _>>()?;
    for (key, value) in &records {
        if *value != value.trim() {
            return Err(format!("whitespace around value in line {key:?}"));
        }
    }

    // The first 11 singletons are shared with v1 (everything up to but not
    // including `received`).
    let batch_singletons = &SINGLETONS[..11];
    let mut fields = std::collections::HashMap::new();
    let mut cursor = 0;
    let mut index = 0;
    while index < records.len() && cursor < batch_singletons.len() {
        let (key, value) = records[index];
        if key != batch_singletons[cursor] {
            return Err(format!(
                "expected key {:?} but found {key:?}",
                batch_singletons[cursor]
            ));
        }
        if fields.insert(key, value).is_some() {
            return Err(format!("duplicate singleton {key:?}"));
        }
        cursor += 1;
        index += 1;
    }
    if cursor != batch_singletons.len() {
        return Err(format!(
            "missing required key {:?}",
            batch_singletons[cursor]
        ));
    }

    let field = match fields["field"] {
        "gf8" => FieldTag::Gf8,
        "gf16" => FieldTag::Gf16,
        other => return Err(format!("unknown field {other:?}")),
    };
    match fields["domain"] {
        "arbitrary" | "additive" | "affine" => {}
        other => return Err(format!("unknown domain {other:?}")),
    }
    let field_definition = fields["field-definition"].to_string();
    if field_definition != field.canonical_definition() {
        return Err(format!(
            "field-definition {field_definition:?} is not the canonical {:?}",
            field.canonical_definition()
        ));
    }
    let n = parse_usize(fields["n"], "n")?;
    let k = parse_usize(fields["k"], "k")?;
    let target_radius = parse_usize(fields["target-radius"], "target-radius")?;
    let multiplicity = parse_usize(fields["multiplicity"], "multiplicity")?;
    let y_degree = parse_usize(fields["y-degree"], "y-degree")?;
    let weighted_degree = parse_usize(fields["weighted-degree"], "weighted-degree")?;
    let support = parse_elements(fields["support"], field, "support")?;
    if support.len() != n {
        return Err(format!("support has {} elements, expected n={n}", support.len()));
    }

    // `received-words=<count>` then exactly that many `received-word=` records,
    // then `expected-candidate=` records.
    let (rw_key, rw_value) = records
        .get(index)
        .ok_or("missing received-words record")?;
    if *rw_key != "received-words" {
        return Err(format!(
            "expected key \"received-words\" but found {rw_key:?}"
        ));
    }
    let word_count = parse_usize(rw_value, "received-words")?;
    if word_count == 0 {
        return Err("received-words must be at least 1".into());
    }
    index += 1;

    let mut received_words: Vec<Vec<Vec<u8>>> = Vec::with_capacity(word_count);
    let mut seen_word = 0;
    while index < records.len() && seen_word < word_count {
        let (key, value) = records[index];
        if key != "received-word" {
            return Err(format!("expected \"received-word\" but found {key:?}"));
        }
        let word = parse_elements(value, field, "received-word")?;
        if word.len() != n {
            return Err(format!(
                "received-word {} has {} elements, expected n={n}",
                seen_word + 1,
                word.len()
            ));
        }
        received_words.push(word);
        seen_word += 1;
        index += 1;
    }
    if seen_word != word_count {
        return Err(format!(
            "received-words={word_count} but only {seen_word} received-word records present"
        ));
    }

    // Remaining records: expected-candidate (shared), then optional codewords.
    let mut expected_candidates = Vec::new();
    let mut expected_codewords = Vec::new();
    let mut seen_codeword = false;
    for &(key, value) in &records[index..] {
        match key {
            "expected-candidate" => {
                if seen_codeword {
                    return Err("expected-candidate after expected-codeword".into());
                }
                let candidate = parse_elements(value, field, "expected-candidate")?;
                validate_candidate(&candidate, field)?;
                expected_candidates.push(candidate);
            }
            "expected-codeword" => {
                seen_codeword = true;
                let codeword = parse_elements(value, field, "expected-codeword")?;
                if codeword.len() != n {
                    return Err(format!(
                        "expected-codeword has {} elements, expected n={n}",
                        codeword.len()
                    ));
                }
                expected_codewords.push(codeword);
            }
            other => return Err(format!("unknown key {other:?}")),
        }
    }
    if expected_candidates.is_empty() {
        return Err("at least one expected-candidate is required".into());
    }

    let inner = Fixture {
        name: fields["name"].to_string(),
        field,
        field_definition,
        domain: fields["domain"].to_string(),
        n,
        k,
        target_radius,
        multiplicity,
        y_degree,
        weighted_degree,
        support,
        // The inner `received` carries the first word so existing single-word
        // code paths (e.g. `reference::decode`) work unchanged on it.
        received: received_words[0].clone(),
        expected_candidates,
        expected_codewords,
    };
    Ok(BatchFixture {
        inner,
        received_words,
    })
}

fn parse_usize(value: &str, key: &str) -> Result<usize, String> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("{key} must be a non-negative decimal, got {value:?}"));
    }
    value
        .parse::<usize>()
        .map_err(|error| format!("{key} out of range: {error}"))
}

fn parse_elements(value: &str, field: FieldTag, key: &str) -> Result<Vec<Vec<u8>>, String> {
    if value.is_empty() {
        return Err(format!("{key} is empty"));
    }
    let mut elements = Vec::new();
    for token in value.split(',') {
        elements.push(parse_element(token, field, key)?);
    }
    Ok(elements)
}

fn parse_element(token: &str, field: FieldTag, key: &str) -> Result<Vec<u8>, String> {
    if token.len() != field.hex_len() {
        return Err(format!(
            "{key} element {token:?} must be {} lowercase hex digits",
            field.hex_len()
        ));
    }
    if !token.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(format!("{key} element {token:?} has non-lowercase-hex digits"));
    }
    let mut bytes = Vec::with_capacity(field.width());
    for pair in token.as_bytes().chunks_exact(2) {
        let hi = (pair[0] as char).to_digit(16).unwrap() as u8;
        let lo = (pair[1] as char).to_digit(16).unwrap() as u8;
        bytes.push((hi << 4) | lo);
    }
    Ok(bytes)
}

fn validate_candidate(candidate: &[Vec<u8>], field: FieldTag) -> Result<(), String> {
    let zero = vec![0_u8; field.width()];
    if candidate.len() == 1 {
        return Ok(());
    }
    if candidate.last() == Some(&zero) {
        return Err("candidate has a trailing zero coefficient".into());
    }
    Ok(())
}
