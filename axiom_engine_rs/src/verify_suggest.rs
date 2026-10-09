//! Verifier suggestion for the agent task loop.
//!
//! The task loop's safety depends entirely on the verifier command the agent
//! provides. A bad verifier means AXIOM confidently commits bad code. This
//! module suggests sensible verifiers from changed files and diff content using
//! pure heuristics: project-language detection (via [`crate::fault_locate`]),
//! file-extension mapping, and security-sensitive pattern matching.
//!
//! This is pattern matching, not magic. Confidence levels say how much to
//! trust each suggestion, and when nothing confident can be suggested the
//! result says so instead of guessing.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::fault_locate::{self, Language};

/// How much we trust a suggestion. `High` means the project layout directly
/// supports it (marker file present, matching files changed). `Medium` means
/// it is the conventional check for the detected language but we could not
/// confirm the exact target. `Low` means it is a reasonable guess the caller
/// should sanity-check before using as a task-loop verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    High,
    Medium,
    Low,
}

/// One suggested verifier command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuggestedVerifier {
    /// The suggested command as a single shell string,
    /// e.g. `"cargo test --quiet"`.
    pub command: String,
    /// Why this verifier was suggested, in one sentence.
    pub reason: String,
    pub confidence: Confidence,
}

/// One heuristically-detected weak spot in test coverage.
///
/// These are hints, not proofs. A flagged spot means "worth a second look",
/// and every entry carries a confidence level saying how much to trust it.
/// False positives are expected; the confidence label is the honesty
/// mechanism.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeakSpot {
    /// File and 1-based line, e.g. `"axiom_engine/response_cache.py:132"`.
    pub location: String,
    /// What looks weak, in one sentence.
    pub issue: String,
    pub confidence: Confidence,
}

/// The JSON envelope printed by `axiom verify-suggest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifySuggestions {
    pub suggested_verifiers: Vec<SuggestedVerifier>,
    /// Present when we could not determine a good verifier; explains why
    /// instead of guessing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Heuristic weak spots found by scanning changed source files against
    /// test files. Hints, not proofs; see [`WeakSpot`].
    pub weak_spots: Vec<WeakSpot>,
}

/// Input to the suggestion engine.
#[derive(Debug, Clone, Default)]
pub struct SuggestInput {
    /// Changed file paths (relative or absolute).
    pub files: Vec<PathBuf>,
    /// Optional unified-diff text; scanned for security-sensitive patterns.
    pub diff_text: Option<String>,
    /// Project root used for language detection.
    pub root: PathBuf,
}

/// Security-sensitive pattern groups. Each entry is `(group name, keywords)`.
/// Matching is case-insensitive substring search over the diff text and the
/// changed file paths.
const SECURITY_PATTERNS: &[(&str, &[&str])] = &[
    (
        "authentication",
        &[
            "password",
            "passwd",
            "authenticate",
            "login",
            "session_token",
            "jwt",
            "oauth",
            "api_key",
            "apikey",
        ],
    ),
    (
        "cryptography",
        &[
            "hmac",
            "encrypt",
            "decrypt",
            "private_key",
            "secret_key",
            "aes",
            "sha256",
            "sha-256",
            "digital_signature",
            "signing",
        ],
    ),
    (
        "ssrf/network",
        &[
            "ssrf",
            "private_range",
            "169.254",
            "dns_rebind",
            "rebind",
            "allowlist",
            "denylist",
        ],
    ),
    (
        "unsafe execution",
        &[
            "unsafe",
            "eval(",
            "exec(",
            "shell=True",
            "pickle.loads",
            "subprocess",
        ],
    ),
];

/// Suggest verifier commands for the given changed files / diff.
pub fn suggest_verifiers(input: &SuggestInput) -> VerifySuggestions {
    // Nothing to analyze: refuse to suggest rather than guess. Language
    // detection alone is not enough — a verifier must be grounded in what
    // actually changed.
    let has_diff = input
        .diff_text
        .as_deref()
        .is_some_and(|d| !d.trim().is_empty());
    if input.files.is_empty() && !has_diff {
        return VerifySuggestions {
            suggested_verifiers: Vec::new(),
            note: Some(
                "no changed files or diff provided; pass file paths or a \
                 unified diff so a verifier can be grounded in the change"
                    .to_string(),
            ),
            weak_spots: Vec::new(),
        };
    }

    let mut out: Vec<SuggestedVerifier> = Vec::new();

    let language = fault_locate::detect_language(&input.root);
    let exts = changed_extensions(&input.files);

    // Baseline compile check for the detected language. This always comes
    // first: a verifier that does not even compile-check is a weak gate.
    if let Some(lang) = language {
        if let Some((cmd, reason)) = baseline_compile(lang, &exts) {
            out.push(SuggestedVerifier {
                command: cmd,
                reason,
                confidence: Confidence::High,
            });
        }
    }

    // Test commands, scoped to what changed where we can.
    out.extend(test_suggestions(language, &input.files, &input.root));

    // Lint as a secondary gate (never the primary verifier).
    if let Some(lang) = language {
        if let Some((cmd, reason)) = lint_suggestion(lang, &exts) {
            out.push(SuggestedVerifier {
                command: cmd,
                reason,
                confidence: Confidence::Medium,
            });
        }
    }

    // Security-sensitive changes get an explicit callout.
    out.extend(security_suggestions(input));

    if out.is_empty() {
        return VerifySuggestions {
            suggested_verifiers: Vec::new(),
            note: Some(
                "could not determine a verifier: no recognized project language \
                 (Cargo.toml / go.mod / pyproject.toml / package.json / …) and no \
                 recognized source files among the changed paths"
                    .to_string(),
            ),
            weak_spots: detect_weak_spots(input),
        };
    }

    VerifySuggestions {
        suggested_verifiers: out,
        note: None,
        weak_spots: detect_weak_spots(input),
    }
}

/// Lowercase extensions of the changed files, e.g. `{"rs", "toml"}`.
fn changed_extensions(files: &[PathBuf]) -> std::collections::HashSet<String> {
    files
        .iter()
        .filter_map(|p| p.extension())
        .filter_map(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .collect()
}

/// The fast compile check for a language, when the changed files include
/// that language's sources. Returns `None` when the language has no
/// meaningful compile step or no relevant files changed.
fn baseline_compile(
    lang: Language,
    exts: &std::collections::HashSet<String>,
) -> Option<(String, String)> {
    let has = |e: &str| exts.contains(e);
    match lang {
        Language::Rust if has("rs") => Some((
            "cargo check --all-targets".to_string(),
            "Rust sources changed; compile-check every target before testing".to_string(),
        )),
        Language::Go if has("go") => Some((
            "go build ./...".to_string(),
            "Go sources changed; build all packages".to_string(),
        )),
        Language::TypeScript if has("ts") || has("tsx") => Some((
            "npx tsc --noEmit".to_string(),
            "TypeScript sources changed; type-check without emitting".to_string(),
        )),
        Language::Java if has("java") => Some((
            "mvn -q compile".to_string(),
            "Java sources changed; compile with Maven".to_string(),
        )),
        Language::Gradle if has("java") || has("kt") => Some((
            "gradle compileJava --quiet".to_string(),
            "JVM sources changed; compile with Gradle".to_string(),
        )),
        // Python, JavaScript and Ruby have no separate compile step worth
        // gating on; their test runners surface syntax errors directly.
        _ => None,
    }
}

/// Test commands for the detected language, scoped to changed files when the
/// layout makes that possible.
fn test_suggestions(
    language: Option<Language>,
    files: &[PathBuf],
    root: &Path,
) -> Vec<SuggestedVerifier> {
    let mut out = Vec::new();
    let Some(lang) = language else {
        return out;
    };
    match lang {
        Language::Rust => {
            // A changed integration test file maps directly to --test <name>.
            let mut scoped = false;
            for f in files {
                if let Some(stem) = integration_test_name(f) {
                    out.push(SuggestedVerifier {
                        command: format!("cargo test --test {stem}"),
                        reason: format!(
                            "integration test `{}` changed; run it directly",
                            f.display()
                        ),
                        confidence: Confidence::High,
                    });
                    scoped = true;
                }
            }
            if !scoped {
                out.push(SuggestedVerifier {
                    command: "cargo test".to_string(),
                    reason: "Rust sources changed; run the full test suite".to_string(),
                    confidence: Confidence::High,
                });
            }
        }
        Language::Python => {
            let test_files: Vec<&PathBuf> = files
                .iter()
                .filter(|p| {
                    p.extension().and_then(|e| e.to_str()) == Some("py") && is_python_test_file(p)
                })
                .collect();
            if test_files.is_empty() {
                out.push(SuggestedVerifier {
                    command: "pytest -q".to_string(),
                    reason: "Python sources changed; run pytest".to_string(),
                    confidence: Confidence::Medium,
                });
            } else {
                let paths = test_files
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                out.push(SuggestedVerifier {
                    command: format!("pytest -q {paths}"),
                    reason: "test files changed; run them directly".to_string(),
                    confidence: Confidence::High,
                });
            }
            let _ = root;
        }
        Language::Go => {
            out.push(SuggestedVerifier {
                command: "go test ./...".to_string(),
                reason: "Go sources changed; test all packages".to_string(),
                confidence: Confidence::High,
            });
        }
        Language::JavaScript | Language::TypeScript => {
            out.push(SuggestedVerifier {
                command: "npm test --silent".to_string(),
                reason: "JS/TS sources changed; run the npm test script (verify it exists)"
                    .to_string(),
                confidence: Confidence::Medium,
            });
        }
        Language::Java => {
            out.push(SuggestedVerifier {
                command: "mvn -q test".to_string(),
                reason: "Java sources changed; run Maven tests".to_string(),
                confidence: Confidence::High,
            });
        }
        Language::Gradle => {
            out.push(SuggestedVerifier {
                command: "gradle test --quiet".to_string(),
                reason: "JVM sources changed; run Gradle tests".to_string(),
                confidence: Confidence::High,
            });
        }
        Language::Ruby => {
            out.push(SuggestedVerifier {
                command: "rake test".to_string(),
                reason: "Ruby sources changed; run the test task".to_string(),
                confidence: Confidence::Medium,
            });
        }
        Language::Chimera => {
            out.push(SuggestedVerifier {
                command: "axiom chimera check".to_string(),
                reason: "ChimeraLang project; type-check the sources".to_string(),
                confidence: Confidence::Medium,
            });
        }
    }
    out
}

/// If `path` looks like `tests/<name>.rs` (or `tests/<name>/...`), return
/// `<name>` for `cargo test --test <name>`.
fn integration_test_name(path: &Path) -> Option<String> {
    let mut comps = path.components();
    let first = comps.next()?.as_os_str().to_str()?;
    if first != "tests" {
        return None;
    }
    let second = comps.next()?.as_os_str().to_str()?;
    let stem = second.strip_suffix(".rs").unwrap_or(second);
    if stem.is_empty() {
        None
    } else {
        Some(stem.to_string())
    }
}

/// `test_*.py`, `*_test.py`, or anything under a `tests/`/`test/` directory.
fn is_python_test_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if name.starts_with("test_") || name.ends_with("_test.py") {
        return true;
    }
    path.components().any(|c| {
        matches!(
            c.as_os_str()
                .to_str()
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("tests") | Some("test")
        )
    })
}

/// Lint suggestions: secondary gates, never the primary verifier.
fn lint_suggestion(
    lang: Language,
    exts: &std::collections::HashSet<String>,
) -> Option<(String, String)> {
    let has = |e: &str| exts.contains(e);
    match lang {
        Language::Rust if has("rs") => Some((
            "cargo clippy --all-targets -- -D warnings".to_string(),
            "lint Rust changes with clippy; denies warnings".to_string(),
        )),
        Language::Python if has("py") => Some((
            "ruff check .".to_string(),
            "lint Python changes with ruff (skip if the project uses another linter)".to_string(),
        )),
        _ => None,
    }
}

/// Security callouts: when the diff or file paths touch security-sensitive
/// areas, suggest running the related tests explicitly so a too-broad
/// verifier cannot silently skip them.
fn security_suggestions(input: &SuggestInput) -> Vec<SuggestedVerifier> {
    let haystack = {
        let mut h = String::new();
        if let Some(d) = &input.diff_text {
            h.push_str(d);
            h.push('\n');
        }
        for f in &input.files {
            h.push_str(&f.display().to_string());
            h.push('\n');
        }
        h.to_ascii_lowercase()
    };
    if haystack.trim().is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (group, keywords) in SECURITY_PATTERNS {
        let hits: Vec<&&str> = keywords.iter().filter(|k| haystack.contains(**k)).collect();
        if hits.is_empty() {
            continue;
        }
        let shown: Vec<String> = hits.iter().take(3).map(|k| format!("`{k}`")).collect();
        // Try to scope to a matching test target; fall back to an explicit
        // reminder when we cannot construct one honestly.
        let scoped = scope_security_test(input, group);
        match scoped {
            Some(cmd) => out.push(SuggestedVerifier {
                command: cmd.clone(),
                reason: format!(
                    "change touches {group} (matched {}); run the related tests explicitly",
                    shown.join(", ")
                ),
                confidence: Confidence::Medium,
            }),
            None => out.push(SuggestedVerifier {
                command: String::new(),
                reason: format!(
                    "change touches {group} (matched {}); no related test target \
                     detected — add or name one explicitly rather than relying on \
                     the broad suite",
                    shown.join(", ")
                ),
                confidence: Confidence::Low,
            }),
        }
    }
    // Drop low-confidence entries that carry no command: they are advice, and
    // the JSON contract is a list of commands. Keep the advice in `note`-style
    // callers instead. (We keep them out here to keep the schema honest.)
    out.retain(|s| !s.command.is_empty());
    out
}

/// Try to build a concrete test command for a security group by matching
/// changed test files against the group name. Returns `None` when nothing
/// maps cleanly — the caller then stays silent instead of guessing.
fn scope_security_test(input: &SuggestInput, group: &str) -> Option<String> {
    let needle = match group {
        "authentication" => "auth",
        "cryptography" => "crypto",
        "ssrf/network" => "ssrf",
        "unsafe execution" => return None,
        _ => return None,
    };
    for f in &input.files {
        let lower = f.display().to_string().to_ascii_lowercase();
        if !lower.contains(needle) {
            continue;
        }
        if lower.ends_with(".rs") {
            if let Some(stem) = integration_test_name(f) {
                return Some(format!("cargo test --test {stem}"));
            }
            // Do not fall back to a `cargo test <needle>` name filter: it can
            // match zero tests and exit 0, producing a passing verifier that
            // checked nothing. Only real test targets get a scoped command.
            continue;
        }
        if lower.ends_with(".py") && is_python_test_file(f) {
            return Some(format!("pytest -q {}", f.display()));
        }
    }
    None
}

/// Scan changed source files against test files for heuristic coverage weak
/// spots.
///
/// Reads files from disk (relative paths resolve against `input.root`);
/// unreadable files are skipped silently. Test files are taken from the
/// changed files plus conventional auto-discovered locations
/// (`tests/test_<stem>.py`, …). Returns hints, not proofs.
fn detect_weak_spots(input: &SuggestInput) -> Vec<WeakSpot> {
    let mut sources: Vec<(String, String)> = Vec::new(); // (display path, content)
    let mut test_corpus = String::new();
    let mut seen_tests = std::collections::HashSet::new();

    for f in &input.files {
        let full = if f.is_absolute() {
            f.clone()
        } else {
            input.root.join(f)
        };
        let Ok(content) = std::fs::read_to_string(&full) else {
            continue;
        };
        let display = f.display().to_string();
        if is_test_path(f) {
            if seen_tests.insert(display.clone()) {
                test_corpus.push_str(&content);
                test_corpus.push('\n');
            }
        } else if is_analyzable_source(f) {
            sources.push((display, content));
        }
    }

    // Auto-discover conventional Python test files for sources, e.g.
    // `axiom_engine/foo.py` -> `tests/test_foo.py`. Only Python: the
    // detectors below are Python-first (Rust gets a best-effort pass).
    for (display, _) in &sources {
        let path = Path::new(display);
        if path.extension().and_then(|e| e.to_str()) != Some("py") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        for candidate in [
            format!("tests/test_{stem}.py"),
            format!("test/test_{stem}.py"),
            format!("tests/{stem}_test.py"),
        ] {
            if seen_tests.contains(&candidate) {
                continue;
            }
            if let Ok(content) = std::fs::read_to_string(input.root.join(&candidate)) {
                seen_tests.insert(candidate);
                test_corpus.push_str(&content);
                test_corpus.push('\n');
            }
        }
    }

    let mut out = Vec::new();
    for (display, content) in &sources {
        out.extend(weak_spots_for_file(display, content, &test_corpus));
    }
    out
}

/// True for test files: `test_*.py`, `*_test.<ext>`, or anything under a
/// `tests/`/`test/` directory.
fn is_test_path(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if name.starts_with("test_") || name.contains("_test.") {
        return true;
    }
    path.components().any(|c| {
        matches!(
            c.as_os_str()
                .to_str()
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("tests") | Some("test")
        )
    })
}

/// Source extensions the weak-spot detectors understand.
fn is_analyzable_source(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("py") | Some("rs")
    )
}

/// Dispatch per-file detectors by extension.
fn weak_spots_for_file(display: &str, content: &str, test_corpus: &str) -> Vec<WeakSpot> {
    let ext = Path::new(display)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let lines: Vec<&str> = content.lines().collect();
    let mut out = Vec::new();
    match ext {
        "py" => {
            out.extend(python_untested_exceptions(display, &lines, test_corpus));
            out.extend(python_unpinned_arithmetic(display, &lines, test_corpus));
            out.extend(python_single_case_coverage(display, &lines, test_corpus));
            out.extend(boundary_identifier_checks(display, &lines, test_corpus));
        }
        "rs" => {
            out.extend(rust_untested_error_paths(display, &lines, test_corpus));
            out.extend(boundary_identifier_checks(display, &lines, test_corpus));
        }
        _ => {}
    }
    out
}

/// True if `word` appears as a whole word in `haystack`.
fn contains_word(haystack: &str, word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    haystack
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .any(|w| w == word)
}

/// Strip string literals and comments from one line of code, crudely.
/// `python_style` selects `#` comments (Python) vs `//` comments (Rust);
/// note `//` is NOT treated as a comment in Python mode (floor division).
fn strip_strings_and_comments(line: &str, python_style: bool) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;
    while let Some(c) = chars.next() {
        if in_single {
            if c == '\'' {
                in_single = false;
            }
            continue;
        }
        if in_double {
            if c == '"' {
                in_double = false;
            }
            continue;
        }
        match c {
            '\'' => in_single = true,
            '"' => in_double = true,
            '#' if python_style => break,
            '/' if !python_style && chars.peek() == Some(&'/') => break,
            _ => out.push(c),
        }
    }
    out
}

/// Detector 1: Python `except` handlers whose exception types are never
/// referenced by any test. A handler the tests never name is a handler the
/// tests never exercise.
fn python_untested_exceptions(display: &str, lines: &[&str], test_corpus: &str) -> Vec<WeakSpot> {
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        let Some(rest) = t.strip_prefix("except") else {
            continue;
        };
        // `except`, `except:`, `except E`, `except (E1, E2)`, `except E as x`
        // — but not identifiers that merely start with "except".
        if !(rest.is_empty() || rest.starts_with([':', ' ', '('])) {
            continue;
        }
        let mut clause = rest.trim().trim_end_matches(':').trim();
        if let Some(idx) = clause.find(" as ") {
            clause = clause[..idx].trim();
        }
        if clause.is_empty() {
            out.push(WeakSpot {
                location: format!("{display}:{}", i + 1),
                issue: "bare `except:` handler; its error path cannot be tied to a \
                        specific exception type in tests"
                    .to_string(),
                confidence: Confidence::Low,
            });
            continue;
        }
        let types: Vec<&str> = clause
            .trim_matches(|c| c == '(' || c == ')')
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        let tested = types.iter().any(|ty| {
            contains_word(test_corpus, ty)
                || ty
                    .rsplit('.')
                    .next()
                    .is_some_and(|last| contains_word(test_corpus, last))
        });
        if !tested {
            out.push(WeakSpot {
                location: format!("{display}:{}", i + 1),
                issue: format!(
                    "exception handler for {} never referenced by tests",
                    types.join(", ")
                ),
                confidence: Confidence::High,
            });
        }
    }
    out
}

/// Detector 2: functions containing float-relevant arithmetic (`/`, `*`,
/// `%`, `**`) whose names appear in tests, while the test corpus has no
/// approximate-float assertions (`approx`, `almost_equal`, `isclose`) and no
/// exact float-literal comparisons. Mirrors the real testteeth finding where
/// `hits / total` -> `hits * total` survived because no test pinned the value.
fn python_unpinned_arithmetic(display: &str, lines: &[&str], test_corpus: &str) -> Vec<WeakSpot> {
    // function name -> (def line, first arithmetic line)
    let mut arith: Vec<(String, usize)> = Vec::new();
    let mut current_fn: Option<(String, usize)> = None; // (name, indent)
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("def ") {
            let name: String = rest
                .chars()
                .take_while(|c| *c != '(' && !c.is_whitespace())
                .collect();
            if !name.is_empty() {
                let indent = line.len() - t.len();
                current_fn = Some((name, indent));
            }
            continue;
        }
        let (fn_name, fn_indent) = match &current_fn {
            Some(v) => v.clone(),
            None => continue,
        };
        // A non-blank line at indent <= the def ends the function body.
        let indent = line.len() - line.trim_start().len();
        if !t.is_empty() && indent <= fn_indent {
            current_fn = None;
            continue;
        }
        let code = strip_strings_and_comments(line, true);
        // Float-relevant operators: `/`, `%`, and `*` as multiplication or
        // power (not `*,` markers or `*args` unpacking).
        let has_dangerous = has_div_or_mod(&code) || has_arithmetic_star(&code);
        if has_dangerous {
            // Avoid double-counting the same function; keep the first line.
            if !arith.iter().any(|(n, _)| n == &fn_name) {
                arith.push((fn_name, i + 1));
            }
        }
    }

    let corpus_has_float_assert = test_corpus.contains("approx")
        || test_corpus.contains("almost_equal")
        || test_corpus.contains("assertAlmostEqual")
        || test_corpus.contains("isclose")
        || has_float_literal_comparison(test_corpus);

    let mut out = Vec::new();
    for (fn_name, line_no) in arith {
        let named_in_tests = contains_word(test_corpus, &fn_name);
        // With no related tests on disk we cannot call the value "unpinned"
        // (it may simply be untested, a different issue).
        if !named_in_tests && test_corpus.trim().is_empty() {
            continue;
        }
        if !corpus_has_float_assert {
            // Medium when the function is named by tests (it runs under
            // test, value not pinned); Low when it is not directly named
            // (it may run indirectly, e.g. via another method).
            let confidence = if named_in_tests {
                Confidence::Medium
            } else {
                Confidence::Low
            };
            out.push(WeakSpot {
                location: format!("{display}:{line_no}"),
                issue: format!(
                    "arithmetic in `{fn_name}()` not pinned by approximate float \
                     assertions in tests; operator swaps (e.g. `/` -> `*`) would \
                     go unnoticed"
                ),
                confidence,
            });
        }
    }
    out
}

/// True if `code` contains a `/` or `%` operator (division/modulo).
/// Callers strip strings and comments first.
fn has_div_or_mod(code: &str) -> bool {
    code.contains('/') || code.contains('%')
}

/// True if `code` contains a `*` used as multiplication or power — not a
/// `*,` keyword-only marker and not `*args` / `**kwargs` unpacking.
fn has_arithmetic_star(code: &str) -> bool {
    let chars: Vec<char> = code.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if chars[i] != '*' {
            i += 1;
            continue;
        }
        // `*,` is a keyword-only argument marker, not multiplication.
        if chars.get(i + 1) == Some(&',') {
            i += 1;
            continue;
        }
        // `**`: unpacking (`f(**kwargs)`) vs power (`a ** b`). Unpacking has
        // an identifier immediately after the stars; power has an operand
        // (possibly after whitespace).
        if chars.get(i + 1) == Some(&'*') {
            let after2 = chars.get(i + 2).copied();
            let before = if i > 0 { Some(chars[i - 1]) } else { None };
            let is_unpack = after2.is_some_and(|c| c.is_alphabetic() || c == '_')
                && matches!(before, None | Some('(') | Some(',') | Some(' '));
            if is_unpack {
                i += 2;
                continue;
            }
            return true; // power operator
        }
        // Single `*`: unpacking (`f(*args)`) has the star directly after
        // `(` or `,` and directly before an identifier.
        let before = if i > 0 { Some(chars[i - 1]) } else { None };
        let next = chars.get(i + 1).copied();
        let is_unpack = next.is_some_and(|c| c.is_alphabetic() || c == '_')
            && matches!(before, Some('(') | Some(','));
        if is_unpack {
            i += 1;
            continue;
        }
        return true; // multiplication
    }
    false
}

/// Crude check for `== <float literal>` in test code.
fn has_float_literal_comparison(corpus: &str) -> bool {
    let bytes = corpus.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'=' && bytes[i + 1] == b'=' {
            let rest = &corpus[i + 2..];
            let rest = rest.trim_start();
            // optional unary minus, digits, dot, digits
            let mut chars = rest.chars();
            if chars.next() == Some('-') {
                // keep going
            }
            let digit_run: String = rest
                .trim_start_matches('-')
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if digit_run.contains('.')
                && digit_run.chars().any(|c| c.is_ascii_digit())
                && digit_run != "."
            {
                return true;
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    false
}

/// Detector 3: functions taking dict/list-like parameters where every test
/// call uses single-element literals. Boundary and multi-key behavior then
/// goes untested (e.g. `sort_keys` only matters with multiple keys).
fn python_single_case_coverage(display: &str, lines: &[&str], test_corpus: &str) -> Vec<WeakSpot> {
    const DICT_LIKE: &[&str] = &[
        "dict", "list", "mapping", "dict[", "list[", "mapping[", "dict,", "list,",
    ];
    const DICT_LIKE_NAMES: &[&str] = &[
        "data", "mapping", "items", "records", "entries", "payload", "obj", "values",
    ];
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        let Some(rest) = t.strip_prefix("def ") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| *c != '(' && !c.is_whitespace())
            .collect();
        if name.is_empty() {
            continue;
        }
        // Extract the parameter list with paren-depth counting.
        let Some(open) = rest.find('(') else {
            continue;
        };
        let mut depth = 0;
        let mut close = None;
        for (j, c) in rest.char_indices().skip(open) {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(j);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else { continue };
        let params = &rest[open + 1..close];
        // Split params on top-level commas.
        let mut params_vec = Vec::new();
        let mut depth = 0;
        let mut start = 0;
        for (j, c) in params.char_indices() {
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                ',' if depth == 0 => {
                    params_vec.push(params[start..j].trim());
                    start = j + 1;
                }
                _ => {}
            }
        }
        params_vec.push(params[start..].trim());
        let takes_collection = params_vec.iter().any(|p| {
            let lower = p.to_ascii_lowercase();
            // Skip `self`/`cls`.
            if lower == "self" || lower == "cls" {
                return false;
            }
            let ann = lower.split(':').nth(1).unwrap_or("");
            DICT_LIKE.iter().any(|d| ann.contains(d)) || {
                let pname = lower
                    .split(':')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .trim_start_matches('*');
                DICT_LIKE_NAMES.contains(&pname)
            }
        });
        if !takes_collection {
            continue;
        }
        // Find test calls to this function and count literal elements.
        let call_sizes = test_call_literal_sizes(test_corpus, &name);
        if call_sizes.is_empty() {
            continue; // Not called in tests: different problem.
        }
        if call_sizes.iter().all(|&n| n <= 1) {
            out.push(WeakSpot {
                location: format!("{display}:{}", i + 1),
                issue: format!(
                    "`{name}()` takes dict/list input but every test call uses \
                     single-element (or no) literals; multi-element behavior \
                     untested"
                ),
                confidence: Confidence::Medium,
            });
        }
    }
    out
}

/// For each call `name(` in the corpus, count top-level elements of the
/// first dict/list literal argument. Returns one count per call; calls
/// without a literal argument contribute 0.
fn test_call_literal_sizes(corpus: &str, name: &str) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut search = corpus;
    let needle = format!("{name}(");
    while let Some(idx) = search.find(&needle) {
        // Word-boundary check: the char before must not be an identifier
        // char. A `.` before is fine — `c.put(...)` is a genuine call to
        // something named `put` (a method).
        let ok_boundary = idx == 0 || {
            let prev = search.as_bytes()[idx - 1] as char;
            !(prev.is_alphanumeric() || prev == '_')
        };
        let args_start = idx + needle.len();
        if !ok_boundary {
            search = &search[args_start..];
            continue;
        }
        // Find matching close paren.
        let args = &search[args_start..];
        let mut depth = 1;
        let mut end = None;
        let mut in_s = false;
        let mut in_d = false;
        for (j, c) in args.char_indices() {
            if in_s {
                if c == '\'' {
                    in_s = false;
                }
                continue;
            }
            if in_d {
                if c == '"' {
                    in_d = false;
                }
                continue;
            }
            match c {
                '\'' => in_s = true,
                '"' => in_d = true,
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(j);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else {
            search = &search[args_start..];
            continue;
        };
        let arg_text = &args[..end];
        sizes.push(largest_literal_size(arg_text));
        search = &search[args_start + end..];
    }
    sizes
}

/// Size (element count) of the largest dict/list literal in `text`;
/// 0 when there is none.
fn largest_literal_size(text: &str) -> usize {
    let mut best = 0;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '{' || c == '[' {
            let open = c;
            let close = if open == '{' { '}' } else { ']' };
            let mut depth = 0;
            let mut commas = 0;
            let mut non_empty = false;
            let mut in_s = false;
            let mut in_d = false;
            let mut j = i;
            let mut closed = false;
            while j < bytes.len() {
                let d = bytes[j] as char;
                if in_s {
                    if d == '\'' {
                        in_s = false;
                    }
                    j += 1;
                    continue;
                }
                if in_d {
                    if d == '"' {
                        in_d = false;
                    }
                    j += 1;
                    continue;
                }
                match d {
                    '\'' => in_s = true,
                    '"' => in_d = true,
                    _ if d == open => depth += 1,
                    _ if d == close => {
                        depth -= 1;
                        if depth == 0 {
                            closed = true;
                            break;
                        }
                    }
                    ',' if depth == 1 => commas += 1,
                    _ if depth >= 1 && !d.is_whitespace() => non_empty = true,
                    _ => {}
                }
                j += 1;
            }
            if closed {
                let size = if non_empty { commas + 1 } else { 0 };
                if size > best {
                    best = size;
                }
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    best
}

/// Detector 4 (shared): boundary identifiers (`limit`, `threshold`,
/// `max_*`, `min_*`, `capacity`, …) used in source but never referenced by
/// tests. Low confidence by design: absence of the identifier is a weak
/// signal, but a cheap one worth surfacing.
fn boundary_identifier_checks(display: &str, lines: &[&str], test_corpus: &str) -> Vec<WeakSpot> {
    const BOUNDARY_IDENTS: &[&str] = &[
        "limit",
        "threshold",
        "capacity",
        "max_entries",
        "max_size",
        "min_entries",
        "min_size",
        "evict",
    ];
    let lower_corpus = test_corpus.to_ascii_lowercase();
    let mut out = Vec::new();
    let mut flagged: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (i, line) in lines.iter().enumerate() {
        let code = strip_strings_and_comments(line, display.ends_with(".py")).to_ascii_lowercase();
        for ident in BOUNDARY_IDENTS {
            if !code.contains(ident) || !flagged.insert(ident.to_string()) {
                continue;
            }
            if !lower_corpus.contains(ident) {
                out.push(WeakSpot {
                    location: format!("{display}:{}", i + 1),
                    issue: format!(
                        "boundary logic around `{ident}` never referenced by tests; \
                         limit/eviction edge cases likely untested"
                    ),
                    confidence: Confidence::Low,
                });
            }
        }
    }
    out
}

/// Rust best-effort: `Err(<Variant>)` constructed or matched in source while
/// the variant name never appears in tests. Low confidence: naming alone is
/// a weak signal for error-path coverage.
fn rust_untested_error_paths(display: &str, lines: &[&str], test_corpus: &str) -> Vec<WeakSpot> {
    let mut out = Vec::new();
    let mut flagged: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (i, line) in lines.iter().enumerate() {
        let code = strip_strings_and_comments(line, false);
        let mut search = code.as_str();
        while let Some(idx) = search.find("Err(") {
            let after = &search[idx + 4..];
            // Capture a possibly qualified path like `Error::NotFound`.
            let path: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == ':')
                .collect();
            search = after;
            // The variant is the last path segment: `Error::NotFound` -> `NotFound`.
            let variant = path.rsplit("::").next().unwrap_or("").trim_matches(':');
            if variant.is_empty() || !flagged.insert(variant.to_string()) {
                continue;
            }
            if !contains_word(test_corpus, variant) {
                out.push(WeakSpot {
                    location: format!("{display}:{}", i + 1),
                    issue: format!(
                        "error variant `{variant}` never referenced by tests; \
                         its error path is likely untested"
                    ),
                    confidence: Confidence::Low,
                });
            }
        }
    }
    out
}

/// Human-readable rendering for `--explain`.
pub fn explain(suggestions: &VerifySuggestions) -> String {
    let mut s = String::new();
    if suggestions.suggested_verifiers.is_empty() {
        s.push_str("No verifier suggestions.\n");
    } else {
        s.push_str("Suggested verifiers (strongest gate first):\n");
        for (i, v) in suggestions.suggested_verifiers.iter().enumerate() {
            let conf = match v.confidence {
                Confidence::High => "high",
                Confidence::Medium => "medium",
                Confidence::Low => "low",
            };
            s.push_str(&format!(
                "  {}. `{}`\n     [confidence: {conf}] {}\n",
                i + 1,
                v.command,
                v.reason
            ));
        }
    }
    if !suggestions.weak_spots.is_empty() {
        s.push_str("\nWeak spots (heuristic — worth a second look):\n");
        for (i, w) in suggestions.weak_spots.iter().enumerate() {
            let conf = match w.confidence {
                Confidence::High => "high",
                Confidence::Medium => "medium",
                Confidence::Low => "low",
            };
            s.push_str(&format!(
                "  {}. {} [confidence: {conf}]\n     {}\n",
                i + 1,
                w.location,
                w.issue
            ));
        }
    }
    if let Some(note) = &suggestions.note {
        s.push_str(&format!("\nNote: {note}\n"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Build a temp project root with the given marker files.
    fn temp_root(markers: &[&str]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "axiom-verify-suggest-test-{}-{}",
            std::process::id(),
            rand_suffix()
        ));
        fs::create_dir_all(&dir).unwrap();
        for m in markers {
            fs::write(dir.join(m), "").unwrap();
        }
        dir
    }

    fn rand_suffix() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        format!("{n}")
    }

    fn input(files: &[&str], root: &Path, diff: Option<&str>) -> SuggestInput {
        SuggestInput {
            files: files.iter().map(PathBuf::from).collect(),
            diff_text: diff.map(str::to_string),
            root: root.to_path_buf(),
        }
    }

    #[test]
    fn rust_change_suggests_check_test_clippy() {
        let root = temp_root(&["Cargo.toml"]);
        let s = suggest_verifiers(&input(&["src/lib.rs"], &root, None));
        let cmds: Vec<&str> = s
            .suggested_verifiers
            .iter()
            .map(|v| v.command.as_str())
            .collect();
        assert!(
            cmds.contains(&"cargo check --all-targets"),
            "missing check: {cmds:?}"
        );
        assert!(cmds.contains(&"cargo test"), "missing test: {cmds:?}");
        assert!(
            cmds.contains(&"cargo clippy --all-targets -- -D warnings"),
            "missing clippy: {cmds:?}"
        );
        assert!(s.suggested_verifiers[0].confidence == Confidence::High);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rust_integration_test_scopes_to_test_target() {
        let root = temp_root(&["Cargo.toml"]);
        let s = suggest_verifiers(&input(&["tests/axiom_integration.rs"], &root, None));
        let cmds: Vec<&str> = s
            .suggested_verifiers
            .iter()
            .map(|v| v.command.as_str())
            .collect();
        assert!(
            cmds.contains(&"cargo test --test axiom_integration"),
            "missing scoped test: {cmds:?}"
        );
        // No unscoped `cargo test` when we scoped successfully.
        assert!(
            !cmds.contains(&"cargo test"),
            "should not also suggest bare cargo test"
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn python_test_file_scopes_pytest() {
        let root = temp_root(&["pyproject.toml"]);
        let s = suggest_verifiers(&input(&["tests/test_auth.py"], &root, None));
        let cmds: Vec<&str> = s
            .suggested_verifiers
            .iter()
            .map(|v| v.command.as_str())
            .collect();
        assert!(
            cmds.iter()
                .any(|c| c.contains("pytest") && c.contains("test_auth.py")),
            "missing scoped pytest: {cmds:?}"
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn go_change_suggests_build_and_test() {
        let root = temp_root(&["go.mod"]);
        let s = suggest_verifiers(&input(&["main.go"], &root, None));
        let cmds: Vec<&str> = s
            .suggested_verifiers
            .iter()
            .map(|v| v.command.as_str())
            .collect();
        assert!(
            cmds.contains(&"go build ./..."),
            "missing go build: {cmds:?}"
        );
        assert!(cmds.contains(&"go test ./..."), "missing go test: {cmds:?}");
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn ssrf_diff_triggers_security_callout() {
        let root = temp_root(&["Cargo.toml"]);
        let diff = "fn check_host(h: &str) {\n // allowlist bypass via dns_rebind\n}";
        // Non-test source path: must not produce a `cargo test <needle>` name
        // filter, which can match zero tests and exit 0 while looking green.
        let s = suggest_verifiers(&input(&["src/safety/ssrf.rs"], &root, Some(diff)));
        let cmds: Vec<&str> = s
            .suggested_verifiers
            .iter()
            .map(|v| v.command.as_str())
            .collect();
        assert!(
            !cmds.iter().any(|c| c.contains("ssrf")),
            "non-test source path must not produce an ssrf test command: {cmds:?}"
        );
        // Real integration test target: scoped to it with medium confidence.
        let s = suggest_verifiers(&input(&["tests/ssrf.rs"], &root, Some(diff)));
        let scoped = s.suggested_verifiers.iter().find(|v| {
            v.reason.contains("ssrf/network") && v.command.contains("cargo test --test ssrf")
        });
        assert!(scoped.is_some(), "missing ssrf callout for tests/ssrf.rs");
        assert_eq!(scoped.unwrap().confidence, Confidence::Medium);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn unknown_project_says_so_honestly() {
        let root = temp_root(&[]);
        let s = suggest_verifiers(&input(&["notes.txt"], &root, None));
        assert!(s.suggested_verifiers.is_empty());
        assert!(s.note.is_some());
        assert!(s.note.as_ref().unwrap().contains("could not determine"));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn explain_renders_human_readable() {
        let root = temp_root(&["Cargo.toml"]);
        let s = suggest_verifiers(&input(&["src/lib.rs"], &root, None));
        let text = explain(&s);
        assert!(text.contains("cargo check --all-targets"));
        assert!(text.contains("confidence: high"));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn empty_input_says_so_honestly() {
        let root = temp_root(&["Cargo.toml"]);
        let s = suggest_verifiers(&input(&[], &root, None));
        assert!(s.suggested_verifiers.is_empty());
        assert!(s.note.is_some());
        assert!(s.note.as_ref().unwrap().contains("no changed files"));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn json_envelope_matches_contract() {
        let root = temp_root(&["Cargo.toml"]);
        let s = suggest_verifiers(&input(&["src/lib.rs"], &root, None));
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"suggested_verifiers\""));
        assert!(json.contains("\"weak_spots\""));
        assert!(json.contains("\"command\""));
        assert!(json.contains("\"reason\""));
        assert!(json.contains("\"confidence\""));
        assert!(json.contains("\"high\""));
        // Round-trips.
        let back: VerifySuggestions = serde_json::from_str(&json).unwrap();
        assert_eq!(back.suggested_verifiers.len(), s.suggested_verifiers.len());
        assert_eq!(back.weak_spots.len(), s.weak_spots.len());
        fs::remove_dir_all(&root).ok();
    }

    // ---- weak-spot detector tests (pure functions, no filesystem) ----

    const CACHE_SRC: &str = r#"
import json, logging
logger = logging.getLogger("axiom.cache")

class CacheStats:
    def to_dict(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total else 0.0
        return {"hit_rate": hit_rate}

def fingerprint(model: str, **kwargs) -> str:
    canonical = {"model": model}
    canonical.update(kwargs)
    serialized = json.dumps(canonical, sort_keys=True)
    return serialized

class ResponseCache:
    def __init__(self, max_entries: int = 1024):
        self.max_entries = max_entries

    def _load_from_disk(self) -> None:
        try:
            data = open("c.json").read()
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("nope %s", exc)

    def put(self, key: str, value: str, meta: dict = None) -> None:
        self._cache[key] = value
"#;

    const CACHE_TESTS_WEAK: &str = r#"
def test_fingerprint_stable():
    a = fingerprint(model="m", max_tokens=10)
    b = fingerprint(model="m", max_tokens=10)
    assert a == b

def test_stats_counts():
    s = CacheStats()
    s.hits = 2; s.misses = 1
    d = s.to_dict()
    assert d["hits"] == 2
"#;

    #[test]
    fn untested_exceptions_flagged() {
        let lines: Vec<&str> = CACHE_SRC.lines().collect();
        let spots = python_untested_exceptions("cache.py", &lines, CACHE_TESTS_WEAK);
        // OSError and JSONDecodeError never appear in the test corpus.
        assert_eq!(
            spots.len(),
            1,
            "expected one except line flagged: {spots:?}"
        );
        assert!(spots[0].location.starts_with("cache.py:"));
        assert!(spots[0].issue.contains("OSError"));
        assert_eq!(spots[0].confidence, Confidence::High);
    }

    #[test]
    fn untested_exceptions_quiet_when_referenced() {
        let lines: Vec<&str> = CACHE_SRC.lines().collect();
        let tests =
            format!("{CACHE_TESTS_WEAK}\ndef test_oserror():\n    pytest.raises(OSError)\n");
        let spots = python_untested_exceptions("cache.py", &lines, &tests);
        assert!(
            spots.is_empty(),
            "no flags when OSError is referenced: {spots:?}"
        );
    }

    #[test]
    fn unpinned_arithmetic_flagged() {
        let lines: Vec<&str> = CACHE_SRC.lines().collect();
        let spots = python_unpinned_arithmetic("cache.py", &lines, CACHE_TESTS_WEAK);
        // to_dict does hits/total; tests mention to_dict but never pin floats.
        assert!(
            spots.iter().any(|w| w.issue.contains("to_dict")),
            "expected to_dict flagged: {spots:?}"
        );
        assert!(spots.iter().all(|w| w.confidence == Confidence::Medium));
    }

    #[test]
    fn unpinned_arithmetic_quiet_with_approx() {
        let lines: Vec<&str> = CACHE_SRC.lines().collect();
        let tests = format!("{CACHE_TESTS_WEAK}\nassert d['hit_rate'] == pytest.approx(0.667)\n");
        let spots = python_unpinned_arithmetic("cache.py", &lines, &tests);
        assert!(spots.is_empty(), "approx pins the float: {spots:?}");
    }

    #[test]
    fn single_case_coverage_flagged() {
        let lines: Vec<&str> = CACHE_SRC.lines().collect();
        // put() takes `meta: dict`; tests never call put at all here, so add
        // a single-element call to the corpus.
        let tests = "def test_put():\n    c.put('k', 'v', {'a': 1})\n";
        let spots = python_single_case_coverage("cache.py", &lines, tests);
        assert!(
            spots.iter().any(|w| w.issue.contains("put")),
            "expected put flagged: {spots:?}"
        );
    }

    #[test]
    fn single_case_coverage_quiet_with_multi() {
        let lines: Vec<&str> = CACHE_SRC.lines().collect();
        let tests = "def test_put():\n    c.put('k', 'v', {'a': 1, 'b': 2})\n";
        let spots = python_single_case_coverage("cache.py", &lines, tests);
        assert!(spots.is_empty(), "multi-element call is fine: {spots:?}");
    }

    #[test]
    fn boundary_identifier_flagged_when_absent() {
        let lines: Vec<&str> = CACHE_SRC.lines().collect();
        let spots = boundary_identifier_checks("cache.py", &lines, CACHE_TESTS_WEAK);
        assert!(
            spots.iter().any(|w| w.issue.contains("max_entries")),
            "expected max_entries flagged: {spots:?}"
        );
        assert!(spots.iter().all(|w| w.confidence == Confidence::Low));
    }

    #[test]
    fn rust_error_variant_flagged() {
        let src = "fn load() -> Result<String, Error> {\n    Err(Error::NotFound)\n}\n";
        let lines: Vec<&str> = src.lines().collect();
        let spots = rust_untested_error_paths("load.rs", &lines, "fn test_ok() {}");
        assert_eq!(spots.len(), 1);
        assert!(spots[0].issue.contains("NotFound"));
        assert_eq!(spots[0].confidence, Confidence::Low);
    }

    #[test]
    fn detect_weak_spots_reads_files_and_autodiscovers() {
        // Source file on disk; test file auto-discovered via tests/test_<stem>.py.
        let root = temp_root(&["pyproject.toml"]);
        fs::create_dir_all(root.join("pkg")).unwrap();
        fs::create_dir_all(root.join("tests")).unwrap();
        fs::write(
            root.join("pkg/mod.py"),
            "def work():\n    try:\n        x = 1\n    except OSError:\n        x = 2\n",
        )
        .unwrap();
        fs::write(
            root.join("tests/test_mod.py"),
            "def test_work():\n    assert True\n",
        )
        .unwrap();
        let s = suggest_verifiers(&input(&["pkg/mod.py"], &root, None));
        assert!(
            s.weak_spots.iter().any(|w| w.issue.contains("OSError")),
            "expected OSError weak spot via auto-discovery: {:?}",
            s.weak_spots
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn arithmetic_star_ignores_kwonly_marker() {
        // `*,` (keyword-only marker) must not count as multiplication.
        assert!(!has_arithmetic_star("    *,"));
        assert!(!has_arithmetic_star("def f(*args):"));
        assert!(!has_arithmetic_star("f(**kwargs)"));
        assert!(has_arithmetic_star("x = a * b"));
        assert!(has_arithmetic_star("x = a*b"));
        assert!(has_arithmetic_star("x = a ** 2"));
        assert!(has_div_or_mod("x = a / b"));
        assert!(has_div_or_mod("x = a % b"));
    }

    #[test]
    fn dogfood_response_cache_weak_spots() {
        // Dogfood: the detectors run against the repo's own
        // `axiom_engine/response_cache.py`. PR #202 hardened
        // `tests/test_response_cache.py` (OSError handlers, `to_dict`
        // hit-rate arithmetic), closing the gaps this test originally
        // dogfooded against. The detectors must therefore NOT flag them:
        // no false positives on covered code. If coverage regresses, the
        // detectors will flag the gaps again and this test will fail,
        // which is the intended dogfood signal.
        // Skipped when the repo files are absent (e.g. vendored builds).
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        if !root.join("axiom_engine/response_cache.py").exists()
            || !root.join("tests/test_response_cache.py").exists()
        {
            return;
        }
        let s = suggest_verifiers(&input(
            &[
                "axiom_engine/response_cache.py",
                "tests/test_response_cache.py",
            ],
            &root,
            None,
        ));
        assert!(
            !s.weak_spots
                .iter()
                .any(|w| w.issue.contains("OSError") && w.confidence == Confidence::High),
            "OSError handlers are covered by tests; must not flag: {:?}",
            s.weak_spots
        );
        assert!(
            !s.weak_spots.iter().any(|w| w.issue.contains("to_dict")),
            "to_dict hit-rate arithmetic is pinned by tests; must not flag: {:?}",
            s.weak_spots
        );
    }

    #[test]
    fn explain_renders_weak_spots() {
        let s = VerifySuggestions {
            suggested_verifiers: Vec::new(),
            note: None,
            weak_spots: vec![WeakSpot {
                location: "a.py:3".to_string(),
                issue: "exception handler for OSError never referenced by tests".to_string(),
                confidence: Confidence::High,
            }],
        };
        let text = explain(&s);
        assert!(text.contains("Weak spots"));
        assert!(text.contains("a.py:3"));
        assert!(text.contains("confidence: high"));
    }
}
