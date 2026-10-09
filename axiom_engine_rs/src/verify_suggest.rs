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

/// The JSON envelope printed by `axiom verify-suggest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifySuggestions {
    pub suggested_verifiers: Vec<SuggestedVerifier>,
    /// Present when we could not determine a good verifier; explains why
    /// instead of guessing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
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
        };
    }

    VerifySuggestions {
        suggested_verifiers: out,
        note: None,
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
        assert!(json.contains("\"command\""));
        assert!(json.contains("\"reason\""));
        assert!(json.contains("\"confidence\""));
        assert!(json.contains("\"high\""));
        // Round-trips.
        let back: VerifySuggestions = serde_json::from_str(&json).unwrap();
        assert_eq!(back.suggested_verifiers.len(), s.suggested_verifiers.len());
        fs::remove_dir_all(&root).ok();
    }
}
