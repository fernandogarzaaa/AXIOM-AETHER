//! Language-general fault localization.
//!
//! The autonomous repair loop ([`crate::solve`]) needs to know *which* file to
//! repair. When the caller does not name a source file, we recover candidates
//! directly from the failing tool's own output — every major compiler and test
//! runner already prints the offending path and line. Parsing that output is
//! what lets a single verify-gated loop repair Rust, Python, JS/TS, Go, C/C++,
//! Java, … from one mechanism, instead of being hand-driven and Rust-bound.
//!
//! Everything here is pure (string + filesystem-existence checks) and offline:
//! no model, no network. [`best_source`] is the one the solver calls; the rest
//! are building blocks and are unit-tested directly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Source-file extensions we recognise in tool output. Kept broad on purpose:
/// a false positive is harmless (the file simply won't exist under the root and
/// is skipped), while a miss means we fail to localize a real fault.
const SOURCE_EXTS: &[&str] = &[
    "rs", "py", "pyi", "js", "jsx", "ts", "tsx", "mjs", "cjs", "go", "c", "h", "cc", "cpp", "cxx",
    "hpp", "hh", "java", "kt", "kts", "rb", "php", "cs", "swift", "scala", "sh", "bash", "lua",
    "ml", "mli", "ex", "exs", "dart", "chimera",
];

/// A located source reference parsed from a tool's failure output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultSite {
    /// Path exactly as it appeared in the trace (may be relative or absolute).
    pub path: PathBuf,
    /// 1-based line number reported alongside it.
    pub line: u32,
    /// How many times this path appeared — a relevance signal for ranking.
    pub hits: u32,
}

/// Coarse project language, detected from marker files at the project root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    /// A Maven-built JVM project (`pom.xml`) → `mvn test`.
    Java,
    /// A Gradle-built JVM project (`build.gradle[.kts]`) → `gradle test`. Kept
    /// distinct from [`Language::Java`] because the build tool, not the language,
    /// determines the verify command — running `mvn` in a Gradle-only repo fails.
    Gradle,
    Ruby,
    /// A ChimeraLang project (a `chimera.toml` marker or any `*.chimera` source
    /// at the root). Verified by AXIOM's own in-tree ChimeraLang implementation
    /// ([`crate::chimera`]) rather than an external toolchain.
    Chimera,
}

/// Detect the dominant project language from well-known marker files. Returns
/// `None` when nothing recognizable is present (the caller then relies purely on
/// the trace, which is already language-agnostic).
pub fn detect_language(root: &Path) -> Option<Language> {
    let has = |name: &str| root.join(name).exists();
    if has("Cargo.toml") {
        return Some(Language::Rust);
    }
    if has("go.mod") {
        return Some(Language::Go);
    }
    if has("pyproject.toml") || has("setup.py") || has("setup.cfg") || has("requirements.txt") {
        return Some(Language::Python);
    }
    if has("tsconfig.json") {
        return Some(Language::TypeScript);
    }
    if has("package.json") {
        return Some(Language::JavaScript);
    }
    // Maven before Gradle, but each maps to its own build tool's verify command.
    if has("pom.xml") {
        return Some(Language::Java);
    }
    if has("build.gradle") || has("build.gradle.kts") {
        return Some(Language::Gradle);
    }
    if has("Gemfile") {
        return Some(Language::Ruby);
    }
    if has("chimera.toml") || has_chimera_source(root) {
        return Some(Language::Chimera);
    }
    None
}

/// True if the project root directly contains a `*.chimera` source file.
fn has_chimera_source(root: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(root) else {
        return false;
    };
    entries.flatten().any(|e| {
        e.path()
            .extension()
            .and_then(|x| x.to_str())
            .map(|x| x.eq_ignore_ascii_case("chimera"))
            .unwrap_or(false)
    })
}

/// A sensible default verify command for a language — the thing CI would run.
/// Provided so a future `axiom solve` can self-select a check when the user
/// gives none; the repair mechanism itself is unchanged by the choice.
pub fn default_verify(language: Language) -> (String, Vec<String>) {
    let (prog, args): (&str, &[&str]) = match language {
        Language::Rust => ("cargo", &["test", "--quiet"]),
        Language::Python => ("pytest", &["-q"]),
        Language::JavaScript | Language::TypeScript => ("npm", &["test", "--silent"]),
        Language::Go => ("go", &["test", "./..."]),
        Language::Java => ("mvn", &["-q", "test"]),
        Language::Gradle => ("gradle", &["test", "--quiet"]),
        Language::Ruby => ("rake", &["test"]),
        // Verified in-process by AXIOM's ChimeraLang implementation; the CLI
        // subcommand `axiom chimera check` is the command-line equivalent.
        Language::Chimera => ("axiom", &["chimera", "check"]),
    };
    (
        prog.to_string(),
        args.iter().map(|s| s.to_string()).collect(),
    )
}

fn has_source_ext(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    match lower.rsplit_once('.') {
        Some((_, ext)) => SOURCE_EXTS.contains(&ext),
        None => false,
    }
}

/// Parse a Python-style traceback frame: `File "path/x.py", line 42, in fn`.
fn parse_python_frame(line: &str) -> Option<(PathBuf, u32)> {
    let l = line.trim_start();
    let rest = l.strip_prefix("File \"")?;
    let end = rest.find('"')?;
    let path = &rest[..end];
    if path.is_empty() {
        return None;
    }
    let after = &rest[end..];
    let idx = after.find("line ")?;
    let digits: String = after[idx + 5..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let line_no = digits.parse::<u32>().ok()?;
    Some((PathBuf::from(path), line_no))
}

/// Parse one whitespace-delimited token for an inline `path.ext:line[:col]` or
/// `path.ext(line,col)` reference (Rust/Go/C/C++/JS/TS stack frames, etc.).
fn parse_inline_token(tok: &str) -> Option<(PathBuf, u32)> {
    // Strip wrapping brackets/quotes and trailing sentence punctuation, but keep
    // interior ':' and '/' which carry the path/line structure.
    let t = tok.trim_matches(|c| {
        matches!(
            c,
            '(' | ')' | '[' | ']' | '{' | '}' | '\'' | '"' | ',' | ';' | '`'
        )
    });
    if t.is_empty() {
        return None;
    }

    // Paren form: file.ext(line,col)
    if let Some(open) = t.find('(') {
        let path_part = &t[..open];
        if has_source_ext(path_part) {
            let digits: String = t[open + 1..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(line) = digits.parse::<u32>() {
                return Some((PathBuf::from(path_part), line));
            }
        }
    }

    // Colon form: path.ext:line[:col][: message]
    let mut parts = t.split(':');
    let path_part = parts.next()?;
    if has_source_ext(path_part) {
        let num_tok = parts.next()?;
        let digits: String = num_tok.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(line) = digits.parse::<u32>() {
            return Some((PathBuf::from(path_part), line));
        }
    }
    None
}

/// Extract every source reference from a tool's failure output, deduplicated by
/// path and ranked most-referenced first (ties keep first-seen order — the
/// primary error line usually comes first).
pub fn locate_sites(trace: &str) -> Vec<FaultSite> {
    let mut order: Vec<PathBuf> = Vec::new();
    let mut info: HashMap<PathBuf, (u32, u32)> = HashMap::new(); // path -> (first line, hits)

    let mut record = |path: PathBuf, line: u32| {
        let entry = info.entry(path.clone()).or_insert_with(|| {
            order.push(path.clone());
            (line, 0)
        });
        entry.1 = entry.1.saturating_add(1);
    };

    for raw in trace.lines() {
        if let Some((p, l)) = parse_python_frame(raw) {
            record(p, l);
            continue; // a Python frame carries exactly one site
        }
        for tok in raw.split_whitespace() {
            if let Some((p, l)) = parse_inline_token(tok) {
                record(p, l);
            }
        }
    }

    let mut sites: Vec<FaultSite> = order
        .into_iter()
        .map(|path| {
            let (line, hits) = info[&path];
            FaultSite { path, line, hits }
        })
        .collect();
    // Stable sort by hits desc keeps first-seen order for equal counts.
    sites.sort_by_key(|s| std::cmp::Reverse(s.hits));
    sites
}

/// Resolve a trace-reported path to a real file that lives **under** `root`.
///
/// The returned path is fed straight into the reversible repair writes in
/// [`crate::solve`], so it must never escape the project being repaired. A
/// verifier frequently prints absolute frames for the stdlib, a dependency, or a
/// virtualenv (e.g. a Python traceback through site-packages, or a Rust panic in
/// `~/.cargo/registry/...`); accepting one because it merely exists would let
/// auto-localization rewrite a file outside the project — and could shadow the
/// real app source if the dependency frame is seen first. We therefore
/// canonicalize both sides and require strict containment, which also defeats
/// `..` traversal in relative frames.
fn resolve_under_root(path: &Path, root: &Path) -> Option<PathBuf> {
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    if !candidate.is_file() {
        return None;
    }
    let root_canon = root.canonicalize().ok()?;
    let cand_canon = candidate.canonicalize().ok()?;
    if cand_canon.starts_with(&root_canon) {
        Some(cand_canon)
    } else {
        None
    }
}

/// The single best source file to repair, parsed from `trace` and confirmed to
/// exist under `root`. Returns the highest-ranked reference that resolves to a
/// real file, or `None` if the trace names no existing source. This is the entry
/// point the autonomous solver uses when the caller did not name a file.
pub fn best_source(trace: &str, root: &Path) -> Option<PathBuf> {
    candidate_sources(trace, root).into_iter().next()
}

/// Every trace-referenced source file that resolves to a real file under `root`,
/// ranked most-blamed first and deduplicated (by canonical path). The autonomous
/// solver tries these in order so a failure that points at several files can
/// still be repaired; [`best_source`] is simply the first of these.
pub fn candidate_sources(trace: &str, root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for site in locate_sites(trace) {
        if let Some(p) = resolve_under_root(&site.path, root) {
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// Constrain a (possibly caller-supplied) path to live under `root`, returning a
/// canonicalized path or `None` if it would escape. Unlike [`best_source`], the
/// path need not exist yet: if it doesn't, the **parent** must exist and resolve
/// under `root` (so a brand-new file can be created in-tree, but `..`/absolute
/// escapes are still rejected). This is the safety check for user-named edit
/// targets (e.g. `axiom task --file`).
pub fn contained_path(path: &Path, root: &Path) -> Option<PathBuf> {
    let root_canon = root.canonicalize().ok()?;
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    if let Ok(canon) = joined.canonicalize() {
        // Exists: require canonical containment.
        return canon.starts_with(&root_canon).then_some(canon);
    }
    // Doesn't exist yet: validate via the canonical parent + the final segment.
    let parent = joined.parent()?;
    let file_name = joined.file_name()?;
    let parent_canon = parent.canonicalize().ok()?;
    parent_canon
        .starts_with(&root_canon)
        .then(|| parent_canon.join(file_name))
}

/// Line numbers the trace attributes to `target`, ranked by reference frequency
/// and deduplicated. Intended as an **advisory hint** for a repair prompt — it
/// matches by file name (not a canonical path), so an approximate match is fine:
/// the worst case is pointing the model at a slightly wrong line, never an
/// unsafe write. Returns an empty vec when the trace says nothing about `target`.
pub fn line_hints_for(trace: &str, target: &Path) -> Vec<u32> {
    let name = match target.file_name() {
        Some(n) => n,
        None => return Vec::new(),
    };
    // Parse directly (not via locate_sites, which keeps only one line per path)
    // so every distinct line for the target is captured, ranked by how often the
    // trace cites it (stable: first-seen order breaks ties).
    let mut order: Vec<u32> = Vec::new();
    let mut freq: HashMap<u32, u32> = HashMap::new();
    let mut consider = |p: &Path, line: u32| {
        if p.file_name() == Some(name) {
            let e = freq.entry(line).or_insert_with(|| {
                order.push(line);
                0
            });
            *e = e.saturating_add(1);
        }
    };
    for raw in trace.lines() {
        if let Some((p, l)) = parse_python_frame(raw) {
            consider(&p, l);
            continue;
        }
        for tok in raw.split_whitespace() {
            if let Some((p, l)) = parse_inline_token(tok) {
                consider(&p, l);
            }
        }
    }
    order.sort_by(|a, b| freq[b].cmp(&freq[a]));
    order
}

/// Does this path look like a test or spec file — i.e. the verifier's oracle
/// rather than the code under test? Matches the common conventions across the
/// languages the localizer knows: `test_*.py`, `*_test.go`, `*.test.js`,
/// `*_spec.rb`, `FooTest.java`, a bare `test.<ext>`, or any file under a
/// `test/`, `tests/`, `spec/`, `__tests__/` directory.
///
/// The autonomous solver uses this to **demote** such files: when a failure
/// blames only the test, the repair should land in the module it imports, and a
/// test file is edited only as a last resort. Heuristic by design — a wrong
/// guess reorders candidates, it never widens what may be written.
pub fn is_test_path(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let stem = match name.rsplit_once('.') {
        Some((s, _)) => s.to_string(),
        None => name.clone(),
    };
    let by_name = name.starts_with("test_")
        || name.starts_with("test-")
        || stem == "test"
        || stem == "tests"
        || stem.ends_with("_test")
        || stem.ends_with("-test")
        || stem.ends_with(".test")
        || stem.ends_with("_spec")
        || stem.ends_with(".spec")
        || stem.starts_with("spec_")
        || (stem.ends_with("test") && stem.len() > 4 && name.ends_with(".java"))
        || (stem.ends_with("tests") && stem.len() > 5 && name.ends_with(".java"));
    if by_name {
        return true;
    }
    path.parent()
        .map(|parent| {
            parent.components().any(|c| {
                let c = c.as_os_str().to_string_lossy().to_ascii_lowercase();
                matches!(
                    c.as_str(),
                    "test" | "tests" | "spec" | "specs" | "__tests__" | "testing"
                )
            })
        })
        .unwrap_or(false)
}

/// Source files `file` pulls in — the modules a test exercises — resolved to
/// real files under `root`. Understands the import forms of the languages the
/// localizer knows (Python `import`/`from`, JS `require`/`import`, Ruby
/// `require_relative`, shell `.`/`source`, C `#include "x.h"` (plus the `x.c`
/// sibling), Rust `mod x;`) and, for Go `_test.go` files, every non-test `.go`
/// file in the same package directory. Test-looking targets and `file` itself
/// are excluded; the result is deduplicated in first-seen order.
pub fn import_targets(file: &Path, root: &Path) -> Vec<PathBuf> {
    let Ok(text) = std::fs::read_to_string(file) else {
        return Vec::new();
    };
    let dir = file.parent().map(Path::to_path_buf).unwrap_or_default();
    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    let self_canon = file.canonicalize().ok();

    // Relative specs (without extension, slashes already normalised) and the
    // extensions to try for each.
    let mut specs: Vec<(String, &[&str])> = Vec::new();
    let mut push = |spec: &str, exts: &'static [&'static str]| {
        let spec = spec
            .trim()
            .trim_matches(|c| c == '\'' || c == '"' || c == ';');
        if !spec.is_empty() && !spec.contains("..") {
            specs.push((spec.to_string(), exts));
        }
    };

    match ext.as_str() {
        "py" => {
            for raw in text.lines() {
                let l = raw.trim();
                if let Some(rest) = l.strip_prefix("from ") {
                    if let Some((module, _)) = rest.split_once(" import ") {
                        let module = module.trim().trim_start_matches('.');
                        push(&module.replace('.', "/"), &["py", "/__init__.py"]);
                    }
                } else if let Some(rest) = l.strip_prefix("import ") {
                    for module in rest.split(',') {
                        let module = module.split(" as ").next().unwrap_or("").trim();
                        push(&module.replace('.', "/"), &["py", "/__init__.py"]);
                    }
                }
            }
        }
        "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" => {
            for raw in text.lines() {
                let l = raw.trim();
                let spec = if let Some(i) = l.find("require(") {
                    quoted_after(&l[i + "require(".len()..])
                } else if l.starts_with("import ") {
                    l.find(" from ")
                        .and_then(|i| quoted_after(&l[i + " from ".len()..]))
                        .or_else(|| quoted_after(&l["import ".len()..]))
                } else {
                    None
                };
                if let Some(spec) = spec {
                    if spec.starts_with('.') {
                        let spec = spec.trim_start_matches("./");
                        push(
                            spec,
                            &[
                                "",
                                "js",
                                "mjs",
                                "cjs",
                                "ts",
                                "tsx",
                                "jsx",
                                "/index.js",
                                "/index.ts",
                            ],
                        );
                    }
                }
            }
        }
        "rb" => {
            for raw in text.lines() {
                let l = raw.trim();
                for kw in ["require_relative ", "require "] {
                    if let Some(rest) = l.strip_prefix(kw) {
                        if let Some(spec) = quoted_after(rest) {
                            let spec = spec.trim_start_matches("./");
                            push(spec, &["", "rb"]);
                        }
                    }
                }
            }
        }
        "sh" | "bash" | "zsh" | "ksh" => {
            for raw in text.lines() {
                let l = raw.trim();
                let rest = l.strip_prefix(". ").or_else(|| l.strip_prefix("source "));
                if let Some(rest) = rest {
                    let spec = rest.split_whitespace().next().unwrap_or("");
                    let spec = spec
                        .trim_start_matches("./")
                        .trim_start_matches("\"$(dirname \"$0\")\"/");
                    push(spec, &[""]);
                }
            }
        }
        "c" | "h" | "cc" | "cpp" | "cxx" | "hpp" | "hh" => {
            for raw in text.lines() {
                let l = raw.trim();
                if let Some(rest) = l.strip_prefix("#include") {
                    if let Some(spec) = quoted_after(rest) {
                        let base = spec.rsplit_once('.').map(|(b, _)| b).unwrap_or(spec);
                        // The implementation first — that is where behaviour lives.
                        push(base, &["c", "cc", "cpp", "cxx"]);
                        push(spec, &[""]);
                    }
                }
            }
        }
        "rs" => {
            for raw in text.lines() {
                let l = raw.trim();
                if let Some(rest) = l.strip_prefix("mod ") {
                    if let Some(name) = rest.strip_suffix(';') {
                        push(name.trim(), &["rs", "/mod.rs"]);
                    }
                }
            }
        }
        "go" => {
            // Same-package siblings: a `_test.go` exercises the other files in
            // its directory, no import statement names them.
            let mut out = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&dir) {
                let mut names: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
                names.sort();
                for p in names {
                    let is_go = p.extension().and_then(|e| e.to_str()) == Some("go");
                    if is_go && !is_test_path(&p) {
                        if let Some(r) = resolve_under_root(&p, root) {
                            if Some(&r) != self_canon.as_ref() && !out.contains(&r) {
                                out.push(r);
                            }
                        }
                    }
                }
            }
            return out;
        }
        _ => {}
    }

    let mut out: Vec<PathBuf> = Vec::new();
    for (spec, exts) in specs {
        for ext in exts {
            let rel = if ext.is_empty() {
                spec.clone()
            } else if ext.starts_with('/') {
                format!("{spec}{ext}")
            } else {
                format!("{spec}.{ext}")
            };
            for base in [&dir, root] {
                if let Some(r) = resolve_under_root(Path::new(&rel), base) {
                    if r.starts_with(root.canonicalize().unwrap_or_else(|_| root.to_path_buf()))
                        && Some(&r) != self_canon.as_ref()
                        && !is_test_path(&r)
                        && !out.contains(&r)
                    {
                        out.push(r);
                    }
                }
            }
        }
    }
    out
}

/// First single- or double-quoted string in `s`.
fn quoted_after(s: &str) -> Option<&str> {
    let s = s.trim_start();
    let q = s.chars().next()?;
    if q != '\'' && q != '"' {
        return None;
    }
    let rest = &s[1..];
    let end = rest.find(q)?;
    Some(&rest[..end])
}

/// The files the autonomous solver should try to repair, in order: every
/// non-test file the trace blames (most-blamed first), then the modules each
/// blamed test file imports (the code under test, which an assertion failure
/// in the test never names directly), and only then the test files themselves.
/// This is [`candidate_sources`] plus the test-awareness that keeps the solver
/// from "fixing" a failing test by editing the test.
pub fn repair_targets(trace: &str, root: &Path) -> Vec<PathBuf> {
    let direct = candidate_sources(trace, root);
    let mut out: Vec<PathBuf> = Vec::new();
    for p in &direct {
        if !is_test_path(p) && !out.contains(p) {
            out.push(p.clone());
        }
    }
    for p in &direct {
        if is_test_path(p) {
            for hop in import_targets(p, root) {
                if !out.contains(&hop) {
                    out.push(hop);
                }
            }
        }
    }
    for p in &direct {
        if is_test_path(p) && !out.contains(p) {
            out.push(p.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rust_compiler_and_panic_frames() {
        let trace = "\
error[E0308]: mismatched types
  --> src/widget.rs:42:18
   |
thread 'main' panicked at src/widget.rs:50:9:
assertion failed";
        let sites = locate_sites(trace);
        assert_eq!(sites[0].path, PathBuf::from("src/widget.rs"));
        assert_eq!(sites[0].hits, 2, "two references to the same file");
        assert_eq!(sites[0].line, 42, "first-seen line retained");
    }

    #[test]
    fn parses_python_traceback() {
        let trace = "\
Traceback (most recent call last):
  File \"/app/pkg/runner.py\", line 17, in <module>
    main()
  File \"/app/pkg/core.py\", line 88, in main
ValueError: boom";
        let sites = locate_sites(trace);
        let paths: Vec<_> = sites.iter().map(|s| s.path.clone()).collect();
        assert!(paths.contains(&PathBuf::from("/app/pkg/runner.py")));
        assert!(paths.contains(&PathBuf::from("/app/pkg/core.py")));
        assert_eq!(
            sites
                .iter()
                .find(|s| s.path == Path::new("/app/pkg/core.py"))
                .unwrap()
                .line,
            88
        );
    }

    #[test]
    fn parses_js_stack_and_ts_diagnostic() {
        let js = "    at Object.<anonymous> (/srv/app/index.js:23:11)";
        let ts = "src/server.ts(120,5): error TS2322: Type mismatch";
        let go = "./internal/db/conn.go:31:2: undefined: Foo";
        assert_eq!(locate_sites(js)[0].path, PathBuf::from("/srv/app/index.js"));
        assert_eq!(locate_sites(js)[0].line, 23);
        assert_eq!(locate_sites(ts)[0].path, PathBuf::from("src/server.ts"));
        assert_eq!(locate_sites(ts)[0].line, 120);
        assert_eq!(
            locate_sites(go)[0].path,
            PathBuf::from("./internal/db/conn.go")
        );
        assert_eq!(locate_sites(go)[0].line, 31);
    }

    #[test]
    fn ranks_most_referenced_file_first() {
        let trace = "\
a.rs:1:1: note
b.py:2:1: error
b.py:9:1: error
b.py:14:1: error";
        let sites = locate_sites(trace);
        assert_eq!(sites[0].path, PathBuf::from("b.py"), "3 hits outrank 1");
        assert_eq!(sites[0].hits, 3);
    }

    #[test]
    fn ignores_non_source_and_prose() {
        let trace = "Compiling thing v0.1.0\nFinished in 3:14 minutes\nsee README.md:1 for details";
        // README.md is not a recognized source extension; 3:14 is not a path.
        assert!(locate_sites(trace).is_empty());
    }

    #[test]
    fn best_source_returns_only_existing_files() {
        let dir = std::env::temp_dir().join(format!("axiom_floc_{}", std::process::id()));
        let sub = dir.join("src");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("real.rs"), "fn main() {}").unwrap();

        // The missing file is referenced more, but only the existing one resolves.
        let trace = "\
src/ghost.rs:3:1: error
src/ghost.rs:4:1: error
src/real.rs:9:2: error";
        let found = best_source(trace, &dir);
        assert_eq!(found, Some(sub.join("real.rs").canonicalize().unwrap()));

        // A trace naming no existing file localizes nothing.
        assert_eq!(best_source("src/ghost.rs:1:1: error", &dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn best_source_rejects_existing_files_outside_the_root() {
        // A real file that exists but lives OUTSIDE the project root (e.g. a
        // dependency / stdlib / virtualenv frame) must never be localized — the
        // repair loop writes to the returned path.
        let base = std::env::temp_dir().join(format!("axiom_floc_esc_{}", std::process::id()));
        let project = base.join("project");
        let outside = base.join("vendor");
        std::fs::create_dir_all(project.join("src")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(project.join("src/app.rs"), "fn main() {}").unwrap();
        let dep = outside.join("dep.rs");
        std::fs::write(&dep, "// dependency").unwrap();

        // Absolute dependency frame seen FIRST (more hits) must be skipped in
        // favor of the in-root app source.
        let trace = format!(
            "thread panicked at {dep}:10:5\n{dep}:11:1: note\nsrc/app.rs:3:2: error",
            dep = dep.display()
        );
        let found = best_source(&trace, &project);
        assert_eq!(
            found,
            Some(project.join("src/app.rs").canonicalize().unwrap())
        );

        // An absolute out-of-root frame on its own localizes nothing.
        let only_dep = format!("{}:10:5: boom", dep.display());
        assert_eq!(best_source(&only_dep, &project), None);

        // A `..` traversal that resolves outside the root is rejected too.
        assert_eq!(best_source("../vendor/dep.rs:1:1: x", &project), None);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn contained_path_allows_in_root_and_rejects_escapes() {
        let dir = std::env::temp_dir().join(format!("axiom_floc_contain_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/lib.rs"), "fn main() {}").unwrap();

        // Existing in-root file → canonical path returned.
        let got = contained_path(Path::new("src/lib.rs"), &dir).unwrap();
        assert_eq!(got, dir.join("src/lib.rs").canonicalize().unwrap());
        // New file whose parent exists in-root → allowed (for creation).
        assert!(contained_path(Path::new("src/new.rs"), &dir).is_some());
        // Absolute outside + `..` escape → rejected.
        assert_eq!(contained_path(Path::new("/etc/passwd"), &dir), None);
        assert_eq!(contained_path(Path::new("../escape.rs"), &dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn candidate_sources_lists_all_in_root_files_ranked_and_deduped() {
        let dir = std::env::temp_dir().join(format!("axiom_floc_cands_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/a.rs"), "fn a() {}").unwrap();
        std::fs::write(dir.join("src/b.rs"), "fn b() {}").unwrap();

        // b.rs is referenced more (ranks first); a.rs once; ghost.rs doesn't exist.
        let trace = "\
src/ghost.rs:1:1: error
src/b.rs:2:1: error
src/b.rs:9:1: error
src/a.rs:3:1: error";
        let cands = candidate_sources(trace, &dir);
        let b = dir.join("src/b.rs").canonicalize().unwrap();
        let a = dir.join("src/a.rs").canonicalize().unwrap();
        assert_eq!(cands, vec![b.clone(), a], "ranked, in-root only, deduped");
        // best_source is the top candidate.
        assert_eq!(best_source(trace, &dir), Some(b));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn detect_language_reads_marker_files() {
        let dir = std::env::temp_dir().join(format!("axiom_floc_lang_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("go.mod"), "module x").unwrap();
        assert_eq!(detect_language(&dir), Some(Language::Go));
        let (prog, args) = default_verify(Language::Go);
        assert_eq!(prog, "go");
        assert_eq!(args, vec!["test", "./..."]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn detect_language_recognizes_chimera_projects() {
        // A bare *.chimera source at the root is enough to detect the language.
        let dir = std::env::temp_dir().join(format!("axiom_floc_chimera_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("main.chimera"), "emit \"hi\"").unwrap();
        assert_eq!(detect_language(&dir), Some(Language::Chimera));
        let (prog, args) = default_verify(Language::Chimera);
        assert_eq!(prog, "axiom");
        assert_eq!(args, vec!["chimera", "check"]);
        let _ = std::fs::remove_dir_all(&dir);

        // The explicit chimera.toml marker also works (no source file present).
        let dir2 = std::env::temp_dir().join(format!("axiom_floc_chimtoml_{}", std::process::id()));
        std::fs::create_dir_all(&dir2).unwrap();
        std::fs::write(dir2.join("chimera.toml"), "name = \"x\"").unwrap();
        assert_eq!(detect_language(&dir2), Some(Language::Chimera));
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[test]
    fn line_hints_rank_and_filter_to_the_target_file() {
        let trace = "\
src/widget.rs:42:18: error
src/widget.rs:42:18: error
src/widget.rs:50:9: error
src/unrelated.rs:3:1: error";
        // Matches by file name; the same line dedups; ranked by frequency.
        let hints = line_hints_for(trace, Path::new("/abs/proj/src/widget.rs"));
        assert_eq!(
            hints,
            vec![42, 50],
            "line 42 (2 hits) ranks before line 50 (1)"
        );
        // A file the trace never mentions yields no hint.
        assert!(line_hints_for(trace, Path::new("src/ghost.rs")).is_empty());
    }

    #[test]
    fn gradle_and_maven_select_their_own_build_tool() {
        // A Gradle-only JVM project must verify with gradle, not mvn.
        let gdir = std::env::temp_dir().join(format!("axiom_floc_gradle_{}", std::process::id()));
        std::fs::create_dir_all(&gdir).unwrap();
        std::fs::write(gdir.join("build.gradle.kts"), "plugins {}").unwrap();
        assert_eq!(detect_language(&gdir), Some(Language::Gradle));
        assert_eq!(default_verify(Language::Gradle).0, "gradle");
        let _ = std::fs::remove_dir_all(&gdir);

        // A Maven project still maps to mvn.
        let mdir = std::env::temp_dir().join(format!("axiom_floc_maven_{}", std::process::id()));
        std::fs::create_dir_all(&mdir).unwrap();
        std::fs::write(mdir.join("pom.xml"), "<project/>").unwrap();
        assert_eq!(detect_language(&mdir), Some(Language::Java));
        assert_eq!(default_verify(Language::Java).0, "mvn");
        let _ = std::fs::remove_dir_all(&mdir);
    }

    #[test]
    fn test_paths_are_recognised_by_convention() {
        for p in [
            "test_calc.py",
            "calc_test.go",
            "sum.test.js",
            "calc_spec.rb",
            "CalcTest.java",
            "test.sh",
            "tests/unit/x.py",
            "src/__tests__/a.ts",
        ] {
            assert!(is_test_path(Path::new(p)), "{p} should be a test path");
        }
        for p in [
            "calc.py",
            "calc.go",
            "sum.js",
            "lib.sh",
            "src/util/gate.sh",
            "Gate.java",
            "contest.py",
        ] {
            assert!(!is_test_path(Path::new(p)), "{p} should not be a test path");
        }
    }

    #[test]
    fn import_targets_follow_the_languages_import_forms() {
        let dir = std::env::temp_dir().join(format!("axiom_floc_imports_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("pkg")).unwrap();
        let w = |rel: &str, body: &str| std::fs::write(dir.join(rel), body).unwrap();
        w("calc.py", "def f(): pass\n");
        w("pkg/__init__.py", "");
        w(
            "test_calc.py",
            "import os\nfrom calc import f\nimport pkg\n",
        );
        w("sum.js", "module.exports = 1;\n");
        w(
            "sum.test.js",
            "const assert = require('assert');\nconst sum = require('./sum');\n",
        );
        w("calc.rb", "def f; end\n");
        w("test.rb", "require_relative 'calc'\n");
        w("lib.sh", "f() { :; }\n");
        w("test.sh", "#!/bin/sh\n. ./lib.sh\n");
        w("calc.h", "int f(void);\n");
        w("calc.c", "int f(void) { return 1; }\n");
        w("test.c", "#include <assert.h>\n#include \"calc.h\"\n");
        w("calc.go", "package calc\n");
        w("calc_test.go", "package calc\n");
        w("other_test.go", "package calc\n");

        let c = |rel: &str| dir.join(rel).canonicalize().unwrap();
        assert_eq!(
            import_targets(&dir.join("test_calc.py"), &dir),
            vec![c("calc.py"), c("pkg/__init__.py")]
        );
        assert_eq!(
            import_targets(&dir.join("sum.test.js"), &dir),
            vec![c("sum.js")]
        );
        assert_eq!(
            import_targets(&dir.join("test.rb"), &dir),
            vec![c("calc.rb")]
        );
        assert_eq!(
            import_targets(&dir.join("test.sh"), &dir),
            vec![c("lib.sh")]
        );
        assert_eq!(
            import_targets(&dir.join("test.c"), &dir),
            vec![c("calc.c"), c("calc.h")]
        );
        assert_eq!(
            import_targets(&dir.join("calc_test.go"), &dir),
            vec![c("calc.go")]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repair_targets_put_the_module_under_test_before_the_test() {
        let dir = std::env::temp_dir().join(format!("axiom_floc_targets_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("calc.py"), "def f(): pass\n").unwrap();
        std::fs::write(
            dir.join("test_calc.py"),
            "from calc import f\nassert f() == 1\n",
        )
        .unwrap();
        let c = |rel: &str| dir.join(rel).canonicalize().unwrap();

        // An assertion failure names only the test; the module is reached by hop.
        let trace = "  File \"test_calc.py\", line 2, in <module>\nAssertionError";
        assert_eq!(
            repair_targets(trace, &dir),
            vec![c("calc.py"), c("test_calc.py")]
        );

        // A directly blamed non-test file outranks everything.
        let trace = "  File \"test_calc.py\", line 2, in <module>\n  File \"calc.py\", line 1, in f\nIndexError";
        assert_eq!(
            repair_targets(trace, &dir),
            vec![c("calc.py"), c("test_calc.py")]
        );

        // Without test-awareness the plain ranking would have led with the test.
        assert_eq!(
            candidate_sources("  File \"test_calc.py\", line 2, in <module>", &dir)[0],
            c("test_calc.py")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
