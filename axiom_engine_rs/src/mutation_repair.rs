//! Deterministic mutation repair — Phase 2.1 of the autonomous solve loop.
//!
//! The legacy Poly-JIT patch set ([`crate::poly_jit::synthesize_patch`]) knows a
//! handful of fixture-shaped string flips (`exit 1` → `exit 0`, a marker
//! rename, one hard-coded `assert_eq!`). This module is its general,
//! language-aware successor. Given the failing verifier's output and the file
//! blamed by [`crate::fault_locate`], it enumerates a **bounded, ranked set of
//! single-edit source mutations** of the kinds that account for a large share of
//! real one-line bugs — comparison-operator and off-by-one slips, inverted
//! booleans, `and`/`or` mix-ups, arithmetic-operator slips, swapped call
//! arguments, missing `return`s, shell `test` operators — and hands them to the
//! verifier-gated, reversible apply loop in [`crate::solve`]. A candidate is kept
//! only if the real verifier goes green; everything else is rolled back
//! byte-for-byte. No model, no network: the intelligence is the verifier, and
//! this module's job is to spend its runs where the evidence points.
//!
//! Three rules keep it honest:
//!
//! 1. **Focus.** Lines the failure trace blames are mutated first; the rest of
//!    the file follows, class-major, so the budget goes where the evidence is.
//!    The trace also reorders the mutation classes: an `IndexError` tries
//!    off-by-one fixes first, an `expected 6 got 5` likewise, a `NoneType` /
//!    `undefined` result tries a missing `return` first.
//! 2. **Never edit the oracle.** A line that *asserts* (`assert`, `expect`,
//!    `t.Fatal`, `raise`, `throw`, `exit 1`, …) is never mutated — flipping an
//!    assertion to make a test pass is the textbook overfit, not a repair.
//! 3. **Bounded.** At most [`DEFAULT_MAX_CANDIDATES`] candidates
//!    (`AXIOM_REPAIR_MAX_CANDIDATES`), each costing exactly one verifier run.
//!
//! Everything here is pure string manipulation; [`plan`] is the entry point and
//! the generators are unit-tested directly.

use std::collections::HashSet;
use std::path::Path;

/// Default cap on candidates per file (one verifier run each).
pub const DEFAULT_MAX_CANDIDATES: usize = 64;

/// Largest integer literal we shift by ±1 (bigger numbers are ports, hashes,
/// timestamps — not loop bounds).
const MAX_SHIFTED_LITERAL: u64 = 10_000;

/// The mutation families, in default priority order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MutationClass {
    /// `<` ↔ `<=`, `>` ↔ `>=`, `==` ↔ `!=`, `<` ↔ `>` (and `=` → `==` in a condition).
    Comparison,
    /// Integer literal ±1, drop/flip a trailing `± 1`, `len(x)` → `len(x) - 1`,
    /// `range(n)` → `range(n + 1)`, Rust `..` ↔ `..=`.
    OffByOne,
    /// `true` ↔ `false`, drop a `not`/`!`, negate a returned expression.
    Boolean,
    /// `and` ↔ `or`, `&&` ↔ `||`.
    Logic,
    /// `+` ↔ `-`, `*` ↔ `/`, `+` → `*`, `*` → `+`, `+=` ↔ `-=`, `*=` ↔ `/=`.
    Arithmetic,
    /// Swap two adjacent arguments of a call.
    SwapArgs,
    /// Prepend `return` to a bare expression statement; drop the trailing `;`
    /// of a Rust block's last statement.
    MissingReturn,
    /// POSIX `test` operators: `-lt`/`-le`/`-gt`/`-ge`/`-eq`/`-ne`, `-z`/`-n`,
    /// `-a`/`-o`, `=`/`!=`.
    ShellTest,
    /// A misspelled import target (`from calcualtor import add`,
    /// `require('./calcualtor')`, `require_relative 'calcualtor'`,
    /// `#include "calcualtor.h"`, `mod calcualtor;`) rewritten to the sibling
    /// source file whose name is closest. Needs the trace's "module not found"
    /// name and the directory listing, so it is planned per file, not per line.
    ImportPath,
}

impl MutationClass {
    pub const ALL: [MutationClass; 9] = [
        MutationClass::Comparison,
        MutationClass::OffByOne,
        MutationClass::Boolean,
        MutationClass::Logic,
        MutationClass::Arithmetic,
        MutationClass::SwapArgs,
        MutationClass::MissingReturn,
        MutationClass::ShellTest,
        MutationClass::ImportPath,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MutationClass::Comparison => "comparison",
            MutationClass::OffByOne => "off-by-one",
            MutationClass::Boolean => "boolean",
            MutationClass::Logic => "logic",
            MutationClass::Arithmetic => "arithmetic",
            MutationClass::SwapArgs => "swap-args",
            MutationClass::MissingReturn => "missing-return",
            MutationClass::ShellTest => "shell-test",
            MutationClass::ImportPath => "import-path",
        }
    }
}

/// Coarse source language, from the file extension. Only used to pick comment
/// and string syntax and to enable language-specific operators; a wrong guess
/// costs wasted verifier runs, never an unsafe write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Python,
    JavaScript,
    Shell,
    Go,
    Rust,
    C,
    Ruby,
    Java,
    Other,
}

impl Lang {
    pub fn from_path(path: &Path) -> Lang {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        match ext.as_str() {
            "py" | "pyi" => Lang::Python,
            "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" => Lang::JavaScript,
            "sh" | "bash" | "zsh" | "ksh" => Lang::Shell,
            "go" => Lang::Go,
            "rs" => Lang::Rust,
            "c" | "h" | "cc" | "cpp" | "cxx" | "hpp" | "hh" => Lang::C,
            "rb" => Lang::Ruby,
            "java" | "kt" | "kts" | "scala" => Lang::Java,
            _ => Lang::Other,
        }
    }

    fn hash_comments(self) -> bool {
        matches!(self, Lang::Python | Lang::Shell | Lang::Ruby)
    }

    fn slash_comments(self) -> bool {
        matches!(
            self,
            Lang::JavaScript | Lang::Go | Lang::Rust | Lang::C | Lang::Java
        )
    }

    fn word_logic(self) -> bool {
        matches!(self, Lang::Python | Lang::Ruby)
    }

    fn c_like(self) -> bool {
        matches!(
            self,
            Lang::JavaScript | Lang::Go | Lang::Rust | Lang::C | Lang::Java
        )
    }
}

/// One candidate repair: the full mutated file plus where/what changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// Human-readable `class:Lnn what -> what`.
    pub label: String,
    pub class: MutationClass,
    /// 1-based line that was edited.
    pub line: usize,
    /// Complete mutated source.
    pub source: String,
}

/// Candidate cap from `AXIOM_REPAIR_MAX_CANDIDATES`, else [`DEFAULT_MAX_CANDIDATES`].
pub fn max_candidates() -> usize {
    std::env::var("AXIOM_REPAIR_MAX_CANDIDATES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_CANDIDATES)
}

/// True if the line *is* (part of) the verifier's oracle — an assertion, a
/// fatal/raise/throw, or an explicit failure exit. Such lines are never mutated:
/// a repair that edits the check instead of the code is an overfit, and the
/// deterministic layer must not be able to produce one.
pub fn is_oracle_line(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    const MARKERS: &[&str] = &[
        "assert",
        "expect(",
        "t.error",
        "t.fatal",
        "t.fail",
        "fatalf(",
        "raise ",
        "throw ",
        "panic!",
        "panic(",
        "exit 1",
        "exit(1)",
        "sys.exit",
        "process.exit",
        "os.exit",
        ".should",
        "fail(",
        "unreachable!",
        "must_equal",
        "verify(",
    ];
    MARKERS.iter().any(|m| l.contains(m))
}

/// Order the mutation classes by what the failure trace suggests. Pure
/// heuristics over the verifier's text; the default order is used when nothing
/// matches. Returned vector always contains every class exactly once.
pub fn class_order(trace: &str) -> Vec<MutationClass> {
    let t = trace.to_ascii_lowercase();
    let has = |needles: &[&str]| needles.iter().any(|n| t.contains(n));

    let mut score: Vec<(i32, usize, MutationClass)> = MutationClass::ALL
        .iter()
        .enumerate()
        .map(|(i, c)| (0i32, i, *c))
        .collect();
    let mut bump = |class: MutationClass, delta: i32| {
        if let Some(s) = score.iter_mut().find(|(_, _, c)| *c == class) {
            s.0 += delta;
        }
    };

    if has(&[
        "index out of range",
        "indexerror",
        "out of bounds",
        "outofbounds",
        "out of range",
        "slice bounds",
        "the len is",
    ]) {
        bump(MutationClass::OffByOne, -30);
        bump(MutationClass::Comparison, -20);
    }
    if has(&["remove this semicolon", "consider removing this semicolon"]) {
        bump(MutationClass::MissingReturn, -40);
    }
    if has(&[
        "nonetype",
        "undefined",
        "<nil>",
        "nil:nilclass",
        "missing return",
        "cannot read propert",
        "expected `i32`, found `()`",
        "expected `i64`, found `()`",
        "expected `usize`, found `()`",
        "found `()`",
        "evaluated but not used",
        "statement with no effect",
        "returned none",
        "is none",
        "is not none",
    ]) {
        bump(MutationClass::MissingReturn, -25);
    }
    if has(&[
        "expected true",
        "expected false",
        "got true",
        "got false",
        "left: true",
        "left: false",
        "right: true",
        "right: false",
        "expected: true",
        "expected: false",
        "actual: true",
        "actual: false",
        "assertion failed: !",
        "== true",
        "== false",
        "to be truthy",
        "to be falsy",
    ]) {
        bump(MutationClass::Boolean, -25);
        bump(MutationClass::Logic, -20);
    }
    if has(&[
        "-lt",
        "-gt",
        "-le",
        "-ge",
        "-eq",
        "-ne",
        "unary operator expected",
        "integer expression expected",
    ]) {
        bump(MutationClass::ShellTest, -30);
    }
    if !missing_module_names(trace).is_empty() {
        bump(MutationClass::ImportPath, -60);
    }
    match numeric_mismatch(&t) {
        Some(1) => bump(MutationClass::OffByOne, -35),
        Some(_) => {
            bump(MutationClass::Arithmetic, -15);
            bump(MutationClass::OffByOne, -5);
        }
        None => {}
    }

    score.sort_by_key(|(s, i, _)| (*s, *i));
    score.into_iter().map(|(_, _, c)| c).collect()
}

/// If the trace reports a pair of mismatched integers (`expected 6 got 5`,
/// `left: 5 / right: 6`, `5 !== 6`, `AssertionError: 5 != 6`, …), return their
/// absolute difference. Looks only at lines that read like an assertion result.
fn numeric_mismatch(lower_trace: &str) -> Option<u64> {
    let mut nums: Vec<u64> = Vec::new();
    for line in lower_trace.lines() {
        let relevant = [
            "expect", "actual", "left", "right", "assert", "got", "!==", "!=", "want", "but was",
            "should",
        ]
        .iter()
        .any(|k| line.contains(k));
        if !relevant {
            continue;
        }
        // Skip the part of a line that is a path:line:col reference.
        for tok in line.split(|c: char| !c.is_ascii_digit()) {
            if tok.is_empty() || tok.len() > 9 {
                continue;
            }
            if let Ok(n) = tok.parse::<u64>() {
                nums.push(n);
            }
        }
    }
    // Any two distinct numbers in the assertion lines: smallest gap wins.
    let mut best: Option<u64> = None;
    for i in 0..nums.len() {
        for j in (i + 1)..nums.len() {
            if nums[i] != nums[j] {
                let d = nums[i].abs_diff(nums[j]);
                best = Some(best.map_or(d, |b| b.min(d)));
            }
        }
    }
    best
}

/// Build the ranked, bounded candidate list for `source` (the contents of
/// `path`) given the verifier's failure `trace`. Blamed lines first (in hint
/// order, every class), then the rest of the file class-major; oracle lines and
/// comment-only lines are never edited; identical results are deduplicated.
pub fn plan(source: &str, path: &Path, trace: &str, max: usize) -> Vec<Candidate> {
    let lang = Lang::from_path(path);
    let order = class_order(trace);
    let class_rank = |c: MutationClass| order.iter().position(|x| *x == c).unwrap_or(usize::MAX);

    let lines: Vec<&str> = source.split_inclusive('\n').collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let n = lines.len();

    // Blamed lines (1-based → 0-based), in the order the trace ranks them.
    let mut hinted: Vec<usize> = Vec::new();
    for l in crate::fault_locate::line_hints_for(trace, path) {
        let idx = l as usize;
        if idx >= 1 && idx <= n && !hinted.contains(&(idx - 1)) {
            hinted.push(idx - 1);
        }
    }
    // Remaining lines: by distance to the nearest hint if any, else file order.
    let mut rest: Vec<usize> = (0..n).filter(|i| !hinted.contains(i)).collect();
    if !hinted.is_empty() {
        rest.sort_by_key(|i| {
            hinted
                .iter()
                .map(|h| h.abs_diff(*i))
                .min()
                .unwrap_or(usize::MAX)
        });
    }

    // (tier, a, b, seq) sort key — see the module docs.
    let mut keyed: Vec<((u8, usize, usize, usize), Candidate)> = Vec::new();
    let mut seq = 0usize;
    let visit = |li: usize, pos: usize, tier: u8, keyed: &mut Vec<_>, seq: &mut usize| {
        let (content, eol) = split_eol(lines[li]);
        if content.trim().is_empty() || is_comment_only(content, lang) || is_oracle_line(content) {
            return;
        }
        let next = lines[li + 1..]
            .iter()
            .map(|l| split_eol(l).0)
            .find(|l| !l.trim().is_empty());
        for class in &order {
            for (what, new_line) in line_mutations(content, lang, *class, next) {
                if new_line == content {
                    continue;
                }
                let mut src = String::with_capacity(source.len() + 16);
                for l in &lines[..li] {
                    src.push_str(l);
                }
                src.push_str(&new_line);
                src.push_str(eol);
                for l in &lines[li + 1..] {
                    src.push_str(l);
                }
                let cr = class_rank(*class);
                let key = if tier == 0 {
                    (0, pos, cr, *seq)
                } else {
                    (1, cr, pos, *seq)
                };
                *seq += 1;
                keyed.push((
                    key,
                    Candidate {
                        label: format!("{}:L{} {}", class.name(), li + 1, what),
                        class: *class,
                        line: li + 1,
                        source: src,
                    },
                ));
            }
        }
    };
    for (pos, li) in hinted.iter().enumerate() {
        visit(*li, pos, 0, &mut keyed, &mut seq);
    }
    for (pos, li) in rest.iter().enumerate() {
        visit(*li, pos, 1, &mut keyed, &mut seq);
    }

    // Import-path repairs are planned per file (they need the trace's missing
    // name and the directory listing); rank them as tier 0 when the trace
    // names a missing module, since nothing else can fix that failure.
    for (li, what, new_line) in import_path_candidates(source, path, trace) {
        let (content, eol) = split_eol(lines[li]);
        if new_line == content {
            continue;
        }
        let mut src = String::with_capacity(source.len() + 16);
        for l in &lines[..li] {
            src.push_str(l);
        }
        src.push_str(&new_line);
        src.push_str(eol);
        for l in &lines[li + 1..] {
            src.push_str(l);
        }
        let cr = class_rank(MutationClass::ImportPath);
        keyed.push((
            (0, 0, cr, seq),
            Candidate {
                label: format!("{}:L{} {}", MutationClass::ImportPath.name(), li + 1, what),
                class: MutationClass::ImportPath,
                line: li + 1,
                source: src,
            },
        ));
        seq += 1;
    }

    keyed.sort_by_key(|(k, _)| *k);
    let mut seen: HashSet<&str> = HashSet::new();
    let mut out = Vec::new();
    for (_, c) in &keyed {
        if c.source == source || !seen.insert(c.source.as_str()) {
            continue;
        }
        out.push(c.clone());
        if out.len() >= max.max(1) {
            break;
        }
    }
    out
}

fn split_eol(l: &str) -> (&str, &str) {
    if let Some(s) = l.strip_suffix("\r\n") {
        (s, "\r\n")
    } else if let Some(s) = l.strip_suffix('\n') {
        (s, "\n")
    } else {
        (l, "")
    }
}

fn is_comment_only(line: &str, lang: Lang) -> bool {
    let t = line.trim_start();
    (lang.hash_comments() && t.starts_with('#'))
        || (lang.slash_comments()
            && (t.starts_with("//") || t.starts_with("/*") || t.starts_with('*')))
        || (lang == Lang::Other && (t.starts_with('#') || t.starts_with("//")))
}

/// Every mutation of `class` applicable to one line, as `(what, new_line)`.
pub fn line_mutations(
    line: &str,
    lang: Lang,
    class: MutationClass,
    next_line: Option<&str>,
) -> Vec<(String, String)> {
    let mask = code_mask(line, lang);
    match class {
        MutationClass::Comparison => comparison(line, lang, &mask),
        MutationClass::OffByOne => off_by_one(line, lang, &mask),
        MutationClass::Boolean => boolean(line, lang, &mask),
        MutationClass::Logic => logic(line, lang, &mask),
        MutationClass::Arithmetic => arithmetic(line, lang, &mask),
        MutationClass::SwapArgs => swap_args(line, &mask),
        MutationClass::MissingReturn => missing_return(line, lang, &mask, next_line),
        MutationClass::ShellTest => shell_test(line, lang, &mask),
        // Planned per file in `plan` (needs the trace and the directory).
        MutationClass::ImportPath => Vec::new(),
    }
}

/// Module / file names the trace says could not be resolved, across the
/// toolchains the localizer knows. Dotted Python paths keep their last segment
/// (`pkg.calcualtor` → `calcualtor`); JS/Ruby relative specs lose their `./`.
pub fn missing_module_names(trace: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |raw: &str| {
        // Last path segment, minus a source extension, then the last dotted
        // segment (Python packages): `./pkg/calcualtor.py` → `calcualtor`.
        let name = raw
            .trim()
            .trim_matches(|c| c == '\'' || c == '"' || c == '`' || c == ';' || c == ',' || c == '.')
            .trim_start_matches("./")
            .rsplit('/')
            .next()
            .unwrap_or("");
        let name = match name.rsplit_once('.') {
            Some((stem, ext)) if ["py", "js", "ts", "rb", "h", "rs", "go", "sh"].contains(&ext) => {
                stem
            }
            _ => name,
        };
        let name = name.rsplit('.').next().unwrap_or("").to_string();
        if !name.is_empty() && !out.contains(&name) {
            out.push(name);
        }
    };
    for line in trace.lines() {
        // Python: No module named 'calcualtor' / No module named 'pkg.calcualtor'
        if let Some(i) = line.find("No module named ") {
            push(&line[i + "No module named ".len()..]);
        }
        // Node: Cannot find module './calcualtor'
        if let Some(i) = line.find("Cannot find module ") {
            push(&line[i + "Cannot find module ".len()..]);
        }
        // Ruby: cannot load such file -- /abs/dir/calcualtor
        if let Some(i) = line.find("cannot load such file -- ") {
            let rest = &line[i + "cannot load such file -- ".len()..];
            push(rest.split_whitespace().next().unwrap_or(""));
        }
        // C: fatal error: calcualtor.h: No such file or directory
        if line.contains("No such file or directory") {
            if let Some(i) = line.find("fatal error: ") {
                let rest = &line[i + "fatal error: ".len()..];
                if let Some(end) = rest.find(':') {
                    push(&rest[..end]);
                }
            }
        }
        // Rust: file not found for module `calcualtor`
        if let Some(i) = line.find("file not found for module ") {
            push(&line[i + "file not found for module ".len()..]);
        }
        // Go: package calc/calcualtor is not in std
        if let Some(i) = line.find("package ") {
            if line.contains(" is not in ") {
                let rest = &line[i + "package ".len()..];
                push(rest.split_whitespace().next().unwrap_or(""));
            }
        }
    }
    out
}

/// Sibling source files (stems) next to `path` that could be what a
/// misspelled import meant: same extension family, Levenshtein distance at
/// most 3 from `missing` (or one a prefix of the other), never the file itself.
fn import_replacements(path: &Path, missing: &str) -> Vec<String> {
    let Some(dir) = path.parent() else {
        return Vec::new();
    };
    let own = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let lang = Lang::from_path(path);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut scored: Vec<(usize, String)> = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        if !p.is_file() {
            continue;
        }
        let same_family = match lang {
            Lang::C => matches!(
                p.extension().and_then(|x| x.to_str()),
                Some("h" | "hpp" | "hh" | "c" | "cc" | "cpp")
            ),
            _ => Lang::from_path(&p) == lang,
        };
        if !same_family {
            continue;
        }
        let stem = p
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if stem.is_empty() || stem == own || stem == missing {
            continue;
        }
        let d = levenshtein(&stem.to_ascii_lowercase(), &missing.to_ascii_lowercase());
        let related = d <= 3 || stem.starts_with(missing) || missing.starts_with(&stem);
        if related && !scored.iter().any(|(_, s)| *s == stem) {
            scored.push((d, stem));
        }
    }
    scored.sort();
    scored.into_iter().map(|(_, s)| s).collect()
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Import-path candidates for a whole file: every line that names a module the
/// trace could not resolve, rewritten to each plausible sibling.
fn import_path_candidates(source: &str, path: &Path, trace: &str) -> Vec<(usize, String, String)> {
    let mut out = Vec::new();
    for missing in missing_module_names(trace) {
        let replacements = import_replacements(path, &missing);
        if replacements.is_empty() {
            continue;
        }
        for (li, raw) in source.split_inclusive('\n').enumerate() {
            let (content, _) = split_eol(raw);
            let mask = code_mask(content, Lang::from_path(path));
            // Imports name the module inside quotes (JS/Ruby/C) or as a bare
            // word (Python/Rust/Go): accept either, but only whole words.
            let hits: Vec<(usize, usize)> = word_spans_any(content, &mask, &missing);
            if hits.is_empty() {
                continue;
            }
            for repl in &replacements {
                let mut new_line = String::with_capacity(content.len() + 8);
                let mut last = 0;
                for (s, e) in &hits {
                    new_line.push_str(&content[last..*s]);
                    new_line.push_str(repl);
                    last = *e;
                }
                new_line.push_str(&content[last..]);
                out.push((li, format!("import `{missing}` -> `{repl}`"), new_line));
            }
        }
    }
    out
}

/// Like [`word_spans`] but also inside string literals (an import spec is a
/// string in several languages), still requiring word boundaries.
fn word_spans_any(line: &str, _mask: &[bool], word: &str) -> Vec<(usize, usize)> {
    let b = line.as_bytes();
    let w = word.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + w.len() <= b.len() {
        if &b[i..i + w.len()] == w
            && (i == 0 || !is_ident(b[i - 1]))
            && (i + w.len() == b.len() || !is_ident(b[i + w.len()]))
        {
            out.push((i, i + w.len()));
            i += w.len();
        } else {
            i += 1;
        }
    }
    out
}

// --- lexical helpers ---------------------------------------------------------

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Per-byte "is code" mask: false inside string/char literals and from a
/// line-comment start to the end of the line. ASCII-only decisions, so every
/// `true` byte is a char boundary we can slice at.
fn code_mask(line: &str, lang: Lang) -> Vec<bool> {
    let b = line.as_bytes();
    let n = b.len();
    let mut mask = vec![true; n];
    let mut i = 0;
    while i < n {
        let c = b[i];
        // Line comments.
        let comment = if lang.hash_comments() || lang == Lang::Other {
            c == b'#' && !(i > 0 && (b[i - 1] == b'$' || b[i - 1] == b'{'))
        } else {
            false
        } || if lang.slash_comments() || lang == Lang::Other {
            c == b'/' && i + 1 < n && b[i + 1] == b'/'
        } else {
            false
        };
        if comment {
            for m in mask.iter_mut().skip(i) {
                *m = false;
            }
            break;
        }
        // String / char literals.
        if c == b'"' || c == b'\'' || c == b'`' {
            if c == b'`' && !matches!(lang, Lang::JavaScript | Lang::Go) {
                i += 1;
                continue;
            }
            if c == b'\'' && lang == Lang::Rust {
                // Char literal only if it closes within a few bytes; otherwise
                // it is a lifetime (`'a`) and not a string at all.
                if let Some(end) = rust_char_end(b, i) {
                    for m in mask.iter_mut().take(end + 1).skip(i) {
                        *m = false;
                    }
                    i = end + 1;
                } else {
                    i += 1;
                }
                continue;
            }
            let escapes = c != b'`' && !(c == b'\'' && lang == Lang::Shell);
            let mut j = i + 1;
            while j < n {
                if escapes && b[j] == b'\\' {
                    j += 2;
                    continue;
                }
                if b[j] == c {
                    break;
                }
                j += 1;
            }
            let end = j.min(n - 1);
            for m in mask.iter_mut().take(end + 1).skip(i) {
                *m = false;
            }
            i = j + 1;
            continue;
        }
        i += 1;
    }
    mask
}

fn rust_char_end(b: &[u8], start: usize) -> Option<usize> {
    // 'x' or '\n' or '\u{..}' (bounded scan).
    let n = b.len();
    let mut j = start + 1;
    if j >= n {
        return None;
    }
    if b[j] == b'\\' {
        j += 2;
        while j < n && j < start + 12 && b[j] != b'\'' {
            j += 1;
        }
        return (j < n && b[j] == b'\'').then_some(j);
    }
    // One UTF-8 scalar then a closing quote.
    let mut k = j + 1;
    while k < n && (b[k] & 0xC0) == 0x80 {
        k += 1;
    }
    (k < n && b[k] == b'\'').then_some(k)
}

/// Operator tokens, longest first so `<=` never scans as `<` + `=`.
const OPS: &[&str] = &[
    "===", "!==", "<<=", ">>=", "..=", "<=", ">=", "==", "!=", "&&", "||", "<<", ">>", "->", "=>",
    "+=", "-=", "*=", "/=", "**", "++", "--", "//", "..", "<", ">", "+", "-", "*", "/", "%", "=",
    "!",
];

/// `(start, end, op)` byte spans of operator tokens in code regions.
fn op_spans(line: &str, mask: &[bool]) -> Vec<(usize, usize, &'static str)> {
    let b = line.as_bytes();
    let n = b.len();
    let mut out = Vec::new();
    let mut i = 0;
    'outer: while i < n {
        if mask[i] {
            for op in OPS {
                let ob = op.as_bytes();
                let e = i + ob.len();
                if e <= n && &b[i..e] == ob && mask[i..e].iter().all(|m| *m) {
                    out.push((i, e, *op));
                    i = e;
                    continue 'outer;
                }
            }
        }
        i += 1;
    }
    out
}

fn replace_span(line: &str, start: usize, end: usize, with: &str) -> String {
    let mut s = String::with_capacity(line.len() + with.len());
    s.push_str(&line[..start]);
    s.push_str(with);
    s.push_str(&line[end..]);
    s
}

/// Index of the previous non-space byte before `i`, if any.
fn prev_non_space(b: &[u8], i: usize) -> Option<usize> {
    let mut k = i;
    while k > 0 {
        k -= 1;
        if b[k] != b' ' && b[k] != b'\t' {
            return Some(k);
        }
    }
    None
}

/// Index of the next non-space byte at or after `i`, if any.
fn next_non_space(b: &[u8], i: usize) -> Option<usize> {
    let mut k = i;
    while k < b.len() {
        if b[k] != b' ' && b[k] != b'\t' {
            return Some(k);
        }
        k += 1;
    }
    None
}

/// Does `b[i]` begin a binary operator (an operand precedes it)?
fn is_binary_position(b: &[u8], i: usize) -> bool {
    match prev_non_space(b, i) {
        Some(p) => is_ident(b[p]) || matches!(b[p], b')' | b']' | b'"' | b'\'' | b'`'),
        None => false,
    }
}

/// Word-boundary occurrences of `word` in code regions.
fn word_spans(line: &str, mask: &[bool], word: &str) -> Vec<(usize, usize)> {
    let b = line.as_bytes();
    let w = word.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + w.len() <= b.len() {
        if &b[i..i + w.len()] == w
            && mask[i..i + w.len()].iter().all(|m| *m)
            && (i == 0 || !is_ident(b[i - 1]))
            && (i + w.len() == b.len() || !is_ident(b[i + w.len()]))
        {
            out.push((i, i + w.len()));
            i += w.len();
        } else {
            i += 1;
        }
    }
    out
}

/// Matching close paren for the `(` at `open`, skipping non-code bytes.
fn matching_paren(b: &[u8], mask: &[bool], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for k in open..b.len() {
        if !mask[k] {
            continue;
        }
        match b[k] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(k);
                }
            }
            _ => {}
        }
    }
    None
}

// --- generators ----------------------------------------------------------------

fn comparison(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let in_condition = {
        let t = line.trim_start();
        t.starts_with("if")
            || t.starts_with("while")
            || t.starts_with("elif")
            || t.starts_with("} else if")
    };
    for (s, e, op) in op_spans(line, mask) {
        let alts: &[&str] = match op {
            "<" => &["<=", ">"],
            "<=" => &["<", ">="],
            ">" => &[">=", "<"],
            ">=" => &[">", "<="],
            "==" => &["!="],
            "!=" => &["=="],
            "===" => &["!=="],
            "!==" => &["==="],
            // `if (x = 5)` — the classic C family slip. Only inside a condition.
            "=" if lang.c_like() && in_condition => &["=="],
            _ => continue,
        };
        if matches!(op, "<" | ">" | "<=" | ">=") {
            if lang == Lang::Shell {
                continue; // redirections, not comparisons
            }
            if lang.c_like() && op.len() == 1 {
                // Generics (`Vec<i32>`, `List<String>`) hug at least one
                // operand; in formatted C-family code a comparison has a space
                // on both sides. Missing an unspaced `i<n` costs a repair, a
                // false hit on a generic would only waste a verifier run.
                let before_ws = s > 0 && (b[s - 1] == b' ' || b[s - 1] == b'\t');
                let after_ws = e < b.len() && (b[e] == b' ' || b[e] == b'\t');
                if !(before_ws && after_ws) {
                    continue;
                }
            }
        }
        for a in alts {
            out.push((format!("{op} -> {a}"), replace_span(line, s, e, a)));
        }
    }
    out
}

fn off_by_one(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    let b = line.as_bytes();
    let n = b.len();
    let mut out = Vec::new();

    // Drop or flip a trailing `± 1` after an operand: `len(xs) - 1` → `len(xs)`.
    for (s, e, op) in op_spans(line, mask) {
        if !(op == "+" || op == "-") || !is_binary_position(b, s) {
            continue;
        }
        let Some(one) = next_non_space(b, e) else {
            continue;
        };
        if b[one] != b'1' || (one + 1 < n && (is_ident(b[one + 1]) || b[one + 1] == b'.')) {
            continue;
        }
        let Some(p) = prev_non_space(b, s) else {
            continue;
        };
        let mut dropped = String::new();
        dropped.push_str(&line[..p + 1]);
        dropped.push_str(&line[one + 1..]);
        out.push((format!("drop `{op} 1`"), dropped));
        let flip = if op == "+" { "-" } else { "+" };
        out.push((
            format!("{op} 1 -> {flip} 1"),
            replace_span(line, s, e, flip),
        ));
    }

    // `len(xs)` / `.len()` / `.length` / `.size` / `.count` → `… - 1`.
    for idiom in ["len(", ".len(", ".size(", ".count("] {
        let mut from = 0;
        while let Some(pos) = line[from..].find(idiom) {
            let s = from + pos;
            from = s + idiom.len();
            let open = s + idiom.len() - 1;
            if !mask[s] || (idiom == "len(" && s > 0 && (is_ident(b[s - 1]) || b[s - 1] == b'.')) {
                continue;
            }
            let Some(close) = matching_paren(b, mask, open) else {
                continue;
            };
            push_minus_one(line, b, close + 1, &mut out);
        }
    }
    for prop in ["length", "size", "count", "Length", "Count"] {
        for (s, e) in word_spans(line, mask, prop) {
            if s == 0 || b[s - 1] != b'.' || !mask[s - 1] {
                continue;
            }
            if e < n && b[e] == b'(' {
                continue; // a method call form, handled above
            }
            push_minus_one(line, b, e, &mut out);
        }
    }

    // Python `range(n)` → `range(n + 1)`.
    if lang == Lang::Python {
        for (_, e) in word_spans(line, mask, "range") {
            if e >= n || b[e] != b'(' {
                continue;
            }
            let Some(close) = matching_paren(b, mask, e) else {
                continue;
            };
            let inner = &line[e + 1..close];
            let top_level_comma = inner_has_top_level_comma(&b[e + 1..close], &mask[e + 1..close]);
            if inner.trim().is_empty() || top_level_comma || inner.trim_end().ends_with("- 1") {
                continue;
            }
            let mut s2 = String::new();
            s2.push_str(&line[..close]);
            s2.push_str(" + 1");
            s2.push_str(&line[close..]);
            out.push(("range(n) -> range(n + 1)".to_string(), s2));
        }
    }

    // Rust ranges: `..` ↔ `..=`.
    if lang == Lang::Rust {
        for (s, e, op) in op_spans(line, mask) {
            match op {
                ".." => out.push((".. -> ..=".to_string(), replace_span(line, s, e, "..="))),
                "..=" => out.push(("..= -> ..".to_string(), replace_span(line, s, e, ".."))),
                _ => {}
            }
        }
    }

    // Integer literals ±1 (small ones only).
    let mut i = 0;
    while i < n {
        if mask[i] && b[i].is_ascii_digit() && (i == 0 || !(is_ident(b[i - 1]) || b[i - 1] == b'.'))
        {
            let mut j = i;
            while j < n && b[j].is_ascii_digit() {
                j += 1;
            }
            if j < n && (is_ident(b[j]) || b[j] == b'.') {
                i = j + 1; // float, hex, identifier tail — leave alone
                continue;
            }
            if let Ok(v) = line[i..j].parse::<u64>() {
                if v <= MAX_SHIFTED_LITERAL {
                    if v >= 1 {
                        out.push((
                            format!("{v} -> {}", v - 1),
                            replace_span(line, i, j, &(v - 1).to_string()),
                        ));
                    }
                    out.push((
                        format!("{v} -> {}", v + 1),
                        replace_span(line, i, j, &(v + 1).to_string()),
                    ));
                }
            }
            i = j;
            continue;
        }
        i += 1;
    }
    out
}

/// Append `line` with ` - 1` inserted at byte `at`, unless an adjustment already
/// follows there.
fn push_minus_one(line: &str, b: &[u8], at: usize, out: &mut Vec<(String, String)>) {
    if let Some(k) = next_non_space(b, at) {
        if b[k] == b'-' || b[k] == b'+' {
            return;
        }
    }
    let mut s = String::with_capacity(line.len() + 4);
    s.push_str(&line[..at]);
    s.push_str(" - 1");
    s.push_str(&line[at..]);
    out.push(("insert `- 1` after length".to_string(), s));
}

fn inner_has_top_level_comma(b: &[u8], mask: &[bool]) -> bool {
    let mut depth = 0i32;
    for (k, c) in b.iter().enumerate() {
        if !mask[k] {
            continue;
        }
        match c {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

fn boolean(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    for (from, to) in [
        ("True", "False"),
        ("False", "True"),
        ("true", "false"),
        ("false", "true"),
        ("TRUE", "FALSE"),
        ("FALSE", "TRUE"),
    ] {
        for (s, e) in word_spans(line, mask, from) {
            out.push((format!("{from} -> {to}"), replace_span(line, s, e, to)));
        }
    }
    if lang.word_logic() {
        for (s, e) in word_spans(line, mask, "not") {
            if e < b.len() && b[e] == b' ' {
                out.push(("drop `not`".to_string(), replace_span(line, s, e + 1, "")));
            }
        }
    }
    for (s, e, op) in op_spans(line, mask) {
        if op != "!" || e >= b.len() {
            continue;
        }
        if !(is_ident(b[e]) || b[e] == b'(') {
            continue;
        }
        if lang == Lang::Rust && s > 0 && is_ident(b[s - 1]) {
            continue; // macro invocation `name!(`
        }
        out.push(("drop `!`".to_string(), replace_span(line, s, e, "")));
    }
    // Negate a returned expression.
    let indent_len = line.len() - line.trim_start().len();
    let t = &line[indent_len..];
    if let Some(expr) = t.strip_prefix("return ") {
        let expr = expr.trim_end();
        if !expr.is_empty() && lang != Lang::Shell {
            let (body, semi) = match expr.strip_suffix(';') {
                Some(x) => (x.trim_end(), ";"),
                None => (expr, ""),
            };
            if !body.is_empty() && !body.starts_with("not ") && !body.starts_with('!') {
                let neg = if lang.word_logic() {
                    format!("return not ({body}){semi}")
                } else {
                    format!("return !({body}){semi}")
                };
                out.push((
                    "negate return".to_string(),
                    format!("{}{}", &line[..indent_len], neg),
                ));
            }
        }
    }
    out
}

fn logic(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (s, e, op) in op_spans(line, mask) {
        match op {
            "&&" => out.push(("&& -> ||".to_string(), replace_span(line, s, e, "||"))),
            "||" => out.push(("|| -> &&".to_string(), replace_span(line, s, e, "&&"))),
            _ => {}
        }
    }
    if lang.word_logic() {
        for (s, e) in word_spans(line, mask, "and") {
            out.push(("and -> or".to_string(), replace_span(line, s, e, "or")));
        }
        for (s, e) in word_spans(line, mask, "or") {
            out.push(("or -> and".to_string(), replace_span(line, s, e, "and")));
        }
    }
    out
}

fn arithmetic(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    if lang == Lang::Shell && !line.contains("$((") {
        return out; // `-` is an option flag in shell, not a minus
    }
    for (s, e, op) in op_spans(line, mask) {
        let alts: &[&str] = match op {
            "+" | "-" | "*" | "/" => {
                if !is_binary_position(b, s) {
                    continue; // unary minus / deref / pointer
                }
                match op {
                    "+" => &["-", "*"],
                    "-" => &["+"],
                    "*" => &["/", "+"],
                    _ => &["*"],
                }
            }
            "+=" => &["-="],
            "-=" => &["+="],
            "*=" => &["/="],
            "/=" => &["*="],
            _ => continue,
        };
        for a in alts {
            out.push((format!("{op} -> {a}"), replace_span(line, s, e, a)));
        }
    }
    out
}

fn swap_args(line: &str, mask: &[bool]) -> Vec<(String, String)> {
    let b = line.as_bytes();
    let n = b.len();
    let mut out = Vec::new();
    for open in 0..n {
        if b[open] != b'(' || !mask[open] || open == 0 || !is_ident(b[open - 1]) {
            continue;
        }
        let Some(close) = matching_paren(b, mask, open) else {
            continue;
        };
        // Top-level argument spans.
        let mut args: Vec<(usize, usize)> = Vec::new();
        let mut depth = 0i32;
        let mut start = open + 1;
        for k in open + 1..close {
            if !mask[k] {
                continue;
            }
            match b[k] {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth -= 1,
                b',' if depth == 0 => {
                    args.push((start, k));
                    start = k + 1;
                }
                _ => {}
            }
        }
        args.push((start, close));
        if !(2..=3).contains(&args.len())
            || args.iter().any(|(s, e)| line[*s..*e].trim().is_empty())
        {
            continue;
        }
        for w in 0..args.len() - 1 {
            let (a_s, a_e) = args[w];
            let (b_s, b_e) = args[w + 1];
            let a = line[a_s..a_e].trim();
            let bb = line[b_s..b_e].trim();
            if a == bb {
                continue;
            }
            let mut s = String::with_capacity(line.len());
            s.push_str(&line[..a_s]);
            s.push(' ');
            s.push_str(bb);
            s.push_str(", ");
            s.push_str(a);
            s.push_str(&line[b_e..]);
            // Normalise the leading space we may have introduced after `(`.
            let s = s.replace("( ", "(");
            out.push((format!("swap args `{a}` <-> `{bb}`"), s));
        }
    }
    out
}

const STATEMENT_KEYWORDS: &[&str] = &[
    "return",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "def",
    "class",
    "import",
    "from",
    "print",
    "assert",
    "raise",
    "pass",
    "break",
    "continue",
    "yield",
    "with",
    "try",
    "except",
    "finally",
    "del",
    "global",
    "nonlocal",
    "lambda",
    "async",
    "await",
    "var",
    "let",
    "const",
    "function",
    "throw",
    "console",
    "require",
    "module",
    "export",
    "switch",
    "case",
    "default",
    "do",
    "new",
    "delete",
    "typeof",
    "puts",
    "end",
    "unless",
    "require_relative",
    "package",
    "func",
    "type",
    "struct",
    "go",
    "defer",
    "select",
    "fmt",
    "log",
    "fn",
    "pub",
    "use",
    "mod",
    "impl",
    "enum",
    "match",
    "loop",
    "static",
    "int",
    "void",
    "char",
    "float",
    "double",
    "long",
    "unsigned",
    "public",
    "private",
    "protected",
    "final",
    "System",
    "printf",
    "fprintf",
    "free",
    "exit",
    "echo",
    "local",
    "sleep",
    "include",
    "describe",
    "it",
    "test",
    "expect",
    "self",
    "super",
    "this",
    "goto",
    "signed",
    "short",
    "bool",
    "size_t",
    "string",
    "auto",
    "extern",
    "typedef",
    "namespace",
    "using",
    "template",
    "catch",
    "interface",
    "abstract",
    "declare",
    "then",
    "begin",
    "rescue",
    "ensure",
    "attr_reader",
    "attr_accessor",
    "module_function",
    "elsif",
    "when",
    "until",
    "next",
    "redo",
    "retry",
    "alias",
    "undef",
    "defined?",
    "in",
    "not",
    "and",
    "or",
    "is",
    "none",
    "true",
    "false",
    "nil",
    "null",
    "True",
    "False",
    "None",
];

fn missing_return(
    line: &str,
    lang: Lang,
    mask: &[bool],
    next_line: Option<&str>,
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let indent_len = line.len() - line.trim_start().len();
    let t = line[indent_len..].trim_end();
    if t.is_empty() {
        return out;
    }

    if lang == Lang::Rust {
        // A block's last statement ends in `;` → it evaluates to `()`.
        let b = line.as_bytes();
        if let Some(semi) = line.rfind(';') {
            let tail = line[semi + 1..].trim();
            let closes_here = tail == "}" || tail.starts_with("} ");
            let next_closes = next_line
                .map(|l| l.trim_start().starts_with('}'))
                .unwrap_or(false);
            let first = t
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .next()
                .unwrap_or("");
            if mask[semi]
                && (tail.is_empty() || closes_here)
                && (closes_here || next_closes)
                && !matches!(first, "let" | "use" | "return" | "break" | "continue")
                && !t.starts_with('}')
                && b[..semi].iter().any(|c| !c.is_ascii_whitespace())
            {
                out.push((
                    "drop trailing `;`".to_string(),
                    replace_span(line, semi, semi + 1, ""),
                ));
            }
        }
        return out;
    }
    if matches!(lang, Lang::Ruby | Lang::Shell) {
        return out; // implicit return / not applicable
    }

    let first_byte = t.as_bytes()[0];
    if !(is_ident(first_byte) || first_byte == b'(') || first_byte.is_ascii_digit() {
        return out;
    }
    let first_word = t
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .next()
        .unwrap_or("");
    if first_word.is_empty() || STATEMENT_KEYWORDS.contains(&first_word) {
        return out;
    }
    // Not an assignment, not a block opener, not a continuation.
    let spans = op_spans(t, &code_mask(t, lang));
    if spans.iter().any(|(_, _, op)| *op == "=") {
        return out;
    }
    let last = t.as_bytes()[t.len() - 1];
    if matches!(last, b'{' | b':' | b',' | b'(' | b'\\' | b'[') {
        return out;
    }
    if lang == Lang::Python && (t.ends_with(')') || t.ends_with(']')) && !t.contains('(') {
        return out;
    }
    out.push((
        "prepend `return`".to_string(),
        format!("{}return {}", &line[..indent_len], t),
    ));
    out
}

fn shell_test(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if lang != Lang::Shell || !(line.contains('[') || line.contains("test ")) {
        return out;
    }
    let table: &[(&str, &[&str])] = &[
        ("-lt", &["-le", "-gt"]),
        ("-le", &["-lt", "-ge"]),
        ("-gt", &["-ge", "-lt"]),
        ("-ge", &["-gt", "-le"]),
        ("-eq", &["-ne"]),
        ("-ne", &["-eq"]),
        ("-z", &["-n"]),
        ("-n", &["-z"]),
        ("-a", &["-o"]),
        ("-o", &["-a"]),
        ("-f", &["-e", "-d"]),
        ("-d", &["-e"]),
    ];
    let b = line.as_bytes();
    for (op, alts) in table {
        let ob = op.as_bytes();
        let mut i = 0;
        while i + ob.len() <= b.len() {
            let e = i + ob.len();
            let bounded =
                (i == 0 || b[i - 1] == b' ') && (e == b.len() || b[e] == b' ' || b[e] == b']');
            if &b[i..e] == ob && mask[i..e].iter().all(|m| *m) && bounded {
                for a in *alts {
                    out.push((format!("{op} -> {a}"), replace_span(line, i, e, a)));
                }
                i = e;
            } else {
                i += 1;
            }
        }
    }
    // String compare inside `[ ]`.
    for (s, e, op) in op_spans(line, mask) {
        let spaced = s > 0 && b[s - 1] == b' ' && e < b.len() && b[e] == b' ';
        match op {
            "=" if spaced => out.push(("= -> !=".to_string(), replace_span(line, s, e, "!="))),
            "!=" if spaced => out.push(("!= -> =".to_string(), replace_span(line, s, e, "="))),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn muts(line: &str, lang: Lang, class: MutationClass) -> Vec<String> {
        line_mutations(line, lang, class, None)
            .into_iter()
            .map(|(_, l)| l)
            .collect()
    }

    #[test]
    fn comparison_flips_respect_strings_and_comments() {
        let m = muts(
            "if x <= 10:  # keep <= here",
            Lang::Python,
            MutationClass::Comparison,
        );
        assert!(m.contains(&"if x < 10:  # keep <= here".to_string()));
        assert!(m.contains(&"if x >= 10:  # keep <= here".to_string()));
        assert!(
            m.iter().all(|l| l.ends_with("# keep <= here")),
            "comment untouched: {m:?}"
        );
        let m = muts(r#"print("a < b")"#, Lang::Python, MutationClass::Comparison);
        assert!(m.is_empty(), "operators inside strings are not code: {m:?}");
    }

    #[test]
    fn generics_are_not_mistaken_for_comparisons() {
        let m = muts(
            "let v: Vec<i32> = Vec::new();",
            Lang::Rust,
            MutationClass::Comparison,
        );
        assert!(m.is_empty(), "{m:?}");
        let m = muts("while i < n {", Lang::Rust, MutationClass::Comparison);
        assert!(m.contains(&"while i <= n {".to_string()));
    }

    #[test]
    fn shell_angle_brackets_are_redirections_not_comparisons() {
        let m = muts("echo boom >&2", Lang::Shell, MutationClass::Comparison);
        assert!(m.is_empty(), "{m:?}");
    }

    #[test]
    fn off_by_one_drops_and_inserts_adjustments() {
        let m = muts(
            "for i in range(len(xs) - 1):",
            Lang::Python,
            MutationClass::OffByOne,
        );
        assert!(m.contains(&"for i in range(len(xs)):".to_string()), "{m:?}");
        assert!(
            m.contains(&"for i in range(len(xs) + 1):".to_string()),
            "{m:?}"
        );
        let m = muts("return xs[len(xs)]", Lang::Python, MutationClass::OffByOne);
        assert!(m.contains(&"return xs[len(xs) - 1]".to_string()), "{m:?}");
        let m = muts(
            "return xs[xs.length]",
            Lang::JavaScript,
            MutationClass::OffByOne,
        );
        assert!(m.contains(&"return xs[xs.length - 1]".to_string()), "{m:?}");
        let m = muts(
            "    for i in 0..xs.len()-1 {",
            Lang::Rust,
            MutationClass::OffByOne,
        );
        assert!(
            m.contains(&"    for i in 0..xs.len() {".to_string()),
            "{m:?}"
        );
        assert!(
            m.contains(&"    for i in 0..=xs.len()-1 {".to_string()),
            "{m:?}"
        );
        let m = muts("x = v[1] + 10", Lang::Python, MutationClass::OffByOne);
        assert!(m.contains(&"x = v[0] + 10".to_string()));
        assert!(m.contains(&"x = v[2] + 10".to_string()));
        assert!(m.contains(&"x = v[1] + 11".to_string()));
        // Floats and huge numbers are left alone.
        let m = muts("t = 0.5 * 86400000", Lang::Python, MutationClass::OffByOne);
        assert!(m.is_empty(), "{m:?}");
    }

    #[test]
    fn boolean_and_logic_mutations() {
        let m = muts(
            "return not len(xs) == 0",
            Lang::Python,
            MutationClass::Boolean,
        );
        assert!(m.contains(&"return len(xs) == 0".to_string()), "{m:?}");
        let m = muts("    return !ok;", Lang::JavaScript, MutationClass::Boolean);
        assert!(m.contains(&"    return ok;".to_string()), "{m:?}");
        let m = muts("    return ok;", Lang::JavaScript, MutationClass::Boolean);
        assert!(m.contains(&"    return !(ok);".to_string()), "{m:?}");
        let m = muts("    assert_eq!(a, b);", Lang::Rust, MutationClass::Boolean);
        assert!(m.is_empty(), "macro bang is not a negation: {m:?}");
        let m = muts(
            "return x >= 0 or x <= 10",
            Lang::Python,
            MutationClass::Logic,
        );
        assert!(m.contains(&"return x >= 0 and x <= 10".to_string()));
        let m = muts("if (a && b) {", Lang::JavaScript, MutationClass::Logic);
        assert!(m.contains(&"if (a || b) {".to_string()));
        let m = muts("ignored = 'a or b'", Lang::Python, MutationClass::Logic);
        assert!(m.is_empty(), "{m:?}");
    }

    #[test]
    fn arithmetic_is_binary_only() {
        let m = muts("return w + h;", Lang::JavaScript, MutationClass::Arithmetic);
        assert!(m.contains(&"return w * h;".to_string()), "{m:?}");
        assert!(m.contains(&"return w - h;".to_string()), "{m:?}");
        let m = muts("x = -1", Lang::Python, MutationClass::Arithmetic);
        assert!(m.is_empty(), "unary minus untouched: {m:?}");
        let m = muts("let p = *q;", Lang::Rust, MutationClass::Arithmetic);
        assert!(m.is_empty(), "deref untouched: {m:?}");
        let m = muts("ls -la", Lang::Shell, MutationClass::Arithmetic);
        assert!(m.is_empty(), "shell flags untouched: {m:?}");
        let m = muts("total += price", Lang::Python, MutationClass::Arithmetic);
        assert!(m.contains(&"total -= price".to_string()));
    }

    #[test]
    fn swap_args_swaps_adjacent_top_level_arguments() {
        let m = muts(
            "    return fullname(last, first)",
            Lang::Python,
            MutationClass::SwapArgs,
        );
        assert_eq!(m, vec!["    return fullname(first, last)".to_string()]);
        let m = muts("f(g(a, b), c)", Lang::JavaScript, MutationClass::SwapArgs);
        assert!(m.contains(&"f(c, g(a, b))".to_string()), "{m:?}");
        assert!(m.contains(&"f(g(b, a), c)".to_string()), "{m:?}");
        let m = muts("f(a)", Lang::Python, MutationClass::SwapArgs);
        assert!(m.is_empty());
    }

    #[test]
    fn missing_return_prepends_only_to_bare_expressions() {
        let m = muts("    x * 2", Lang::Python, MutationClass::MissingReturn);
        assert_eq!(m, vec!["    return x * 2".to_string()]);
        let m = muts("  x * 2;", Lang::JavaScript, MutationClass::MissingReturn);
        assert_eq!(m, vec!["  return x * 2;".to_string()]);
        for stmt in [
            "    y = x * 2",
            "    print(x)",
            "def f(x):",
            "    if x:",
            "    return x",
        ] {
            let m = muts(stmt, Lang::Python, MutationClass::MissingReturn);
            assert!(m.is_empty(), "{stmt:?} -> {m:?}");
        }
        // Rust: drop the semicolon that turns the tail expression into `()`.
        let m = line_mutations(
            "    x + 1;",
            Lang::Rust,
            MutationClass::MissingReturn,
            Some("}"),
        );
        assert_eq!(
            m.into_iter().map(|(_, l)| l).collect::<Vec<_>>(),
            vec!["    x + 1".to_string()]
        );
        let m = muts(
            "pub fn inc(x: i32) -> i32 { x + 1; }",
            Lang::Rust,
            MutationClass::MissingReturn,
        );
        assert_eq!(m, vec!["pub fn inc(x: i32) -> i32 { x + 1 }".to_string()]);
        let m = line_mutations(
            "    let y = 2;",
            Lang::Rust,
            MutationClass::MissingReturn,
            Some("}"),
        );
        assert!(m.is_empty());
    }

    #[test]
    fn shell_test_operators() {
        let m = muts(
            r#"is_positive() { [ "$1" -lt 0 ]; }"#,
            Lang::Shell,
            MutationClass::ShellTest,
        );
        assert!(
            m.contains(&r#"is_positive() { [ "$1" -gt 0 ]; }"#.to_string()),
            "{m:?}"
        );
        let m = muts("echo -n hi", Lang::Shell, MutationClass::ShellTest);
        assert!(m.is_empty(), "echo -n is not a test: {m:?}");
        let m = muts(
            r#"[ "$a" = "$b" ] || exit 2"#,
            Lang::Shell,
            MutationClass::ShellTest,
        );
        assert!(
            m.contains(&r#"[ "$a" != "$b" ] || exit 2"#.to_string()),
            "{m:?}"
        );
    }

    #[test]
    fn oracle_lines_are_recognised() {
        assert!(is_oracle_line("    assert total([1, 2, 3]) == 6"));
        assert!(is_oracle_line("assert.strictEqual(sum([1,2,3]), 6);"));
        assert!(is_oracle_line(
            "        t.Fatalf(\"expected 6 got %d\", got)"
        ));
        assert!(is_oracle_line(
            "raise \"test.rb:2: expected 3\" unless last([1, 2, 3]) == 3"
        ));
        assert!(is_oracle_line("is_positive 5 || { echo failed; exit 1; }"));
        assert!(!is_oracle_line("    return age > 18"));
        assert!(!is_oracle_line("for (let i = 0; i <= xs.length; i++) {"));
    }

    #[test]
    fn class_order_follows_the_trace() {
        assert_eq!(class_order("")[..2], MutationClass::ALL[..2]);
        let o = class_order("IndexError: list index out of range");
        assert_eq!(o[0], MutationClass::OffByOne);
        assert_eq!(o[1], MutationClass::Comparison);
        let o = class_order("AssertionError: expected 6 got 5");
        assert_eq!(o[0], MutationClass::OffByOne);
        let o = class_order("TypeError: unsupported operand type(s) for ==: 'NoneType' and 'int'");
        assert_eq!(o[0], MutationClass::MissingReturn);
        let o = class_order("assertion `left == right` failed\n  left: false\n right: true");
        assert_eq!(o[0], MutationClass::Boolean);
        let o = class_order("5 !== 30");
        assert_eq!(o[0], MutationClass::Arithmetic);
        let o = class_order("sh: 3: [: -lt: unexpected operator");
        assert_eq!(o[0], MutationClass::ShellTest);
        // Every class appears exactly once.
        let o = class_order("anything");
        assert_eq!(o.len(), MutationClass::ALL.len());
        for c in MutationClass::ALL {
            assert_eq!(o.iter().filter(|x| **x == c).count(), 1);
        }
    }

    #[test]
    fn plan_focuses_blamed_lines_skips_oracles_and_is_bounded() {
        let src =
            "def is_adult(age):\n    return age > 18\n\n\ndef unused(a, b):\n    return a - b\n";
        let path = PathBuf::from("calc.py");
        let trace = "  File \"calc.py\", line 2, in is_adult\nAssertionError";
        let full = plan(src, &path, trace, 64);
        assert!(!full.is_empty());
        assert_eq!(full[0].line, 2, "blamed line first: {}", full[0].label);
        assert!(full.iter().take_while(|c| c.line == 2).count() >= 2);
        assert!(full.iter().any(|c| c.source.contains("return age >= 18")));
        // Dedup + bound.
        let mut seen = std::collections::HashSet::new();
        assert!(full.iter().all(|c| seen.insert(c.source.clone())));
        assert!(full.iter().all(|c| c.source != src));
        assert!(plan(src, &path, trace, 3).len() == 3);

        // An oracle line is never touched, even when it is the blamed line.
        let test_src = "from calc import is_adult\nassert is_adult(18) is True\n";
        let p = plan(
            test_src,
            &PathBuf::from("test_calc.py"),
            "test_calc.py:2: AssertionError",
            64,
        );
        assert!(
            p.iter().all(|c| c.line != 2),
            "{:?}",
            p.iter().map(|c| &c.label).collect::<Vec<_>>()
        );
    }

    #[test]
    fn missing_module_names_cover_the_toolchains() {
        assert_eq!(
            missing_module_names("ModuleNotFoundError: No module named 'calcualtor'"),
            vec!["calcualtor"]
        );
        assert_eq!(
            missing_module_names("ModuleNotFoundError: No module named 'pkg.calcualtor'"),
            vec!["calcualtor"]
        );
        assert_eq!(
            missing_module_names("Error: Cannot find module './calcualtor'\nRequire stack:"),
            vec!["calcualtor"]
        );
        assert_eq!(
            missing_module_names("app.rb:1:in 'require_relative': cannot load such file -- /tmp/x/calcualtor (LoadError)"),
            vec!["calcualtor"]
        );
        assert_eq!(
            missing_module_names(
                "app.c:1:10: fatal error: calcualtor.h: No such file or directory"
            ),
            vec!["calcualtor"]
        );
        assert_eq!(
            missing_module_names("error[E0583]: file not found for module `calcualtor`"),
            vec!["calcualtor"]
        );
        assert!(missing_module_names("AssertionError: expected 6 got 5").is_empty());
        assert_eq!(levenshtein("calcualtor", "calculator"), 2);
        assert_eq!(
            class_order("No module named 'x'")[0],
            MutationClass::ImportPath
        );
    }

    #[test]
    fn import_path_candidates_rewrite_to_the_closest_sibling() {
        let dir = std::env::temp_dir().join(format!("axiom_mut_import_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("calculator.py"),
            "def add(a, b):\n    return a + b\n",
        )
        .unwrap();
        std::fs::write(dir.join("unrelated.py"), "").unwrap();
        std::fs::write(dir.join("notes.txt"), "").unwrap();
        let app = dir.join("app.py");
        let src = "from calcualtor import add\n\n\ndef total(a, b):\n    return add(a, b)\n";
        std::fs::write(&app, src).unwrap();
        let trace = "  File \"app.py\", line 1, in <module>\nModuleNotFoundError: No module named 'calcualtor'";
        let p = plan(src, &app, trace, 64);
        assert_eq!(p[0].class, MutationClass::ImportPath, "{}", p[0].label);
        assert!(
            p[0].source.starts_with("from calculator import add\n"),
            "{}",
            p[0].source
        );
        assert!(
            !p.iter().any(|c| c.source.contains("unrelated")),
            "an unrelated sibling is not a candidate"
        );
        // JS: the spec is a quoted string.
        let js = dir.join("app.js");
        std::fs::write(dir.join("calculator.js"), "module.exports = 1;\n").unwrap();
        let js_src = "const add = require('./calcualtor');\nmodule.exports = add;\n";
        std::fs::write(&js, js_src).unwrap();
        let p = plan(js_src, &js, "Error: Cannot find module './calcualtor'", 64);
        assert!(
            p[0].source
                .starts_with("const add = require('./calculator');"),
            "{}",
            p[0].source
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn plan_preserves_line_endings_and_falls_back_to_whole_file() {
        let src = "a\r\nfor i in range(len(xs) - 1):\r\n    s += xs[i]\r\n";
        let p = plan(src, &PathBuf::from("m.py"), "no hints here", 64);
        assert!(
            p.iter().all(|c| c.source.matches("\r\n").count() == 3),
            "CRLF preserved"
        );
        assert!(p.iter().any(|c| c.source.contains("range(len(xs)):\r\n")));
    }
}
