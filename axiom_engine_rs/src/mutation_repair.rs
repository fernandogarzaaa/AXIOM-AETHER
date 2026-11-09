//! Deterministic mutation repair — Phase 2.1 of the autonomous solve loop.
//!
//! The legacy Poly-JIT patch set ([`crate::poly_jit::synthesize_patch`]) knows a
//! handful of fixture-shaped string flips (`exit 1` → `exit 0`, a marker
//! rename, one hard-coded `assert_eq!`). This module is its general,
//! language-aware successor. Given the failing verifier's output and the file
//! blamed by [`crate::fault_locate`], it enumerates a **bounded, ranked set of
//! single-edit source mutations** of the kinds that account for a large share of
//! real one-line bugs — comparison-operator and off-by-one slips, inverted or
//! dropped negations, `and`/`or` mix-ups, arithmetic-operator slips, swapped
//! call arguments, swapped variables, `min`/`max`-style paired calls,
//! `break`/`continue`, missing `return`s, shell `test` operators — and hands them to the
//! verifier-gated, reversible apply loop in [`crate::solve`]. A candidate is kept
//! only if the real verifier goes green; everything else is rolled back
//! byte-for-byte. No model, no network: the intelligence is the verifier, and
//! this module's job is to spend its runs where the evidence points.
//!
//! Three rules keep it honest:
//!
//! 1. **Focus.** Lines the failure trace blames are mutated first; then the
//!    rest of the function that raised, the bodies of the functions the
//!    failing test calls, what those call, and finally the rest of the file —
//!    each tier class-major, so the budget goes where the evidence is.
//!    Identifier swaps (up to 15 per line) are tried off the blamed lines only
//!    after everything else.
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

/// Tier for identifier swaps off the blamed lines: after every other edit.
const LAST_TIER: u8 = 5;

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
    /// `true` ↔ `false`, drop a `not`/`!` (or unwrap `!( … )`), negate a
    /// returned expression, an `if`/`while` condition, or a closure body.
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
    /// Swap a call or keyword for its paired opposite: `min` ↔ `max`,
    /// `floor` ↔ `ceil`, `first` ↔ `last`, `starts_with` ↔ `ends_with`,
    /// `any` ↔ `all`, `break` ↔ `continue`.
    ApiSwap,
    /// Exchange every occurrence of two identifiers on one line (`a - b` with
    /// `b - a`, a swapped `(row, col)`): the "wrong variable" bug.
    IdentSwap,
    /// Argument- and operand-level slips: `± 1` on a bare identifier argument
    /// or index (`f(mid, hi)` → `f(mid + 1, hi)`), a two-element subscript
    /// swapped (`d[j, k]` → `d[k, j]`), and the operands of a lone `+`
    /// swapped (`s = s + c` → `s = c + s`). Prolific, so tried late.
    ArgShift,
}

impl MutationClass {
    pub const ALL: [MutationClass; 12] = [
        MutationClass::Comparison,
        MutationClass::OffByOne,
        MutationClass::Boolean,
        MutationClass::Logic,
        MutationClass::Arithmetic,
        MutationClass::SwapArgs,
        MutationClass::MissingReturn,
        MutationClass::ShellTest,
        MutationClass::ApiSwap,
        MutationClass::ArgShift,
        MutationClass::IdentSwap,
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
            MutationClass::ApiSwap => "api-swap",
            MutationClass::IdentSwap => "ident-swap",
            MutationClass::ArgShift => "arg-shift",
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

/// Lines that continue a multiline oracle expression. Only the opening line
/// of a multiline assertion carries an oracle marker, so line-local
/// `is_oracle_line` misses the comparison on a continuation line and `plan`
/// would mutate it (Codex P1 on #247: `1 == 2` -> `1 != 2`, verifier green
/// by reversing the assertion). A bracket-depth scan from each oracle line
/// marks the continuation lines up to and including the line where the
/// depth returns to zero. Counting is naive (no string/comment awareness);
/// over-extension only ever skips a mutation, which is the safe direction.
fn oracle_continuation_lines(lines: &[&str]) -> HashSet<usize> {
    let depth_of = |s: &str| -> i32 {
        s.chars().fold(0i32, |d, c| match c {
            '(' | '[' | '{' => d + 1,
            ')' | ']' | '}' => d - 1,
            _ => d,
        })
    };
    let mut out = HashSet::new();
    for (i, line) in lines.iter().enumerate() {
        let (content, _) = split_eol(line);
        if !is_oracle_line(content) {
            continue;
        }
        let mut depth = depth_of(content);
        if depth <= 0 {
            continue;
        }
        let mut j = i + 1;
        while j < lines.len() {
            let (next, _) = split_eol(lines[j]);
            depth += depth_of(next);
            out.insert(j);
            j += 1;
            if depth <= 0 {
                break;
            }
        }
    }
    out
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
    let oracle_continuations = oracle_continuation_lines(&lines);

    // In-file test region (Rust `#[cfg(test)]`): the oracle. Nothing at or
    // below it is ever mutated, and a blamed line inside it (the assertion
    // that panicked) is not a repair site but a pointer to the functions the
    // test calls — see `callee_focus_lines`.
    let oracle = oracle_ranges(&lines, lang);
    let in_scope = |i: usize| !in_ranges(&oracle, i);

    // Blamed lines (1-based → 0-based), in the order the trace ranks them.
    let mut hinted: Vec<usize> = Vec::new();
    let mut hinted_in_oracle: Vec<usize> = Vec::new();
    for l in crate::fault_locate::line_hints_for(trace, path) {
        let idx = l as usize;
        if idx == 0 || idx > n {
            continue;
        }
        let i = idx - 1;
        if in_scope(i) {
            if !hinted.contains(&i) {
                hinted.push(i);
            }
        } else if !hinted_in_oracle.contains(&i) {
            hinted_in_oracle.push(i);
        }
    }
    // The rest of the function that raised (an in-scope blamed line is
    // usually the symptom: an index that went out of bounds a few lines after
    // the bound was computed). Nearest lines first, earlier lines before later.
    let enclosing: Vec<usize> = enclosing_fn_lines(&lines, lang, &hinted)
        .into_iter()
        .filter(|i| in_scope(*i) && !hinted.contains(i))
        .collect();
    // Functions the failing test exercises, by name: their bodies are the
    // most likely home of the fault when the trace only blames the assertion.
    // Direct callees first, then what those call.
    let (direct, indirect) = callee_focus_lines(&lines, lang, &oracle, &hinted_in_oracle);
    let taken = |i: &usize| hinted.contains(i) || enclosing.contains(i);
    let focus: Vec<usize> = direct.into_iter().filter(|i| !taken(i)).collect();
    let focus2: Vec<usize> = indirect
        .into_iter()
        .filter(|i| !taken(i) && !focus.contains(i))
        .collect();
    // Remaining lines: by distance to the nearest hint/focus line if any, else
    // file order.
    let anchors: Vec<usize> = hinted
        .iter()
        .chain(enclosing.iter())
        .chain(focus.iter())
        .chain(focus2.iter())
        .copied()
        .collect();
    let mut rest: Vec<usize> = (0..n)
        .filter(|i| in_scope(*i) && !anchors.contains(i))
        .collect();
    if !anchors.is_empty() {
        rest.sort_by_key(|i| {
            anchors
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
        if content.trim().is_empty()
            || is_comment_only(content, lang)
            || is_oracle_line(content)
            || oracle_continuations.contains(&li)
        {
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
                // Tier 0 (blamed lines): line order, every class. Tiers 1+
                // (callee focus, then the rest of the file): class-major so
                // the trace-derived class order decides what is tried first.
                // Identifier swaps are the most prolific class (up to 15 per
                // line), so off the blamed lines they wait until every other
                // edit has been tried. A swapped variable usually breaks the
                // build, and the compiler then blames the line itself.
                let tier = if matches!(*class, MutationClass::IdentSwap | MutationClass::ArgShift)
                    && tier > 0
                {
                    LAST_TIER
                } else {
                    tier
                };
                let key = if tier == 0 {
                    (0, pos, cr, *seq)
                } else {
                    (tier, cr, pos, *seq)
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
    for (pos, li) in enclosing.iter().enumerate() {
        visit(*li, pos, 1, &mut keyed, &mut seq);
    }
    for (pos, li) in focus.iter().enumerate() {
        visit(*li, pos, 2, &mut keyed, &mut seq);
    }
    for (pos, li) in focus2.iter().enumerate() {
        visit(*li, pos, 3, &mut keyed, &mut seq);
    }
    for (pos, li) in rest.iter().enumerate() {
        visit(*li, pos, 4, &mut keyed, &mut seq);
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

/// Inclusive line ranges of every `#[cfg(test)]`-gated item: the oracle.
/// Each range runs from the attribute line through the end of the item it
/// gates — the matching `}` for a block item (`mod tests { … }`, a test-only
/// `fn`), or the first line ending in `;` for a one-line item (a test-only
/// `use`). Only the gated item is excluded: production code *after* a
/// test-only helper or import stays repairable. Rust only; other languages
/// keep tests in separate files.
fn oracle_ranges(lines: &[&str], lang: Lang) -> Vec<(usize, usize)> {
    if lang != Lang::Rust {
        return Vec::new();
    }
    let n = lines.len();
    let content = |i: usize| split_eol(lines[i]).0;
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if content(i).trim() != "#[cfg(test)]" {
            i += 1;
            continue;
        }
        // The gated item starts at the next non-blank, non-attribute line.
        let mut item = i + 1;
        while item < n && {
            let t = content(item).trim();
            t.is_empty() || (t.starts_with("#[") && t.ends_with(']'))
        } {
            item += 1;
        }
        if item >= n {
            out.push((i, n - 1));
            break;
        }
        // A block item opens `{` before any `;`; a one-line item hits `;`.
        let mut end = item;
        let mut is_block = false;
        'scan: for (j, raw) in lines.iter().enumerate().skip(item) {
            let l = split_eol(raw).0;
            let mask = code_mask(l, Lang::Rust);
            for (k, b) in l.bytes().enumerate() {
                if !mask[k] {
                    continue;
                }
                match b {
                    b'{' => {
                        is_block = true;
                        end = j;
                        break 'scan;
                    }
                    b';' => {
                        end = j;
                        break 'scan;
                    }
                    _ => {}
                }
            }
            end = j;
        }
        if is_block {
            end = body_end(lines, end, n);
        }
        out.push((i, end));
        i = end + 1;
    }
    out
}

fn in_ranges(ranges: &[(usize, usize)], i: usize) -> bool {
    ranges.iter().any(|(s, e)| *s <= i && i <= *e)
}

/// Lines of the non-test functions that the failing test calls, in the order
/// the test calls them. The blamed lines are assertions inside an oracle range
/// (`hinted_in_oracle`); each is widened to its enclosing test function, the
/// identifiers that function calls are collected, and every `fn <name>`
/// defined outside the oracle ranges with one of those names contributes its
/// body. Cheap, static, deterministic: the call-graph half of spectrum-based
/// fault localization without running the tests one by one.
fn callee_focus_lines(
    lines: &[&str],
    lang: Lang,
    oracle: &[(usize, usize)],
    hinted_in_oracle: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    if lang != Lang::Rust || oracle.is_empty() || hinted_in_oracle.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let n = lines.len();
    let content = |i: usize| split_eol(lines[i]).0;
    let is_fn_line = |i: usize| {
        let t = content(i).trim_start();
        t.starts_with("fn ") || t.starts_with("pub fn ") || t.contains(" fn ")
    };
    let is_test_attr = |i: usize| content(i).trim() == "#[test]";

    // Definitions outside the oracle: name -> (first line, last line).
    let mut defs: Vec<(String, usize, usize)> = Vec::new();
    for i in 0..n {
        if in_ranges(oracle, i) || !is_fn_line(i) {
            continue;
        }
        let Some(name) = fn_name(content(i)) else {
            continue;
        };
        let end = body_end(lines, i, n);
        defs.push((name, i, end));
    }
    if defs.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Called names, in order of appearance across the blamed tests. Each scan
    // stays inside the oracle range that holds the blamed line.
    let mut called: Vec<String> = Vec::new();
    for &h in hinted_in_oracle {
        let Some(&(rs, re)) = oracle.iter().find(|(s, e)| *s <= h && h <= *e) else {
            continue;
        };
        let mut start = h;
        while start > rs && !is_fn_line(start) {
            start -= 1;
        }
        let mut end = h + 1;
        while end <= re && !is_test_attr(end) && !(end > start && is_fn_line(end)) {
            end += 1;
        }
        for i in start..end.min(n) {
            for name in called_identifiers(content(i)) {
                if !called.contains(&name) {
                    called.push(name);
                }
            }
        }
    }

    let bodies_of = |names: &[String], skip: &[usize]| -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        for name in names {
            for (def, s, e) in &defs {
                if def == name {
                    for i in *s..=*e {
                        if !out.contains(&i) && !skip.contains(&i) {
                            out.push(i);
                        }
                    }
                }
            }
        }
        out
    };
    let direct = bodies_of(&called, &[]);
    // One hop further: what the direct callees call (a fault in a helper
    // shows up as a wrong result from the function the test calls).
    let mut called2: Vec<String> = Vec::new();
    for &i in &direct {
        for name in called_identifiers(content(i)) {
            if !called.contains(&name) && !called2.contains(&name) {
                called2.push(name);
            }
        }
    }
    let indirect = bodies_of(&called2, &direct);
    (direct, indirect)
}

/// Lines of the function enclosing each blamed line (Rust brace bodies,
/// Python indentation bodies), ordered by distance to the blamed line with
/// earlier lines first (the faulty computation usually precedes the panic).
fn enclosing_fn_lines(lines: &[&str], lang: Lang, hinted: &[usize]) -> Vec<usize> {
    let n = lines.len();
    let content = |i: usize| split_eol(lines[i]).0;
    let indent = |i: usize| content(i).len() - content(i).trim_start().len();
    let mut out: Vec<usize> = Vec::new();
    for &h in hinted {
        let range = match lang {
            Lang::Rust => {
                let is_fn = |i: usize| {
                    fn_name(content(i).trim_start()).is_some()
                        && (content(i).trim_start().starts_with("fn ")
                            || content(i).contains(" fn ")
                            || content(i).trim_start().starts_with("pub fn "))
                };
                let mut s = h;
                while s > 0 && !is_fn(s) {
                    s -= 1;
                }
                if !is_fn(s) {
                    continue;
                }
                let e = body_end(lines, s, n);
                if e < h {
                    continue; // the hint is not inside that body
                }
                (s, e)
            }
            Lang::Python => {
                let mut s = h;
                while s > 0 && !content(s).trim_start().starts_with("def ") {
                    s -= 1;
                }
                if !content(s).trim_start().starts_with("def ") || s == h {
                    continue;
                }
                let base = indent(s);
                let mut e = s + 1;
                while e < n && (content(e).trim().is_empty() || indent(e) > base) {
                    e += 1;
                }
                if e <= h {
                    continue;
                }
                (s, e - 1)
            }
            _ => continue,
        };
        let mut body: Vec<usize> = (range.0..=range.1).filter(|i| *i != h).collect();
        body.sort_by_key(|i| if *i < h { 2 * (h - i) } else { 2 * (i - h) + 1 });
        for i in body {
            if !out.contains(&i) {
                out.push(i);
            }
        }
    }
    out
}

/// `fn name(` / `fn name<` → `name`.
fn fn_name(line: &str) -> Option<String> {
    let idx = line.find("fn ")?;
    let rest = &line[idx + 3..];
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Last line of the block that opens on `start` (brace depth returns to zero),
/// bounded by `limit`.
fn body_end(lines: &[&str], start: usize, limit: usize) -> usize {
    let mut depth = 0i32;
    let mut opened = false;
    for (i, raw) in lines.iter().enumerate().take(limit).skip(start) {
        let l = split_eol(raw).0;
        let mask = code_mask(l, Lang::Rust);
        for (k, b) in l.bytes().enumerate() {
            if !mask[k] {
                continue;
            }
            match b {
                b'{' => {
                    depth += 1;
                    opened = true;
                }
                b'}' => depth -= 1,
                _ => {}
            }
        }
        if opened && depth <= 0 {
            return i;
        }
    }
    limit.saturating_sub(1)
}

/// Identifiers immediately followed by `(` on a code line, macros (`name!(`)
/// excluded, in order of appearance.
fn called_identifiers(line: &str) -> Vec<String> {
    let b = line.as_bytes();
    let mask = code_mask(line, Lang::Rust);
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if mask[i] && (b[i].is_ascii_alphabetic() || b[i] == b'_') {
            let s = i;
            while i < b.len() && is_ident(b[i]) {
                i += 1;
            }
            let mut j = i;
            while j < b.len() && (b[j] == b' ' || b[j] == b'\t') {
                j += 1;
            }
            let macro_call = j < b.len() && b[j] == b'!';
            let turbofish = j + 1 < b.len() && b[j] == b':' && b[j + 1] == b':';
            if !macro_call && !turbofish && j < b.len() && b[j] == b'(' {
                out.push(line[s..i].to_string());
            }
            continue;
        }
        i += 1;
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
        MutationClass::Arithmetic => {
            let mut v = arithmetic(line, lang, &mask);
            v.extend(bitwise(line, lang, &mask));
            v
        }
        MutationClass::SwapArgs => swap_args(line, &mask),
        MutationClass::ArgShift => {
            let mut v = shift_argument(line, lang, &mask);
            v.extend(swap_subscript(line, &mask));
            v.extend(swap_sum_operands(line, lang, &mask));
            v
        }
        MutationClass::MissingReturn => missing_return(line, lang, &mask, next_line),
        MutationClass::ShellTest => shell_test(line, lang, &mask),
        // Planned per file in `plan` (needs the trace and the directory).
        MutationClass::ImportPath => Vec::new(),
        MutationClass::ApiSwap => api_swap(line, lang, &mask),
        MutationClass::IdentSwap => ident_swap(line, lang, &mask),
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
    if lang.c_like() {
        out.extend(unwrap_negated_group(line, mask));
    }
    out.extend(negate_condition(line, lang, mask));
    out.extend(negate_closure_body(line, lang, mask));
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

/// At most this many identifiers per line get a `± 1` shift.
const MAX_SHIFTED_ARGS: usize = 4;

/// `f(mid, end)` → `f(mid + 1, end)`, `range(0, r)` → `range(0, r + 1)`,
/// `dp[i, j]` → `dp[i, j - 1]`: ± 1 on a bare identifier that is a whole
/// call argument or subscript element (the developer off-by-one that is not on
/// a literal or a length).
fn shift_argument(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    let b = line.as_bytes();
    let n = b.len();
    let mut out = Vec::new();
    if lang == Lang::Shell {
        return out;
    }
    let mut shifted = 0;
    let mut i = 0;
    while i < n && shifted < MAX_SHIFTED_ARGS {
        if !(mask[i]
            && (b[i].is_ascii_alphabetic() || b[i] == b'_')
            && (i == 0 || !is_ident(b[i - 1])))
        {
            i += 1;
            continue;
        }
        let mut e = i;
        while e < n && is_ident(b[e]) {
            e += 1;
        }
        let word = &line[i..e];
        let before = prev_non_space(b, i).map(|k| b[k]);
        let after = next_non_space(b, e).map(|k| b[k]);
        let is_arg = matches!(before, Some(b'(') | Some(b',') | Some(b'['))
            && matches!(after, Some(b')') | Some(b',') | Some(b']'))
            && !(i > 0 && b[i - 1] == b'.');
        if is_arg
            && !NOT_SWAPPABLE.contains(&word)
            && !word.chars().all(|c| c.is_ascii_uppercase() || c == '_')
        {
            for (op, label) in [(" + 1", "+ 1"), (" - 1", "- 1")] {
                out.push((
                    format!("`{word}` -> `{word} {label}`"),
                    format!("{}{word}{op}{}", &line[..i], &line[e..]),
                ));
            }
            shifted += 1;
        }
        i = e;
    }
    out
}

/// Bitwise operator slips: `^=` / `&=` / `|=` among themselves, and a spaced
/// binary `a ^ b` / `a & b` / `a | b` (spaces on both sides, so a Rust borrow
/// `&x` or closure `|x|` is not one).
fn bitwise(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    let b = line.as_bytes();
    let n = b.len();
    let mut out = Vec::new();
    if lang == Lang::Shell {
        return out;
    }
    let mut i = 0;
    while i < n {
        if !mask[i] || !matches!(b[i], b'^' | b'&' | b'|') {
            i += 1;
            continue;
        }
        let c = b[i];
        let doubled = (i + 1 < n && b[i + 1] == c) || (i > 0 && b[i - 1] == c);
        if doubled {
            i += 1;
            continue;
        }
        let compound = i + 1 < n && b[i + 1] == b'=' && !(i + 2 < n && b[i + 2] == b'=');
        let spaced = i > 0 && b[i - 1] == b' ' && i + 1 < n && b[i + 1] == b' ';
        if compound || spaced {
            let from = c as char;
            for to in ['&', '|', '^'] {
                if to as u8 == c {
                    continue;
                }
                let new_line = replace_span(line, i, i + 1, &to.to_string());
                out.push((
                    if compound {
                        format!("{from}= -> {to}=")
                    } else {
                        format!("{from} -> {to}")
                    },
                    new_line,
                ));
            }
        }
        i += 1;
    }
    out
}

/// Swap the two elements of a two-element subscript: `d[j, k]` → `d[k, j]`.
fn swap_subscript(line: &str, mask: &[bool]) -> Vec<(String, String)> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    for open in 0..b.len() {
        if b[open] != b'['
            || !mask[open]
            || open == 0
            || !(is_ident(b[open - 1]) || b[open - 1] == b']' || b[open - 1] == b')')
        {
            continue;
        }
        let Some(close) = matching_paren(b, mask, open) else {
            continue;
        };
        let mut depth = 0i32;
        let mut commas = Vec::new();
        for k in open + 1..close {
            if !mask[k] {
                continue;
            }
            match b[k] {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth -= 1,
                b',' if depth == 0 => commas.push(k),
                _ => {}
            }
        }
        if commas.len() != 1 {
            continue;
        }
        let (x, y) = (
            line[open + 1..commas[0]].trim(),
            line[commas[0] + 1..close].trim(),
        );
        if x.is_empty() || y.is_empty() || x == y {
            continue;
        }
        out.push((
            format!("swap subscript `{x}` <-> `{y}`"),
            format!("{}[{y}, {x}]{}", &line[..open], &line[close + 1..]),
        ));
    }
    out
}

/// `result = result + c` → `result = c + result`: swap the operands of a
/// single top-level `+` on the right of an assignment or `return`
/// (concatenation is not commutative).
fn swap_sum_operands(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if lang == Lang::Shell {
        return out;
    }
    let b = line.as_bytes();
    let indent = line.len() - line.trim_start().len();
    let t = &line[indent..];
    // Start of the right-hand side.
    let rhs_start = if let Some(r) = t.strip_prefix("return ") {
        indent + (t.len() - r.len())
    } else {
        let mut eq = None;
        for k in indent..b.len() {
            if mask[k]
                && b[k] == b'='
                && !matches!(b.get(k + 1), Some(b'='))
                && !matches!(
                    b.get(k.wrapping_sub(1)),
                    Some(b'=' | b'!' | b'<' | b'>' | b'+' | b'-' | b'*' | b'/')
                )
            {
                eq = Some(k + 1);
                break;
            }
        }
        match eq {
            Some(k) => k,
            None => return out,
        }
    };
    let end = line.trim_end().trim_end_matches(';').len();
    if rhs_start >= end {
        return out;
    }
    let mut depth = 0i32;
    let mut plus = Vec::new();
    for k in rhs_start..end {
        if !mask[k] {
            continue;
        }
        match b[k] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'+' if depth == 0 && b.get(k + 1) != Some(&b'=') && is_binary_position(b, k) => {
                plus.push(k)
            }
            b'-' | b'*' | b'/' | b'%' | b'<' | b'>' | b'&' | b'|' if depth == 0 => return out,
            _ => {}
        }
    }
    if plus.len() != 1 {
        return out;
    }
    let (a, c) = (
        line[rhs_start..plus[0]].trim(),
        line[plus[0] + 1..end].trim(),
    );
    if a.is_empty() || c.is_empty() || a == c {
        return out;
    }
    let lead = &line[..rhs_start];
    let sep = if lead.ends_with(' ') { "" } else { " " };
    out.push((
        format!("swap operands `{a}` + `{c}`"),
        format!("{lead}{sep}{c} + {a}{}", &line[end..]),
    ));
    out
}

/// Paired opposites: a call to one is a classic single-token slip for the other.
/// `call` pairs must be followed by `(` (a call, not a variable that happens to
/// be named `max`); keyword pairs stand alone.
const API_PAIRS_CALL: &[(&str, &str)] = &[
    ("min", "max"),
    ("max", "min"),
    ("floor", "ceil"),
    ("ceil", "floor"),
    ("first", "last"),
    ("last", "first"),
    ("starts_with", "ends_with"),
    ("ends_with", "starts_with"),
    ("startswith", "endswith"),
    ("endswith", "startswith"),
    ("startsWith", "endsWith"),
    ("endsWith", "startsWith"),
    ("any", "all"),
    ("all", "any"),
    ("min_by_key", "max_by_key"),
    ("max_by_key", "min_by_key"),
];

fn api_swap(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    for (from, to) in API_PAIRS_CALL {
        for (s, e) in word_spans(line, mask, from) {
            if next_non_space(b, e).map(|k| b[k]) != Some(b'(') {
                continue;
            }
            out.push((format!("{from} -> {to}"), replace_span(line, s, e, to)));
        }
    }
    if lang != Lang::Shell {
        for (from, to) in [("break", "continue"), ("continue", "break")] {
            for (s, e) in word_spans(line, mask, from) {
                out.push((format!("{from} -> {to}"), replace_span(line, s, e, to)));
            }
        }
    }
    out
}

/// Words that are never a variable worth swapping.
const NOT_SWAPPABLE: &[&str] = &[
    "as",
    "async",
    "await",
    "break",
    "const",
    "continue",
    "crate",
    "def",
    "del",
    "dyn",
    "elif",
    "else",
    "enum",
    "extern",
    "false",
    "fn",
    "for",
    "from",
    "func",
    "function",
    "if",
    "impl",
    "import",
    "in",
    "is",
    "lambda",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "new",
    "nil",
    "not",
    "null",
    "or",
    "and",
    "pub",
    "ref",
    "return",
    "self",
    "static",
    "struct",
    "super",
    "this",
    "trait",
    "true",
    "type",
    "typeof",
    "undefined",
    "unsafe",
    "use",
    "var",
    "where",
    "while",
    "with",
    "yield",
    "None",
    "True",
    "False",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "usize",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "isize",
    "f32",
    "f64",
    "bool",
    "char",
    "str",
    "int",
    "float",
    "string",
];

/// At most this many distinct identifiers per line take part in swaps, so a
/// long line costs at most `C(6, 2) = 15` candidates.
const MAX_SWAP_IDENTS: usize = 6;

fn ident_swap(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    if lang == Lang::Shell {
        return Vec::new();
    }
    let b = line.as_bytes();
    let n = b.len();
    // (start, end) spans of every code identifier, and the distinct names in
    // first-seen order.
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut names: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < n {
        if mask[i]
            && (b[i].is_ascii_alphabetic() || b[i] == b'_')
            && (i == 0 || !is_ident(b[i - 1]))
        {
            let mut e = i;
            while e < n && is_ident(b[e]) && mask[e] {
                e += 1;
            }
            let w = &line[i..e];
            let after_path = i >= 2 && &b[i - 2..i] == b"::";
            let macro_call = e < n && b[e] == b'!';
            let upper = b[i].is_ascii_uppercase();
            if !after_path && !macro_call && !upper && !NOT_SWAPPABLE.contains(&w) {
                spans.push((i, e));
                if !names.contains(&w) {
                    names.push(w);
                }
            }
            i = e;
        } else {
            i += 1;
        }
    }
    names.truncate(MAX_SWAP_IDENTS);
    let mut out = Vec::new();
    for x in 0..names.len() {
        for y in x + 1..names.len() {
            let (a, c) = (names[x], names[y]);
            let mut s = String::with_capacity(line.len());
            let mut last = 0;
            for (st, en) in &spans {
                let w = &line[*st..*en];
                let rep = if w == a {
                    c
                } else if w == c {
                    a
                } else {
                    continue;
                };
                s.push_str(&line[last..*st]);
                s.push_str(rep);
                last = *en;
            }
            s.push_str(&line[last..]);
            out.push((format!("swap `{a}` <-> `{c}`"), s));
        }
    }
    out
}

/// Is `expr` a single postfix operand (`a`, `a.b(c)`, `xs[i]`, `f(x, y)`) that
/// can take a prefix `!` / `not ` without parentheses?
fn is_simple_operand(expr: &str, lang: Lang) -> bool {
    let b = expr.as_bytes();
    if b.is_empty() || !(b[0].is_ascii_alphabetic() || b[0] == b'_') {
        return false;
    }
    let mask = code_mask(expr, lang);
    let mut depth = 0i32;
    for (k, c) in b.iter().enumerate() {
        if !mask[k] {
            continue;
        }
        match c {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            _ if depth == 0 && !(is_ident(*c) || *c == b'.' || *c == b':' || *c == b'?') => {
                return false
            }
            _ => {}
        }
    }
    depth == 0
}

fn negated(expr: &str, lang: Lang) -> String {
    let simple = is_simple_operand(expr, lang);
    match (lang.word_logic(), simple) {
        (true, true) => format!("not {expr}"),
        (true, false) => format!("not ({expr})"),
        (false, true) => format!("!{expr}"),
        (false, false) => format!("!({expr})"),
    }
}

/// Negate a whole `if` / `while` condition (the dropped `!`): `if x {` →
/// `if !x {`, `if a || b {` → `if !(a || b) {`, `if (x) {` → `if (!x) {`,
/// `if x:` → `if not x:`. Conditions that already start with a negation are
/// left to the `drop !` operator; `if let` / `while let` are not conditions.
fn negate_condition(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if lang == Lang::Shell {
        return out;
    }
    let t_end = line.trim_end().len();
    for kw in ["if", "while", "elif"] {
        for (_, e) in word_spans(line, mask, kw) {
            let Some(cs) = next_non_space(line.as_bytes(), e) else {
                continue;
            };
            // Condition runs to the block opener at the end of the line.
            let opener = if lang.word_logic() { ':' } else { '{' };
            let body = line[cs..t_end].trim_end();
            let Some(cond) = body.strip_suffix(opener) else {
                continue;
            };
            let cond = cond.trim_end();
            if cond.is_empty()
                || cond.starts_with("let ")
                || cond.starts_with('!')
                || cond.starts_with("not ")
                || !mask[cs..cs + cond.len()].iter().any(|m| *m)
            {
                continue;
            }
            let ce = cs + cond.len();
            // `if (cond) {` (C, JS, Java): negate inside the parentheses.
            let bb = line.as_bytes();
            let new_cond = if bb[cs] == b'(' && matching_paren(bb, mask, cs) == Some(ce - 1) {
                let inner = &line[cs + 1..ce - 1];
                if inner.starts_with('!') {
                    continue;
                }
                format!("({})", negated(inner, lang))
            } else {
                negated(cond, lang)
            };
            out.push((
                format!("negate `{kw}` condition"),
                replace_span(line, cs, ce, &new_cond),
            ));
        }
    }
    out
}

/// Negate the body of a one-expression closure passed as the last argument:
/// `.filter(|&p| claimed[p])` → `.filter(|&p| !claimed[p])`,
/// `.filter(x => x.ok)` → `.filter(x => !x.ok)`.
fn negate_closure_body(line: &str, lang: Lang, mask: &[bool]) -> Vec<(String, String)> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let push = |body_s: usize, close: usize, out: &mut Vec<(String, String)>| {
        let body = line[body_s..close].trim();
        if body.is_empty() || body.starts_with('!') || body.starts_with('{') {
            return;
        }
        let s = body_s + (line[body_s..close].len() - line[body_s..close].trim_start().len());
        let e = s + body.len();
        out.push((
            "negate closure body".to_string(),
            replace_span(line, s, e, &negated(body, lang)),
        ));
    };
    for open in 0..b.len() {
        if b[open] != b'(' || !mask[open] {
            continue;
        }
        let Some(close) = matching_paren(b, mask, open) else {
            continue;
        };
        let Some(first) = next_non_space(b, open + 1) else {
            continue;
        };
        if lang == Lang::Rust && b[first] == b'|' {
            // `|args| body` — the second `|` ends the parameter list.
            if let Some(rel) = line[first + 1..close].find('|') {
                push(first + 1 + rel + 1, close, &mut out);
            }
        } else if lang == Lang::JavaScript {
            if let Some(rel) = line[open + 1..close].find("=>") {
                push(open + 1 + rel + 2, close, &mut out);
            }
        }
    }
    out
}

/// `!(a || b)` → `a || b` when the parenthesised group is the whole condition
/// (keeps the code warning-free, unlike leaving `(a || b)` behind).
fn unwrap_negated_group(line: &str, mask: &[bool]) -> Vec<(String, String)> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    for (s, e, op) in op_spans(line, mask) {
        if op != "!" || e >= b.len() || b[e] != b'(' || (s > 0 && is_ident(b[s - 1])) {
            continue;
        }
        let Some(close) = matching_paren(b, mask, e) else {
            continue;
        };
        let before = prev_non_space(b, s).map(|k| b[k]);
        let after = next_non_space(b, close + 1).map(|k| b[k]);
        let opens_condition = before.is_none_or(|c| is_ident(c) || c == b'(' || c == b'=');
        let ends_condition = matches!(
            after,
            None | Some(b'{') | Some(b')') | Some(b';') | Some(b',')
        );
        if opens_condition && ends_condition {
            let inner = &line[e + 1..close];
            out.push((
                "unwrap `!( )`".to_string(),
                format!("{}{}{}", &line[..s], inner, &line[close + 1..]),
            ));
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
    fn negation_is_added_to_conditions_and_closure_bodies() {
        let has = |m: &[String], s: &str| m.iter().any(|l| l == s);
        // A dropped `!` on an `if` condition: simple operand, no parentheses.
        let m = muts(
            "            if eligible_idx.contains(&i) {",
            Lang::Rust,
            MutationClass::Boolean,
        );
        assert!(
            has(&m, "            if !eligible_idx.contains(&i) {"),
            "{m:?}"
        );
        // A compound condition is wrapped.
        let m = muts("if a != 0 || b {", Lang::Rust, MutationClass::Boolean);
        assert!(has(&m, "if !(a != 0 || b) {"), "{m:?}");
        // C-style parenthesised condition: negate inside the parentheses.
        let m = muts("while (ok) {", Lang::JavaScript, MutationClass::Boolean);
        assert!(has(&m, "while (!ok) {"), "{m:?}");
        // Python uses `not`.
        let m = muts("    if xs:", Lang::Python, MutationClass::Boolean);
        assert!(has(&m, "    if not xs:"), "{m:?}");
        // `if let` is a pattern match, not a condition.
        let m = muts("if let Some(x) = y {", Lang::Rust, MutationClass::Boolean);
        assert!(!m.iter().any(|l| l.contains("if !")), "{m:?}");
        // Closure bodies.
        let m = muts(
            "            .filter(|&p| payload_claimed[p])",
            Lang::Rust,
            MutationClass::Boolean,
        );
        assert!(
            has(&m, "            .filter(|&p| !payload_claimed[p])"),
            "{m:?}"
        );
        let m = muts(
            "xs.filter(x => x.ok)",
            Lang::JavaScript,
            MutationClass::Boolean,
        );
        assert!(has(&m, "xs.filter(x => !x.ok)"), "{m:?}");
    }

    #[test]
    fn negated_group_unwraps_without_leaving_parentheses() {
        let m = muts(
            "        if !(self.gain != 0.0 || self.explore != 0.0) {",
            Lang::Rust,
            MutationClass::Boolean,
        );
        assert!(
            m.contains(&"        if self.gain != 0.0 || self.explore != 0.0 {".to_string()),
            "{m:?}"
        );
        // A `!( )` that is one operand of a larger expression keeps its scope
        // (only the plain `drop !` applies there).
        let m = muts("if !(a) && b {", Lang::Rust, MutationClass::Boolean);
        assert!(
            !m.contains(&"if a && b {".to_string()) || m.contains(&"if (a) && b {".to_string())
        );
        assert!(!m.iter().any(|l| l == "if a) && b {"), "{m:?}");
    }

    #[test]
    fn variable_argument_and_index_shift_by_one() {
        let has = |m: &[String], x: &str| m.iter().any(|l| l == x);
        let m = muts(
            "    return search(mid, hi)",
            Lang::Python,
            MutationClass::ArgShift,
        );
        assert!(has(&m, "    return search(mid + 1, hi)"), "{m:?}");
        assert!(has(&m, "    return search(mid, hi - 1)"), "{m:?}");
        let m = muts(
            "for c in range(0, r):",
            Lang::Python,
            MutationClass::ArgShift,
        );
        assert!(has(&m, "for c in range(0, r + 1):"), "{m:?}");
        let m = muts(
            "t[i, j] = t[i - 1, j] + 1",
            Lang::Python,
            MutationClass::ArgShift,
        );
        assert!(has(&m, "t[i, j] = t[i - 1, j - 1] + 1"), "{m:?}");
        // Attribute access and keywords are not bare arguments.
        let m = muts("f(self.x, None)", Lang::Python, MutationClass::ArgShift);
        assert!(
            !m.iter().any(|l| l.contains("x + 1") || l.contains("None")),
            "{m:?}"
        );
    }

    #[test]
    fn bitwise_operators_swap_but_logic_and_borrows_do_not() {
        let m = muts(
            "        n ^= n - 1",
            Lang::Python,
            MutationClass::Arithmetic,
        );
        assert!(m.contains(&"        n &= n - 1".to_string()), "{m:?}");
        let m = muts("mask = a | b", Lang::Python, MutationClass::Arithmetic);
        assert!(m.contains(&"mask = a & b".to_string()), "{m:?}");
        let m = muts("if a && b || c {", Lang::Rust, MutationClass::Arithmetic);
        assert!(
            !m.iter().any(|l| l.contains('^')),
            "logic operators untouched: {m:?}"
        );
        let m = muts(
            "let r = f(&x, |y| y);",
            Lang::Rust,
            MutationClass::Arithmetic,
        );
        assert!(
            !m.iter().any(|l| l.contains('^')),
            "borrow and closure untouched: {m:?}"
        );
    }

    #[test]
    fn subscript_elements_and_sum_operands_swap() {
        let m = muts(
            "d = w[i, k] + w[j, k]",
            Lang::Python,
            MutationClass::ArgShift,
        );
        assert!(m.contains(&"d = w[i, k] + w[k, j]".to_string()), "{m:?}");
        let m = muts(
            "        out = out + digits[i]",
            Lang::Python,
            MutationClass::ArgShift,
        );
        assert!(
            m.contains(&"        out = digits[i] + out".to_string()),
            "{m:?}"
        );
        let m = muts(
            "    return head + rest;",
            Lang::JavaScript,
            MutationClass::ArgShift,
        );
        assert!(m.contains(&"    return rest + head;".to_string()), "{m:?}");
        // Mixed operators: precedence would change, so no swap.
        let m = muts("x = a + b * c", Lang::Python, MutationClass::ArgShift);
        assert!(!m.iter().any(|l| l.starts_with("x = b")), "{m:?}");
    }

    #[test]
    fn api_swap_flips_paired_calls_and_loop_control() {
        let m = muts(
            "    let fan_out = self.fan_out.min(self.nodes.len()).max(1);",
            Lang::Rust,
            MutationClass::ApiSwap,
        );
        assert!(
            m.contains(&"    let fan_out = self.fan_out.max(self.nodes.len()).max(1);".to_string())
        );
        assert!(
            m.contains(&"    let fan_out = self.fan_out.min(self.nodes.len()).min(1);".to_string())
        );
        let m = muts("x = math.floor(y)", Lang::Python, MutationClass::ApiSwap);
        assert_eq!(m, vec!["x = math.ceil(y)".to_string()]);
        let m = muts("                break;", Lang::Rust, MutationClass::ApiSwap);
        assert_eq!(m, vec!["                continue;".to_string()]);
        // A variable named `max` is not a call.
        let m = muts("let max = 3;", Lang::Rust, MutationClass::ApiSwap);
        assert!(m.is_empty(), "{m:?}");
        // Strings are untouched.
        let m = muts(r#"log("min(a)")"#, Lang::JavaScript, MutationClass::ApiSwap);
        assert!(m.is_empty(), "{m:?}");
    }

    #[test]
    fn ident_swap_exchanges_every_occurrence_of_a_pair() {
        let m = muts(
            "        let tau = self.effective_tau(residual);",
            Lang::Rust,
            MutationClass::IdentSwap,
        );
        assert!(
            m.contains(&"        let residual = self.effective_tau(tau);".to_string()),
            "{m:?}"
        );
        // Both occurrences of each name move together.
        let m = muts("d = b - a + b", Lang::Python, MutationClass::IdentSwap);
        assert!(m.contains(&"d = a - b + a".to_string()), "{m:?}");
        assert!(m.contains(&"b = d - a + d".to_string()), "{m:?}");
        assert_eq!(m.len(), 3, "C(3, 2) pairs: {m:?}");
        // Keywords, `Type::path` segments, macros and capitalised names stay.
        let m = muts("let x = Foo::new(y);", Lang::Rust, MutationClass::IdentSwap);
        assert_eq!(m, vec!["let y = Foo::new(x);".to_string()]);
        let m = muts(
            "println!(\"{a}\", b);",
            Lang::Rust,
            MutationClass::IdentSwap,
        );
        assert!(m.is_empty(), "{m:?}");
        // At most C(6, 2) candidates per line.
        let m = muts(
            "f(a, b, c, d, e, g, h, i, j)",
            Lang::Python,
            MutationClass::IdentSwap,
        );
        assert!(m.len() <= 15, "{}", m.len());
    }

    #[test]
    fn ident_swaps_off_the_blamed_line_come_after_every_other_class() {
        let src = "fn f(a: i32, b: i32) -> i32 {\n    let d = a - b;\n    d + 1\n}\n";
        let plan = plan(src, &PathBuf::from("lib.rs"), "", 1_000);
        let first_swap = plan
            .iter()
            .position(|c| c.class == MutationClass::IdentSwap)
            .expect("an ident swap is planned");
        assert!(
            plan[first_swap..]
                .iter()
                .all(|c| c.class == MutationClass::IdentSwap),
            "{:?}",
            plan.iter().map(|c| &c.label).collect::<Vec<_>>()
        );
        // On a blamed line it is tried with that line's other edits.
        let plan = plan_with(src, "error[E0425]: --> lib.rs:2:13");
        let pos = plan
            .iter()
            .position(|c| c.class == MutationClass::IdentSwap && c.line == 2)
            .unwrap();
        assert!(pos < 10, "{pos}");
    }

    fn plan_with(src: &str, trace: &str) -> Vec<Candidate> {
        plan(src, &PathBuf::from("lib.rs"), trace, 1_000)
    }

    #[test]
    fn enclosing_function_of_an_in_scope_panic_comes_before_the_rest() {
        // The panic is at line 4 (the index); the faulty bound is line 3.
        let src = "fn other(x: usize) -> usize {\n    x + 1\n}\nfn get(xs: &[u8]) -> u8 {\n    let i = xs.len();\n    xs[i]\n}\n";
        let trace = "thread 'tests::t' panicked at src/lib.rs:6:5:\nindex out of bounds: the len is 3 but the index is 3";
        let plan = plan(src, &PathBuf::from("src/lib.rs"), trace, 1_000);
        let fix = "fn other(x: usize) -> usize {\n    x + 1\n}\nfn get(xs: &[u8]) -> u8 {\n    let i = xs.len() - 1;\n    xs[i]\n}\n";
        let pos = plan
            .iter()
            .position(|c| c.source == fix)
            .expect("fix planned");
        let first_other = plan.iter().position(|c| c.line <= 3).unwrap_or(usize::MAX);
        assert!(pos < first_other, "fix at {pos}, other fn at {first_other}");
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
    fn plan_never_mutates_continuation_lines_of_multiline_assertions() {
        // Codex P1 on #247: only the opening line of a multiline assertion
        // carries an oracle marker, so the comparison on a continuation line
        // was mutated (`1 == 2` -> `1 != 2`) and the verifier went green by
        // reversing the assertion. The whole assertion expression is the
        // oracle, including its closing line — even when the blamed line is
        // the continuation itself.
        let src = "def check():\n    assert (\n        1 == 2\n    )\n";
        let p = plan(
            src,
            &PathBuf::from("test_check.py"),
            "test_check.py:3: AssertionError",
            64,
        );
        assert!(
            !p.iter().any(|c| c.source.contains("1 != 2")),
            "continuation line mutated: {:?}",
            p.iter().map(|c| &c.label).collect::<Vec<_>>()
        );
        assert!(
            p.iter().all(|c| c.line != 3 && c.line != 4),
            "{:?}",
            p.iter().map(|c| &c.label).collect::<Vec<_>>()
        );

        // Same shape in Rust: `assert!` opens the expression, the
        // comparison lives on the next line.
        let src = "fn check(a: i32, b: i32) {\n    assert!(\n        a == b\n    );\n}\n";
        let p = plan(
            src,
            &PathBuf::from("lib.rs"),
            "lib.rs:3: assertion failed",
            64,
        );
        assert!(
            !p.iter().any(|c| c.source.contains("a != b")),
            "continuation line mutated: {:?}",
            p.iter().map(|c| &c.label).collect::<Vec<_>>()
        );
        assert!(
            p.iter().all(|c| c.line != 3 && c.line != 4),
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
    fn rust_test_region_is_oracle_and_focus_follows_the_failing_tests_callees() {
        let src = "\
pub fn clamp_add(a: i32, b: i32) -> i32 {
    let s = a + b;
    if s > 100 {
        100
    } else {
        s
    }
}

pub fn unrelated(x: i32) -> i32 {
    x * 3
}

#[cfg(test)]
mod tests {
    #[test]
    fn adds_and_clamps() {
        let r = super::clamp_add(60, 50);
        assert_eq!(r, 100);
    }
}
";
        let path = PathBuf::from("src/lib.rs");
        // The panic blames the assertion (line 19), which is in the test region.
        let trace = "thread 'tests::adds_and_clamps' panicked at src/lib.rs:19:9:\nassertion `left == right` failed\n  left: 110\n right: 100";
        let p = plan(src, &path, trace, 64);
        assert!(!p.is_empty());
        // Nothing in the test region is ever a candidate.
        assert!(
            p.iter().all(|c| c.line < 14),
            "{:?}",
            p.iter().map(|c| c.line).collect::<Vec<_>>()
        );
        // The focus (clamp_add's body, lines 1-8) comes before `unrelated`.
        let first_unrelated = p
            .iter()
            .position(|c| (10..=12).contains(&c.line))
            .unwrap_or(p.len());
        // (Identifier swaps and arg-shifts off the blamed line deliberately come last.)
        let last_focus = p
            .iter()
            .rposition(|c| {
                (1..=8).contains(&c.line)
                    && !matches!(c.class, MutationClass::IdentSwap | MutationClass::ArgShift)
            })
            .unwrap();
        assert!(
            last_focus < first_unrelated,
            "focus lines must precede unrelated ones: {:?}",
            p.iter()
                .map(|c| (c.line, c.label.clone()))
                .collect::<Vec<_>>()
        );
        assert!(
            p.iter().any(|c| c.source.contains("if s >= 100")),
            "expected the comparison flip among candidates"
        );

        // Helpers.
        assert_eq!(
            fn_name("    pub fn clamp_add(a: i32) -> i32 {"),
            Some("clamp_add".into())
        );
        assert_eq!(
            called_identifiers(
                "        let r = super::clamp_add(60, 50); assert_eq!(r, 1); foo::<T>(x)"
            ),
            vec!["clamp_add".to_string()]
        );
        let lines: Vec<&str> = src.split_inclusive('\n').collect();
        assert_eq!(oracle_ranges(&lines, Lang::Rust), vec![(13, 20)]);
        assert!(oracle_ranges(&lines, Lang::Python).is_empty());
        assert_eq!(body_end(&lines, 0, 13), 7);
    }

    #[test]
    fn oracle_excludes_only_the_cfg_test_item_not_the_rest_of_the_file() {
        // The layout Codex flagged in cvm_store.rs: a test-only import and a
        // test-only helper method precede production code. Everything after
        // them must stay repairable.
        let src = "\
#[cfg(test)]
use std::collections::HashMap;

pub struct Store {
    n: usize,
}

impl Store {
    #[cfg(test)]
    pub(crate) fn len_for_test(&self) -> usize {
        self.n
    }
}

pub fn evicted_bytes(a: usize, b: usize) -> usize {
    a - b
}

#[cfg(test)]
mod tests {
    #[test]
    fn sums() {
        assert_eq!(super::evicted_bytes(2, 3), 5);
    }
}
";
        let lines: Vec<&str> = src.split_inclusive('\n').collect();
        let ranges = oracle_ranges(&lines, Lang::Rust);
        assert_eq!(ranges, vec![(0, 1), (8, 11), (18, 24)], "{ranges:?}");
        let trace = "thread 'tests::sums' panicked at src/lib.rs:23:9:\nassertion `left == right` failed\n  left: 18446744073709551615";
        let p = plan(src, &PathBuf::from("src/lib.rs"), trace, 64);
        assert!(
            p.iter().any(|c| c.source.contains("    a + b\n")),
            "production code after the test-only items must be repairable: {:?}",
            p.iter().map(|c| c.label.clone()).collect::<Vec<_>>()
        );
        for c in &p {
            assert!(
                !in_ranges(&ranges, c.line - 1),
                "edited an oracle line: {}",
                c.label
            );
        }
        // The fix is reached through callee focus, ahead of unrelated lines.
        assert_eq!(
            p[0].line, 16,
            "first candidate should be inside evicted_bytes: {}",
            p[0].label
        );
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
