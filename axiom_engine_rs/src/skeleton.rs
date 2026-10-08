//! skeleton.rs — Claude-READABLE compression of heavy context.
//!
//! The neural fingerprint (vocab indices + Frobenius norms) is meaningful to the
//! Axiom model but opaque to a *different* model like Claude — shipping it
//! upstream wastes tokens and degrades answers. Instead we ship a compact
//! structural *skeleton*: doc summary, imports, and declaration signatures with
//! bodies elided. It is small (≈80% smaller than the source) AND readable, so
//! Claude can answer accurately from real signatures.
//!
//! Axiom's neural capability is untouched: the TTT session already absorbed the
//! heavy context into its fast-weights (adapt_session), and the drift signal
//! (recall_norm + state_hash) rides along as tiny attributes on the digest.

use std::cmp::Ordering;
use std::collections::HashSet;

use tree_sitter::{Node, Parser, Query};

/// Visibility / async prefixes stripped before testing for a declaration.
const VIS_PREFIXES: [&str; 9] = [
    "pub ",
    "public ",
    "private ",
    "protected ",
    "export ",
    "default ",
    "async ",
    "static ",
    "final ",
];

/// Keywords that begin a declaration whose signature we keep. Covers Rust, Go,
/// Python, JS/TS, Java/C#, C/C++.
const DECL_KEYWORDS: [&str; 18] = [
    "fn ",
    "func ",
    "def ",
    "function ",
    "struct ",
    "enum ",
    "trait ",
    "impl ",
    "impl<",
    "interface ",
    "class ",
    "type ",
    "const ",
    "static ",
    "mod ",
    "namespace ",
    "package ",
    "module ",
];

/// Control-flow keywords that can also end a line with `{` — never signatures.
const CONTROL_KEYWORDS: [&str; 13] = [
    "if ", "if(", "for ", "for(", "while ", "while(", "switch ", "switch(", "match ", "else",
    "do ", "try ", "catch",
];

/// Heuristic: a method/function signature with no leading keyword (JS/TS class
/// methods, Java/C# methods). The line opens a block `{`, has a parameter list
/// `(...)`, and is not control flow. Keeps brace-language methods that the
/// keyword list alone would miss.
fn looks_like_signature(t: &str) -> bool {
    let trimmed = t.trim_end();
    if !trimmed.ends_with('{') {
        return false;
    }
    if !t.contains('(') || !t.contains(')') {
        return false;
    }
    if t.starts_with('}') || t.starts_with("//") || t.starts_with('*') || t.starts_with('@') {
        return false;
    }
    !CONTROL_KEYWORDS.iter().any(|c| t.starts_with(c))
}

/// Largest char boundary <= idx (safe string slicing).
fn floor_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Smallest char boundary >= idx (safe string slicing).
fn ceil_boundary(s: &str, mut idx: usize) -> usize {
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

/// Prose / non-code fallback: a signature skeleton would destroy plain text, so
/// keep a head + tail excerpt instead. Still compresses large prose heavily.
fn prose_excerpt(text: &str, head_budget: usize, tail_budget: usize) -> String {
    if text.len() <= head_budget + tail_budget + 64 {
        return text.trim().to_string();
    }
    let head_end = floor_boundary(text, head_budget);
    let tail_start = ceil_boundary(text, text.len().saturating_sub(tail_budget));
    let elided = tail_start.saturating_sub(head_end);
    format!(
        "{}\n… [{elided} chars of prose elided] …\n{}",
        text[..head_end].trim_end(),
        text[tail_start..].trim_start()
    )
}

fn is_import(t: &str) -> bool {
    ["use ", "import ", "from ", "#include", "require("]
        .iter()
        .any(|p| t.starts_with(p))
}

fn is_doc(t: &str) -> bool {
    t.starts_with("///")
        || t.starts_with("//!")
        || t.starts_with("# ")
        || t.starts_with("\"\"\"")
        || t.starts_with("/**")
}

fn rust_ast_body(heavy: &str, max_doc_lines: usize) -> Option<(String, usize)> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .ok()?;
    let tree = parser.parse(heavy, None)?;
    let language = tree_sitter_rust::LANGUAGE.into();
    let query = Query::new(
        &language,
        r#"
        (use_declaration) @import
        (function_item) @decl
        (struct_item) @decl
        (enum_item) @decl
        (trait_item) @decl
        (impl_item) @decl
        (type_item) @decl
        (const_item) @decl
        (static_item) @decl
        (mod_item) @decl
        (macro_definition) @decl
        (line_comment) @doc
        (block_comment) @doc
        "#,
    )
    .ok()?;

    let _capture_names = query.capture_names();
    let mut captured: Vec<(usize, &'static str, Node)> = Vec::new();
    collect_rust_captures(tree.root_node(), &mut captured);
    captured.sort_by_key(|(start, _, _)| *start);

    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut doc_budget = max_doc_lines;
    let mut elided = 0usize;
    let mut decl_count = 0usize;

    for (_, capture_name, node) in captured {
        if capture_name == "decl" {
            decl_count += 1;
        }
        let rendered = match capture_name {
            "import" => node_text(node, heavy)
                .map(str::trim_end)
                .map(str::to_string),
            "doc" => {
                let text = node_text(node, heavy)?.trim();
                if !is_doc(text) || doc_budget == 0 {
                    elided += 1;
                    None
                } else {
                    doc_budget -= 1;
                    Some(text.to_string())
                }
            }
            "decl" => render_rust_decl(node, heavy),
            _ => None,
        };
        if let Some(line) = rendered {
            push_unique_structural(&line, &mut out, &mut seen, &mut elided);
        } else {
            elided += 1;
        }
    }

    if out.is_empty() {
        return None;
    }
    if decl_count > 0 || elided > 0 || tree.root_node().has_error() {
        out.push(format!(
            "// … {} implementation lines elided …",
            elided.max(1)
        ));
    }
    Some((out.join("\n"), elided))
}

fn collect_rust_captures<'tree>(
    node: Node<'tree>,
    captured: &mut Vec<(usize, &'static str, Node<'tree>)>,
) {
    let capture = match node.kind() {
        "use_declaration" => Some("import"),
        "line_comment" | "block_comment" => Some("doc"),
        "function_item" | "struct_item" | "enum_item" | "trait_item" | "impl_item"
        | "type_item" | "const_item" | "static_item" | "mod_item" | "macro_definition" => {
            Some("decl")
        }
        _ => None,
    };
    if let Some(name) = capture {
        captured.push((node.start_byte(), name, node));
    }
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            collect_rust_captures(child, captured);
        }
    }
}

fn node_text<'a>(node: Node, source: &'a str) -> Option<&'a str> {
    node.utf8_text(source.as_bytes()).ok()
}

fn render_rust_decl(node: Node, source: &str) -> Option<String> {
    match node.kind() {
        "function_item" => render_until_child(node, source, &["block"], " { … }"),
        "impl_item" | "trait_item" | "mod_item" => {
            render_until_child(node, source, &["declaration_list", "block"], " { … }")
        }
        "struct_item" | "enum_item" | "macro_definition" => render_until_child(
            node,
            source,
            &[
                "field_declaration_list",
                "ordered_field_declaration_list",
                "enum_variant_list",
                "token_tree",
            ],
            " { … }",
        ),
        "const_item" | "static_item" => render_until_equals(node, source),
        _ => node_text(node, source).map(|s| s.trim_end().to_string()),
    }
}

fn render_until_child(
    node: Node,
    source: &str,
    child_kinds: &[&str],
    suffix: &str,
) -> Option<String> {
    let child = first_named_child_kind(node, child_kinds);
    let end = child
        .map(|n| n.start_byte())
        .unwrap_or_else(|| node.end_byte());
    let start = node.start_byte();
    let prefix = source.get(start..end)?.trim_end();
    Some(format!(
        "{}{}",
        prefix.trim_end_matches('{').trim_end(),
        suffix
    ))
}

fn first_named_child_kind<'tree>(node: Node<'tree>, child_kinds: &[&str]) -> Option<Node<'tree>> {
    for i in 0..node.named_child_count() {
        let child = node.named_child(i as u32)?;
        if child_kinds.iter().any(|kind| child.kind() == *kind) {
            return Some(child);
        }
    }
    None
}

/// Detect a Python-style module-level constant: zero-indentation
/// `UPPER_SNAKE_CASE = ...` assignment. Returns the constant name.
///
/// Rationale (P0 safety): these carry bug-relevant values (e.g.
/// `RISK_ORDER = {"low": 3, ...}`) but match no declaration keyword, so the
/// generic extractor silently drops them. The digest then hides the defect
/// entirely -- worse than an elided body.
fn is_module_constant(line: &str, trimmed: &str) -> Option<String> {
    // Must be at module level (no leading whitespace).
    if line.len() != trimmed.len() {
        return None;
    }
    // Must be an assignment, not a comparison or annotation.
    let eq = trimmed.find('=')?;
    // Exclude `==`, `!=`, `<=`, `>=`, `=>`, `:=`.
    let bytes = trimmed.as_bytes();
    if eq > 0 {
        match bytes[eq - 1] {
            b'=' | b'!' | b'<' | b'>' | b':' => return None,
            _ => {}
        }
    }
    if bytes.get(eq + 1) == Some(&b'=') {
        return None;
    }
    let name = trimmed[..eq].trim_end();
    // UPPER_SNAKE_CASE: starts uppercase, rest uppercase/digits/underscore,
    // at least 2 chars to avoid single-letter noise.
    if name.len() < 2
        || !name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        || !name.chars().next().unwrap().is_ascii_uppercase()
    {
        return None;
    }
    Some(name.to_string())
}

fn render_until_equals(node: Node, source: &str) -> Option<String> {
    let text = node_text(node, source)?.trim_end();
    if let Some(eq) = text.find('=') {
        return Some(format!("{} = …;", text[..eq].trim_end()));
    }
    Some(text.to_string())
}

/// Strip stacked visibility/async prefixes (e.g. "pub async fn") so keyword
/// tests see the real declaration kind.
fn strip_decl_prefix(mut s: &str) -> &str {
    loop {
        let mut changed = false;
        for p in VIS_PREFIXES {
            if let Some(rest) = s.strip_prefix(p) {
                s = rest.trim_start();
                changed = true;
            }
        }
        // pub(crate) / pub(super) etc.
        if let Some(rest) = s.strip_prefix("pub(") {
            if let Some(i) = rest.find(')') {
                s = rest[i + 1..].trim_start();
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    s
}

fn is_decl(t: &str) -> bool {
    let s = strip_decl_prefix(t);
    DECL_KEYWORDS.iter().any(|k| s.starts_with(k))
}

/// Function/method declarations: the units diagnostic elision filters.
/// Classes, structs, enums, etc. stay dumb signatures even in diagnostic mode
/// (their interesting content lives in the methods, which are handled
/// separately — this also avoids duplicating method bodies inside the
/// class/impl rendering).
fn is_function_decl(t: &str) -> bool {
    let s = strip_decl_prefix(t);
    const FN_KEYWORDS: [&str; 4] = ["fn ", "func ", "def ", "function "];
    if FN_KEYWORDS.iter().any(|k| s.starts_with(k)) {
        return true;
    }
    looks_like_signature(t)
}

/// True if `t` starts with keyword `kw` followed by a word boundary
/// (so "except" doesn't match "exceptional").
fn starts_with_kw(t: &str, kw: &str) -> bool {
    if let Some(rest) = t.strip_prefix(kw) {
        rest.is_empty() || !is_ident_byte(rest.as_bytes()[0])
    } else {
        false
    }
}

/// Build the compact readable digest from heavy context text.
///
/// `original_tokens`, `recall_norm`, and `state_hash` ride along as attributes so
/// Axiom keeps its drift / session-continuity signal at negligible token cost.
pub fn build_digest(
    heavy: &str,
    session_id: &str,
    original_tokens: usize,
    recall_norm: f32,
    state_hash: &str,
    max_doc_lines: usize,
) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut seen_structural: HashSet<String> = HashSet::new();
    let mut doc_budget = max_doc_lines as i32;
    let mut elided = 0usize;
    let mut code_lines = 0usize; // imports + declarations + brace-method signatures

    if let Some((body, _)) = rust_ast_body(heavy, max_doc_lines) {
        return format!(
            "<axiom_context_digest session=\"{session_id}\" kind=\"structural-skeleton\" \
original_tokens=\"{original_tokens}\" recall_norm=\"{recall_norm:.3}\" state=\"{state_hash}\">\n\
# Lossy digest of elided heavy context. For code: signatures kept, bodies dropped.\n\
# Ask Axiom to expand a named symbol if you need its body.\n\
state_hash={state_hash}\n\
{body}\n\
</axiom_context_digest>"
        );
    }

    for line in heavy.lines() {
        let t = line.trim_start();
        if t.is_empty() {
            continue;
        }
        if is_import(t) {
            push_unique_structural(line.trim_end(), &mut out, &mut seen_structural, &mut elided);
            code_lines += 1;
        } else if is_decl(t) || looks_like_signature(t) {
            let sig = line.split('{').next().unwrap_or(line).trim_end();
            let suffix = if line.contains('{') { " { … }" } else { "" };
            let structural = format!("{sig}{suffix}");
            push_unique_structural(&structural, &mut out, &mut seen_structural, &mut elided);
            code_lines += 1;
        } else if is_doc(t) && doc_budget > 0 {
            let before = seen_structural.len();
            push_unique_structural(line.trim_end(), &mut out, &mut seen_structural, &mut elided);
            if seen_structural.len() > before {
                doc_budget -= 1;
            }
        } else {
            elided += 1;
        }
    }

    // If nothing structural was found, this is prose/data — a signature skeleton
    // would erase it. Keep a head+tail excerpt instead so meaning survives.
    let (kind, body) = if code_lines == 0 {
        ("prose-excerpt", prose_excerpt(heavy, 1200, 500))
    } else {
        if elided > 0 {
            out.push(format!("// … {elided} implementation lines elided …"));
        }
        ("structural-skeleton", out.join("\n"))
    };

    format!(
        "<axiom_context_digest session=\"{session_id}\" kind=\"{kind}\" \
original_tokens=\"{original_tokens}\" recall_norm=\"{recall_norm:.3}\" state=\"{state_hash}\">\n\
# Lossy digest of elided heavy context. For code: signatures kept, bodies dropped.\n\
# Ask Axiom to expand a named symbol if you need its body.\n\
state_hash={state_hash}\n\
{body}\n\
</axiom_context_digest>"
    )
}

/// Build just the skeleton body (no XML wrapper) for direct human/agent
/// consumption. Returns the structural skeleton: imports, doc comments, and
/// declaration signatures with bodies elided. For prose, returns a head+tail
/// excerpt. This is what `axiom skeleton` prints per file.
pub fn skeleton_body(heavy: &str, max_doc_lines: usize) -> String {
    if let Some((body, _)) = rust_ast_body(heavy, max_doc_lines) {
        return body;
    }

    let mut out: Vec<String> = Vec::new();
    let mut seen_structural: HashSet<String> = HashSet::new();
    let mut doc_budget = max_doc_lines as i32;
    let mut elided = 0usize;
    let mut code_lines = 0usize;

    for line in heavy.lines() {
        let t = line.trim_start();
        if t.is_empty() {
            continue;
        }
        if is_import(t) {
            push_unique_structural(line.trim_end(), &mut out, &mut seen_structural, &mut elided);
            code_lines += 1;
        } else if is_decl(t) || looks_like_signature(t) {
            let sig = line.split('{').next().unwrap_or(line).trim_end();
            let suffix = if line.contains('{') { " { … }" } else { "" };
            let structural = format!("{sig}{suffix}");
            push_unique_structural(&structural, &mut out, &mut seen_structural, &mut elided);
            code_lines += 1;
        } else if is_doc(t) && doc_budget > 0 {
            let before = seen_structural.len();
            push_unique_structural(line.trim_end(), &mut out, &mut seen_structural, &mut elided);
            if seen_structural.len() > before {
                doc_budget -= 1;
            }
        } else {
            elided += 1;
        }
    }

    if code_lines == 0 {
        prose_excerpt(heavy, 1200, 500)
    } else {
        if elided > 0 {
            out.push(format!("// … {elided} implementation lines elided …"));
        }
        out.join("\n")
    }
}

fn push_unique_structural(
    line: &str,
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
    elided: &mut usize,
) {
    let normalized = line.trim();
    if normalized.is_empty() {
        return;
    }
    if seen.insert(normalized.to_string()) {
        out.push(line.to_string());
    } else {
        *elided += 1;
    }
}

/// Leading-whitespace width of a line (tabs count as 4) — for indent matching.
fn indent_of(line: &str) -> usize {
    let mut w = 0;
    for ch in line.chars() {
        match ch {
            ' ' => w += 1,
            '\t' => w += 4,
            _ => break,
        }
    }
    w
}

/// True if `name` appears in `line` as a whole identifier (not a substring of a
/// larger identifier).
fn contains_symbol(line: &str, name: &str) -> bool {
    let bytes = line.as_bytes();
    let nb = name.as_bytes();
    if nb.is_empty() {
        return false;
    }
    let mut i = 0;
    while let Some(pos) = line[i..].find(name) {
        let start = i + pos;
        let end = start + name.len();
        let before_ok = start == 0 || !is_ident_byte(bytes[start - 1]);
        let after_ok = end >= bytes.len() || !is_ident_byte(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        i = end;
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Capture a declaration's full block starting at `start`. Brace-delimited blocks
/// are brace-matched; Python-style `:` headers use indentation; otherwise the
/// single declaration line is returned. (Braces inside strings/comments are not
/// specially handled — a known approximation.)
fn capture_block(lines: &[&str], start: usize) -> String {
    let first = lines[start];
    if first.contains('{') {
        let mut depth: i32 = 0;
        let mut out: Vec<&str> = Vec::new();
        for line in &lines[start..] {
            out.push(line);
            for ch in line.chars() {
                if ch == '{' {
                    depth += 1;
                } else if ch == '}' {
                    depth -= 1;
                }
            }
            if depth <= 0 && line.contains('}') {
                break;
            }
        }
        out.join("\n")
    } else if first.trim_end().ends_with(':') {
        let base = indent_of(first);
        let mut out: Vec<&str> = vec![first];
        for line in &lines[start + 1..] {
            if line.trim().is_empty() {
                out.push(line);
                continue;
            }
            if indent_of(line) <= base {
                break;
            }
            out.push(line);
        }
        // Trim trailing blank lines.
        while out.last().map(|l| l.trim().is_empty()).unwrap_or(false) {
            out.pop();
        }
        out.join("\n")
    } else {
        first.to_string()
    }
}

/// Extract the full declaration + body of `name` from `source`. Returns None when
/// no declaration of that symbol is found. This is the retrieval half of the
/// skeleton round-trip: the digest drops bodies, and the proxy can expand any one
/// back on demand from the stored source.
pub fn expand_symbol(source: &str, name: &str) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if (is_decl(t) || looks_like_signature(t)) && contains_symbol(line, name) {
            return Some(capture_block(&lines, i));
        }
        // Module-level constants (e.g. `RISK_ORDER = {...}`): the digest
        // shows only `RISK_ORDER = …`, so expansion must return the full
        // assignment including a multi-line value.
        if let Some(const_name) = is_module_constant(line, t) {
            if const_name == name {
                return Some(capture_constant_block(&lines, i));
            }
        }
    }
    None
}

/// Capture a module-level constant's full assignment, including multi-line
/// values (dicts, lists). Stops at the first subsequent line at zero
/// indentation that is not a continuation of the value.
fn capture_constant_block(lines: &[&str], start: usize) -> String {
    let mut out = vec![lines[start].to_string()];
    // Track bracket depth to handle multi-line dict/list values.
    let mut depth = 0i32;
    for c in lines[start].chars() {
        match c {
            '{' | '[' | '(' => depth += 1,
            '}' | ']' | ')' => depth -= 1,
            _ => {}
        }
    }
    for line in &lines[start + 1..] {
        // A new zero-indentation line outside any brackets ends the value.
        if depth <= 0 && !line.starts_with(' ') && !line.starts_with('\t') && !line.trim().is_empty() {
            break;
        }
        for c in line.chars() {
            match c {
                '{' | '[' | '(' => depth += 1,
                '}' | ']' | ')' => depth -= 1,
                _ => {}
            }
        }
        out.push(line.to_string());
        if depth <= 0 {
            break;
        }
    }
    out.join("\n")
}

// ---------------------------------------------------------------------------
// PageRank-ranked skeletonization.
//
// `build_digest` keeps every declaration equally: under a token budget the
// output truncates arbitrarily. The ranked variant below builds an intra-file
// reference graph (symbol A -> symbol B when A's body mentions B) and orders
// declarations by PageRank, so a budget keeps the most load-bearing symbols
// first and the skeleton degrades gracefully instead of cutting off mid-file.
// ---------------------------------------------------------------------------

/// One extracted declaration: display signature, full body text (reference
/// scanning only — never rendered), and its PageRank importance score.
struct RankedSymbol {
    name: String,
    signature: String,
    body: String,
    score: f64,
}

/// Approximate token count, matching the Python prototype's chars/4 rule.
fn approx_tokens(s: &str) -> usize {
    (s.len() / 4).max(1)
}

/// PageRank over `adj` (adjacency lists, edge i -> j). Dangling nodes
/// distribute their rank uniformly. Standard power iteration.
fn pagerank(adj: &[Vec<usize>], damping: f64, iterations: usize) -> Vec<f64> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    let mut rank = vec![1.0 / nf; n];
    let teleport = (1.0 - damping) / nf;
    for _ in 0..iterations {
        let mut next = vec![teleport; n];
        for (i, outs) in adj.iter().enumerate() {
            if outs.is_empty() {
                let share = damping * rank[i] / nf;
                for r in next.iter_mut() {
                    *r += share;
                }
            } else {
                let share = damping * rank[i] / outs.len() as f64;
                for &j in outs {
                    next[j] += share;
                }
            }
        }
        rank = next;
    }
    rank
}

/// Edge i -> j when symbol i's body mentions symbol j's name as a whole
/// identifier (heuristic reference extraction; v1, no type resolution).
fn build_reference_graph(symbols: &[RankedSymbol]) -> Vec<Vec<usize>> {
    let n = symbols.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i == j || symbols[j].name.is_empty() {
                continue;
            }
            if contains_symbol(&symbols[i].body, &symbols[j].name) {
                adj[i].push(j);
            }
        }
    }
    adj
}

/// Best-effort declaration name from a tree-sitter Rust node: the grammar's
/// `name` field when present, else the first identifier-like direct child.
/// For `impl` blocks, prefers the implemented type (`impl Trait for Type`
/// yields `Type`).
fn rust_decl_name(node: Node, source: &str) -> Option<String> {
    if let Some(n) = node.child_by_field_name("name") {
        if let Some(t) = node_text(n, source) {
            let t = t.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    // Direct identifier-like children, in source order, stopping at the body.
    let mut cands: Vec<String> = Vec::new();
    for i in 0..node.named_child_count() {
        let c = node.named_child(i as u32)?;
        match c.kind() {
            "declaration_list" | "block" | "field_declaration_list"
            | "ordered_field_declaration_list" | "enum_variant_list" | "token_tree" => break,
            "identifier" | "type_identifier" => {
                if let Some(t) = node_text(c, source) {
                    let t = t.trim();
                    if !t.is_empty() {
                        cands.push(t.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    if node.kind() == "impl_item" {
        // `impl Trait for Type` — the type is what other code references.
        cands.pop()
    } else {
        cands.into_iter().next()
    }
}

/// Extract (imports, declarations) from Rust via tree-sitter.
fn extract_symbols_rust(heavy: &str) -> (Vec<String>, Vec<RankedSymbol>) {
    let mut imports = Vec::new();
    let mut symbols = Vec::new();
    let mut parser = Parser::new();
    let Ok(()) = parser.set_language(&tree_sitter_rust::LANGUAGE.into()) else {
        return (imports, symbols);
    };
    let Some(tree) = parser.parse(heavy, None) else {
        return (imports, symbols);
    };
    let mut captured: Vec<(usize, &'static str, Node)> = Vec::new();
    collect_rust_captures(tree.root_node(), &mut captured);
    captured.sort_by_key(|(start, _, _)| *start);
    // Key dedup on node start byte, not signature text: methods with
    // identical signatures in different impl blocks are distinct symbols.
    let mut seen: HashSet<usize> = HashSet::new();
    for (_, kind, node) in captured {
        if kind == "import" {
            if let Some(t) = node_text(node, heavy) {
                let t = t.trim_end().to_string();
                if !imports.contains(&t) {
                    imports.push(t);
                }
            }
        } else if kind == "decl" {
            let Some(signature) = render_rust_decl(node, heavy) else {
                continue;
            };
            if !seen.insert(node.start_byte()) {
                continue;
            }
            let name = rust_decl_name(node, heavy).unwrap_or_default();
            let body = node_text(node, heavy).unwrap_or("").to_string();
            symbols.push(RankedSymbol {
                name,
                signature,
                body,
                score: 0.0,
            });
        }
    }
    (imports, symbols)
}

/// Heuristic declaration name from a source line, mirroring `is_decl`'s
/// prefix stripping. `fn add(a: i32)` -> `add`; `class Foo:` -> `Foo`.
fn extract_decl_name(line: &str) -> Option<String> {
    let mut s = line.trim_start();
    // Strip stacked visibility/async prefixes (same as is_decl).
    loop {
        let mut changed = false;
        for p in VIS_PREFIXES {
            if let Some(rest) = s.strip_prefix(p) {
                s = rest.trim_start();
                changed = true;
            }
        }
        if let Some(rest) = s.strip_prefix("pub(") {
            if let Some(i) = rest.find(')') {
                s = rest[i + 1..].trim_start();
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Case 1: parameter list present — name is the identifier before '('.
    if let Some(paren) = s.find('(') {
        let before = s[..paren].trim_end();
        if let Some(tok) = before.split_whitespace().last() {
            let name = tok.trim_matches(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
            if !name.is_empty() && !is_decl_keyword(name) {
                return Some(name.to_string());
            }
        }
        // Fell through: the pre-paren token was a keyword (e.g. `const f: fn()`).
    }
    // Case 2: no parens, or keyword fallback — first identifier after the
    // declaration keyword. Skip `impl<...>` generics.
    if let Some(rest) = s.strip_prefix("impl<") {
        if let Some(gt) = rest.find('>') {
            s = rest[gt + 1..].trim_start();
        }
    }
    for kw in DECL_KEYWORDS {
        if let Some(rest) = s.strip_prefix(kw) {
            let mut toks = rest.split_whitespace();
            let mut tok = toks.next()?;
            if tok.starts_with('<') {
                while !tok.contains('>') {
                    tok = toks.next()?;
                }
                tok = toks.next()?;
            }
            let name = tok.trim_matches(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

/// True if `name` is itself a declaration keyword (so `const f: fn()` yields
/// `f`, not `fn`).
fn is_decl_keyword(name: &str) -> bool {
    DECL_KEYWORDS
        .iter()
        .any(|k| k.trim_end() == name || k.trim_end_matches('<') == name)
}

/// Extract (imports, declarations) with the language-agnostic line heuristic
/// used by `build_digest`'s non-Rust path. Bodies come from `capture_block`.
fn extract_symbols_generic(heavy: &str) -> (Vec<String>, Vec<RankedSymbol>) {
    let mut imports = Vec::new();
    let mut symbols = Vec::new();
    let lines: Vec<&str> = heavy.lines().collect();
    let mut seen: HashSet<String> = HashSet::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if t.is_empty() {
            continue;
        }
        if is_import(t) {
            let s = line.trim_end().to_string();
            if !imports.contains(&s) {
                imports.push(s);
            }
        } else if is_decl(t) || looks_like_signature(t) {
            let base = line.split('{').next().unwrap_or(line).trim_end();
            let signature = if line.contains('{') {
                format!("{base} {{ … }}")
            } else {
                base.to_string()
            };
            if !seen.insert(signature.clone()) {
                continue;
            }
            let name = extract_decl_name(line).unwrap_or_default();
            let body = capture_block(&lines, i);
            symbols.push(RankedSymbol {
                name,
                signature,
                body,
                score: 0.0,
            });
        } else if let Some(const_name) = is_module_constant(line, t) {
            // Safety (P0): Python-style UPPER_SNAKE_CASE module constants
            // (e.g. `RISK_ORDER = {...}`) carry bug-relevant values but match
            // no declaration keyword. Capture the name with an elided-value
            // stub so it stays visible under a token budget.
            let signature = format!("{const_name} = …");
            if !seen.insert(signature.clone()) {
                continue;
            }
            symbols.push(RankedSymbol {
                name: const_name,
                signature,
                body: String::new(),
                score: 0.0,
            });
        }
    }
    (imports, symbols)
}

/// Ranked skeletonization: declarations ordered by PageRank importance over the
/// intra-file reference graph, so a token budget keeps the most load-bearing
/// symbols first.
///
/// `language` selects the extractor: `"rust"` (or `"rs"`) uses tree-sitter;
/// anything else uses the language-agnostic line heuristic. `token_budget`
/// caps approximate output tokens (chars/4); `None` keeps everything.
/// Imports always lead (deduped, source order); declarations follow sorted by
/// score descending. At least one line is emitted whenever there is content.
///
/// This is additive: `build_digest` and `expand_symbol` are untouched.
pub fn skeletonize_ranked(
    text: &str,
    language: &str,
    token_budget: Option<usize>,
) -> String {
    let (imports, mut symbols) = if language.eq_ignore_ascii_case("rust")
        || language.eq_ignore_ascii_case("rs")
    {
        extract_symbols_rust(text)
    } else {
        extract_symbols_generic(text)
    };

    if !symbols.is_empty() {
        let adj = build_reference_graph(&symbols);
        let scores = pagerank(&adj, 0.85, 20);
        for (sym, sc) in symbols.iter_mut().zip(scores.iter()) {
            sym.score = *sc;
        }
        symbols.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
        });
    }

    // No code structure found: fall back to a prose excerpt like build_digest.
    if imports.is_empty() && symbols.is_empty() {
        return prose_excerpt(text, 1200, 500);
    }

    let mut out: Vec<String> = Vec::new();
    let mut used = 0usize;
    let mut elided = 0usize;
    // Cap imports at 25% of budget so declarations (the ranked content)
    // always get room. Without a budget, emit all imports.
    let import_budget = token_budget.map(|b| b / 4);
    let mut import_used = 0usize;
    for imp in &imports {
        let t = approx_tokens(imp);
        if let Some(ibudget) = import_budget {
            // Skip oversized imports even if first: don't let one long
            // `use` line consume the budget needed for ranked declarations.
            if import_used + t > ibudget {
                elided += 1;
                continue;
            }
            import_used += t;
        }
        if let Some(budget) = token_budget {
            if used + t > budget && !out.is_empty() {
                elided += 1;
                continue;
            }
            used += t;
        }
        out.push(imp.clone());
    }
    for sym in &symbols {
        let t = approx_tokens(&sym.signature);
        if let Some(budget) = token_budget {
            if used + t > budget && !out.is_empty() {
                // Safety (P0): never let a symbol vanish entirely. Emit a
                // name-only stub so the agent knows it exists and can expand
                // it. An agent that sees `// charge (… elided …)` knows to
                // ask for the body; an agent that never sees `charge` cannot.
                // The stub is ~1-2 tokens vs the full signature.
                if !sym.name.is_empty() {
                    let stub = format!("// {} (… body elided …)", sym.name);
                    let stub_t = approx_tokens(&stub);
                    if used + stub_t <= budget {
                        out.push(stub);
                        used += stub_t;
                        continue;
                    }
                }
                elided += 1;
                continue;
            }
            used += t;
        }
        out.push(sym.signature.clone());
    }
    if elided > 0 {
        out.push(format!(
            "// … {elided} lower-ranked lines elided by token budget …"
        ));
    }
    out.join("\n")
}

// ---------------------------------------------------------------------------
// Diagnostic elision.
//
// Dumb skeletons drop every function body, which erases diagnostic signal:
// an agent fixing a bug inside a body must read the full file anyway. Diagnostic
// mode keeps the lines that carry diagnostic signal and elides the rest:
//
//   1. Error handling paths: try/except/catch/finally, raise/throw,
//      panic!/unwrap()/expect()/unreachable!, Rust's `?` operator, Err.
//   2. Boundary conditions: comparisons (<,>,<=,>=,==,!=) on values.
//   3. Suspicious patterns: TODO/FIXME/XXX/HACK comments, bare `except:`,
//      asserts, todo!/unimplemented!.
//   4. Complex conditionals: if/while with 3+ conditions (and/or/&&/||).
//   5. Return statements (and Rust tail expressions): a function's contract.
//
// A kept line that opens a block keeps its whole block (the except body, the
// if body, the returned dict literal). Target: 40-60% of source — noticeably
// larger than a dumb skeleton (~20%), still far from full source.
// ---------------------------------------------------------------------------

/// True when the trimmed line carries diagnostic signal worth keeping.
fn is_interesting_line(t: &str) -> bool {
    is_error_handling(t)
        || is_suspicious(t)
        || is_boundary_condition(t)
        || is_complex_conditional(t)
        || is_return_stmt(t)
}

fn is_error_handling(t: &str) -> bool {
    // Python
    if starts_with_kw(t, "try") || starts_with_kw(t, "except") || starts_with_kw(t, "finally") {
        return true;
    }
    if starts_with_kw(t, "raise") {
        return true;
    }
    // JS/TS/Java/C#
    if starts_with_kw(t, "catch") || starts_with_kw(t, "throw") {
        return true;
    }
    // Rust
    if t.contains("panic!") || t.contains("unreachable!") {
        return true;
    }
    if t.contains(".unwrap()") || t.contains(".expect(") {
        return true;
    }
    if t.contains("bail!") || t.contains("ensure!") {
        return true;
    }
    if t.starts_with("Err(") || t.contains("=> Err(") || t.contains("return Err(") {
        return true;
    }
    // Rust `?` try operator: a line ending in `?`. (A `?` inside a comment is
    // a harmless false positive; other languages don't end lines with `?`.)
    if t.ends_with('?') || t.ends_with("?,") || t.ends_with("?;") {
        return true;
    }
    false
}

fn is_suspicious(t: &str) -> bool {
    for m in [
        "TODO",
        "FIXME",
        "XXX",
        "HACK",
        "BUG",
        "KLUDGE",
        "WORKAROUND",
    ] {
        if contains_symbol(t, m) {
            return true;
        }
    }
    // Bare `except:` swallows everything — always worth a look.
    if t == "except:" {
        return true;
    }
    if starts_with_kw(t, "assert") || t.starts_with("debug_assert") {
        return true;
    }
    if t.contains("todo!") || t.contains("unimplemented!") {
        return true;
    }
    false
}

/// If `b[i]` is `<` opening a generic type argument list (e.g. `Vec<T>`,
/// `HashMap<String, Vec<u8>>`, `foo::<T>`), return the index of the matching
/// `>`. None otherwise. Only type-like content may appear between the
/// brackets; anything else (a `;`, `{`, …) means this is not a generic.
fn generic_close(b: &[u8], i: usize) -> Option<usize> {
    // A generic `<` follows an identifier, a nested `>`, or `::` (turbofish).
    let prev = if i > 0 { b[i - 1] } else { return None };
    if !(prev.is_ascii_alphanumeric() || prev == b'_' || prev == b'>' || prev == b':') {
        return None;
    }
    let mut depth = 0i32;
    let mut j = i;
    while j < b.len() {
        match b[j] {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    return Some(j);
                }
            }
            b if b.is_ascii_alphanumeric()
                || b == b'_'
                || b == b' '
                || b == b'\t'
                || b == b','
                || b == b':'
                || b == b'&'
                || b == b'\''
                || b == b'['
                || b == b']'
                || b == b'('
                || b == b')'
                || b == b'*'
                || b == b'+'
                || b == b'.'
                || b == b'?'
                || b == b'!'
                || b == b'|' => {}
            _ => return None,
        }
        j += 1;
    }
    None
}

/// True if the line contains a `<` or `>` that is a comparison, not part of
/// `->`, `=>`, `>>`, `<<`, `>=`, `<=`, `<-`, or a generic type argument list
/// like `Vec<T>`.
fn has_bare_comparison(t: &str) -> bool {
    let b = t.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'<' || c == b'>' {
            let prev = if i > 0 { b[i - 1] } else { 0 };
            let next = if i + 1 < b.len() { b[i + 1] } else { 0 };
            let prev_ok = !(prev == b'<' || prev == b'>' || prev == b'-' || prev == b'=');
            let next_ok = !(next == b'<' || next == b'>' || next == b'=');
            if prev_ok && next_ok {
                if c == b'<' {
                    // Skip generic type argument lists entirely.
                    if let Some(close) = generic_close(b, i) {
                        i = close + 1;
                        continue;
                    }
                }
                return true;
            }
        }
        i += 1;
    }
    false
}

fn is_boundary_condition(t: &str) -> bool {
    t.contains("<=")
        || t.contains(">=")
        || t.contains("==")
        || t.contains("!=")
        || has_bare_comparison(t)
}

/// Count boolean connectors with word boundaries for and/or
/// (so "candle" doesn't count as "and").
fn count_bool_connectors(t: &str) -> usize {
    let mut count = t.matches("&&").count() + t.matches("||").count();
    for tok in t.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if tok == "and" || tok == "or" {
            count += 1;
        }
    }
    count
}

fn is_complex_conditional(t: &str) -> bool {
    let is_cond =
        starts_with_kw(t, "if") || starts_with_kw(t, "elif") || starts_with_kw(t, "while");
    if !is_cond {
        return false;
    }
    // 3+ conditions joined by boolean connectors.
    count_bool_connectors(t) >= 2
}

fn is_return_stmt(t: &str) -> bool {
    if !starts_with_kw(t, "return") {
        return false;
    }
    // Bare `return` is structural noise; `return <expr>` is the contract.
    t.len() > "return".len()
}

/// If the trimmed line at `lines[start]` opens a block, return the index one
/// past the block's last line. Python-style `:` blocks use indentation;
/// brace blocks use brace balance. None for single-line statements.
fn block_end(lines: &[&str], start: usize) -> Option<usize> {
    let t = lines[start].trim();
    if t.is_empty() || t.starts_with('#') || t.starts_with("//") {
        return None;
    }
    let n = lines.len();
    if t.ends_with(':') {
        // Python-style indent block. (A `:` inside brackets, e.g. slices or
        // dict displays, is never line-final — acceptable approximation.)
        let base = indent_of(lines[start]);
        let mut i = start + 1;
        while i < n {
            let l = lines[i];
            if !l.trim().is_empty() && indent_of(l) <= base {
                break;
            }
            i += 1;
        }
        if i > start + 1 {
            Some(i)
        } else {
            None
        }
    } else if t.contains('{') {
        let mut depth = 0i32;
        let mut i = start;
        loop {
            for ch in lines[i].chars() {
                if ch == '{' {
                    depth += 1;
                } else if ch == '}' {
                    depth -= 1;
                }
            }
            i += 1;
            if depth <= 0 || i >= n {
                break;
            }
        }
        if i > start + 1 {
            Some(i)
        } else {
            None
        }
    } else {
        None
    }

}


/// Filter one function/method body: keep the signature header, interesting
/// lines (plus their blocks), and the closing brace; collapse boring runs
/// into elision markers.
fn diagnostic_symbol_body(body: &str) -> String {
    let lines: Vec<&str> = body.lines().collect();
    let n = lines.len();
    if n == 0 {
        return String::new();
    }
    let mut keep = vec![false; n];

    // Header: signature lines up to and including the block opener.
    let mut idx = 0;
    let mut header_end = 0;
    while idx < n {
        keep[idx] = true;
        header_end = idx + 1;
        let te = lines[idx].trim_end();
        if te.ends_with('{') || te.ends_with(':') {
            break;
        }
        idx += 1;
        if idx > 8 {
            break;
        }
    }

    // Mark interesting lines.
    for (i, line) in lines.iter().enumerate() {
        if keep[i] {
            continue;
        }
        if is_interesting_line(line.trim_start()) {
            keep[i] = true;
        }
    }

    // Interesting block-openers keep their whole block (the except body, the
    // if body, the returned dict literal).
    let mut i = header_end;
    while i < n {
        if keep[i] {
            if let Some(end) = block_end(&lines, i) {
                let end = end.min(n);
                for k in keep.iter_mut().take(end).skip(i) {
                    *k = true;
                }
                // The absorbed block's own nested openers are subsumed.
                i = end;
                continue;
            }
        }
        i += 1;
    }

    // Always keep the final closing brace — it balances the header's opener.
    // (Python bodies have no closing brace; this is a no-op for them.)
    if n >= 1 && lines[n - 1].trim_start().starts_with('}') {
        keep[n - 1] = true;
    }

    // Rust tail expression: the last value before the closing brace is the
    // return value. Keep it when it's a bare expression, not a statement.
    // (Python uses explicit `return`, already covered above.)
    if n >= 2 {
        let last = lines[n - 1].trim();
        if last == "}" || last == "};" {
            let prev_trimmed = lines[n - 2].trim();
            if !prev_trimmed.is_empty()
                && !prev_trimmed.ends_with(';')
                && !prev_trimmed.ends_with('{')
                && !prev_trimmed.ends_with('}')
                && !keep[n - 2]
                && !is_decl(lines[n - 2].trim_start())
            {
                keep[n - 2] = true;
            }
        }
    }

    // Render, collapsing boring runs into markers.
    let mut out: Vec<String> = Vec::new();
    let mut boring = 0usize;
    let mut boring_indent = 0usize;
    for (i, line) in lines.iter().enumerate() {
        if keep[i] {
            if boring > 0 {
                let pad = " ".repeat(boring_indent);
                out.push(format!("{pad}// … [{boring} lines elided] …"));
                boring = 0;
            }
            out.push(line.to_string());
        } else if boring == 0 {
            boring_indent = indent_of(line);
            boring = 1;
        } else {
            boring += 1;
        }
    }
    // Trailing boring lines drop silently — the function just ends.
    out.join("\n")
}

/// Diagnostic skeleton for Rust via tree-sitter: function bodies keep diagnostic
/// signal; other declarations stay dumb signatures. None when no functions
/// are found (caller falls back to the generic path).
fn rust_diagnostic_body(heavy: &str, max_doc_lines: usize) -> Option<String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .ok()?;
    let tree = parser.parse(heavy, None)?;

    let mut captured: Vec<(usize, &'static str, Node)> = Vec::new();
    collect_rust_captures(tree.root_node(), &mut captured);
    captured.sort_by_key(|(start, _, _)| *start);

    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut doc_budget = max_doc_lines;
    let mut elided = 0usize;
    let mut fn_count = 0usize;

    for (_, capture_name, node) in captured {
        let rendered = match capture_name {
            "import" => node_text(node, heavy)
                .map(str::trim_end)
                .map(str::to_string),
            "doc" => {
                let text = node_text(node, heavy)?.trim();
                if !is_doc(text) || doc_budget == 0 {
                    elided += 1;
                    None
                } else {
                    doc_budget -= 1;
                    Some(text.to_string())
                }
            }
            "decl" => {
                if node.kind() == "function_item" {
                    fn_count += 1;
                    node_text(node, heavy).map(diagnostic_symbol_body)
                } else {
                    render_rust_decl(node, heavy)
                }
            }
            _ => None,
        };
        if let Some(line) = rendered {
            push_unique_structural(&line, &mut out, &mut seen, &mut elided);
        } else {
            elided += 1;
        }
    }

    if out.is_empty() || fn_count == 0 {
        return None;
    }
    if elided > 0 {
        out.push(format!("// … {elided} fully-elided lines …"));
    }
    Some(out.join("\n"))
}

/// Diagnostic skeleton via the language-agnostic line heuristic (Python, JS/TS,
/// Go, …): function/method bodies keep diagnostic signal; classes and other
/// declarations stay dumb signatures.
fn generic_diagnostic_body(heavy: &str, max_doc_lines: usize) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut seen_structural: HashSet<String> = HashSet::new();
    let mut doc_budget = max_doc_lines as i32;
    let mut elided = 0usize;
    let mut code_lines = 0usize;

    let lines: Vec<&str> = heavy.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let t = line.trim_start();
        if t.is_empty() {
            i += 1;
            continue;
        }
        if is_import(t) {
            push_unique_structural(line.trim_end(), &mut out, &mut seen_structural, &mut elided);
            code_lines += 1;
            i += 1;
        } else if is_decl(t) || looks_like_signature(t) {
            if is_function_decl(t) {
                // Fold leading decorators into the diagnostic body.
                let mut start = i;
                while start > 0 && lines[start - 1].trim_start().starts_with('@') {
                    start -= 1;
                }
                let block = capture_block(&lines, i);
                let consumed = block.lines().count();
                let body = if start < i {
                    format!("{}\n{block}", lines[start..i].join("\n"))
                } else {
                    block
                };
                let diagnostic = diagnostic_symbol_body(&body);
                push_unique_structural(&diagnostic, &mut out, &mut seen_structural, &mut elided);
                code_lines += 1;
                i += consumed.max(1);
            } else {
                let sig = line.split('{').next().unwrap_or(line).trim_end();
                let suffix = if line.contains('{') { " { … }" } else { "" };
                let structural = format!("{sig}{suffix}");
                push_unique_structural(&structural, &mut out, &mut seen_structural, &mut elided);
                code_lines += 1;
                i += 1;
            }
        } else if is_doc(t) && doc_budget > 0 {
            let before = seen_structural.len();
            push_unique_structural(line.trim_end(), &mut out, &mut seen_structural, &mut elided);
            if seen_structural.len() > before {
                doc_budget -= 1;
            }
            i += 1;
        } else {
            elided += 1;
            i += 1;
        }
    }

    if code_lines == 0 {
        prose_excerpt(heavy, 1200, 500)
    } else {
        if elided > 0 {
            out.push(format!("// … {elided} fully-elided lines …"));
        }
        out.join("\n")
    }
}

/// Diagnostic skeleton body (no XML wrapper) for `axiom skeleton --diagnostic`: like
/// [`skeleton_body`] but function/method bodies keep diagnostic signal
/// (error paths, boundary conditions, suspicious patterns, complex
/// conditionals, return values) instead of being fully elided.
pub fn skeleton_body_diagnostic(heavy: &str, max_doc_lines: usize) -> String {
    if let Some(body) = rust_diagnostic_body(heavy, max_doc_lines) {
        return body;
    }
    generic_diagnostic_body(heavy, max_doc_lines)
}

#[cfg(test)]
mod tests {
    #[test]
    fn ranked_keeps_name_stub_when_signature_exceeds_budget() {
        // P0 safety: a low-ranked symbol must not vanish entirely. With a
        // tiny budget, full signatures don't fit, but names must survive.
        let txt = "def alpha():\n    pass\ndef beta():\n    pass\ndef gamma():\n    pass\n";
        let out = skeletonize_ranked(txt, "python", Some(8));
        // At least the names must appear as stubs, even if signatures don't fit.
        let has_stub = out.contains("(… body elided …)");
        let all_names_visible = ["alpha", "beta", "gamma"]
            .iter()
            .all(|n| out.contains(n));
        assert!(
            has_stub || all_names_visible,
            "budget-elided symbols must keep names: {out}"
        );
    }

    #[test]
    fn ranked_emits_all_names_without_budget() {
        let txt = "def alpha():\n    pass\ndef beta():\n    pass\n";
        let out = skeletonize_ranked(txt, "python", None);
        assert!(out.contains("alpha"), "{out}");
        assert!(out.contains("beta"), "{out}");
    }

    #[test]
    fn module_constant_detected() {
        assert_eq!(
            is_module_constant("RISK_ORDER = {\"low\": 1}", "RISK_ORDER = {\"low\": 1}"),
            Some("RISK_ORDER".to_string())
        );
        assert_eq!(
            is_module_constant("MAX_RETRIES = 3", "MAX_RETRIES = 3"),
            Some("MAX_RETRIES".to_string())
        );
    }

    #[test]
    fn module_constant_rejects_non_constants() {
        // Indented (not module-level).
        assert_eq!(is_module_constant("    X = 1", "X = 1"), None);
        // Lowercase (not UPPER_SNAKE_CASE).
        assert_eq!(is_module_constant("risk_order = 1", "risk_order = 1"), None);
        // Comparison, not assignment.
        assert_eq!(is_module_constant("X == 1", "X == 1"), None);
        // Single char (noise).
        assert_eq!(is_module_constant("X = 1", "X = 1"), None);
        // Walrus / annotated.
        assert_eq!(is_module_constant("X := 1", "X := 1"), None);
    }

    #[test]
    fn ranked_captures_python_module_constant() {
        let txt = "RISK_ORDER = {\"low\": 1, \"high\": 3}\ndef helper():\n    pass\n";
        let out = skeletonize_ranked(txt, "python", None);
        assert!(out.contains("RISK_ORDER"), "{out}");
    }

    #[test]
    fn expand_symbol_returns_full_constant_value() {
        let txt = "RISK_ORDER = {\n    \"low\": 1,\n    \"high\": 3,\n}\ndef helper():\n    pass\n";
        let expanded = expand_symbol(txt, "RISK_ORDER").expect("must expand");
        assert!(expanded.contains("\"low\": 1"), "{expanded}");
        assert!(expanded.contains("\"high\": 3"), "{expanded}");
        assert!(!expanded.contains("def helper"), "{expanded}");
    }
    use super::*;

    const SAMPLE: &str = r#"
//! A small module.
use std::collections::HashMap;
pub fn add(a: i32, b: i32) -> i32 {
    let s = a + b;
    s
}
struct Point {
    x: f64,
    y: f64,
}
impl Point {
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}
"#;

    #[test]
    fn keeps_signatures_drops_bodies() {
        let d = build_digest(SAMPLE, "s1", 100, 1.5, "sha256:abc", 3);
        assert!(d.contains("pub fn add(a: i32, b: i32) -> i32 { … }"));
        assert!(d.contains("struct Point { … }"));
        assert!(d.contains("use std::collections::HashMap;"));
        assert!(d.contains("//! A small module."));
        // bodies gone
        assert!(!d.contains("let s = a + b"));
        assert!(!d.contains("sqrt()"));
        // attributes preserved
        assert!(d.contains("recall_norm=\"1.500\""));
        assert!(d.contains("state=\"sha256:abc\""));
    }

    #[test]
    fn elision_counter_present() {
        let d = build_digest(SAMPLE, "s1", 100, 0.0, "h", 3);
        assert!(d.contains("implementation lines elided"));
    }

    #[test]
    fn strips_stacked_visibility() {
        let txt = "pub async fn handler() -> Result<()> {\n  ok()\n}";
        let d = build_digest(txt, "s", 10, 0.0, "h", 0);
        assert!(d.contains("pub async fn handler() -> Result<()> { … }"));
    }

    #[test]
    fn rust_ast_keeps_static_mut_declarations() {
        let txt = "static mut GLOBAL_COUNTER_XQZ: i64 = 0;\nfn run() { unsafe { GLOBAL_COUNTER_XQZ += 1; } }";
        let d = build_digest(txt, "s", 10, 0.0, "h", 0);
        assert!(d.contains("static mut GLOBAL_COUNTER_XQZ: i64 = …;"), "{d}");
        assert!(d.contains("fn run() { … }"));
        assert!(!d.contains("GLOBAL_COUNTER_XQZ += 1"));
    }

    #[test]
    fn go_func_kept() {
        let txt = "package main\nimport \"fmt\"\nfunc Add(a, b int) int {\n  return a + b\n}";
        let d = build_digest(txt, "s", 10, 0.0, "h", 3);
        assert!(d.contains("func Add(a, b int) int { … }"));
        assert!(d.contains("package main"));
        assert!(!d.contains("return a + b"));
    }

    #[test]
    fn python_def_class_kept() {
        let txt = "import os\nclass Foo:\n    def bar(self, x):\n        return x * 2\n";
        let d = build_digest(txt, "s", 10, 0.0, "h", 3);
        assert!(d.contains("class Foo:"));
        assert!(d.contains("def bar(self, x):"));
        assert!(!d.contains("return x * 2"));
    }

    #[test]
    fn skeleton_body_has_no_xml_wrapper() {
        // The CLI-readable body must not include the digest XML envelope.
        let b = skeleton_body(SAMPLE, 3);
        assert!(!b.contains("<axiom_context_digest"));
        assert!(!b.contains("recall_norm="));
        assert!(b.contains("pub fn add(a: i32, b: i32) -> i32 { … }"));
        assert!(b.contains("struct Point { … }"));
        assert!(!b.contains("let s = a + b"));
    }

    #[test]
    fn skeleton_body_python_readable() {
        let txt = "import os\nclass Foo:\n    def bar(self, x):\n        return x * 2\n";
        let b = skeleton_body(txt, 3);
        assert!(b.contains("class Foo:"));
        assert!(b.contains("def bar(self, x):"));
        assert!(!b.contains("return x * 2"));
        assert!(!b.contains("<axiom_context_digest"));
    }

    #[test]
    fn skeleton_body_prose_falls_back_to_excerpt() {
        let txt = "This is just plain prose with no code at all. ".repeat(100);
        let b = skeleton_body(&txt, 3);
        assert!(b.contains("elided"));
        assert!(!b.contains("<axiom_context_digest"));
    }

    #[test]
    fn js_class_method_without_keyword_kept() {
        // Methods with no leading keyword must be caught by looks_like_signature.
        let txt = "class Api {\n  async handle(req, res) {\n    res.send(req.body)\n  }\n}";
        let d = build_digest(txt, "s", 10, 0.0, "h", 3);
        assert!(d.contains("async handle(req, res) { … }"));
        assert!(!d.contains("res.send(req.body)"));
    }

    #[test]
    fn control_flow_not_treated_as_signature() {
        let txt = "fn run() {\n    if (x > 0) {\n        go()\n    }\n    for (i in xs) {\n        step()\n    }\n}";
        let d = build_digest(txt, "s", 10, 0.0, "h", 0);
        assert!(d.contains("fn run() { … }"));
        // control headers must be elided, not kept as signatures
        assert!(!d.contains("if (x > 0) { … }"));
        assert!(!d.contains("for (i in xs) { … }"));
    }

    #[test]
    fn repeated_declarations_are_deduplicated_in_digest() {
        let repeated = "fn calculate_invoice_total(customer_id: &str) -> Result<Money> {\n    lookup_contract_discount(customer_id)?;\n}\n\n".repeat(3);
        let d = build_digest(&repeated, "s", 100, 0.0, "h", 0);
        assert_eq!(d.matches("fn calculate_invoice_total").count(), 1);
        assert!(d.contains("implementation lines elided"));
    }

    #[test]
    fn prose_falls_back_to_excerpt_not_destroyed() {
        // Long plain text with no code structure: must keep readable content,
        // not collapse to "N lines elided".
        let para = "The quarterly report shows revenue grew across all regions. ".repeat(60);
        let d = build_digest(&para, "s", 500, 0.0, "h", 3);
        assert!(d.contains("kind=\"prose-excerpt\""));
        assert!(d.contains("quarterly report shows revenue"));
        assert!(d.contains("chars of prose elided"));
    }

    #[test]
    fn short_prose_kept_whole() {
        let txt = "Just a short note, nothing structural here.";
        let d = build_digest(txt, "s", 10, 0.0, "h", 3);
        assert!(d.contains("Just a short note"));
    }

    #[test]
    fn expand_brace_symbol_returns_full_body() {
        let body = expand_symbol(SAMPLE, "add").unwrap();
        assert!(body.contains("pub fn add(a: i32, b: i32) -> i32 {"));
        assert!(body.contains("let s = a + b;"));
        assert!(body.trim_end().ends_with('}'));
    }

    #[test]
    fn expand_python_uses_indentation() {
        let src = "class Foo:\n    def bar(self, x):\n        y = x * 2\n        return y\n\ndef other():\n    pass\n";
        let body = expand_symbol(src, "bar").unwrap();
        assert!(body.contains("def bar(self, x):"));
        assert!(body.contains("return y"));
        // must stop before the next top-level def
        assert!(!body.contains("def other"));
    }

    #[test]
    fn expand_unknown_symbol_is_none() {
        assert!(expand_symbol(SAMPLE, "does_not_exist").is_none());
    }

    #[test]
    fn expand_matches_whole_identifier_only() {
        // "add" must not match inside "readd" or "address"
        let src = "fn readd() {\n  q()\n}\nfn add() {\n  real()\n}\n";
        let body = expand_symbol(src, "add").unwrap();
        assert!(body.contains("fn add()"));
        assert!(body.contains("real()"));
        assert!(!body.contains("readd"));
    }

    // --- PageRank-ranked skeletonization -----------------------------------

    #[test]
    fn pagerank_orders_by_inbound_references() {
        // 0 -> 1, 0 -> 2, 1 -> 2 : node 2 has the most inbound references.
        let adj = vec![vec![1, 2], vec![2], vec![]];
        let r = pagerank(&adj, 0.85, 20);
        assert!(r[2] > r[1], "{r:?}");
        assert!(r[1] > r[0], "{r:?}");
        let sum: f64 = r.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "{sum}");
    }

    #[test]
    fn pagerank_uniform_when_no_edges() {
        let adj = vec![vec![], vec![], vec![]];
        let r = pagerank(&adj, 0.85, 20);
        for v in &r {
            assert!((*v - 1.0 / 3.0).abs() < 1e-9, "{r:?}");
        }
    }

    #[test]
    fn pagerank_empty_graph() {
        let r = pagerank(&[], 0.85, 20);
        assert!(r.is_empty());
    }

    #[test]
    fn ranked_rust_skeleton_orders_callees_first() {
        // util <- helper <- main : the most-depended-upon symbol leads.
        let src = "fn util() -> i32 {\n    1\n}\nfn helper() -> i32 {\n    util() + util()\n}\nfn main() {\n    let x = helper();\n}\n";
        let out = skeletonize_ranked(src, "rust", None);
        let pu = out.find("fn util()").expect("util missing");
        let ph = out.find("fn helper()").expect("helper missing");
        let pm = out.find("fn main()").expect("main missing");
        assert!(pu < ph && ph < pm, "{out}");
    }

    #[test]
    fn ranked_rust_keeps_imports_first() {
        let src = "use std::collections::HashMap;\nfn b() {\n    a();\n}\nfn a() {}\n";
        let out = skeletonize_ranked(src, "rust", None);
        let pi = out.find("use std::collections::HashMap;").unwrap();
        let pa = out.find("fn a()").unwrap();
        assert!(pi < pa, "{out}");
        // a is referenced by b, so a outranks b
        let pb = out.find("fn b()").unwrap();
        assert!(pa < pb, "{out}");
    }

    #[test]
    fn token_budget_keeps_top_symbols() {
        let src = "fn aaa() {\n    bbb();\n}\nfn bbb() {\n    ccc();\n}\nfn ccc() -> i32 {\n    42\n}\n";
        let out = skeletonize_ranked(src, "rust", Some(10));
        assert!(out.contains("fn ccc()"), "{out}");
        assert!(!out.contains("fn aaa()"), "{out}");
        assert!(out.contains("elided by token budget"), "{out}");
    }

    #[test]
    fn token_budget_none_keeps_everything() {
        let src = "fn aaa() {\n    bbb();\n}\nfn bbb() {\n    ccc();\n}\nfn ccc() -> i32 {\n    42\n}\n";
        let out = skeletonize_ranked(src, "rust", None);
        assert!(out.contains("fn aaa()"));
        assert!(out.contains("fn bbb()"));
        assert!(out.contains("fn ccc()"));
        assert!(!out.contains("elided by token budget"));
    }

    #[test]
    fn ranked_generic_python_path() {
        let src = "import os\nclass Foo:\n    def bar(self):\n        return baz()\ndef baz():\n    return 1\n";
        let out = skeletonize_ranked(src, "python", None);
        assert!(out.contains("import os"), "{out}");
        let pb = out.find("def baz():").expect("baz missing");
        let pf = out.find("def bar(self):").expect("bar missing");
        assert!(pb < pf, "{out}");
    }

    #[test]
    fn ranked_prose_falls_back_to_excerpt() {
        let para = "The quarterly report shows revenue grew. ".repeat(80);
        let out = skeletonize_ranked(&para, "english", None);
        assert!(out.contains("quarterly report"), "{out}");
    }

    #[test]
    fn extract_decl_name_cases() {
        assert_eq!(
            extract_decl_name("pub fn add(a: i32) -> i32 {"),
            Some("add".to_string())
        );
        assert_eq!(
            extract_decl_name("class Foo:"),
            Some("Foo".to_string())
        );
        assert_eq!(
            extract_decl_name("    async handle(req, res) {"),
            Some("handle".to_string())
        );
        assert_eq!(
            extract_decl_name("const MAX: usize = 5;"),
            Some("MAX".to_string())
        );
        assert_eq!(
            extract_decl_name("impl Point {"),
            Some("Point".to_string())
        );
        // `fn` after a colon is a type, not the name.
        assert_eq!(
            extract_decl_name("const f: fn() = g;"),
            Some("f".to_string())
        );
    }

    // --- Diagnostic elision --------------------------------------------------------

    const DIAGNOSTIC_PY: &str = r#"import os

class Loader:
    def load(self, path):
        # TODO: support remote paths
        count = 0
        total = 0
        try:
            data = read(path)
        except OSError as e:
            log(e)
            raise
        if count > total and total != 0:
            count += 1
        name = "loader"
        return data
"#;

    #[test]
    fn diagnostic_keeps_error_paths() {
        let b = skeleton_body_diagnostic(DIAGNOSTIC_PY, 3);
        assert!(b.contains("try:"), "{b}");
        assert!(b.contains("except OSError as e:"), "{b}");
        assert!(b.contains("log(e)"), "{b}"); // except-block extension
        assert!(b.contains("raise"), "{b}");
    }

    #[test]
    fn diagnostic_keeps_todo_and_boundary_condition() {
        let b = skeleton_body_diagnostic(DIAGNOSTIC_PY, 3);
        assert!(b.contains("TODO"), "{b}");
        assert!(b.contains("if count > total and total != 0:"), "{b}");
        assert!(b.contains("count += 1"), "{b}"); // if-block extension
    }

    #[test]
    fn diagnostic_elides_boring_assignments() {
        let b = skeleton_body_diagnostic(DIAGNOSTIC_PY, 3);
        assert!(!b.contains("total = 0"), "{b}");
        assert!(!b.contains("name = \"loader\""), "{b}");
        assert!(b.contains("lines elided"), "{b}");
    }

    #[test]
    fn diagnostic_keeps_return_value() {
        let b = skeleton_body_diagnostic(DIAGNOSTIC_PY, 3);
        assert!(b.contains("return data"), "{b}");
    }

    #[test]
    fn diagnostic_keeps_returned_dict_literal() {
        // The SSRF case: a missing key hides inside a returned dict literal.
        let src = "class P:\n    def to_dict(self):\n        tmp = 1\n        return {\n            \"allowed\": self.allowed,\n            \"blocked\": self.blocked,\n        }\n";
        let b = skeleton_body_diagnostic(src, 3);
        assert!(b.contains("\"allowed\": self.allowed"), "{b}");
        assert!(b.contains("\"blocked\": self.blocked"), "{b}");
        assert!(!b.contains("tmp = 1"), "{b}");
    }

    #[test]
    fn diagnostic_keeps_complex_conditional() {
        let src = "def f(a, b, c, d):\n    x = 1\n    if a and b or c and d:\n        return 1\n    return 0\n";
        let b = skeleton_body_diagnostic(src, 3);
        assert!(b.contains("if a and b or c and d:"), "{b}");
        assert!(!b.contains("x = 1"), "{b}");
    }

    #[test]
    fn diagnostic_and_or_need_word_boundaries() {
        // "candle" contains "and" but is not a boolean connector.
        assert_eq!(count_bool_connectors("if candle and wax:"), 1);
        assert_eq!(count_bool_connectors("if a && b || c:"), 2);
    }

    #[test]
    fn diagnostic_keeps_assert_and_bare_except() {
        let src = "def f(x):\n    assert x > 0\n    try:\n        g()\n    except:\n        pass\n    return x\n";
        let b = skeleton_body_diagnostic(src, 3);
        assert!(b.contains("assert x > 0"), "{b}");
        assert!(b.contains("except:"), "{b}");
    }

    #[test]
    fn diagnostic_rust_keeps_unwrap_and_try_operator() {
        let src = "use std::collections::HashMap;\nfn get(m: &HashMap<String, i32>, k: &str) -> i32 {\n    let v = m.get(k).unwrap();\n    let w = v + 1;\n    w\n}\nfn load(path: &str) -> Result<String, std::io::Error> {\n    let s = std::fs::read_to_string(path)?;\n    Ok(s)\n}\n";
        let b = skeleton_body_diagnostic(src, 3);
        assert!(b.contains(".unwrap()"), "{b}");
        assert!(b.contains("?;"), "{b}");
        assert!(!b.contains("let w = v + 1;"), "{b}");
    }

    #[test]
    fn diagnostic_rust_keeps_tail_expression() {
        // Rust's tail expression is the return value.
        let src = "fn inc(x: i32) -> i32 {\n    let y = x + 1;\n    y\n}\n";
        let b = skeleton_body_diagnostic(src, 3);
        assert!(b.contains("\n    y\n"), "{b}");
        assert!(!b.contains("let y = x + 1;"), "{b}");
    }

    #[test]
    fn diagnostic_rust_keeps_err_and_panic() {
        let src = "fn div(a: i32, b: i32) -> Result<i32, String> {\n    let q = a / b;\n    if b == 0 {\n        return Err(\"zero\".to_string());\n    }\n    if q < 0 {\n        panic!(\"negative\");\n    }\n    Ok(q)\n}\n";
        let b = skeleton_body_diagnostic(src, 3);
        assert!(b.contains("return Err("), "{b}");
        assert!(b.contains("panic!"), "{b}");
        assert!(b.contains("if b == 0 {"), "{b}");
        assert!(!b.contains("let q = a / b;"), "{b}");
    }

    #[test]
    fn diagnostic_bare_comparison_ignores_arrows() {
        assert!(!has_bare_comparison("let f = |x| x -> i32;"));
        assert!(!has_bare_comparison("Some(x) => foo(),"));
        assert!(has_bare_comparison("if x < limit {"));
        assert!(has_bare_comparison("while n > 0:"));
    }

    #[test]
    fn diagnostic_generics_are_not_comparisons() {
        assert!(!has_bare_comparison("let x: Vec<u8> = Vec::new();"));
        assert!(!has_bare_comparison("let r: Result<T, E> = foo();"));
        assert!(!has_bare_comparison(
            "let m: HashMap<String, Vec<u8>> = HashMap::new();"
        ));
        assert!(!has_bare_comparison("let y = foo::<T>(a);"));
        // Real comparisons still fire, even unspaced.
        assert!(has_bare_comparison("if x<y {"));
        assert!(has_bare_comparison("while n>0 {"));
        assert!(has_bare_comparison("if x < y {"));
    }

    #[test]
    fn diagnostic_non_function_decls_stay_signatures() {
        let b = skeleton_body_diagnostic(DIAGNOSTIC_PY, 3);
        assert!(b.contains("class Loader:"), "{b}");
        assert!(b.contains("import os"), "{b}");
        // No method bodies leak into the class rendering.
        assert_eq!(b.matches("def load").count(), 1);
    }

    // Boring-heavy sample: realistic code where most lines are assignments
    // and straightforward logic, with a few diagnostic hot spots.
    const DIAGNOSTIC_SIZE_PY: &str = r#"import os
import sys

CONFIG_PATH = "/etc/app.conf"
DEFAULT_TIMEOUT = 30

class Processor:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.items = []
        self.cache = {}
        self.enabled = True
        self.retries = 3

    def process(self, items, limit):
        results = []
        batch = []
        seen = set()
        mapping = {}
        for item in items:
            batch.append(item)
            seen.add(item.id)
            transformed = item.value * 2
            mapping[item.id] = transformed
        if len(results) > limit:
            raise ValueError("over limit")
        total = sum(batch)
        average = total / max(1, len(batch))
        summary = f"{average:.2f}"
        return results
"#;

    #[test]
    fn diagnostic_size_between_dumb_and_full() {
        let dumb = skeleton_body(DIAGNOSTIC_SIZE_PY, 3);
        let diagnostic = skeleton_body_diagnostic(DIAGNOSTIC_SIZE_PY, 3);
        assert!(
            dumb.len() < diagnostic.len(),
            "dumb={} diagnostic={}",
            dumb.len(),
            diagnostic.len()
        );
        assert!(
            diagnostic.len() < DIAGNOSTIC_SIZE_PY.len(),
            "diagnostic={} full={}",
            diagnostic.len(),
            DIAGNOSTIC_SIZE_PY.len()
        );
        let ratio = diagnostic.len() as f64 / DIAGNOSTIC_SIZE_PY.len() as f64;
        assert!(ratio < 0.8, "diagnostic/full ratio={ratio:.2}");
    }

    #[test]
    fn diagnostic_prose_falls_back_to_excerpt() {
        let txt = "Just plain prose, no code here. ".repeat(100);
        let b = skeleton_body_diagnostic(&txt, 3);
        assert!(b.contains("elided"), "{b}");
    }
}
