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

fn render_until_equals(node: Node, source: &str) -> Option<String> {
    let text = node_text(node, source)?.trim_end();
    if let Some(eq) = text.find('=') {
        return Some(format!("{} = …;", text[..eq].trim_end()));
    }
    Some(text.to_string())
}

fn is_decl(t: &str) -> bool {
    let mut s = t;
    // Strip stacked visibility/async prefixes (e.g. "pub async fn").
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
    DECL_KEYWORDS.iter().any(|k| s.starts_with(k))
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
{body}\n\
</axiom_context_digest>"
    )
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
    }
    None
}

#[cfg(test)]
mod tests {
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
}
