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

/// Visibility / async prefixes stripped before testing for a declaration.
const VIS_PREFIXES: [&str; 5] = ["pub ", "public ", "export ", "default ", "async "];

/// Keywords that begin a declaration whose signature we keep.
const DECL_KEYWORDS: [&str; 15] = [
    "fn ", "def ", "function ", "struct ", "enum ", "trait ", "impl ", "impl<",
    "interface ", "class ", "type ", "const ", "static ", "mod ", "namespace ",
];

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
    let mut doc_budget = max_doc_lines as i32;
    let mut elided = 0usize;

    for line in heavy.lines() {
        let t = line.trim_start();
        if t.is_empty() {
            continue;
        }
        if is_import(t) {
            out.push(line.trim_end().to_string());
        } else if is_decl(t) {
            let sig = line.split('{').next().unwrap_or(line).trim_end();
            let suffix = if line.contains('{') { " { … }" } else { "" };
            out.push(format!("{sig}{suffix}"));
        } else if is_doc(t) && doc_budget > 0 {
            out.push(line.trim_end().to_string());
            doc_budget -= 1;
        } else {
            elided += 1;
        }
    }
    if elided > 0 {
        out.push(format!("// … {elided} implementation lines elided …"));
    }
    let body = out.join("\n");

    format!(
        "<axiom_context_digest session=\"{session_id}\" kind=\"structural-skeleton\" \
original_tokens=\"{original_tokens}\" recall_norm=\"{recall_norm:.3}\" state=\"{state_hash}\">\n\
# Readable skeleton of elided heavy context: signatures kept, bodies dropped.\n\
# Ask Axiom to expand a named symbol if you need its body.\n\
{body}\n\
</axiom_context_digest>"
    )
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
}
