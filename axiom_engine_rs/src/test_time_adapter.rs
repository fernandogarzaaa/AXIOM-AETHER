//! B.1 — per-bug test-time adaptation for the repair loop.
//!
//! This mirrors the Akyürek/MIT ARC recipe (per-task adaptation + augmentation +
//! hierarchical self-consistency voting) adapted to AXIOM's *verifier-gated,
//! candidate-based* repair. For a single fault we:
//!
//!   1. **Augment** each candidate fix with semantics-preserving transforms
//!      (alpha-renaming of identifiers, whitespace normalization), collapsing
//!      candidates that differ only superficially onto one canonical form.
//!   2. **Vote** across candidates: equivalent-under-augmentation candidates
//!      reinforce each other (the test-time signal — a fix that recurs across
//!      independent phrasings is likelier the real fix, not an artifact of one
//!      sampling), weighted by any prior (e.g. fleet `verified_count`).
//!   3. **Re-rank** so the consensus candidate is tried *first*.
//!
//! ## Safety invariant (unchanged)
//!
//! This adapter only **reorders** which candidates the verifier tries — it never
//! trusts a candidate. The verifier still gates every patch before it is applied
//! or recorded, so a wrong consensus simply gets rejected and the next candidate
//! is tried. Re-verify-before-trust is fully preserved.

use std::collections::HashMap;

/// One candidate fix considered for a fault, plus any prior confidence (for
/// fleet patches this is the `verified_count`; for freshly generated candidates
/// it can be `1.0`).
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Full proposed file contents.
    pub content: String,
    /// Prior weight (≥ 0). Used as the per-candidate vote mass and as a
    /// tie-breaker. `verified_count` maps here directly.
    pub prior: f32,
}

impl Candidate {
    /// Convenience constructor.
    pub fn new(content: impl Into<String>, prior: f32) -> Self {
        Self {
            content: content.into(),
            prior: prior.max(0.0),
        }
    }
}

/// A small Rust/C-family keyword set kept verbatim during alpha-renaming, so the
/// canonical form preserves control-flow structure while abstracting away the
/// arbitrary names a model might pick. (Not exhaustive — unknown keywords are
/// simply renamed like any identifier, which is harmless for the equivalence
/// test as long as it is applied consistently.)
const PRESERVED_KEYWORDS: &[&str] = &[
    "fn", "let", "mut", "if", "else", "for", "while", "loop", "match", "return",
    "struct", "enum", "impl", "trait", "pub", "use", "mod", "const", "static",
    "self", "Self", "true", "false", "in", "as", "ref", "move", "where", "type",
    "def", "class", "import", "from", "return", "elif", "None", "True", "False",
];

/// Compute the augmentation-invariant canonical form of a source string:
/// identifiers are alpha-renamed to positional placeholders (`V0`, `V1`, …) by
/// first occurrence, keywords are preserved, and runs of whitespace collapse to a
/// single space. Two candidates that differ only by variable names and
/// formatting produce the *same* canonical form.
pub fn canonical_form(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut mapping: HashMap<String, String> = HashMap::new();
    let mut next_id = 0usize;
    let bytes = src.as_bytes();
    let mut i = 0usize;
    // Tracks whether the previously emitted token was a "word" (run of
    // [A-Za-z0-9_]). Two adjacent words must keep a single separating space
    // (`return x` ≠ `returnx`); around symbols/punctuation, whitespace is
    // insignificant and dropped — so spacing like `x:` vs `x :` canonicalizes
    // identically.
    let mut prev_was_word = false;

    let is_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';

    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if is_word(c) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_word(bytes[i]) {
                i += 1;
            }
            let token = &src[start..i];
            if prev_was_word {
                out.push(' ');
            }
            if token.as_bytes()[0].is_ascii_digit() {
                // Numeric / literal-led token: keep verbatim (no aliasing).
                out.push_str(token);
            } else if PRESERVED_KEYWORDS.contains(&token) {
                out.push_str(token);
            } else {
                let placeholder = mapping.entry(token.to_string()).or_insert_with(|| {
                    let p = format!("V{next_id}");
                    next_id += 1;
                    p
                });
                out.push_str(placeholder);
            }
            prev_was_word = true;
        } else {
            // Punctuation / operators: copy verbatim, no surrounding space.
            out.push(c as char);
            prev_was_word = false;
            i += 1;
        }
    }
    out
}

/// Re-rank candidates by augmentation-invariant self-consistency voting.
///
/// Returns candidate indices (into `candidates`) ordered best-first:
///   1. by **cluster vote** — the summed prior of all candidates sharing the
///      candidate's canonical form (consensus across superficial variations),
///   2. then by the candidate's own **prior** (fleet verification weight),
///   3. then by original index (stable — preserves the incoming ranking on ties).
///
/// An empty input yields an empty ranking.
pub fn rerank_by_self_consistency(candidates: &[Candidate]) -> Vec<usize> {
    if candidates.is_empty() {
        return Vec::new();
    }
    // Cluster by canonical form; each cluster's vote is the summed prior mass.
    let mut cluster_vote: HashMap<String, f32> = HashMap::new();
    let canon: Vec<String> = candidates.iter().map(|c| canonical_form(&c.content)).collect();
    for (idx, key) in canon.iter().enumerate() {
        *cluster_vote.entry(key.clone()).or_insert(0.0) += candidates[idx].prior.max(0.0);
    }

    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| {
        let va = cluster_vote[&canon[a]];
        let vb = cluster_vote[&canon[b]];
        // Higher cluster vote first.
        vb.partial_cmp(&va)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Then higher individual prior.
            .then(
                candidates[b]
                    .prior
                    .partial_cmp(&candidates[a].prior)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            // Then stable by original index.
            .then(a.cmp(&b))
    });
    order
}

/// Apply [`rerank_by_self_consistency`] and return the candidates' contents in
/// best-first order — the order the verifier should try them.
pub fn order_candidate_contents(candidates: &[Candidate]) -> Vec<String> {
    rerank_by_self_consistency(candidates)
        .into_iter()
        .map(|i| candidates[i].content.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_form_is_invariant_to_renaming_and_whitespace() {
        let a = "fn add(x: i32, y: i32) -> i32 { x + y }";
        let b = "fn   add( alpha : i32 , beta : i32 ) -> i32 {  alpha + beta  }";
        // `add` is not in the keyword set, so it renames consistently; the
        // structural form must match across both phrasings.
        assert_eq!(canonical_form(a), canonical_form(b));
    }

    #[test]
    fn canonical_form_distinguishes_real_structural_differences() {
        let a = "fn f() { return x + y; }";
        let b = "fn f() { return x - y; }";
        assert_ne!(canonical_form(a), canonical_form(b), "+ vs - must differ");
    }

    #[test]
    fn keywords_are_preserved_so_control_flow_is_not_aliased() {
        // `if` (keyword) must not collapse with a renamed identifier.
        let with_if = canonical_form("if cond { a }");
        let without = canonical_form("xx cond { a }");
        assert!(with_if.starts_with("if "));
        assert_ne!(with_if, without);
    }

    #[test]
    fn voting_promotes_the_consensus_candidate() {
        // Three candidates: two are the same fix up to variable names (consensus),
        // one is a lone different fix with a slightly higher individual prior.
        // The consensus cluster (vote 1+1=2) must outrank the lone one (vote 1.5).
        let cands = vec![
            Candidate::new("fn g() { return a + b; }", 1.0),
            Candidate::new("fn g() { return foo - bar; }", 1.5), // lone, higher prior
            Candidate::new("fn g() { return p + q; }", 1.0),     // == #0 up to renames
        ];
        let order = rerank_by_self_consistency(&cands);
        // The two consensus candidates (#0, #2) come before the lone one (#1).
        assert_eq!(order[2], 1, "lone higher-prior candidate must rank last");
        assert!(
            (order[0] == 0 && order[1] == 2) || (order[0] == 2 && order[1] == 0),
            "consensus cluster must occupy the top two slots, got {order:?}"
        );
    }

    #[test]
    fn prior_breaks_ties_within_equal_votes() {
        // Two singleton clusters with different priors: higher prior wins.
        let cands = vec![
            Candidate::new("fn a() { x }", 0.5),
            Candidate::new("fn b() { y - z }", 3.0),
        ];
        let order = rerank_by_self_consistency(&cands);
        assert_eq!(order, vec![1, 0]);
    }

    #[test]
    fn empty_input_yields_empty_ranking() {
        assert!(rerank_by_self_consistency(&[]).is_empty());
    }

    #[test]
    fn order_contents_returns_best_first() {
        let cands = vec![
            Candidate::new("fn g() { a + b }", 1.0),
            Candidate::new("fn g() { c + d }", 1.0), // consensus with #0
            Candidate::new("fn g() { lonely() }", 1.0),
        ];
        let ordered = order_candidate_contents(&cands);
        assert_eq!(ordered.len(), 3);
        // The lonely candidate must not be first.
        assert_ne!(ordered[0], "fn g() { lonely() }");
    }
}
