//! Grounding verification — flag response claims unsupported by the evidence.
//!
//! This is Axiom's honest answer to "hallucination": not omniscient
//! fact-checking, but **grounding verification** — does each factual claim in a
//! response have support in the supplied evidence/context? Axiom is positioned
//! for exactly this: the proxy already absorbs the heavy context, so the same
//! material can ground the model's answer.
//!
//! The default tier is **lexical** (deterministic, no model): a claim is
//! supported when enough of its content tokens appear in some evidence span.
//! Confidence is a [`BetaBelief`] so it carries uncertainty (a short claim with
//! one matched token is *tentatively* supported, not certainly). An optional
//! neural surprisal score (computed server-side against the context-adapted W̃)
//! can be layered on top — this module keeps the deterministic core.
//!
//! ## Honest limitations (do not oversell)
//!
//! The lexical tier shares the well-measured weakness of every lexical
//! verifier (incl. chimeralang-mcp's, ~0 recall on contradictions): a claim
//! that *reuses the evidence's vocabulary but negates or alters a value* can
//! still score as "supported". Lexical grounding flags **unsupported** claims
//! (no overlap); it does **not** reliably catch fluent contradictions. The
//! `verdict_contradiction_blind_spot` test documents this explicitly.

use crate::belief::BetaBelief;

/// A claim's grounding outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Well-supported by the evidence.
    Supported,
    /// No meaningful support found — a likely hallucination relative to context.
    Unsupported,
    /// Partial support — neither confidently grounded nor clearly absent.
    Unverified,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Verdict::Supported => "SUPPORTED",
            Verdict::Unsupported => "UNSUPPORTED",
            Verdict::Unverified => "UNVERIFIED",
        })
    }
}

/// One claim and how well the evidence grounds it.
#[derive(Debug, Clone)]
pub struct ClaimVerdict {
    pub claim: String,
    /// Fraction of the claim's content tokens found in the best evidence span.
    pub support: f32,
    pub verdict: Verdict,
    /// Confidence in the verdict, carrying uncertainty (Beta over support).
    pub confidence: BetaBelief,
    /// Optional neural grounding-lift: how much absorbing the context reduced
    /// the claim's surprisal under W̃ (`CE_base − CE_context`). Positive ⇒ the
    /// context predicts the claim (grounded); ≤0 ⇒ the context did not help —
    /// the contradiction signature the lexical tier alone misses. `None` when
    /// the neural tier was not run.
    pub lift: Option<f32>,
}

/// Aggregate grounding report for a response.
#[derive(Debug, Clone)]
pub struct GroundingReport {
    pub claims: Vec<ClaimVerdict>,
    pub supported: usize,
    pub unsupported: usize,
    pub unverified: usize,
    /// supported / total claims (0 when there are no checkable claims).
    pub grounded_fraction: f32,
}

impl GroundingReport {
    /// Claims that found no support — the ones to surface to the user/model.
    pub fn flagged(&self) -> Vec<&ClaimVerdict> {
        self.claims
            .iter()
            .filter(|c| c.verdict == Verdict::Unsupported)
            .collect()
    }
}

/// At/above this share of claim tokens present in an evidence span → supported.
const SUPPORT_HIGH: f32 = 0.60;
/// Below this → unsupported; between the two → unverified.
const SUPPORT_LOW: f32 = 0.30;

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
    "at", "for", "and", "or", "but", "if", "then", "that", "this", "these", "those", "it", "its",
    "as", "by", "with", "from", "into", "about", "than", "so", "such", "can", "will", "would",
    "should", "may", "might", "do", "does", "did", "has", "have", "had", "not", "no", "yes",
];

fn content_tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_ascii_alphanumeric())
        .filter_map(|w| {
            let w = w.trim().to_ascii_lowercase();
            if w.len() >= 2 && !STOPWORDS.contains(&w.as_str()) {
                Some(w)
            } else {
                None
            }
        })
        .collect()
}

/// Split prose into sentence-like spans (on `. ! ?` and newlines).
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
            let t = cur.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
            cur.clear();
        } else {
            cur.push(ch);
        }
    }
    let t = cur.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
    out
}

/// Extract checkable factual claims from a response: declarative sentences with
/// enough content. Questions (segments terminated by `?`) and trivially short
/// fragments are skipped.
pub fn extract_claims(response: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |seg: &str, out: &mut Vec<String>| {
        let s = seg.trim().trim_end_matches(['!', '.']).trim();
        if !s.is_empty() && content_tokens(s).len() >= 3 {
            out.push(s.to_string());
        }
    };
    for ch in response.chars() {
        match ch {
            '.' | '!' | '\n' => {
                flush(&cur, &mut out);
                cur.clear();
            }
            // A '?' marks the pending segment as a question — discard it.
            '?' => cur.clear(),
            _ => cur.push(ch),
        }
    }
    flush(&cur, &mut out);
    out
}

/// Max containment of a claim's content tokens within any single evidence span.
fn best_support(claim_tokens: &[String], evidence_spans: &[Vec<String>]) -> f32 {
    if claim_tokens.is_empty() {
        return 0.0;
    }
    let mut best = 0.0f32;
    for span in evidence_spans {
        let matched = claim_tokens.iter().filter(|t| span.contains(t)).count();
        let containment = matched as f32 / claim_tokens.len() as f32;
        if containment > best {
            best = containment;
        }
    }
    best
}

/// Recompute the aggregate report fields from a claim list.
fn tally(claims: Vec<ClaimVerdict>) -> GroundingReport {
    let (mut supported, mut unsupported, mut unverified) = (0usize, 0usize, 0usize);
    for c in &claims {
        match c.verdict {
            Verdict::Supported => supported += 1,
            Verdict::Unsupported => unsupported += 1,
            Verdict::Unverified => unverified += 1,
        }
    }
    let total = claims.len();
    let grounded_fraction = if total == 0 {
        0.0
    } else {
        supported as f32 / total as f32
    };
    GroundingReport {
        claims,
        supported,
        unsupported,
        unverified,
        grounded_fraction,
    }
}

/// Verify a response's claims against `evidence`, returning a grounding report.
pub fn verify(response: &str, evidence: &str) -> GroundingReport {
    let evidence_spans: Vec<Vec<String>> = sentences(evidence)
        .iter()
        .map(|s| content_tokens(s))
        .filter(|t| !t.is_empty())
        .collect();

    let mut claims = Vec::new();
    for claim in extract_claims(response) {
        let toks = content_tokens(&claim);
        let support = best_support(&toks, &evidence_spans);
        let verdict = if support >= SUPPORT_HIGH {
            Verdict::Supported
        } else if support < SUPPORT_LOW {
            Verdict::Unsupported
        } else {
            Verdict::Unverified
        };
        // Confidence in support, with evidence strength ∝ claim length so a
        // short claim's verdict carries more uncertainty.
        let strength = (toks.len() as f32).clamp(2.0, 12.0);
        claims.push(ClaimVerdict {
            claim,
            support,
            verdict,
            confidence: BetaBelief::from_confidence(support, strength),
            lift: None,
        });
    }
    tally(claims)
}

/// A lexically-"supported" claim is demoted only when absorbing the context
/// made it *meaningfully more* surprising (lift below `-DEMOTE_MARGIN`) — the
/// signature of a contradiction that merely reuses the vocabulary. A near-zero
/// lift means "no signal" (e.g. a weak/untrained model) and must NOT override
/// the lexical verdict, so the margin is strictly negative.
const DEMOTE_MARGIN: f32 = 0.5;

/// Verify, then refine with the **neural-surprisal tier**: `lift(claim)` returns
/// `CE_base − CE_context` for the claim under W̃ (positive ⇒ the absorbed
/// context predicts it). A lexically-Supported claim with non-positive lift is
/// downgraded to Unverified — this is how Axiom catches the vocabulary-sharing
/// *contradictions* the lexical tier alone cannot. `lift` returning `None`
/// leaves a claim's lexical verdict untouched.
pub fn verify_with_signals<S>(response: &str, evidence: &str, mut lift: S) -> GroundingReport
where
    S: FnMut(&str) -> Option<f32>,
{
    let base = verify(response, evidence);
    let claims = base
        .claims
        .into_iter()
        .map(|mut c| {
            if let Some(l) = lift(&c.claim) {
                c.lift = Some(l);
                if c.verdict == Verdict::Supported && l < -DEMOTE_MARGIN {
                    c.verdict = Verdict::Unverified;
                }
            }
            c
        })
        .collect();
    tally(claims)
}

/// Result of verifying with grounding-gated expansion: the report before any
/// expansion, the report after expanding only the unsupported claims'
/// dependencies, and what that targeted expansion cost/resolved.
#[derive(Debug, Clone)]
pub struct GatedVerifyReport {
    pub before: GroundingReport,
    pub after: GroundingReport,
    /// Symbols whose bodies were expanded to ground a flagged claim.
    pub expanded_symbols: Vec<String>,
    /// Claims that went from not-Supported to Supported after expansion.
    pub resolved_claims: Vec<String>,
    /// Bytes of evidence added by expansion — the *only* tokens spent back, and
    /// only on claims grounding could not confirm from the skeleton alone.
    pub expansion_bytes: usize,
}

/// Cap on symbol expansions per verification so a pathological response cannot
/// trigger unbounded un-compression.
const MAX_EXPANSIONS: usize = 8;

/// Identifier-like candidate symbols referenced in a claim (snake_case,
/// CamelCase, or any alnum token ≥3 chars). Over-generation is fine: the
/// resolver returns `None` for non-symbols.
fn candidate_symbols(claim: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for raw in claim.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        let t = raw.trim();
        if t.len() >= 3 && t.chars().any(|c| c.is_ascii_alphabetic()) && seen.insert(t.to_string()) {
            out.push(t.to_string());
        }
    }
    out
}

/// Verify with **grounding-gated expansion**: the keystone that saves tokens
/// *while* reducing hallucination. Compress aggressively (the caller passes the
/// lean skeleton as `evidence`); then, for each claim the skeleton could not
/// ground, expand *only* the symbols that claim references (via `resolve`) and
/// re-verify. Tokens are spent back surgically — only where grounding was
/// uncertain — never across the board.
pub fn verify_with_gated_expansion<F>(
    response: &str,
    evidence: &str,
    mut resolve: F,
) -> GatedVerifyReport
where
    F: FnMut(&str) -> Option<String>,
{
    let before = verify(response, evidence);
    let mut expanded_symbols = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut extra = String::new();

    for claim in before.claims.iter().filter(|c| c.verdict != Verdict::Supported) {
        for sym in candidate_symbols(&claim.claim) {
            if expanded_symbols.len() >= MAX_EXPANSIONS {
                break;
            }
            if !seen.insert(sym.clone()) {
                continue;
            }
            if let Some(body) = resolve(&sym) {
                extra.push('\n');
                extra.push_str(&body);
                expanded_symbols.push(sym);
            }
        }
    }

    let after = if extra.is_empty() {
        before.clone()
    } else {
        verify(response, &format!("{evidence}\n{extra}"))
    };

    // Claims that expansion rescued: not-Supported before, Supported after.
    let before_bad: std::collections::HashSet<&str> = before
        .claims
        .iter()
        .filter(|c| c.verdict != Verdict::Supported)
        .map(|c| c.claim.as_str())
        .collect();
    let resolved_claims = after
        .claims
        .iter()
        .filter(|c| c.verdict == Verdict::Supported && before_bad.contains(c.claim.as_str()))
        .map(|c| c.claim.clone())
        .collect();

    GatedVerifyReport {
        before,
        after,
        expanded_symbols,
        resolved_claims,
        expansion_bytes: extra.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EVIDENCE: &str = "Axiom is an inference engine with online test-time training. \
        Every token updates the per-layer fast-weight matrices. The drift gate \
        separates clean code from anomalies with a positive margin.";

    #[test]
    fn supported_claim_is_grounded() {
        let r = verify("Axiom uses online test-time training.", EVIDENCE);
        assert_eq!(r.claims.len(), 1);
        assert_eq!(r.claims[0].verdict, Verdict::Supported);
        assert!(r.grounded_fraction > 0.99);
        assert!(r.flagged().is_empty());
    }

    #[test]
    fn unsupported_claim_is_flagged_as_hallucination() {
        let r = verify("Axiom was written in Haskell by a team at MIT in 1998.", EVIDENCE);
        assert_eq!(r.claims[0].verdict, Verdict::Unsupported);
        assert_eq!(r.flagged().len(), 1);
        assert!(r.claims[0].confidence.mean() < 0.3);
    }

    #[test]
    fn mixed_response_reports_partial_grounding() {
        let resp = "Axiom uses online test-time training. It also mines dogecoin on weekends.";
        let r = verify(resp, EVIDENCE);
        assert_eq!(r.claims.len(), 2);
        assert_eq!(r.supported, 1);
        assert_eq!(r.unsupported, 1);
        assert!((r.grounded_fraction - 0.5).abs() < 1e-6);
    }

    #[test]
    fn questions_and_fragments_are_not_claims() {
        assert!(extract_claims("What is Axiom? Ok. Yes.").is_empty());
    }

    #[test]
    fn empty_response_has_no_claims() {
        let r = verify("", EVIDENCE);
        assert_eq!(r.claims.len(), 0);
        assert_eq!(r.grounded_fraction, 0.0);
    }

    #[test]
    fn verdict_contradiction_blind_spot() {
        // DOCUMENTED LIMITATION: a claim that reuses evidence vocabulary but
        // negates the meaning still scores as supported under lexical grounding.
        // This test pins the known weakness so we never overstate the tier.
        let r = verify("The drift gate does not separate clean code from anomalies.", EVIDENCE);
        assert_eq!(
            r.claims[0].verdict,
            Verdict::Supported,
            "lexical grounding cannot catch vocabulary-sharing contradictions"
        );
    }

    #[test]
    fn gated_expansion_grounds_a_claim_by_expanding_only_its_dependency() {
        // Skeleton (lean evidence): only signatures, bodies elided.
        let skeleton = "fn parse_header(buf: &[u8]) -> Header. fn checksum(data: &[u8]) -> u32.";
        // The answer makes a claim about an implementation detail not in the skeleton.
        let response = "checksum computes the crc32 polynomial fold over the data.";

        // Resolver stands in for /v1/expand over the stored source: returns the
        // elided body whose tokens ground the claim.
        let resolve = |sym: &str| -> Option<String> {
            match sym {
                "checksum" => Some(
                    "fn checksum(data: &[u8]) -> u32 { computes crc32 polynomial fold over data }"
                        .to_string(),
                ),
                _ => None,
            }
        };

        let report = verify_with_gated_expansion(response, skeleton, resolve);
        // Before expansion the claim is not grounded by the skeleton alone…
        assert_ne!(report.before.claims[0].verdict, Verdict::Supported);
        // …expansion pulled ONLY the referenced symbol…
        assert_eq!(report.expanded_symbols, vec!["checksum".to_string()]);
        assert!(report.expansion_bytes > 0);
        // …and re-verification now grounds it.
        assert_eq!(report.after.claims[0].verdict, Verdict::Supported);
        assert_eq!(report.resolved_claims.len(), 1);
        assert!(report.after.grounded_fraction > report.before.grounded_fraction);
    }

    #[test]
    fn neural_tier_catches_the_contradiction_lexical_misses() {
        // The contradiction reuses the evidence's vocabulary, so the lexical tier
        // alone scores it SUPPORTED (see verdict_contradiction_blind_spot). The
        // neural tier supplies a non-positive grounding lift (the trained model
        // is NOT helped by the context for this claim) → it is demoted.
        let contradiction = "The drift gate does not separate clean code from anomalies.";
        // Mock the lift the trained model would produce: ≤0 for the contradiction.
        let report = verify_with_signals(contradiction, EVIDENCE, |_claim| Some(-1.2));
        assert_eq!(
            report.claims[0].verdict,
            Verdict::Unverified,
            "neural tier must demote a lexically-supported contradiction"
        );
        assert_eq!(report.claims[0].lift, Some(-1.2));
    }

    #[test]
    fn neural_tier_no_signal_does_not_override_lexical() {
        // A weak/untrained model yields ~0 lift for everything; that must NOT
        // demote a lexically-grounded claim (no false positives on no signal).
        let report = verify_with_signals(
            "Axiom uses online test-time training.",
            EVIDENCE,
            |_claim| Some(0.0),
        );
        assert_eq!(report.claims[0].verdict, Verdict::Supported);
    }

    #[test]
    fn neural_tier_keeps_genuinely_grounded_claims() {
        // A real, grounded claim: positive lift (context predicts it) → stays Supported.
        let report = verify_with_signals(
            "Axiom uses online test-time training.",
            EVIDENCE,
            |_claim| Some(2.5),
        );
        assert_eq!(report.claims[0].verdict, Verdict::Supported);
        assert_eq!(report.claims[0].lift, Some(2.5));
    }

    #[test]
    fn gated_expansion_spends_nothing_when_already_grounded() {
        // The skeleton already grounds the claim → no expansion, no tokens spent.
        let mut resolver_calls = 0;
        let report = verify_with_gated_expansion(
            "Axiom uses online test-time training.",
            EVIDENCE,
            |_sym| {
                resolver_calls += 1;
                None
            },
        );
        assert_eq!(report.before.claims[0].verdict, Verdict::Supported);
        assert!(report.expanded_symbols.is_empty());
        assert_eq!(report.expansion_bytes, 0, "no tokens spent when already grounded");
        assert_eq!(resolver_calls, 0, "resolver not consulted for supported claims");
    }
}
