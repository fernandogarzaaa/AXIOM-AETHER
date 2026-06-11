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
    let mut flush = |seg: &str, out: &mut Vec<String>| {
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

/// Verify a response's claims against `evidence`, returning a grounding report.
pub fn verify(response: &str, evidence: &str) -> GroundingReport {
    let evidence_spans: Vec<Vec<String>> = sentences(evidence)
        .iter()
        .map(|s| content_tokens(s))
        .filter(|t| !t.is_empty())
        .collect();

    let mut claims = Vec::new();
    let (mut supported, mut unsupported, mut unverified) = (0usize, 0usize, 0usize);

    for claim in extract_claims(response) {
        let toks = content_tokens(&claim);
        let support = best_support(&toks, &evidence_spans);
        let verdict = if support >= SUPPORT_HIGH {
            supported += 1;
            Verdict::Supported
        } else if support < SUPPORT_LOW {
            unsupported += 1;
            Verdict::Unsupported
        } else {
            unverified += 1;
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
        });
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
}
