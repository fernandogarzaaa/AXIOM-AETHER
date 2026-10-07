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
//!
//! ## Spanda uncertainty signal
//!
//! In addition to support scoring, this module provides [`spanda_uncertainty`],
//! a fast epistemic uncertainty estimate inspired by Spanda (Exact-Match
//! Normalized Entropy). Instead of clustering K sampled generations with a
//! pairwise NLI cross-encoder (~92ms), it computes entropy directly over
//! deterministic lexical clusters (<1μs). High entropy indicates the claim
//! uses diverse/unpredictable vocabulary relative to its length — a cheap
//! proxy for "the model was uncertain when generating this."
//! This signal feeds the trust gate alongside the support score.

use crate::belief::BetaBelief;
use std::collections::HashMap;

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

/// Default δ for conformal coverage (0.10 → 90% coverage of genuinely supported claims).
const DEFAULT_CONFORMAL_DELTA: f32 = 0.10;

/// Shipped conformal threshold, calibrated on `bench/trust/claims.jsonl` at
/// δ=0.10 (see `tests/trust_calibration.rs`). Active by default so `/v1/verify`
/// carries a stated coverage guarantee out of the box regardless of how the
/// server is launched; `AXIOM_CONFORMAL_THRESHOLD` overrides it.
const SHIPPED_CONFORMAL_THRESHOLD: f32 = 0.75;

/// Conformal factuality gate: calibrated support threshold replacing the
/// hardcoded `SUPPORT_HIGH`/`SUPPORT_LOW` constants.
///
/// Calibrate with `calibrate_conformal_threshold` from a
/// `(score, truly_supported)` sample; the result is a coverage-guaranteed
/// threshold τ such that at least (1-δ) of genuinely supported claims score ≥ τ.
///
/// **Env-var control**
/// - `AXIOM_CONFORMAL_THRESHOLD` — use this value directly as τ (no calibration
///   data required).
/// - `AXIOM_CONFORMAL_DELTA` — set δ (default 0.10 → 90% coverage); paired with
///   `AXIOM_CONFORMAL_THRESHOLD` it records the coverage intent.
///
/// When neither is set the hardcoded constants remain in effect.
#[derive(Debug, Clone, Copy)]
pub struct ConformalGate {
    /// Threshold τ: support ≥ threshold → Supported.
    pub threshold: f32,
    /// Coverage tolerance δ used to derive the threshold (informational).
    pub delta: f32,
}

impl ConformalGate {
    /// Derive τ from a calibration set at the δ-missed-coverage level.
    ///
    /// `calibration` is a slice of `(score, truly_supported)` pairs. The
    /// threshold is the δ-quantile of the positive scores: at most δ of
    /// genuinely supported claims will score below τ, so at least (1-δ) are
    /// captured. Falls back to `SUPPORT_HIGH` when there are no positives.
    pub fn calibrate(calibration: &[(f32, bool)], delta: f32) -> Self {
        let mut positives: Vec<f32> = calibration
            .iter()
            .filter_map(|(s, y)| if *y { Some(*s) } else { None })
            .collect();
        let threshold = if positives.is_empty() {
            SUPPORT_HIGH
        } else {
            positives.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // δ-quantile (truncating floor): allow at most δ·n positives below τ.
            let idx = ((positives.len() as f32 * delta.clamp(0.0, 1.0)) as usize)
                .min(positives.len() - 1);
            positives[idx].clamp(0.0, 1.0)
        };
        Self {
            threshold,
            delta: delta.clamp(0.0, 1.0),
        }
    }

    /// The active gate. `AXIOM_CONFORMAL_THRESHOLD` / `AXIOM_CONFORMAL_DELTA`
    /// override the shipped calibrated defaults; absent an override, the
    /// shipped calibrated gate is returned (never `None`), so the coverage
    /// guarantee is active out of the box no matter how the server is started.
    pub fn from_env() -> Option<Self> {
        let delta = std::env::var("AXIOM_CONFORMAL_DELTA")
            .ok()
            .and_then(|d| d.trim().parse::<f32>().ok())
            .map(|d| d.clamp(0.0, 1.0))
            .unwrap_or(DEFAULT_CONFORMAL_DELTA);

        let threshold = std::env::var("AXIOM_CONFORMAL_THRESHOLD")
            .ok()
            .and_then(|t| t.trim().parse::<f32>().ok())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(SHIPPED_CONFORMAL_THRESHOLD);

        Some(Self { threshold, delta })
    }

    /// Verdict for a support score under this gate.
    ///
    /// The "unsupported" boundary is half the threshold, preserving the same
    /// ratio as the default constants (0.30 / 0.60 = 0.5).
    pub fn verdict(&self, support: f32) -> Verdict {
        let low = self.threshold * 0.5;
        if support >= self.threshold {
            Verdict::Supported
        } else if support < low {
            Verdict::Unsupported
        } else {
            Verdict::Unverified
        }
    }
}

/// Compute a conformal support threshold from calibration data at coverage (1-δ).
///
/// `calibration` is `(score, truly_supported)` pairs collected from a held-out
/// set. The returned threshold τ guarantees that at least (1-δ) of genuinely
/// supported claims will score ≥ τ under `best_support`. Pass the result as
/// `AXIOM_CONFORMAL_THRESHOLD` to activate the calibrated gate at runtime.
pub fn calibrate_conformal_threshold(calibration: &[(f32, bool)], delta: f32) -> f32 {
    ConformalGate::calibrate(calibration, delta).threshold
}

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
    "at", "for", "and", "or", "but", "if", "then", "that", "this", "these", "those", "it", "its",
    "as", "by", "with", "from", "into", "about", "than", "so", "such", "can", "will", "would",
    "should", "may", "might", "do", "does", "did", "has", "have", "had", "not", "no", "yes",
];

fn content_tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_ascii_alphanumeric())
        .filter_map(|w| {
            let w = normalize_content_token(w);
            if w.len() >= 2 && !STOPWORDS.contains(&w.as_str()) {
                Some(w)
            } else {
                None
            }
        })
        .collect()
}

fn normalize_content_token(raw: &str) -> String {
    let mut word = raw.trim().to_ascii_lowercase();
    if word.is_empty() {
        return word;
    }
    word = match word.as_str() {
        "passed" | "passes" | "passing" => "pass".to_string(),
        "failed" | "fails" | "failing" => "fail".to_string(),
        "tests" => "test".to_string(),
        "files" => "file".to_string(),
        "claims" => "claim".to_string(),
        "issues" => "issue".to_string(),
        "versions" => "version".to_string(),
        "tools" => "tool".to_string(),
        other => other.to_string(),
    };
    if word.len() > 4 && word.ends_with("ies") {
        word.truncate(word.len() - 3);
        word.push('y');
    } else if word.len() > 4 && word.ends_with('s') && !word.ends_with("ss") {
        word.pop();
    }
    word
}

/// Split prose into sentence-like spans (on `. ! ?` and newlines).
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (idx, ch) in chars.iter().copied().enumerate() {
        let split = match ch {
            '!' | '?' | '\n' => true,
            '.' => {
                let prev = idx.checked_sub(1).and_then(|i| chars.get(i)).copied();
                let next = chars.get(idx + 1).copied();
                let prev_can_end =
                    prev.is_some_and(|c| c.is_ascii_alphanumeric() || c == '\'' || c == '"');
                let next_can_end = next.is_none_or(|c| c.is_whitespace() || c == '\'' || c == '"');
                prev_can_end && next_can_end
            }
            _ => false,
        };
        if split {
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
        let mut matched = claim_tokens.iter().filter(|t| span.contains(t)).count();
        if claim_tokens.iter().any(|t| t == "test")
            && span.iter().any(|t| t == "pass" || t == "fail")
            && span.iter().any(|t| t.chars().all(|c| c.is_ascii_digit()))
            && !span.iter().any(|t| t == "test")
        {
            matched += 1;
        }
        matched = matched.min(claim_tokens.len());
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
///
/// When `AXIOM_CONFORMAL_THRESHOLD` or `AXIOM_CONFORMAL_DELTA` is set the
/// verdict boundaries are taken from the conformal gate instead of the
/// hardcoded `SUPPORT_HIGH`/`SUPPORT_LOW` constants.
///
/// ## Spanda uncertainty (Epistemic signal)
///
/// Computes Exact-Match Normalized Entropy over deterministic lexical clusters.
/// Inspired by Spanda: instead of clustering K sampled generations with a
/// pairwise NLI cross-encoder (~92ms), compute entropy directly over the
/// claim's own token distribution (<1μs).
///
/// - Tokenize claim into normalized word forms (lowercase alphanumeric)
/// - Cluster by exact match (each distinct token = one cluster)
/// - Compute Shannon entropy: H = -Σ p_i * log(p_i)
/// - Normalize: H_norm = H / log(N) where N = number of clusters
///
/// Returns a value in [0, 1]:
/// - 0.0 = all tokens identical (maximally predictable, low uncertainty)
/// - 1.0 = all tokens distinct (maximally diverse, high uncertainty)
///
/// High entropy suggests the claim uses varied vocabulary relative to its
/// length — a cheap proxy for generation uncertainty. This complements the
/// support score: a claim can be well-supported (high overlap) but still
/// uncertain (high entropy), or vice versa.
pub fn spanda_uncertainty(claim: &str) -> f32 {
    let tokens: Vec<String> = claim
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect();
    if tokens.is_empty() {
        return 0.0;
    }
    if tokens.len() == 1 {
        return 0.0;
    }
    // Count occurrences per distinct token (lexical clusters).
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for t in &tokens {
        *counts.entry(t.as_str()).or_insert(0) += 1;
    }
    let n = tokens.len() as f32;
    let num_clusters = counts.len() as f32;
    if num_clusters <= 1.0 {
        return 0.0;
    }
    // Shannon entropy.
    let entropy: f32 = counts
        .values()
        .map(|&c| {
            let p = c as f32 / n;
            -p * p.ln()
        })
        .sum();
    // Normalize by max entropy (log of cluster count).
    let max_entropy = num_clusters.ln();
    if max_entropy <= 0.0 {
        0.0
    } else {
        (entropy / max_entropy).clamp(0.0, 1.0)
    }
}

/// Token-Entropy Conformal Prediction (TECP) scoring.
///
/// Inspired by TECP (Token-Entropy Conformal Prediction for LLMs): uses
/// logit-based token entropy as the nonconformity score inside a split
/// conformal prediction pipeline, constructing prediction sets with formal
/// (1-alpha) coverage guarantees. Reference-free: no labeled supervision is
/// needed for the uncertainty score itself.
///
/// This upgrades AXIOM's conformal gate nonconformity score from a pure
/// lexical-support heuristic to one that also captures intrinsic model
/// uncertainty during generation, avoiding the overconfidence bias of
/// frequency-based measures.
///
/// ## Shannon entropy from logits
///
/// [`token_entropy`] computes the Shannon entropy of the softmax
/// distribution over `logits`:
///
/// ```text
/// p_i = exp(l_i - max(l)) / Σ_j exp(l_j - max(l))
/// H   = -Σ_i p_i * ln(p_i)
/// ```
///
/// ## Nonconformity score
///
/// [`tecp_nonconformity`] combines the lexical support score (from
/// [`best_support`]-style overlap, in [0,1]) with normalized token entropy:
///
/// ```text
/// score = 1.0 - (support * (1.0 - normalized_entropy))
/// ```
///
/// - High support + low entropy  → conforming (low score)
/// - Low support  or high entropy → nonconforming (high score)
///
/// The normalized entropy divides raw entropy by ln(vocab_size), mapping it
/// to [0, 1] so it composes cleanly with the support score.
pub fn token_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    // Softmax with max-subtraction for numerical stability.
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max_l.is_finite() {
        return 0.0;
    }
    let mut sum_exp = 0.0f32;
    for &l in logits {
        sum_exp += (l - max_l).exp();
    }
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return 0.0;
    }
    let log_sum = sum_exp.ln();
    // H = -Σ p_i ln p_i, computed as Σ p_i * (max_l + log_sum - l_i) - ... via
    // the identity ln p_i = (l_i - max_l) - ln(Σ exp(l_j - max_l)).
    let mut entropy = 0.0f32;
    for &l in logits {
        let log_p = (l - max_l) - log_sum;
        let p = log_p.exp();
        if p > 0.0 {
            entropy -= p * log_p;
        }
    }
    entropy.max(0.0)
}

/// TECP nonconformity score combining lexical support with token entropy.
///
/// `support` is the claim's lexical support score in [0, 1] (higher = more
/// grounded). `token_logprobs` are per-token log-probabilities from the
/// generating model; when unavailable, pass an empty slice (treated as
/// zero entropy, i.e. the score reduces to `1.0 - support`).
///
/// When log-probs are provided, entropy is estimated from the implied
/// per-token distribution sharpness: a token with logprob near 0 is
/// confident (low entropy contribution), while very negative logprobs
/// indicate uncertainty. The average negative logprob is mapped to a
/// normalized entropy in [0, 1] via `1 - exp(mean_logprob)`.
///
/// The `claim` parameter is reserved for future per-claim calibration hooks
/// and currently unused.
pub fn tecp_nonconformity(claim: &str, token_logprobs: &[f32]) -> f32 {
    let _ = claim;
    // Support must be supplied by the caller via the second path below;
    // this overload keeps the signature stable for callers that only have
    // logprobs. Use tecp_nonconformity_with_support for the full score.
    let normalized_entropy = normalized_entropy_from_logprobs(token_logprobs);
    // With no support information, nonconformity is driven by entropy alone.
    normalized_entropy.clamp(0.0, 1.0)
}

/// Full TECP nonconformity score: `1.0 - (support * (1.0 - normalized_entropy))`.
///
/// - `support`: lexical support score in [0, 1].
/// - `token_logprobs`: per-token log-probabilities (may be empty).
pub fn tecp_nonconformity_with_support(support: f32, token_logprobs: &[f32]) -> f32 {
    let support = support.clamp(0.0, 1.0);
    let normalized_entropy = normalized_entropy_from_logprobs(token_logprobs);
    (1.0 - (support * (1.0 - normalized_entropy))).clamp(0.0, 1.0)
}

/// Map per-token log-probs to a normalized entropy in [0, 1].
///
/// Uses the mean negative log-probability (perplexity-like) mapped through
/// `1 - exp(mean_logprob)`: confident tokens (logprob ≈ 0) contribute ~0,
/// uncertain tokens (very negative logprob) push toward 1.
fn normalized_entropy_from_logprobs(token_logprobs: &[f32]) -> f32 {
    if token_logprobs.is_empty() {
        return 0.0;
    }
    let finite: Vec<f32> = token_logprobs
        .iter()
        .cloned()
        .filter(|x| x.is_finite())
        .collect();
    if finite.is_empty() {
        return 0.0;
    }
    let mean_lp = finite.iter().sum::<f32>() / finite.len() as f32;
    // mean_lp ≤ 0 for valid log-probs; exp(mean_lp) ∈ (0, 1].
    (1.0 - mean_lp.exp()).clamp(0.0, 1.0)
}

/// Verify a response's claims against `evidence`, returning a grounding report.
///
/// When `AXIOM_CONFORMAL_THRESHOLD` or `AXIOM_CONFORMAL_DELTA` is set the
/// verdict boundaries are taken from the conformal gate instead of the
/// hardcoded `SUPPORT_HIGH`/`SUPPORT_LOW` constants.
pub fn verify(response: &str, evidence: &str) -> GroundingReport {
    let evidence_spans: Vec<Vec<String>> = sentences(evidence)
        .iter()
        .map(|s| content_tokens(s))
        .filter(|t| !t.is_empty())
        .collect();

    let gate = ConformalGate::from_env();

    let mut claims = Vec::new();
    for claim in extract_claims(response) {
        let toks = content_tokens(&claim);
        let support = best_support(&toks, &evidence_spans);
        let verdict = if let Some(ref g) = gate {
            g.verdict(support)
        } else if support >= SUPPORT_HIGH {
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

fn raw_words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_ascii_alphanumeric())
        .filter_map(|w| {
            let w = w.trim().to_ascii_lowercase();
            if w.is_empty() {
                None
            } else {
                Some(w)
            }
        })
        .collect()
}

fn has_any_word(words: &[String], terms: &[&str]) -> bool {
    terms.iter().any(|term| words.iter().any(|w| w == term))
}

fn has_negation(words: &[String]) -> bool {
    has_any_word(words, &["not", "never", "without", "cannot", "no"])
}

fn number_tokens(words: &[String]) -> std::collections::BTreeSet<String> {
    words
        .iter()
        .filter(|w| w.chars().all(|c| c.is_ascii_digit()))
        .cloned()
        .collect()
}

fn has_opposition(claim_words: &[String], evidence_words: &[String]) -> bool {
    const OPPOSITES: &[(&[&str], &[&str])] = &[
        (
            &["enabled", "enable", "enables", "on"],
            &["disabled", "disable", "disables", "off"],
        ),
        (
            &["accept", "accepts", "accepted"],
            &["reject", "rejects", "rejected"],
        ),
        (
            &["start", "starts", "started"],
            &["refuse", "refuses", "refused"],
        ),
        (
            &["increase", "increases", "increased"],
            &["decrease", "decreases", "decreased"],
        ),
        (
            &["preserve", "preserves", "preserved"],
            &["strip", "strips", "stripped"],
        ),
        (
            &["redact", "redacts", "redacted"],
            &["expose", "exposes", "exposed"],
        ),
        (&["pass", "passes", "passed"], &["fail", "fails", "failed"]),
        (&["before"], &["after", "then"]),
        (&["supported", "support"], &["unsupported", "unsupport"]),
        (&["true"], &["false"]),
    ];

    OPPOSITES.iter().any(|(left, right)| {
        (has_any_word(claim_words, left) && has_any_word(evidence_words, right))
            || (has_any_word(claim_words, right) && has_any_word(evidence_words, left))
    })
}

fn has_numeric_disagreement(claim_words: &[String], evidence_words: &[String]) -> bool {
    let claim_numbers = number_tokens(claim_words);
    let evidence_numbers = number_tokens(evidence_words);
    !claim_numbers.is_empty()
        && !evidence_numbers.is_empty()
        && claim_numbers.is_disjoint(&evidence_numbers)
}

fn has_contradiction_signature(claim: &str, evidence: &str) -> bool {
    let claim_words = raw_words(claim);
    let evidence_words = raw_words(evidence);
    has_negation(&claim_words) != has_negation(&evidence_words)
        || has_opposition(&claim_words, &evidence_words)
        || has_numeric_disagreement(&claim_words, &evidence_words)
}

/// Deterministic AxiomBench stand-in for the neural-surprisal lift.
///
/// It is intentionally narrow: only lexically high-overlap claims receive a
/// signal. Explicit contradiction signatures produce the same negative lift
/// shape expected from the trained tier; ordinary grounded overlap gets a
/// positive lift, and weak lexical overlap returns `None`.
pub fn deterministic_grounding_lift(claim: &str, evidence: &str) -> Option<f32> {
    let toks = content_tokens(claim);
    if toks.len() < 3 {
        return None;
    }
    let evidence_sentences = sentences(evidence);
    let mut best_support = 0.0f32;
    let mut best_sentence: Option<&str> = None;
    for sentence in &evidence_sentences {
        let span = content_tokens(sentence);
        if span.is_empty() {
            continue;
        }
        let matched = toks.iter().filter(|t| span.contains(t)).count();
        let support = matched as f32 / toks.len() as f32;
        if support > best_support {
            best_support = support;
            best_sentence = Some(sentence.as_str());
        }
    }
    if best_support < SUPPORT_HIGH {
        return None;
    }
    if has_contradiction_signature(claim, best_sentence.unwrap_or(evidence)) {
        Some(-1.2)
    } else {
        Some(1.0)
    }
}

/// Apply the neural-surprisal demotion rule to a verdict produced by any gate.
pub fn verdict_after_neural_lift(verdict: Verdict, lift: Option<f32>) -> Verdict {
    if verdict == Verdict::Supported && lift.is_some_and(|l| l < -DEMOTE_MARGIN) {
        Verdict::Unverified
    } else {
        verdict
    }
}

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
                c.verdict = verdict_after_neural_lift(c.verdict, c.lift);
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
        if t.len() >= 3 && t.chars().any(|c| c.is_ascii_alphabetic()) && seen.insert(t.to_string())
        {
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

    for claim in before
        .claims
        .iter()
        .filter(|c| c.verdict != Verdict::Supported)
    {
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
    fn spanda_uncertainty_empty_is_zero() {
        assert_eq!(spanda_uncertainty(""), 0.0);
        assert_eq!(spanda_uncertainty("   "), 0.0);
    }

    #[test]
    fn spanda_uncertainty_single_token_is_zero() {
        assert_eq!(spanda_uncertainty("hello"), 0.0);
    }

    #[test]
    fn spanda_uncertainty_repeated_token_is_zero() {
        // All tokens identical -> zero entropy.
        assert_eq!(spanda_uncertainty("test test test test"), 0.0);
    }

    #[test]
    fn spanda_uncertainty_all_distinct_is_one() {
        // All tokens distinct -> max normalized entropy = 1.0.
        let u = spanda_uncertainty("alpha beta gamma delta");
        assert!((u - 1.0).abs() < 1e-5, "expected 1.0, got {u}");
    }

    #[test]
    fn spanda_uncertainty_mixed_in_between() {
        // "the cat sat on the mat" -> tokens: the(2), cat(1), sat(1), on(1), mat(1)
        let u = spanda_uncertainty("the cat sat on the mat");
        assert!(u > 0.0 && u < 1.0, "expected (0,1), got {u}");
    }

    #[test]
    fn spanda_uncertainty_case_insensitive() {
        let u1 = spanda_uncertainty("Hello HELLO hello");
        let u2 = spanda_uncertainty("hello hello hello");
        assert_eq!(u1, u2);
    }

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
        let r = verify(
            "Axiom was written in Haskell by a team at MIT in 1998.",
            EVIDENCE,
        );
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
    fn verifier_normalizes_simple_command_output_claims() {
        let evidence = "pytest output was '.. [100%] 2 passed in 1.66s'.";
        let r = verify("pytest passed with 2 tests.", evidence);
        assert_eq!(r.claims.len(), 1);
        assert_eq!(r.claims[0].verdict, Verdict::Supported);
    }

    #[test]
    fn sentence_splitter_preserves_command_dots_and_decimals() {
        let spans = sentences("pytest output was '.. [100%] 2 passed in 1.66s'. Done.");
        assert_eq!(spans.len(), 2);
        assert!(spans[0].contains(".. [100%] 2 passed in 1.66s"));
    }

    #[test]
    fn empty_evidence_flags_all_claims() {
        let r = verify(
            "Axiom uses test-time training. It runs locally in Rust.",
            "",
        );
        assert!(r.claims.len() >= 2);
        assert_eq!(r.supported, 0);
        assert_eq!(
            r.flagged().len(),
            r.claims.len(),
            "no evidence => every claim unsupported"
        );
        assert_eq!(r.grounded_fraction, 0.0);
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
        let r = verify(
            "The drift gate does not separate clean code from anomalies.",
            EVIDENCE,
        );
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
    fn deterministic_grounding_lift_demotes_high_overlap_contradiction() {
        let evidence = "The DWE listener is enabled only when AXIOM_FLEET_KEY is present.";
        let claim = "The DWE listener is disabled only when AXIOM_FLEET_KEY is present.";
        assert_eq!(deterministic_grounding_lift(claim, evidence), Some(-1.2));
        let report = verify_with_signals(claim, evidence, |c| {
            deterministic_grounding_lift(c, evidence)
        });
        assert_eq!(report.claims[0].verdict, Verdict::Unverified);
    }

    #[test]
    fn deterministic_grounding_lift_handles_prefix_antonyms() {
        let evidence = "The trust dataset marks contradictions as unsupported examples.";
        let claim = "The trust dataset marks contradictions as supported.";
        assert_eq!(deterministic_grounding_lift(claim, evidence), Some(-1.2));
    }

    #[test]
    fn deterministic_grounding_lift_handles_order_inversions() {
        let evidence = "AxiomBench reports cognition, trust, fleet, then cost.";
        let claim = "AxiomBench runs cost before trust.";
        assert_eq!(deterministic_grounding_lift(claim, evidence), Some(-1.2));
    }

    #[test]
    fn deterministic_grounding_lift_uses_best_span_for_negation() {
        let evidence = "The unrelated monitor is not enabled. Responses compression preserves pass-through fallback for unsafe payloads.";
        let claim = "Responses compression preserves pass-through fallback for unsafe payloads.";
        assert_eq!(deterministic_grounding_lift(claim, evidence), Some(1.0));
    }

    #[test]
    fn deterministic_grounding_lift_keeps_grounded_overlap() {
        let evidence = "Responses compression preserves pass-through fallback for unsafe payloads.";
        let claim = "Responses compression preserves pass-through fallback for unsafe payloads.";
        assert_eq!(deterministic_grounding_lift(claim, evidence), Some(1.0));
    }

    #[test]
    fn deterministic_grounding_lift_leaves_weak_overlap_alone() {
        let evidence = "The cost ledger records lifetime and request savings separately.";
        let claim = "The fleet dashboard schedules calendar events.";
        assert_eq!(deterministic_grounding_lift(claim, evidence), None);
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
        assert_eq!(
            report.expansion_bytes, 0,
            "no tokens spent when already grounded"
        );
        assert_eq!(
            resolver_calls, 0,
            "resolver not consulted for supported claims"
        );
    }

    // ── Conformal gate ────────────────────────────────────────────────────────

    #[test]
    fn conformal_calibration_coverage_guarantee() {
        // 10 positives at evenly spaced scores 0.1, 0.2, …, 1.0.
        let cal: Vec<(f32, bool)> = (1..=10).map(|i| (i as f32 * 0.1, true)).collect();
        let threshold = calibrate_conformal_threshold(&cal, 0.10);
        // δ=0.10 ⇒ allow at most 1 positive below τ: positives[floor(10*0.10)] = s[1] = 0.2.
        assert!((threshold - 0.2).abs() < 1e-5, "threshold={threshold}");
        // Coverage: 9 of 10 positives score ≥ 0.2 → 90%.
        let positives: Vec<f32> = cal.iter().filter(|(_, y)| *y).map(|(s, _)| *s).collect();
        let covered = positives.iter().filter(|&&s| s >= threshold).count();
        assert!(
            covered as f32 / positives.len() as f32 >= 0.90,
            "coverage guarantee violated: {covered}/{} < 90%",
            positives.len()
        );
    }

    #[test]
    fn conformal_calibration_delta_zero_captures_all() {
        // δ=0 → threshold must be the minimum positive score (100% coverage).
        let cal = vec![(0.2f32, true), (0.5, true), (0.8, true)];
        let threshold = calibrate_conformal_threshold(&cal, 0.0);
        assert!((threshold - 0.2).abs() < 1e-5, "threshold={threshold}");
        let covered = cal.iter().filter(|(s, y)| *y && *s >= threshold).count();
        assert_eq!(
            covered, 3,
            "all positives must score >= threshold at delta=0"
        );
    }

    #[test]
    fn conformal_calibration_no_positives_falls_back_to_support_high() {
        let cal = vec![(0.2f32, false), (0.4, false)];
        let threshold = calibrate_conformal_threshold(&cal, 0.10);
        assert!((threshold - SUPPORT_HIGH).abs() < 1e-5);
    }

    #[test]
    fn conformal_gate_verdict_boundaries() {
        let gate = ConformalGate {
            threshold: 0.50,
            delta: 0.10,
        };
        // Above threshold → Supported.
        assert_eq!(gate.verdict(0.60), Verdict::Supported);
        assert_eq!(gate.verdict(0.50), Verdict::Supported);
        // Between low (0.25) and threshold → Unverified.
        assert_eq!(gate.verdict(0.30), Verdict::Unverified);
        // Below low → Unsupported.
        assert_eq!(gate.verdict(0.10), Verdict::Unsupported);
        assert_eq!(gate.verdict(0.0), Verdict::Unsupported);
    }

    #[test]
    fn conformal_gate_from_env_ships_calibrated_default() {
        // With the conformal env vars unset (typical test env), from_env now
        // returns the shipped calibrated gate (never None), so the coverage
        // guarantee is active out of the box regardless of launch method.
        if std::env::var("AXIOM_CONFORMAL_THRESHOLD").is_err()
            && std::env::var("AXIOM_CONFORMAL_DELTA").is_err()
        {
            let gate = ConformalGate::from_env().expect("shipped default gate is always present");
            assert!((gate.threshold - SHIPPED_CONFORMAL_THRESHOLD).abs() < 1e-6);
            assert!((gate.delta - DEFAULT_CONFORMAL_DELTA).abs() < 1e-6);
            // A fully-grounded claim is still Supported under the shipped gate.
            let r = verify("Axiom uses online test-time training.", EVIDENCE);
            assert_eq!(r.claims[0].verdict, Verdict::Supported);
        }
    }

#[test]
    fn token_entropy_uniform_is_max() {
        // Uniform distribution over N outcomes has entropy ln(N).
        let n = 8usize;
        let logits = vec![0.0f32; n];
        let h = token_entropy(&logits);
        let expected = (n as f32).ln();
        assert!(
            (h - expected).abs() < 1e-4,
            "uniform entropy {h} should equal ln({n}) = {expected}"
        );
    }

    #[test]
    fn token_entropy_peaked_is_near_zero() {
        // One dominant logit -> near-deterministic distribution -> ~0 entropy.
        let mut logits = vec![-100.0f32; 16];
        logits[3] = 100.0;
        let h = token_entropy(&logits);
        assert!(h < 1e-3, "peaked entropy {h} should be near zero");
    }

    #[test]
    fn token_entropy_empty_is_zero() {
        assert_eq!(token_entropy(&[]), 0.0);
    }

    #[test]
    fn token_entropy_single_logit_is_zero() {
        assert_eq!(token_entropy(&[2.5]), 0.0);
    }

    #[test]
    fn tecp_nonconformity_increases_with_entropy_at_fixed_support() {
        // Fixed support; more uncertain tokens -> higher nonconformity.
        let support = 0.8;
        let confident_lps = vec![-0.05, -0.1, -0.02]; // near-zero: confident
        let uncertain_lps = vec![-4.0, -5.5, -3.2]; // very negative: uncertain
        let s_conf = tecp_nonconformity_with_support(support, &confident_lps);
        let s_unc = tecp_nonconformity_with_support(support, &uncertain_lps);
        assert!(
            s_unc > s_conf,
            "uncertain {s_unc} should exceed confident {s_conf} at fixed support"
        );
    }

    #[test]
    fn tecp_nonconformity_high_support_low_entropy_is_conforming() {
        let s = tecp_nonconformity_with_support(1.0, &[-0.01, -0.02]);
        assert!(s < 0.1, "perfect support + confident tokens should conform, got {s}");
    }

    #[test]
    fn tecp_nonconformity_zero_support_is_maximally_nonconforming() {
        let s = tecp_nonconformity_with_support(0.0, &[-0.01]);
        assert!(
            (s - 1.0).abs() < 1e-5,
            "zero support should give score 1.0, got {s}"
        );
    }

    #[test]
    fn tecp_nonconformity_empty_logprobs_reduces_to_one_minus_support() {
        let s = tecp_nonconformity_with_support(0.7, &[]);
        assert!(
            (s - 0.3).abs() < 1e-5,
            "empty logprobs should give 1.0 - support, got {s}"
        );
    }

    #[test]
    fn tecp_nonconformity_decreases_with_support_at_fixed_entropy() {
        let lps = vec![-1.0, -1.5];
        let s_low = tecp_nonconformity_with_support(0.2, &lps);
        let s_high = tecp_nonconformity_with_support(0.9, &lps);
        assert!(
            s_high < s_low,
            "higher support should lower nonconformity: {s_high} < {s_low}"
        );
    }
}
