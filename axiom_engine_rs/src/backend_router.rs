//! Multi-provider generation router — make several LLM backends work together.
//!
//! Axiom's own model handles compression/recall/drift; *generation* is delegated
//! to a "brain". This module lets that brain be **more than one provider at
//! once** — e.g. OpenAI (GPT) and Anthropic (Claude) — with three behaviors:
//!
//!   * **Routing** — pick a provider by task kind (default: code repair → Claude,
//!     everything else → GPT), so each model does what it is best at.
//!   * **Failover** — if the chosen provider errors, fall through to the next in
//!     a deterministic order instead of failing the request.
//!   * **Consensus** (opt-in) — ask two providers and *fuse* their answers using
//!     Axiom's own [`crate::belief::BetaBelief`] evidence combination, so two
//!     models cross-check each other rather than trusting one blindly. This is
//!     the capability unique to Axiom: it already has the belief math to combine
//!     independent sources honestly.
//!
//! The orchestration here is pure and synchronous so it is fully unit-testable
//! with mock backends; real backends wrap their (blocking `reqwest`) HTTP client
//! behind the [`ChatBackend`] trait.

use std::collections::BTreeMap;

use crate::belief::BetaBelief;

/// Which provider serves a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Provider {
    /// OpenAI (or any OpenAI-compatible endpoint, incl. a local OpenDrop server).
    OpenAi,
    /// Anthropic (Claude).
    Anthropic,
    /// A local model (bootstrap TTT or OpenDrop) used as a last-resort fallback.
    Local,
}

/// Coarse task class used for routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    /// Source repair / coding (the `solve`/`task` loop).
    CodeRepair,
    /// Step-by-step reasoning (e.g. ChimeraLang `inquire`/beliefs).
    Reasoning,
    /// Everything else (general generation, compression-proxy passthrough).
    General,
}

/// Why a backend call failed (so the router can fail over).
#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    /// Provider not configured (no key / not selected).
    NotConfigured,
    /// Transport / upstream error, with a message.
    Upstream(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::NotConfigured => f.write_str("provider not configured"),
            BackendError::Upstream(m) => write!(f, "upstream error: {m}"),
        }
    }
}

/// One generation backend. Implemented by the real OpenAI/Anthropic/local
/// adapters (wrapping their blocking client) and by mocks in tests. `Send + Sync`
/// so a [`Router`] can be shared in the axum server state across requests.
pub trait ChatBackend: Send + Sync {
    /// Generate a completion for `prompt`. Returns the text or a [`BackendError`].
    fn complete(&self, prompt: &str) -> Result<String, BackendError>;
}

/// Routing + failover + consensus policy.
#[derive(Debug, Clone)]
pub struct RoutePolicy {
    /// Per-task primary provider.
    pub code_repair: Provider,
    pub reasoning: Provider,
    pub general: Provider,
    /// Deterministic failover order tried after the primary.
    pub failover: Vec<Provider>,
    /// When true, [`Router::generate`] asks the primary *and* the first failover
    /// provider and fuses the two answers (consensus). Off by default.
    pub consensus: bool,
}

impl Default for RoutePolicy {
    /// The recommended default: code repair → Claude, everything else → GPT,
    /// failover GPT→Claude→Local, consensus off (opt-in per request).
    fn default() -> Self {
        Self {
            code_repair: Provider::Anthropic,
            reasoning: Provider::OpenAi,
            general: Provider::OpenAi,
            failover: vec![Provider::OpenAi, Provider::Anthropic, Provider::Local],
            consensus: false,
        }
    }
}

impl RoutePolicy {
    /// Primary provider for a task kind.
    pub fn primary_for(&self, task: TaskKind) -> Provider {
        match task {
            TaskKind::CodeRepair => self.code_repair,
            TaskKind::Reasoning => self.reasoning,
            TaskKind::General => self.general,
        }
    }

    /// Provider try-order for a task: primary first, then the failover list with
    /// the primary de-duplicated out.
    pub fn order_for(&self, task: TaskKind) -> Vec<Provider> {
        let primary = self.primary_for(task);
        let mut order = vec![primary];
        for p in &self.failover {
            if *p != primary {
                order.push(*p);
            }
        }
        order
    }
}

/// Outcome of a routed generation.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedAnswer {
    /// The chosen answer text.
    pub text: String,
    /// Provider whose answer was returned.
    pub provider: Provider,
    /// Confidence in [0,1]. For single-provider routing this is the provider's
    /// nominal trust; for consensus it reflects cross-model agreement.
    pub confidence: f32,
    /// Providers consulted (for observability).
    pub consulted: Vec<Provider>,
}

/// Holds the configured backends and the policy.
pub struct Router {
    backends: BTreeMap<Provider, Box<dyn ChatBackend>>,
    policy: RoutePolicy,
}

impl Router {
    pub fn new(policy: RoutePolicy) -> Self {
        Self {
            backends: BTreeMap::new(),
            policy,
        }
    }

    /// Register a backend for a provider.
    pub fn with(mut self, provider: Provider, backend: Box<dyn ChatBackend>) -> Self {
        self.backends.insert(provider, backend);
        self
    }

    fn call(&self, provider: Provider, prompt: &str) -> Result<String, BackendError> {
        match self.backends.get(&provider) {
            Some(b) => b.complete(prompt),
            None => Err(BackendError::NotConfigured),
        }
    }

    /// Generate an answer for `task`, honoring routing, failover, and (if the
    /// policy enables it) consensus. Returns the last error only if *every*
    /// candidate provider fails.
    pub fn generate(&self, task: TaskKind, prompt: &str) -> Result<RoutedAnswer, BackendError> {
        if self.policy.consensus {
            if let Some(ans) = self.try_consensus(task, prompt) {
                return Ok(ans);
            }
            // Consensus unavailable (only one provider responded) → fall back to
            // plain failover below.
        }

        let order = self.policy.order_for(task);
        let mut last_err = BackendError::NotConfigured;
        let mut consulted = Vec::new();
        for p in order {
            consulted.push(p);
            match self.call(p, prompt) {
                Ok(text) => {
                    return Ok(RoutedAnswer {
                        text,
                        provider: p,
                        confidence: 0.8,
                        consulted,
                    })
                }
                Err(e) => last_err = e,
            }
        }
        Err(last_err)
    }

    /// Ask the two highest-priority available providers and fuse their answers.
    /// Returns `None` if fewer than two providers respond (caller then fails over).
    fn try_consensus(&self, task: TaskKind, prompt: &str) -> Option<RoutedAnswer> {
        let order = self.policy.order_for(task);
        let mut answers: Vec<(Provider, String)> = Vec::new();
        let mut consulted = Vec::new();
        for p in order {
            if answers.len() == 2 {
                break;
            }
            consulted.push(p);
            if let Ok(text) = self.call(p, prompt) {
                answers.push((p, text));
            }
        }
        if answers.len() < 2 {
            return None;
        }
        let (pa, a) = &answers[0];
        let (_pb, b) = &answers[1];
        // Fuse: each provider is one independent "success" of evidence; if the
        // two answers agree, evidence compounds (high confidence); if they
        // disagree, the combined belief stays uncertain. We return the primary's
        // answer with a confidence derived from agreement.
        let agree = answers_agree(a, b);
        let mut belief = BetaBelief::uniform();
        belief.reinforce(); // provider A produced an answer
        if agree {
            belief.reinforce(); // provider B corroborated → compounding evidence
        } else {
            // Two independent models actively disagree: that is evidence the
            // answer is *unreliable*, so drive confidence below the
            // single-provider baseline rather than merely neutral.
            belief.penalize();
            belief.penalize();
        }
        Some(RoutedAnswer {
            text: a.clone(),
            provider: *pa,
            confidence: belief.mean(),
            consulted,
        })
    }
}

/// Cheap textual agreement check for consensus: normalize whitespace/case and
/// compare. Good enough to tell "both models said the same thing" from "they
/// diverged"; a semantic check is a later refinement.
fn answers_agree(a: &str, b: &str) -> bool {
    let norm = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    let (na, nb) = (norm(a), norm(b));
    if na == nb {
        return true;
    }
    // Token-overlap (Jaccard) ≥ 0.6 counts as agreement for longer answers.
    let sa: std::collections::BTreeSet<&str> = na.split(' ').collect();
    let sb: std::collections::BTreeSet<&str> = nb.split(' ').collect();
    let inter = sa.intersection(&sb).count();
    let union = sa.union(&sb).count().max(1);
    (inter as f32 / union as f32) >= 0.6
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(&'static str);
    impl ChatBackend for Fixed {
        fn complete(&self, _p: &str) -> Result<String, BackendError> {
            Ok(self.0.to_string())
        }
    }
    struct Fails;
    impl ChatBackend for Fails {
        fn complete(&self, _p: &str) -> Result<String, BackendError> {
            Err(BackendError::Upstream("boom".into()))
        }
    }

    #[test]
    fn default_policy_routes_code_to_claude_else_gpt() {
        let p = RoutePolicy::default();
        assert_eq!(p.primary_for(TaskKind::CodeRepair), Provider::Anthropic);
        assert_eq!(p.primary_for(TaskKind::General), Provider::OpenAi);
        assert_eq!(p.primary_for(TaskKind::Reasoning), Provider::OpenAi);
    }

    #[test]
    fn order_puts_primary_first_without_duplication() {
        let p = RoutePolicy::default();
        assert_eq!(
            p.order_for(TaskKind::CodeRepair),
            vec![Provider::Anthropic, Provider::OpenAi, Provider::Local]
        );
    }

    #[test]
    fn routes_to_primary_provider() {
        let r = Router::new(RoutePolicy::default())
            .with(Provider::Anthropic, Box::new(Fixed("claude-fix")))
            .with(Provider::OpenAi, Box::new(Fixed("gpt-answer")));
        let a = r.generate(TaskKind::CodeRepair, "fix this").unwrap();
        assert_eq!(a.provider, Provider::Anthropic);
        assert_eq!(a.text, "claude-fix");
    }

    #[test]
    fn fails_over_when_primary_errors() {
        let r = Router::new(RoutePolicy::default())
            .with(Provider::Anthropic, Box::new(Fails))
            .with(Provider::OpenAi, Box::new(Fixed("gpt-backup")));
        let a = r.generate(TaskKind::CodeRepair, "x").unwrap();
        assert_eq!(a.provider, Provider::OpenAi, "should fail over to GPT");
        assert_eq!(a.text, "gpt-backup");
    }

    #[test]
    fn errors_only_when_all_providers_fail() {
        let r = Router::new(RoutePolicy::default())
            .with(Provider::Anthropic, Box::new(Fails))
            .with(Provider::OpenAi, Box::new(Fails));
        assert!(r.generate(TaskKind::General, "x").is_err());
    }

    #[test]
    fn consensus_agreement_raises_confidence() {
        let mut policy = RoutePolicy::default();
        policy.consensus = true;
        let r = Router::new(policy)
            .with(Provider::OpenAi, Box::new(Fixed("the answer is 42")))
            .with(Provider::Anthropic, Box::new(Fixed("The answer is 42")));
        let a = r.generate(TaskKind::General, "q").unwrap();
        // Two providers agreed → confidence above the single-provider baseline.
        assert!(a.confidence > 0.6, "agreement should raise confidence, got {}", a.confidence);
        assert_eq!(a.consulted.len(), 2);
    }

    #[test]
    fn consensus_conflict_lowers_confidence() {
        let mut policy = RoutePolicy::default();
        policy.consensus = true;
        let r = Router::new(policy)
            .with(Provider::OpenAi, Box::new(Fixed("totally different alpha beta")))
            .with(Provider::Anthropic, Box::new(Fixed("unrelated gamma delta epsilon")));
        let a = r.generate(TaskKind::General, "q").unwrap();
        assert!(a.confidence < 0.5, "conflict should lower confidence, got {}", a.confidence);
    }

    #[test]
    fn consensus_falls_back_to_single_when_only_one_responds() {
        let mut policy = RoutePolicy::default();
        policy.consensus = true;
        let r = Router::new(policy).with(Provider::OpenAi, Box::new(Fixed("solo")));
        let a = r.generate(TaskKind::General, "q").unwrap();
        assert_eq!(a.text, "solo");
    }
}
