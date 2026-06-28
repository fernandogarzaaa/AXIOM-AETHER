//! Per-session token-awareness and self-awareness state.
//!
//! Tracks token budget, tool call history, and compression metrics so Axiom
//! can autonomously adapt response verbosity and compression targeting.
//! Exposed via `POST /v1/budget` (agent reports remaining tokens) and
//! `GET /v1/awareness/{id}` (read back the current state).

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};

use dashmap::DashMap;

/// Per-session awareness state — lock-free where possible.
#[derive(Debug)]
pub struct AwarenessState {
    /// Agent-reported remaining token budget.
    budget_remaining: AtomicUsize,
    /// True once the agent has reported a budget.
    budget_set: AtomicBool,
    /// Target model identifier, e.g. "claude-sonnet-4-6".
    pub target_model: Mutex<Option<String>>,
    /// Tokens Axiom's own responses have consumed from the agent's budget.
    pub tokens_spent: AtomicUsize,
    /// Total MCP tool calls recorded this session.
    pub tool_calls_total: AtomicUsize,
    /// `axiom_expand` calls — each is a signal that compression dropped too much.
    pub expansion_calls: AtomicUsize,
    /// Running sum of bytes fed into the compressor (denominator for ratio).
    pub bytes_compressed_in: AtomicUsize,
    /// Running sum of bytes emitted by the compressor (numerator for ratio).
    pub bytes_compressed_out: AtomicUsize,
}

impl Default for AwarenessState {
    fn default() -> Self {
        Self {
            budget_remaining: AtomicUsize::new(0),
            budget_set: AtomicBool::new(false),
            target_model: Mutex::new(None),
            tokens_spent: AtomicUsize::new(0),
            tool_calls_total: AtomicUsize::new(0),
            expansion_calls: AtomicUsize::new(0),
            bytes_compressed_in: AtomicUsize::new(0),
            bytes_compressed_out: AtomicUsize::new(0),
        }
    }
}

impl AwarenessState {
    /// Record a token budget reported by the agent.
    pub fn set_budget(&self, remaining: usize, model: Option<String>) {
        self.budget_remaining.store(remaining, Ordering::Relaxed);
        self.budget_set.store(true, Ordering::Relaxed);
        if let Some(m) = model {
            if let Ok(mut g) = self.target_model.lock() {
                *g = Some(m);
            }
        }
    }

    /// Record that Axiom spent `cost` tokens on a single tool response.
    /// Also increments the tool-call counter and the expand counter when applicable.
    pub fn record_tool_response(&self, tool: &str, token_cost: usize) {
        self.tokens_spent.fetch_add(token_cost, Ordering::Relaxed);
        self.tool_calls_total.fetch_add(1, Ordering::Relaxed);
        if tool == "axiom_expand" {
            self.expansion_calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a compression event so the running ratio stays current.
    pub fn record_compression(&self, bytes_in: usize, bytes_out: usize) {
        self.bytes_compressed_in
            .fetch_add(bytes_in, Ordering::Relaxed);
        self.bytes_compressed_out
            .fetch_add(bytes_out, Ordering::Relaxed);
    }

    /// Current budget remaining, or `None` if the agent has not reported one.
    pub fn budget(&self) -> Option<usize> {
        if self.budget_set.load(Ordering::Relaxed) {
            Some(self.budget_remaining.load(Ordering::Relaxed))
        } else {
            None
        }
    }

    /// Token target for the compressor: 60 % of remaining budget, minimum 512.
    /// Returns `None` when no budget has been set.
    pub fn compression_target_tokens(&self) -> Option<usize> {
        self.budget()
            .map(|b| ((b as f64 * 0.6) as usize).max(512))
    }

    /// Running compression ratio (bytes_out / bytes_in), or `None` if no data yet.
    pub fn compression_ratio(&self) -> Option<f32> {
        let bytes_in = self.bytes_compressed_in.load(Ordering::Relaxed);
        if bytes_in == 0 {
            None
        } else {
            Some(
                self.bytes_compressed_out.load(Ordering::Relaxed) as f32
                    / bytes_in as f32,
            )
        }
    }

    /// True when the remaining budget is below 20 k tokens.
    pub fn is_tight(&self) -> bool {
        self.budget().map(|b| b < 20_000).unwrap_or(false)
    }

    /// A short English recommendation for the agent, if anything is noteworthy.
    pub fn recommendation(&self) -> Option<String> {
        let budget = self.budget()?;
        let spent = self.tokens_spent.load(Ordering::Relaxed);
        let pct = if budget + spent > 0 {
            spent * 100 / (budget + spent)
        } else {
            0
        };
        let expansions = self.expansion_calls.load(Ordering::Relaxed);
        let mut msgs = Vec::new();
        if budget < 20_000 {
            msgs.push(
                "Budget < 20 k — Axiom is in compact-response mode.".to_string(),
            );
        } else if budget < 50_000 {
            msgs.push(format!(
                "Budget at {} k remaining ({pct}% spent on Axiom responses).                  Consider pre-compressing large paths.",
                budget / 1_000
            ));
        }
        if expansions > 2 {
            msgs.push(format!(
                "{expansions} symbol expansions — compression may be too aggressive;                  raise AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS."
            ));
        }
        if msgs.is_empty() {
            None
        } else {
            Some(msgs.join(" "))
        }
    }
}

/// Thread-safe store of per-session (or per-client) awareness states.
#[derive(Clone, Default)]
pub struct AwarenessStore {
    inner: Arc<DashMap<String, Arc<AwarenessState>>>,
}

impl AwarenessStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
        }
    }

    /// Retrieve an existing state or insert a fresh default.
    pub fn get_or_create(&self, id: &str) -> Arc<AwarenessState> {
        self.inner
            .entry(id.to_string())
            .or_insert_with(|| Arc::new(AwarenessState::default()))
            .clone()
    }

    /// Retrieve an existing state, returning `None` if not yet created.
    pub fn get(&self, id: &str) -> Option<Arc<AwarenessState>> {
        self.inner.get(id).map(|r| r.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_set_and_read() {
        let s = AwarenessState::default();
        assert!(s.budget().is_none());
        s.set_budget(40_000, Some("claude-sonnet-4-6".into()));
        assert_eq!(s.budget(), Some(40_000));
        assert_eq!(
            s.target_model.lock().unwrap().as_deref(),
            Some("claude-sonnet-4-6")
        );
    }

    #[test]
    fn compression_target_is_sixty_pct() {
        let s = AwarenessState::default();
        s.set_budget(50_000, None);
        assert_eq!(s.compression_target_tokens(), Some(30_000));
    }

    #[test]
    fn compression_target_floors_at_512() {
        let s = AwarenessState::default();
        s.set_budget(100, None);
        assert_eq!(s.compression_target_tokens(), Some(512));
    }

    #[test]
    fn is_tight_below_20k() {
        let s = AwarenessState::default();
        s.set_budget(15_000, None);
        assert!(s.is_tight());
        s.set_budget(25_000, None);
        assert!(!s.is_tight());
    }

    #[test]
    fn compression_ratio_none_until_data() {
        let s = AwarenessState::default();
        assert!(s.compression_ratio().is_none());
        s.record_compression(1000, 400);
        let ratio = s.compression_ratio().unwrap();
        assert!((ratio - 0.4).abs() < 1e-5);
    }

    #[test]
    fn token_spend_accumulates() {
        let s = AwarenessState::default();
        s.record_tool_response("axiom_recall", 50);
        s.record_tool_response("axiom_expand", 30);
        assert_eq!(s.tokens_spent.load(Ordering::Relaxed), 80);
        assert_eq!(s.expansion_calls.load(Ordering::Relaxed), 1);
        assert_eq!(s.tool_calls_total.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn store_get_or_create_is_idempotent() {
        let store = AwarenessStore::new();
        let a = store.get_or_create("sess-1");
        let b = store.get_or_create("sess-1");
        a.set_budget(10_000, None);
        assert_eq!(b.budget(), Some(10_000));
    }
}
