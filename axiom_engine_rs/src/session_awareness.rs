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
    /// Dollar-true cost accounting (see `cost_ledger`). USD is stored as
    /// micro-dollars (1e-6 USD) in an integer atomic so accumulation across
    /// many turns stays exact instead of drifting under repeated f64 adds.
    cost_usd_micros: AtomicUsize,
    uncached_equivalent_usd_micros: AtomicUsize,
    uncached_input_tokens: AtomicUsize,
    cache_write_tokens: AtomicUsize,
    cache_read_tokens: AtomicUsize,
    cost_output_tokens: AtomicUsize,
    cost_estimated: AtomicBool,
    /// S4 (CVM cost stack): running total of prefix-diet tokens removed
    /// (see `prefix_diet::DietReport`) -- the dedup tier's own contribution
    /// to S0's uncached-equivalent counterfactual.
    prefix_diet_tokens_removed: AtomicUsize,
    /// S6 (CVM cost stack) actuarial keepalive: pings sent and their
    /// estimated $ saved (a cache-read re-price avoided a full cache-write
    /// re-price) -- always labeled `estimated` since the counterfactual it
    /// avoided never actually happens when the ping works.
    keepalive_pings_sent: AtomicUsize,
    keepalive_estimated_usd_saved_micros: AtomicUsize,
    /// S3 (CVM cost stack) digest admission control: running totals of
    /// digested tool_result blocks and their bytes in/out.
    digest_blocks: AtomicUsize,
    digest_bytes_in: AtomicUsize,
    digest_bytes_out: AtomicUsize,
    /// P0 (Prolonged-Session Stack): subscription quota units consumed this
    /// session (see `cost_ledger::quota_units`), stored as units x 1e6 in an
    /// integer atomic so accumulation stays exact. The subscription-side
    /// analogue of `cost_usd_micros`.
    quota_units_micros: AtomicUsize,
}

/// A snapshot of a session's accumulated dollar-true cost, for `/metrics` and
/// `axiom_status`/`GET /v1/awareness/:id`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CostSummary {
    pub usd_total: f64,
    /// What the accumulated turns would have cost with zero caching (every
    /// cached token billed as full-price input). The counterfactual that
    /// makes "how much is caching/CVM saving" answerable in dollars.
    pub usd_uncached_equivalent: f64,
    pub uncached_input_tokens: u64,
    pub cache_write_tokens: u64,
    pub cache_read_tokens: u64,
    pub output_tokens: u64,
    /// True if any accumulated turn used an estimated (non-table) price.
    pub estimated: bool,
    /// S4: running total of tokens removed by prefix-diet dedup.
    pub prefix_diet_tokens_removed: u64,
    /// S6: running total of keepalive pings sent this session.
    pub keepalive_pings_sent: u64,
    /// S6: estimated $ saved by those pings (always an estimate).
    pub keepalive_estimated_usd_saved: f64,
    /// S3: running total of tool_result blocks digested.
    pub digest_blocks: u64,
    /// S3: running total of original bytes before digestion.
    pub digest_bytes_in: u64,
    /// S3: running total of bytes after digestion (stub + digest text).
    pub digest_bytes_out: u64,
    /// P0 (PSS): subscription quota units consumed this session.
    pub quota_units_total: f64,
}

impl CostSummary {
    /// Fraction of input-side tokens served from cache (reads / (reads +
    /// writes + uncached)). `None`-equivalent is 0.0 when there is no input
    /// traffic yet, which is the correct "no signal" value for a ratio.
    pub fn cache_hit_rate(&self) -> f64 {
        let denom = self.cache_read_tokens + self.cache_write_tokens + self.uncached_input_tokens;
        if denom == 0 {
            0.0
        } else {
            self.cache_read_tokens as f64 / denom as f64
        }
    }
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
            cost_usd_micros: AtomicUsize::new(0),
            uncached_equivalent_usd_micros: AtomicUsize::new(0),
            uncached_input_tokens: AtomicUsize::new(0),
            cache_write_tokens: AtomicUsize::new(0),
            cache_read_tokens: AtomicUsize::new(0),
            cost_output_tokens: AtomicUsize::new(0),
            cost_estimated: AtomicBool::new(false),
            prefix_diet_tokens_removed: AtomicUsize::new(0),
            keepalive_pings_sent: AtomicUsize::new(0),
            keepalive_estimated_usd_saved_micros: AtomicUsize::new(0),
            digest_blocks: AtomicUsize::new(0),
            digest_bytes_in: AtomicUsize::new(0),
            digest_bytes_out: AtomicUsize::new(0),
            quota_units_micros: AtomicUsize::new(0),
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

    /// Record one priced API turn (see `cost_ledger::turn_cost`) into the
    /// session's running dollar total. `prices` is the price table the turn
    /// was actually priced under (i.e. the same one `turn_cost` resolved for
    /// this turn's model) -- passed in rather than re-resolved from a model
    /// string so this function stays a pure accumulator.
    pub fn record_turn_cost(
        &self,
        tc: &crate::cost_ledger::TurnCost,
        prices: &crate::cost_ledger::PriceTable,
    ) {
        // usd is always >= 0 and bounded by realistic per-turn spend, so the
        // micro-dollar conversion cannot meaningfully overflow a usize here.
        let micros = (tc.usd * 1_000_000.0).round() as usize;
        self.cost_usd_micros.fetch_add(micros, Ordering::Relaxed);
        let uncached_micros = (tc.uncached_equivalent_usd(prices) * 1_000_000.0).round() as usize;
        self.uncached_equivalent_usd_micros
            .fetch_add(uncached_micros, Ordering::Relaxed);
        self.uncached_input_tokens
            .fetch_add(tc.uncached_in as usize, Ordering::Relaxed);
        self.cache_write_tokens
            .fetch_add(tc.cache_write as usize, Ordering::Relaxed);
        self.cache_read_tokens
            .fetch_add(tc.cache_read as usize, Ordering::Relaxed);
        self.cost_output_tokens
            .fetch_add(tc.output as usize, Ordering::Relaxed);
        if tc.estimated {
            self.cost_estimated.store(true, Ordering::Relaxed);
        }
    }

    /// Current dollar-true cost summary for this session.
    pub fn cost_summary(&self) -> CostSummary {
        CostSummary {
            usd_total: self.cost_usd_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            usd_uncached_equivalent: self.uncached_equivalent_usd_micros.load(Ordering::Relaxed)
                as f64
                / 1_000_000.0,
            uncached_input_tokens: self.uncached_input_tokens.load(Ordering::Relaxed) as u64,
            cache_write_tokens: self.cache_write_tokens.load(Ordering::Relaxed) as u64,
            cache_read_tokens: self.cache_read_tokens.load(Ordering::Relaxed) as u64,
            output_tokens: self.cost_output_tokens.load(Ordering::Relaxed) as u64,
            estimated: self.cost_estimated.load(Ordering::Relaxed),
            prefix_diet_tokens_removed: self.prefix_diet_tokens_removed.load(Ordering::Relaxed)
                as u64,
            keepalive_pings_sent: self.keepalive_pings_sent.load(Ordering::Relaxed) as u64,
            keepalive_estimated_usd_saved: self
                .keepalive_estimated_usd_saved_micros
                .load(Ordering::Relaxed) as f64
                / 1_000_000.0,
            digest_blocks: self.digest_blocks.load(Ordering::Relaxed) as u64,
            digest_bytes_in: self.digest_bytes_in.load(Ordering::Relaxed) as u64,
            digest_bytes_out: self.digest_bytes_out.load(Ordering::Relaxed) as u64,
            quota_units_total: self.quota_units_micros.load(Ordering::Relaxed) as f64
                / 1_000_000.0,
        }
    }

    /// Record one priced turn's subscription quota units (P0/PSS). Pass the
    /// value from `cost_ledger::quota_units` for this turn.
    pub fn record_turn_quota(&self, units: f64) {
        let micros = (units.max(0.0) * 1_000_000.0).round() as usize;
        self.quota_units_micros
            .fetch_add(micros, Ordering::Relaxed);
    }

    /// Record one request's prefix-diet dedup savings (S4).
    pub fn record_prefix_diet(&self, tokens_removed: usize) {
        self.prefix_diet_tokens_removed
            .fetch_add(tokens_removed, Ordering::Relaxed);
    }

    /// Record one S6 keepalive ping and its estimated $ saved.
    pub fn record_keepalive_ping(&self, estimated_usd_saved: f64) {
        self.keepalive_pings_sent.fetch_add(1, Ordering::Relaxed);
        let micros = (estimated_usd_saved.max(0.0) * 1_000_000.0).round() as usize;
        self.keepalive_estimated_usd_saved_micros
            .fetch_add(micros, Ordering::Relaxed);
    }

    /// Record one request's digest admission control activity (S3).
    pub fn record_digest(&self, blocks: usize, bytes_in: usize, bytes_out: usize) {
        self.digest_blocks.fetch_add(blocks, Ordering::Relaxed);
        self.digest_bytes_in.fetch_add(bytes_in, Ordering::Relaxed);
        self.digest_bytes_out
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
        let pct = (spent * 100).checked_div(budget + spent).unwrap_or(0);
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
    fn record_turn_cost_accumulates_usd_and_uncached_equivalent() {
        let s = AwarenessState::default();
        let tc1 = crate::cost_ledger::TurnCost {
            uncached_in: 1000,
            cache_write: 0,
            cache_read: 0,
            output: 500,
            usd: 0.0105,
            estimated: false,
        };
        let tc2 = crate::cost_ledger::TurnCost {
            uncached_in: 0,
            cache_write: 0,
            cache_read: 80_000,
            output: 300,
            usd: 0.0285,
            estimated: false,
        };
        let (prices, _) = crate::cost_ledger::PriceTable::for_model("claude-sonnet-4-6");
        s.record_turn_cost(&tc1, &prices);
        s.record_turn_cost(&tc2, &prices);
        let summary = s.cost_summary();
        assert!((summary.usd_total - (0.0105 + 0.0285)).abs() < 1e-9);
        assert_eq!(summary.cache_read_tokens, 80_000);
        assert_eq!(summary.uncached_input_tokens, 1000);
        assert_eq!(summary.output_tokens, 800);
        assert!(!summary.estimated, "no turn was estimated");
    }

    #[test]
    fn record_turn_cost_marks_summary_estimated_if_any_turn_was() {
        let s = AwarenessState::default();
        let (prices, _) = crate::cost_ledger::PriceTable::for_model("claude-sonnet-4-6");
        s.record_turn_cost(
            &crate::cost_ledger::TurnCost {
                estimated: true,
                ..Default::default()
            },
            &prices,
        );
        assert!(s.cost_summary().estimated);
    }

    #[test]
    fn cache_hit_rate_is_reads_over_reads_plus_writes_plus_uncached() {
        let s = AwarenessState::default();
        let (prices, _) = crate::cost_ledger::PriceTable::for_model("claude-sonnet-4-6");
        s.record_turn_cost(
            &crate::cost_ledger::TurnCost {
                uncached_in: 100,
                cache_write: 100,
                cache_read: 800,
                ..Default::default()
            },
            &prices,
        );
        let rate = s.cost_summary().cache_hit_rate();
        assert!((rate - 0.8).abs() < 1e-9);
    }

    #[test]
    fn cost_summary_uncached_equivalent_exceeds_actual_and_accumulates() {
        let s = AwarenessState::default();
        let (prices, _) = crate::cost_ledger::PriceTable::for_model("claude-sonnet-4-6");
        let tc = crate::cost_ledger::TurnCost {
            uncached_in: 1000,
            cache_write: 2000,
            cache_read: 77_000,
            output: 500,
            usd: 0.0, // irrelevant to this test, which only checks the counterfactual
            estimated: false,
        };
        s.record_turn_cost(&tc, &prices);
        let summary = s.cost_summary();
        let expected_uncached = tc.uncached_equivalent_usd(&prices);
        assert!(
            (summary.usd_uncached_equivalent - expected_uncached).abs() < 1e-6,
            "{} vs {}",
            summary.usd_uncached_equivalent,
            expected_uncached
        );
        assert!(summary.usd_uncached_equivalent > 0.0);
    }

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
