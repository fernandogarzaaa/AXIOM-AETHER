//! Dollar-true, cache-aware cost accounting.
//!
//! Axiom's byte-savings counters (`axiom_savings_*`) measure bytes removed
//! from the outbound request, not dollars spent. Under Anthropic prompt
//! caching, rewriting a byte that was going to be a 0.1x-priced cache read
//! turns it into a 1.0x (or 1.25x write) charge -- a "savings" that costs
//! 10x more. This module reads the ground truth Anthropic already returns
//! in every response (`usage.input_tokens`, `usage.cache_creation_input_tokens`,
//! `usage.cache_read_input_tokens`, `usage.output_tokens`) and turns it into
//! real USD, plus a counterfactual (what this turn would have cost with zero
//! caching), so every later CVM mechanism is judged by money, not bytes.
//!
//! See docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S0.

use serde_json::Value;

/// Per-million-token USD pricing for one model, one cache-write tier (5-minute
/// TTL; 1-hour writes are priced separately by Anthropic but Claude Code does
/// not request them, so this table only carries what Axiom's traffic uses).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriceTable {
    pub input_per_mtok: f64,
    pub cache_write_5m_per_mtok: f64,
    pub cache_read_per_mtok: f64,
    pub output_per_mtok: f64,
}

impl PriceTable {
    const SONNET: PriceTable = PriceTable {
        input_per_mtok: 3.00,
        cache_write_5m_per_mtok: 3.75,
        cache_read_per_mtok: 0.30,
        output_per_mtok: 15.00,
    };
    const OPUS: PriceTable = PriceTable {
        input_per_mtok: 5.00,
        cache_write_5m_per_mtok: 6.25,
        cache_read_per_mtok: 0.50,
        output_per_mtok: 25.00,
    };
    const HAIKU: PriceTable = PriceTable {
        input_per_mtok: 1.00,
        cache_write_5m_per_mtok: 1.25,
        cache_read_per_mtok: 0.10,
        output_per_mtok: 5.00,
    };

    /// Look up pricing for a model id. Falls back to Sonnet-tier pricing with
    /// `estimated = true` for unrecognized model ids (new/renamed models),
    /// so a price-table gap degrades to an honest estimate, never a silent
    /// zero-cost or a hard failure.
    pub fn for_model(model: &str) -> (PriceTable, bool) {
        let m = model.to_ascii_lowercase();
        if m.contains("haiku") {
            (Self::HAIKU, false)
        } else if m.contains("opus") {
            (Self::OPUS, false)
        } else if m.contains("sonnet") {
            (Self::SONNET, false)
        } else {
            (Self::SONNET, true)
        }
    }
}

/// The token/dollar breakdown of a single API turn.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TurnCost {
    /// Tokens billed at full (uncached) input price.
    pub uncached_in: u64,
    /// Tokens billed at the cache-write premium (new prefix written this turn).
    pub cache_write: u64,
    /// Tokens billed at the cache-read discount (prefix reused from cache).
    pub cache_read: u64,
    /// Output tokens generated this turn.
    pub output: u64,
    /// Total USD cost of this turn under the resolved price table.
    pub usd: f64,
    /// True when the model id was not in the static price table and Sonnet
    /// pricing was used as an estimate.
    pub estimated: bool,
}

impl TurnCost {
    /// What this turn would have cost with no caching at all: every token
    /// that was cached (read or write) becomes a full-price input token.
    /// This is the denominator for "how much is caching/CVM actually saving".
    pub fn uncached_equivalent_usd(&self, prices: &PriceTable) -> f64 {
        let total_in = self.uncached_in + self.cache_write + self.cache_read;
        total_in as f64 / 1_000_000.0 * prices.input_per_mtok
            + self.output as f64 / 1_000_000.0 * prices.output_per_mtok
    }
}

/// Parse an Anthropic `usage` object into a priced `TurnCost`.
///
/// Missing individual fields are treated as zero (Anthropic omits
/// `cache_creation_input_tokens`/`cache_read_input_tokens` entirely when a
/// request does not use caching at all). Returns `None` only when `usage`
/// itself is absent or not a JSON object, since that means there is nothing
/// to account.
pub fn turn_cost(model: &str, usage: &Value) -> Option<TurnCost> {
    let obj = usage.as_object()?;
    let field = |k: &str| obj.get(k).and_then(Value::as_u64).unwrap_or(0);

    let uncached_in = field("input_tokens");
    let cache_write = field("cache_creation_input_tokens");
    let cache_read = field("cache_read_input_tokens");
    let output = field("output_tokens");

    let (prices, estimated) = PriceTable::for_model(model);
    let usd = uncached_in as f64 / 1_000_000.0 * prices.input_per_mtok
        + cache_write as f64 / 1_000_000.0 * prices.cache_write_5m_per_mtok
        + cache_read as f64 / 1_000_000.0 * prices.cache_read_per_mtok
        + output as f64 / 1_000_000.0 * prices.output_per_mtok;

    Some(TurnCost {
        uncached_in,
        cache_write,
        cache_read,
        output,
        usd,
        estimated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn turn_cost_prices_all_four_fields_sonnet() {
        let usage = json!({
            "input_tokens": 1000,
            "cache_creation_input_tokens": 2000,
            "cache_read_input_tokens": 500_000,
            "output_tokens": 1500,
        });
        let tc = turn_cost("claude-sonnet-4-6", &usage).unwrap();
        assert_eq!(tc.uncached_in, 1000);
        assert_eq!(tc.cache_write, 2000);
        assert_eq!(tc.cache_read, 500_000);
        assert_eq!(tc.output, 1500);
        assert!(!tc.estimated);
        let expected = 1000.0 / 1e6 * 3.00
            + 2000.0 / 1e6 * 3.75
            + 500_000.0 / 1e6 * 0.30
            + 1500.0 / 1e6 * 15.00;
        assert!((tc.usd - expected).abs() < 1e-9, "{} vs {}", tc.usd, expected);
    }

    #[test]
    fn turn_cost_missing_fields_default_to_zero() {
        let usage = json!({ "input_tokens": 800, "output_tokens": 200 });
        let tc = turn_cost("claude-sonnet-4-6", &usage).unwrap();
        assert_eq!(tc.cache_write, 0);
        assert_eq!(tc.cache_read, 0);
        let expected = 800.0 / 1e6 * 3.00 + 200.0 / 1e6 * 15.00;
        assert!((tc.usd - expected).abs() < 1e-9);
    }

    #[test]
    fn turn_cost_returns_none_without_usage_object() {
        assert!(turn_cost("claude-sonnet-4-6", &json!(null)).is_none());
        assert!(turn_cost("claude-sonnet-4-6", &json!("not an object")).is_none());
    }

    #[test]
    fn turn_cost_unknown_model_falls_back_to_sonnet_and_flags_estimated() {
        let usage = json!({ "input_tokens": 1000, "output_tokens": 100 });
        let tc = turn_cost("claude-super-6-hypothetical", &usage).unwrap();
        assert!(tc.estimated);
        let expected = 1000.0 / 1e6 * 3.00 + 100.0 / 1e6 * 15.00;
        assert!((tc.usd - expected).abs() < 1e-9);
    }

    #[test]
    fn turn_cost_resolves_haiku_and_opus_tiers() {
        let usage = json!({ "input_tokens": 1_000_000, "output_tokens": 0 });
        let haiku = turn_cost("claude-haiku-4-5", &usage).unwrap();
        assert!((haiku.usd - 1.00).abs() < 1e-9);
        let opus = turn_cost("claude-opus-4-8", &usage).unwrap();
        assert!((opus.usd - 5.00).abs() < 1e-9);
    }

    #[test]
    fn uncached_equivalent_prices_all_input_tiers_as_full_price() {
        let usage = json!({
            "input_tokens": 1000,
            "cache_creation_input_tokens": 2000,
            "cache_read_input_tokens": 77_000,
            "output_tokens": 500,
        });
        let tc = turn_cost("claude-sonnet-4-6", &usage).unwrap();
        let (prices, _) = PriceTable::for_model("claude-sonnet-4-6");
        let uncached = tc.uncached_equivalent_usd(&prices);
        let expected = (1000.0 + 2000.0 + 77_000.0) / 1e6 * 3.00 + 500.0 / 1e6 * 15.00;
        assert!((uncached - expected).abs() < 1e-9);
        assert!(uncached > tc.usd, "uncached equivalent must exceed the actual (cached) cost");
    }
}
