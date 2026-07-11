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
    /// Current-era Sonnet pricing (Sonnet 5, verified 2026-07). Also the
    /// fallback table for genuinely unrecognized model ids, since it is the
    /// most likely tier for traffic this proxy actually serves.
    const SONNET: PriceTable = PriceTable {
        input_per_mtok: 2.00,
        cache_write_5m_per_mtok: 2.50,
        cache_read_per_mtok: 0.20,
        output_per_mtok: 10.00,
    };
    /// Legacy Sonnet 4.x family pricing -- genuinely different from Sonnet 5;
    /// matched separately so the two are never silently conflated.
    const SONNET_LEGACY: PriceTable = PriceTable {
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
    /// Fable 5 (Mythos-class, above Opus). Real Anthropic list rate, verified
    /// 2026-07 (platform.claude.com/docs pricing). Mythos 5 shares this rate.
    const FABLE: PriceTable = PriceTable {
        input_per_mtok: 10.00,
        cache_write_5m_per_mtok: 12.50,
        cache_read_per_mtok: 1.00,
        output_per_mtok: 50.00,
    };
    /// Sonnet 5 list pricing from 2026-09-01 onward (Anthropic scheduled
    /// increase; the introductory rate in `SONNET` applies before that date).
    const SONNET5_POST_SEP2026: PriceTable = PriceTable {
        input_per_mtok: 3.00,
        cache_write_5m_per_mtok: 3.75,
        cache_read_per_mtok: 0.30,
        output_per_mtok: 15.00,
    };

    /// Unix seconds for 2026-09-01 00:00:00 UTC, when Sonnet-5 list pricing
    /// rises from the introductory rate to the standard rate.
    const SONNET5_PRICE_CHANGE_UNIX: u64 = 1_788_220_800;

    /// Current-date pricing for a model id. Thin wrapper over
    /// [`Self::for_model_at`] using the system clock, so the date-sensitive
    /// branch (Sonnet 5's 2026-09-01 increase) stays testable without a clock
    /// dependency.
    pub fn for_model(model: &str) -> (PriceTable, bool) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self::for_model_at(model, now)
    }

    /// Look up pricing for a model id as of `now_unix` (Unix seconds). Matches
    /// specific known versions (not just a family keyword) so a genuinely new
    /// or unrecognized version falls through to the estimated branch instead
    /// of silently inheriting a possibly-stale sibling version's rate. Sonnet
    /// 5 is date-aware: its list price rises on 2026-09-01. Unrecognized ids
    /// fall back to the current (date-aware) Sonnet 5 rate with
    /// `estimated = true` -- an honest guess, never a silent zero-cost or a
    /// hard failure.
    pub fn for_model_at(model: &str, now_unix: u64) -> (PriceTable, bool) {
        let m = model.to_ascii_lowercase();
        let current_sonnet5 = || {
            if now_unix >= Self::SONNET5_PRICE_CHANGE_UNIX {
                Self::SONNET5_POST_SEP2026
            } else {
                Self::SONNET
            }
        };
        if m.contains("haiku-4") || m.contains("haiku-3") {
            (Self::HAIKU, false)
        } else if m.contains("fable-5") || m.contains("mythos-5") {
            (Self::FABLE, false)
        } else if m.contains("opus-4") {
            (Self::OPUS, false)
        } else if m.contains("sonnet-5") {
            (current_sonnet5(), false)
        } else if m.contains("sonnet-4") {
            (Self::SONNET_LEGACY, false)
        } else {
            (current_sonnet5(), true)
        }
    }
}

/// Normalization anchor for quota units: 1 unit == 1 Sonnet-5 uncached input
/// token at the introductory ($2/MTok) rate. A FIXED unit definition, not the
/// live price -- so post-2026-09-01 Sonnet-5 input (at $3) correctly costs 1.5
/// quota units per token, reflecting its genuinely higher quota weight.
const SONNET5_INPUT_ANCHOR_PER_MTOK: f64 = 2.00;

/// Subscription "quota units" for a priced turn, normalized to the Sonnet-5
/// input anchor. Anthropic's usage windows weight tokens roughly like price
/// (cache reads cheap, output heaviest, higher tiers heavier), so we reuse the
/// per-tier price ratios as quota weights. This is the subscription-side
/// analogue of `TurnCost::usd` for the Prolonged-Session Stack.
pub fn quota_units(tc: &TurnCost, prices: &PriceTable) -> f64 {
    let w = |per_mtok: f64| per_mtok / SONNET5_INPUT_ANCHOR_PER_MTOK;
    tc.uncached_in as f64 * w(prices.input_per_mtok)
        + tc.cache_write as f64 * w(prices.cache_write_5m_per_mtok)
        + tc.cache_read as f64 * w(prices.cache_read_per_mtok)
        + tc.output as f64 * w(prices.output_per_mtok)
}

/// Counterfactual quota units for a turn with zero caching: every cached token
/// (read or write) is re-priced as a full uncached input token. The quota-side
/// analogue of [`TurnCost::uncached_equivalent_usd`] -- the denominator for
/// "how many quota units is caching / the CVM+PSS stack actually saving".
pub fn quota_units_uncached(tc: &TurnCost, prices: &PriceTable) -> f64 {
    let w = |per_mtok: f64| per_mtok / SONNET5_INPUT_ANCHOR_PER_MTOK;
    let total_in = tc.uncached_in + tc.cache_write + tc.cache_read;
    total_in as f64 * w(prices.input_per_mtok) + tc.output as f64 * w(prices.output_per_mtok)
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

    // ---- P0 (Prolonged-Session Stack): pricing + quota ledger ------------

    fn sonnet5_intro_table() -> PriceTable {
        PriceTable {
            input_per_mtok: 2.00,
            cache_write_5m_per_mtok: 2.50,
            cache_read_per_mtok: 0.20,
            output_per_mtok: 10.00,
        }
    }

    #[test]
    fn fable_5_is_priced_first_class_not_estimated() {
        let (p, estimated) = PriceTable::for_model("claude-fable-5");
        assert!(!estimated, "fable-5 must be a known model, not an estimate");
        assert!((p.input_per_mtok - 10.00).abs() < 1e-9);
        assert!((p.output_per_mtok - 50.00).abs() < 1e-9);
        assert!((p.cache_read_per_mtok - 1.00).abs() < 1e-9);
        // mythos-5 shares Fable's rate
        let (pm, em) = PriceTable::for_model("claude-mythos-5");
        assert!(!em);
        assert_eq!(pm, p);
    }

    #[test]
    fn sonnet5_pricing_is_date_aware_across_the_sep_2026_change() {
        // 2026-08-31 23:59:59 UTC vs 2026-09-01 00:00:00 UTC
        let before = PriceTable::for_model_at("claude-sonnet-5", 1_788_220_799);
        let after = PriceTable::for_model_at("claude-sonnet-5", 1_788_220_800);
        assert!(!before.1 && !after.1, "sonnet-5 is a known model either side");
        assert!((before.0.input_per_mtok - 2.00).abs() < 1e-9);
        assert!((after.0.input_per_mtok - 3.00).abs() < 1e-9);
        assert!((after.0.output_per_mtok - 15.00).abs() < 1e-9);
    }

    #[test]
    fn quota_units_normalize_sonnet5_anchor_input_to_one() {
        // 1M Sonnet-5 (intro-rate) uncached input tokens == 1_000_000 units.
        let tc = TurnCost {
            uncached_in: 1_000_000,
            cache_write: 0,
            cache_read: 0,
            output: 0,
            usd: 0.0,
            estimated: false,
        };
        let u = quota_units(&tc, &sonnet5_intro_table());
        assert!((u - 1_000_000.0).abs() < 1e-6, "got {u}");
    }

    #[test]
    fn quota_units_weight_output_heaviest() {
        let s = sonnet5_intro_table();
        let out = TurnCost {
            uncached_in: 0,
            cache_write: 0,
            cache_read: 0,
            output: 1000,
            usd: 0.0,
            estimated: false,
        };
        let inp = TurnCost {
            uncached_in: 1000,
            cache_write: 0,
            cache_read: 0,
            output: 0,
            usd: 0.0,
            estimated: false,
        };
        assert!(quota_units(&out, &s) > quota_units(&inp, &s));
    }

    #[test]
    fn quota_units_uncached_reprices_cached_tokens_as_full_input() {
        let s = sonnet5_intro_table();
        // 500k cache reads (cheap live) become full input in the counterfactual.
        let tc = TurnCost {
            uncached_in: 0,
            cache_write: 0,
            cache_read: 500_000,
            output: 0,
            usd: 0.0,
            estimated: false,
        };
        // live: 500k * (0.20/2.00) = 50k units; uncached: 500k * (2.00/2.00) = 500k units.
        assert!((quota_units(&tc, &s) - 50_000.0).abs() < 1e-6);
        assert!((quota_units_uncached(&tc, &s) - 500_000.0).abs() < 1e-6);
        assert!(quota_units_uncached(&tc, &s) > quota_units(&tc, &s));
    }

    #[test]
    fn quota_units_post_sep_sonnet5_input_costs_one_and_a_half_units() {
        // Post-Sep Sonnet-5 input is $3/MTok; against the fixed $2 anchor that
        // is 1.5 units/token -- the higher real quota weight is preserved.
        let (post, _) = PriceTable::for_model_at("claude-sonnet-5", 1_788_220_800);
        let tc = TurnCost {
            uncached_in: 1_000_000,
            cache_write: 0,
            cache_read: 0,
            output: 0,
            usd: 0.0,
            estimated: false,
        };
        let u = quota_units(&tc, &post);
        assert!((u - 1_500_000.0).abs() < 1e-6, "got {u}");
    }

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
    fn turn_cost_unknown_model_falls_back_to_current_sonnet_and_flags_estimated() {
        let usage = json!({ "input_tokens": 1000, "output_tokens": 100 });
        let tc = turn_cost("claude-super-6-hypothetical", &usage).unwrap();
        assert!(tc.estimated);
        // Falls back to the CURRENT-date Sonnet 5 rate (now date-aware across
        // the 2026-09-01 change), so derive the expected cost from the same
        // resolved table rather than hardcoding a rate that would break in CI
        // after the price change.
        let (prices, _) = PriceTable::for_model("claude-super-6-hypothetical");
        let expected =
            1000.0 / 1e6 * prices.input_per_mtok + 100.0 / 1e6 * prices.output_per_mtok;
        assert!((tc.usd - expected).abs() < 1e-9);
    }

    #[test]
    fn for_model_distinguishes_sonnet_5_from_legacy_sonnet_4_pricing() {
        // These two families have genuinely different real-world prices;
        // conflating them under one "contains sonnet" match was the bug.
        // Pin the date (epoch = pre-2026-09-01) so the intro-rate assertion
        // stays deterministic after Sonnet 5's scheduled price change.
        let (sonnet5, est5) = PriceTable::for_model_at("claude-sonnet-5", 0);
        assert!(!est5);
        assert!((sonnet5.input_per_mtok - 2.00).abs() < 1e-9);

        let (sonnet4, est4) = PriceTable::for_model("claude-sonnet-4-6");
        assert!(!est4);
        assert!((sonnet4.input_per_mtok - 3.00).abs() < 1e-9);

        assert_ne!(sonnet5, sonnet4, "distinct model families must not share a price table");
    }

    #[test]
    fn for_model_flags_a_genuinely_new_sonnet_version_as_estimated() {
        // A hypothetical future version that isn't sonnet-5 or sonnet-4-x
        // must not silently inherit either table's pricing.
        let (_, estimated) = PriceTable::for_model("claude-sonnet-7");
        assert!(estimated, "an unrecognized sonnet version must be flagged estimated");
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
