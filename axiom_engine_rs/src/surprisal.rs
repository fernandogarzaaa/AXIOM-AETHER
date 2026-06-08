//! Surprisal-aware dual-track adaptation.
//!
//! SR-TTT keeps normal, compressible tokens on the fast-weight path and sends
//! high-surprisal exact identifiers into a bounded sparse residual cache. The
//! cache is intentionally token-id based so it can preserve hashes, schema
//! keys, and secrets without forcing lossy matrix updates.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

pub const DEFAULT_SURPRISAL_TAU: f32 = 5.75;
pub const DEFAULT_EXACT_CACHE_TOKENS: usize = 2048;
pub const MAX_EXACT_CACHE_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactResidualEntry {
    pub session_id: String,
    pub token_id: u32,
    pub token_text: String,
    pub surprisal: f32,
    pub sequence_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExactResidualTelemetry {
    pub sessions: usize,
    pub entries: usize,
    pub capacity_tokens: usize,
    pub estimated_bytes: u64,
    pub vram_budget_bytes: u64,
    pub last_session_id: Option<String>,
    pub last_exact_token: Option<String>,
    pub last_surprisal: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurprisalRouteReport {
    pub input_tokens: usize,
    pub fast_weight_tokens: usize,
    pub exact_residual_tokens: usize,
    pub tau: f32,
    pub max_surprisal: f32,
}

#[derive(Clone)]
pub struct ExactAttentionResidualCache {
    inner: Arc<Mutex<ExactCacheInner>>,
    capacity_tokens: usize,
    tau: f32,
}

struct ExactCacheInner {
    entries: VecDeque<ExactResidualEntry>,
    per_session: HashMap<String, usize>,
    telemetry: ExactResidualTelemetry,
}

impl Default for ExactAttentionResidualCache {
    fn default() -> Self {
        Self::new(DEFAULT_EXACT_CACHE_TOKENS, DEFAULT_SURPRISAL_TAU)
    }
}

impl ExactAttentionResidualCache {
    pub fn new(capacity_tokens: usize, tau: f32) -> Self {
        let capacity_tokens = capacity_tokens.max(1);
        let telemetry = ExactResidualTelemetry {
            capacity_tokens,
            vram_budget_bytes: MAX_EXACT_CACHE_BYTES,
            ..ExactResidualTelemetry::default()
        };
        Self {
            inner: Arc::new(Mutex::new(ExactCacheInner {
                entries: VecDeque::new(),
                per_session: HashMap::new(),
                telemetry,
            })),
            capacity_tokens,
            tau,
        }
    }

    pub fn tau(&self) -> f32 {
        self.tau
    }

    pub fn route_tokens(
        &self,
        session_id: &str,
        token_ids: &[u32],
        decoded_text: &str,
    ) -> (Vec<u32>, SurprisalRouteReport) {
        let lexical = lexical_tokens(decoded_text);
        let mut fast_weight_tokens = Vec::with_capacity(token_ids.len());
        let mut exact_entries = Vec::new();
        let mut max_surprisal = 0.0_f32;

        for (idx, token_id) in token_ids.iter().copied().enumerate() {
            let token_text = lexical
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("<tok:{token_id}>"));
            let surprisal = estimate_surprisal(token_id, &token_text);
            max_surprisal = max_surprisal.max(surprisal);
            if surprisal > self.tau {
                exact_entries.push(ExactResidualEntry {
                    session_id: session_id.to_string(),
                    token_id,
                    token_text,
                    surprisal,
                    sequence_index: idx,
                });
            } else {
                fast_weight_tokens.push(token_id);
            }
        }

        let exact_residual_tokens = exact_entries.len();
        self.append_entries(exact_entries);
        (
            fast_weight_tokens,
            SurprisalRouteReport {
                input_tokens: token_ids.len(),
                fast_weight_tokens: token_ids.len().saturating_sub(exact_residual_tokens),
                exact_residual_tokens,
                tau: self.tau,
                max_surprisal,
            },
        )
    }

    pub fn residual_prompt(&self, session_id: &str, max_tokens: usize) -> String {
        let Ok(inner) = self.inner.lock() else {
            return String::new();
        };
        let mut tokens: Vec<String> = inner
            .entries
            .iter()
            .rev()
            .filter(|entry| entry.session_id == session_id)
            .take(max_tokens)
            .map(|entry| entry.token_text.clone())
            .collect();
        tokens.reverse();
        if tokens.is_empty() {
            String::new()
        } else {
            format!(
                "<axiom_exact_attention_residual tokens=\"{}\">\n{}\n</axiom_exact_attention_residual>",
                tokens.len(),
                tokens.join(" ")
            )
        }
    }

    pub fn telemetry(&self) -> ExactResidualTelemetry {
        self.inner
            .lock()
            .map(|inner| inner.telemetry.clone())
            .unwrap_or_default()
    }

    fn append_entries(&self, entries: Vec<ExactResidualEntry>) {
        if entries.is_empty() {
            return;
        }
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        for entry in entries {
            *inner
                .per_session
                .entry(entry.session_id.clone())
                .or_insert(0) += 1;
            inner.telemetry.last_session_id = Some(entry.session_id.clone());
            inner.telemetry.last_exact_token = Some(entry.token_text.clone());
            inner.telemetry.last_surprisal = Some(entry.surprisal);
            inner.entries.push_back(entry);
            while inner.entries.len() > self.capacity_tokens {
                if let Some(evicted) = inner.entries.pop_front() {
                    if let Some(count) = inner.per_session.get_mut(&evicted.session_id) {
                        *count = count.saturating_sub(1);
                    }
                }
            }
        }
        inner.per_session.retain(|_, count| *count > 0);
        inner.telemetry.entries = inner.entries.len();
        inner.telemetry.sessions = inner.per_session.len();
        inner.telemetry.capacity_tokens = self.capacity_tokens;
        inner.telemetry.estimated_bytes = estimate_cache_bytes(inner.entries.len());
        inner.telemetry.vram_budget_bytes = MAX_EXACT_CACHE_BYTES;
    }
}

pub fn estimate_surprisal(token_id: u32, token_text: &str) -> f32 {
    let len = token_text.chars().count();
    let unique = token_text
        .chars()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let entropy_hint = if len == 0 {
        0.0
    } else {
        unique as f32 / len as f32
    };
    let has_digit = token_text.chars().any(|c| c.is_ascii_digit());
    let has_mixed_case = token_text.chars().any(|c| c.is_ascii_lowercase())
        && token_text.chars().any(|c| c.is_ascii_uppercase());
    let has_symbol = token_text
        .chars()
        .any(|c| matches!(c, '_' | '-' | ':' | '/' | '=' | '.'));
    let hash_like = len >= 16
        && token_text.chars().filter(|c| c.is_ascii_hexdigit()).count()
            >= len.saturating_mul(3) / 4;
    let schema_like = token_text.contains("api")
        || token_text.contains("key")
        || token_text.contains("schema")
        || token_text.contains("token");
    let token_noise = (token_id % 997) as f32 / 997.0;
    2.0 + entropy_hint * 2.2
        + if len >= 12 { 1.2 } else { 0.0 }
        + if has_digit { 0.8 } else { 0.0 }
        + if has_mixed_case { 0.8 } else { 0.0 }
        + if has_symbol { 0.7 } else { 0.0 }
        + if hash_like { 2.2 } else { 0.0 }
        + if schema_like { 0.9 } else { 0.0 }
        + token_noise * 0.4
}

fn lexical_tokens(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|tok| {
            tok.trim_matches(|c: char| c.is_ascii_punctuation() && c != '_' && c != '-')
                .to_string()
        })
        .filter(|tok| !tok.is_empty())
        .collect()
}

fn estimate_cache_bytes(entries: usize) -> u64 {
    (entries as u64) * 96
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_entropy_identifier_routes_to_exact_cache() {
        let cache = ExactAttentionResidualCache::new(16, DEFAULT_SURPRISAL_TAU);
        let tokens = vec![1, 2, 3, 4];
        let (fast, report) = cache.route_tokens(
            "sr",
            &tokens,
            "normal words 9f4c2e7a10b34898AABBCCDDEEFF0011",
        );
        assert!(fast.len() < tokens.len());
        assert!(report.exact_residual_tokens > 0);
        assert!(cache.telemetry().entries > 0);
        assert!(cache.residual_prompt("sr", 8).contains("9f4c2e7a"));
    }

    #[test]
    fn cache_stays_bounded() {
        let cache = ExactAttentionResidualCache::new(2, DEFAULT_SURPRISAL_TAU);
        for idx in 0..8 {
            let token = format!("abcdef0123456789{idx}XYZ");
            cache.route_tokens("s", &[idx], &token);
        }
        assert_eq!(cache.telemetry().entries, 2);
    }
}
