//! Static-first cache stack: breakpoint constructor, exact response cache,
//! OpenAI usage parse.
//!
//! Complements (never duplicates) the upstream cost modules:
//! - [`crate::cache_safety`] *honors* client `cache_control` breakpoints.
//!   This module *places* one on bodies that carry none, at the caller-declared
//!   static/dynamic boundary, and refuses when any marker already exists.
//! - `axiom_engine/response_cache.py` owns the Python exact-cache path. The
//!   [`ExactCache`] here ports its fingerprint semantics to Rust for the proxy
//!   path: SHA-256 over canonical JSON
//!   `{"model","system","messages","max_tokens"}`. `temperature` is excluded
//!   from the key by design, so this cache is for deterministic
//!   (temperature ~ 0) traffic only; sampling workloads must bypass it.
//! - [`crate::cost_ledger`] parses Anthropic `usage` into USD. [`OpenAiUsage`]
//!   parses OpenAI `usage.prompt_tokens_details.cached_tokens` into counts and
//!   a hit ratio. No OpenAI price table is carried here; inventing one would be
//!   a fabrication, not a feature.
//!
//! No savings are claimed by this module. Hit rates are measured outputs of a
//! future traffic pilot, never premises.
//!
//! See experiments/cache_stack/manifest.json (frozen 2026-09-07).

use std::collections::{HashMap, VecDeque};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// L0: static-boundary breakpoint constructor (Anthropic `messages` bodies)
// ---------------------------------------------------------------------------

/// Place one ephemeral breakpoint at the last static message.
///
/// `static_len` counts leading entries of `body["messages"]` that are static
/// (system echo, tool schemas, few-shots, stable docs). The marker is attached
/// to message index `static_len - 1`. Returns `true` iff a marker was placed.
///
/// Refusals (returns `false`, body untouched):
/// - any `cache_control` key exists anywhere in the body (the client marks its
///   own breakpoints; moving them would invalidate its prefix cache --
///   see [`crate::cache_safety`]);
/// - `static_len == 0`, no `messages` array, or `static_len > messages.len()`.
/// - the boundary message has empty content.
///
/// Messages at or after `static_len` are never touched.
pub fn ensure_static_breakpoint(body: &mut Value, static_len: usize) -> bool {
    if has_marker(body) || static_len == 0 {
        return false;
    }
    let Some(msgs) = body.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return false;
    };
    if static_len > msgs.len() {
        return false;
    }
    let boundary = &mut msgs[static_len - 1];
    let Some(content) = boundary.get_mut("content") else {
        return false;
    };
    match content {
        Value::Array(blocks) => {
            let Some(last) = blocks.iter_mut().rev().find_map(|b| b.as_object_mut()) else {
                return false;
            };
            last.insert(
                "cache_control".to_string(),
                json!({"type": "ephemeral"}),
            );
            true
        }
        Value::String(text) => {
            if text.is_empty() {
                return false;
            }
            *content = json!([{
                "type": "text",
                "text": std::mem::take(text),
                "cache_control": {"type": "ephemeral"},
            }]);
            true
        }
        _ => false,
    }
}

fn has_marker(v: &Value) -> bool {
    match v {
        Value::Object(map) => {
            map.contains_key("cache_control") || map.values().any(has_marker)
        }
        Value::Array(arr) => arr.iter().any(has_marker),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// L1: exact-match response cache (deterministic traffic only)
// ---------------------------------------------------------------------------

/// Cache key: SHA-256 over canonical JSON of
/// `{"model","system","messages","max_tokens"}`. Missing fields canonicalize
/// to `null`, so two requests hash equal iff these four fields are equal.
pub fn exact_cache_key(model: &str, system: &Value, messages: &Value, max_tokens: &Value) -> String {
    let canon = json!({
        "model": model,
        "system": system,
        "messages": messages,
        "max_tokens": max_tokens,
    });
    // serde_json::Value serializes maps with keys in sorted order, so this is
    // canonical up to the field set above.
    let bytes = serde_json::to_vec(&canon).expect("json");
    format!("{:x}", Sha256::digest(bytes))
}

/// Bounded exact-match store. Oldest-inserted entries evict first when
/// `capacity` is exceeded. Single-threaded; wrap in a Mutex at the call site
/// if shared across threads.
pub struct ExactCache {
    capacity: usize,
    map: HashMap<String, Value>,
    order: VecDeque<String>,
    pub hits: u64,
    pub misses: u64,
}

impl ExactCache {
    /// `capacity == 0` disables storage (every lookup misses).
    pub fn new(capacity: usize) -> Self {
        Self { capacity, map: HashMap::new(), order: VecDeque::new(), hits: 0, misses: 0 }
    }

    pub fn lookup(&mut self, key: &str) -> Option<Value> {
        match self.map.get(key) {
            Some(v) => {
                self.hits += 1;
                Some(v.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    pub fn store(&mut self, key: String, response: Value) {
        if self.capacity == 0 {
            return;
        }
        if !self.map.contains_key(&key) {
            while self.order.len() >= self.capacity {
                if let Some(old) = self.order.pop_front() {
                    self.map.remove(&old);
                }
            }
            self.order.push_back(key.clone());
        }
        self.map.insert(key, response);
    }

    pub fn entries(&self) -> usize {
        self.map.len()
    }

    /// `None` when nothing has been looked up yet (not 0.0: no data, no rate).
    pub fn hit_rate(&self) -> Option<f64> {
        let total = self.hits + self.misses;
        if total == 0 {
            None
        } else {
            Some(self.hits as f64 / total as f64)
        }
    }
}

// ---------------------------------------------------------------------------
// L2: OpenAI usage parse (counts only, no USD)
// ---------------------------------------------------------------------------

/// Token counts from an OpenAI chat-completions response object.
/// Missing fields read as 0, meaning "not reported" -- never as measured zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OpenAiUsage {
    pub input_tokens: u64,
    pub cached_read_tokens: u64,
    pub output_tokens: u64,
}

impl OpenAiUsage {
    /// Fraction of input tokens served from prefix cache. `None` when
    /// `input_tokens == 0` (no data, no ratio).
    pub fn cache_hit_ratio(&self) -> Option<f64> {
        if self.input_tokens == 0 {
            None
        } else {
            Some(self.cached_read_tokens as f64 / self.input_tokens as f64)
        }
    }
}

fn u64_field(v: &Value, key: &str) -> u64 {
    v.get(key).and_then(|x| x.as_u64()).unwrap_or(0)
}

/// Extract [`OpenAiUsage`] from a chat-completions response object.
/// Reads `usage.prompt_tokens`, `usage.prompt_tokens_details.cached_tokens`,
/// `usage.completion_tokens`. Absent `usage` yields all zeros.
pub fn parse_openai_usage(response: &Value) -> OpenAiUsage {
    let usage = response.get("usage");
    let (input, output) = match usage {
        Some(u) => (u64_field(u, "prompt_tokens"), u64_field(u, "completion_tokens")),
        None => (0, 0),
    };
    let cached = usage
        .and_then(|u| u.get("prompt_tokens_details"))
        .map(|d| u64_field(d, "cached_tokens"))
        .unwrap_or(0);
    OpenAiUsage { input_tokens: input, cached_read_tokens: cached.min(input), output_tokens: output }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- L0 --

    #[test]
    fn breakpoint_placed_on_last_static_string_message() {
        let mut body = json!({"model": "m", "messages": [
            {"role": "system", "content": "static docs"},
            {"role": "user", "content": "dynamic q"},
        ]});
        assert!(ensure_static_breakpoint(&mut body, 1));
        let c = &body["messages"][0]["content"];
        assert_eq!(c[0]["cache_control"]["type"], json!("ephemeral"));
        // tail untouched: still a bare string, no marker
        assert!(body["messages"][1]["content"].is_string());
        assert!(!has_marker(&body["messages"][1]));
    }

    #[test]
    fn breakpoint_attaches_to_last_block_of_array_content() {
        let mut body = json!({"model": "m", "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ]},
            {"role": "user", "content": "tail"},
        ]});
        assert!(ensure_static_breakpoint(&mut body, 1));
        let blocks = body["messages"][0]["content"].as_array().unwrap();
        assert!(blocks[0].get("cache_control").is_none());
        assert_eq!(blocks[1]["cache_control"]["type"], json!("ephemeral"));
    }

    #[test]
    fn breakpoint_refuses_when_client_marks_its_own() {
        let mut body = json!({"model": "m", "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
            ]},
        ]});
        let before = body.clone();
        assert!(!ensure_static_breakpoint(&mut body, 1));
        assert_eq!(body, before);
    }

    #[test]
    fn breakpoint_refuses_degenerate_inputs() {
        let mut no_msgs = json!({"model": "m"});
        assert!(!ensure_static_breakpoint(&mut no_msgs, 1));
        let mut body = json!({"model": "m", "messages": [{"role": "u", "content": "x"}]});
        let before = body.clone();
        assert!(!ensure_static_breakpoint(&mut body, 0));
        assert!(!ensure_static_breakpoint(&mut body, 5));
        assert_eq!(body, before);
        let mut empty = json!({"model": "m", "messages": [{"role": "u", "content": ""}]});
        assert!(!ensure_static_breakpoint(&mut empty, 1));
    }

    // -- L1 --

    #[test]
    fn cache_key_stable_and_field_sensitive() {
        let sys = json!("s");
        let msgs = json!([{"role": "user", "content": "hi"}]);
        let mt = json!(100);
        let k1 = exact_cache_key("m", &sys, &msgs, &mt);
        let k2 = exact_cache_key("m", &sys, &msgs, &mt);
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 64);
        assert_ne!(k1, exact_cache_key("m2", &sys, &msgs, &mt));
        assert_ne!(k1, exact_cache_key("m", &sys, &json!([{"role": "user", "content": "bye"}]), &mt));
    }

    #[test]
    fn exact_cache_hit_miss_stats_and_eviction() {
        let mut c = ExactCache::new(2);
        assert_eq!(c.hit_rate(), None);
        assert!(c.lookup("k1").is_none());
        c.store("k1".into(), json!({"a": 1}));
        c.store("k2".into(), json!({"a": 2}));
        assert!(c.lookup("k1").is_some());
        assert_eq!(c.entries(), 2);
        // over capacity: oldest (k1) evicts
        c.store("k3".into(), json!({"a": 3}));
        assert_eq!(c.entries(), 2);
        assert!(c.lookup("k2").is_some());
        // hits=2 (k1,k2), misses=1
        assert!((c.hit_rate().unwrap() - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn zero_capacity_cache_never_stores() {
        let mut c = ExactCache::new(0);
        c.store("k".into(), json!(1));
        assert_eq!(c.entries(), 0);
        assert!(c.lookup("k").is_none());
    }

    // -- L2 --

    #[test]
    fn openai_usage_parses_cached_tokens() {
        let r = json!({"usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 900, "audio_tokens": 0},
        }});
        let u = parse_openai_usage(&r);
        assert_eq!(u, OpenAiUsage { input_tokens: 1000, cached_read_tokens: 900, output_tokens: 50 });
        assert!((u.cache_hit_ratio().unwrap() - 0.9).abs() < 1e-12);
    }

    #[test]
    fn openai_usage_absent_fields_read_zero_no_ratio_without_input() {
        let r = json!({"usage": {"prompt_tokens": 10, "completion_tokens": 2}});
        let u = parse_openai_usage(&r);
        assert_eq!(u.cached_read_tokens, 0);
        assert_eq!(u.cache_hit_ratio(), Some(0.0));
        let empty = parse_openai_usage(&json!({}));
        assert_eq!(empty, OpenAiUsage::default());
        assert_eq!(empty.cache_hit_ratio(), None);
    }

    #[test]
    fn openai_usage_cached_clamped_to_input() {
        // Defensive: a provider reporting cached > prompt must not yield >1.0.
        let r = json!({"usage": {"prompt_tokens": 10, "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 999}}});
        let u = parse_openai_usage(&r);
        assert_eq!(u.cached_read_tokens, 10);
        assert_eq!(u.cache_hit_ratio(), Some(1.0));
    }
}
