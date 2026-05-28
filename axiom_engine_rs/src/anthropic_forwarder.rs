//! Outbound bridge to the real Anthropic Messages API used by the
//! context-compression pipeline.
//!
//! When AXIOM compression mode is on, the server intercepts the
//! incoming `/v1/messages` payload, separates "heavy" context messages
//! from the user's actual query, runs the heavy context through the
//! local TTT engine to produce a [`MemoryFingerprint`], strips the
//! heavy text from the outbound JSON, and prepends the fingerprint
//! to the surviving user prompt before forwarding to Anthropic.
//!
//! All HTTP is `reqwest` async (non-blocking). Streaming responses
//! are piped back to the caller as the bytes arrive — never buffered
//! in full.

use std::time::Duration;

use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};

use crate::context_compressor::MemoryFingerprint;

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Active outbound bridge. Cheap to clone (Arc-internal `reqwest::Client`).
#[derive(Clone)]
pub struct AnthropicForwarder {
    api_key: String,
    base_url: String,
    client: Client,
}

impl AnthropicForwarder {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("reqwest async client should construct");
        Self {
            api_key,
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            client,
        }
    }

    /// Env-driven activation. Returns `Some(forwarder)` only when
    /// `ANTHROPIC_API_KEY` is set. The compression-mode flag is
    /// checked separately by [`CompressorConfig::from_env`].
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").ok()?;
        let base_url = std::env::var("ANTHROPIC_BASE_URL").ok();
        Some(Self::new(api_key, base_url))
    }

    /// POST to `/v1/messages` on the real Anthropic API.
    ///
    /// Returns the raw JSON response when the upstream call succeeds.
    pub async fn forward_messages_json(
        &self,
        payload: &Value,
    ) -> Result<Value, ForwarderError> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(payload)
            .send()
            .await
            .map_err(|e| ForwarderError::Network(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| ForwarderError::Network(format!("body read failed: {e}")))?;
        if !status.is_success() {
            return Err(ForwarderError::Upstream { status: status.as_u16(), body });
        }
        serde_json::from_str(&body).map_err(|e| ForwarderError::Decode(e.to_string()))
    }
}

#[derive(Debug)]
pub enum ForwarderError {
    Network(String),
    Decode(String),
    Upstream { status: u16, body: String },
}

impl std::fmt::Display for ForwarderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForwarderError::Network(m) => write!(f, "network error: {m}"),
            ForwarderError::Decode(m) => write!(f, "decode error: {m}"),
            ForwarderError::Upstream { status, body } => {
                write!(f, "upstream {status}: {body}")
            }
        }
    }
}

impl std::error::Error for ForwarderError {}

// ---------------------------------------------------------------------------
// Payload mutation
// ---------------------------------------------------------------------------

/// One block extracted from a single `messages[*].content` entry.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractedContent {
    pub role: String,
    pub text: String,
    pub token_count: usize,
}

/// Result of separating heavy context blocks from the user-query tail.
#[derive(Debug, Clone)]
pub struct PartitionedMessages {
    /// Content that exceeded the heavy-message threshold — ingested by
    /// the local TTT engine and stripped from the outbound payload.
    pub heavy_context: Vec<ExtractedContent>,
    /// Content that survives in the outbound payload, in order.
    pub surviving: Vec<Value>,
    /// Index in `surviving` of the user message we should prepend the
    /// fingerprint to (the last user turn). `None` if there isn't one;
    /// in that case the caller appends a new synthetic user message.
    pub target_user_index: Option<usize>,
}

/// Walk a `messages` array, splitting each message's content into
/// (heavy text we ingest locally) vs (light text we keep). Block-form
/// content is preserved field-by-field; only text blocks above the
/// threshold are pulled out.
pub fn partition_messages(
    raw_messages: &[Value],
    threshold_tokens: usize,
    token_counter: impl Fn(&str) -> usize,
) -> PartitionedMessages {
    let mut heavy_context: Vec<ExtractedContent> = Vec::new();
    let mut surviving: Vec<Value> = Vec::with_capacity(raw_messages.len());
    let mut target_user_index: Option<usize> = None;

    for raw in raw_messages {
        let role = raw
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("user")
            .to_string();

        let content_value = raw.get("content").cloned().unwrap_or(Value::Null);
        let (kept_content, mut extracted) =
            split_content(&role, &content_value, threshold_tokens, &token_counter);

        heavy_context.append(&mut extracted);

        // Drop entirely-empty messages produced by stripping all blocks.
        let kept_empty = content_is_empty(&kept_content);
        if !kept_empty {
            let mut new_msg = raw.clone();
            new_msg["content"] = kept_content;
            if role == "user" {
                target_user_index = Some(surviving.len());
            }
            surviving.push(new_msg);
        }
    }

    PartitionedMessages {
        heavy_context,
        surviving,
        target_user_index,
    }
}

fn split_content(
    role: &str,
    content: &Value,
    threshold_tokens: usize,
    token_counter: &impl Fn(&str) -> usize,
) -> (Value, Vec<ExtractedContent>) {
    match content {
        Value::String(text) => {
            let count = token_counter(text);
            if count >= threshold_tokens {
                (
                    Value::String(String::new()),
                    vec![ExtractedContent {
                        role: role.to_string(),
                        text: text.clone(),
                        token_count: count,
                    }],
                )
            } else {
                (Value::String(text.clone()), Vec::new())
            }
        }
        Value::Array(blocks) => {
            let mut kept: Vec<Value> = Vec::with_capacity(blocks.len());
            let mut extracted: Vec<ExtractedContent> = Vec::new();
            for block in blocks {
                let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let text_opt = block.get("text").and_then(|v| v.as_str()).map(str::to_string);
                if block_type == "text" {
                    if let Some(text) = text_opt {
                        let count = token_counter(&text);
                        if count >= threshold_tokens {
                            extracted.push(ExtractedContent {
                                role: role.to_string(),
                                text,
                                token_count: count,
                            });
                            continue;
                        }
                    }
                }
                kept.push(block.clone());
            }
            (Value::Array(kept), extracted)
        }
        other => (other.clone(), Vec::new()),
    }
}

fn content_is_empty(v: &Value) -> bool {
    match v {
        Value::String(s) => s.trim().is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Null => true,
        _ => false,
    }
}

/// Build the outbound JSON payload by stripping heavy text and prepending
/// the fingerprint to the surviving user turn (or appending a fresh
/// user turn when none remain).
pub fn build_compressed_payload(
    original: &Value,
    fingerprint: &MemoryFingerprint,
    partitioned: &PartitionedMessages,
) -> Value {
    let mut payload = original.clone();
    let fingerprint_block = fingerprint.to_prompt_block();

    let mut messages = partitioned.surviving.clone();
    match partitioned.target_user_index {
        Some(idx) => {
            if let Some(msg) = messages.get_mut(idx) {
                prepend_to_user_content(msg, &fingerprint_block);
            }
        }
        None => {
            messages.push(json!({
                "role": "user",
                "content": fingerprint_block,
            }));
        }
    }

    payload["messages"] = Value::Array(messages);
    payload
}

fn prepend_to_user_content(msg: &mut Value, prepend_text: &str) {
    let content = msg.get("content").cloned().unwrap_or(Value::Null);
    let new_content = match content {
        Value::String(existing) => Value::String(format!("{prepend_text}\n\n{existing}")),
        Value::Array(mut blocks) => {
            let prepend_block = json!({"type": "text", "text": format!("{prepend_text}\n\n")});
            blocks.insert(0, prepend_block);
            Value::Array(blocks)
        }
        Value::Null => Value::String(prepend_text.to_string()),
        other => Value::String(format!("{prepend_text}\n\n{other}")),
    };
    msg["content"] = new_content;
}

// ---------------------------------------------------------------------------
// Token-count proxy for partitioning
// ---------------------------------------------------------------------------

/// Approximate token count when we want a fast pre-tokenizer estimate.
/// We use whitespace-splitting as a stable, allocation-free proxy: it's
/// roughly 4x undercount vs BPE but consistent enough for thresholding.
pub fn whitespace_token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ws(text: &str) -> usize {
        whitespace_token_count(text)
    }

    #[test]
    fn partition_extracts_heavy_string_content() {
        let messages = vec![
            json!({"role": "user", "content": "short ping"}),
            json!({"role": "user", "content": (0..400).map(|i| format!("tok{i}")).collect::<Vec<_>>().join(" ")}),
        ];
        let part = partition_messages(&messages, 100, ws);
        assert_eq!(part.heavy_context.len(), 1);
        assert_eq!(part.heavy_context[0].role, "user");
        assert!(part.heavy_context[0].token_count >= 100);
        // The heavy message becomes empty and is dropped; only the short ping survives.
        assert_eq!(part.surviving.len(), 1);
        assert_eq!(part.surviving[0]["content"], "short ping");
        assert_eq!(part.target_user_index, Some(0));
    }

    #[test]
    fn partition_handles_block_content() {
        let big_text = (0..300).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "explain this codebase"},
                {"type": "text", "text": big_text.clone()},
            ]
        })];
        let part = partition_messages(&messages, 100, ws);
        assert_eq!(part.heavy_context.len(), 1);
        assert_eq!(part.heavy_context[0].text, big_text);
        assert_eq!(part.surviving.len(), 1);
        let blocks = part.surviving[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["text"], "explain this codebase");
    }

    #[test]
    fn build_compressed_payload_prepends_fingerprint_to_string_content() {
        let original = json!({
            "model": "claude-opus-4-7",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "summarise the diff please"}],
        });
        let partitioned = partition_messages(
            original["messages"].as_array().unwrap(),
            10_000, // nothing crosses the threshold
            ws,
        );
        let fp = MemoryFingerprint {
            schema: "axiom-ttt-context-fingerprint/v1".into(),
            session_id: "sess-z".into(),
            context_tokens_processed: 0,
            n_layers: 1,
            d_model: 4,
            state_hash: "sha256:0000".into(),
            layer_frobenius_norms: vec![0.0],
            recall_norm: 0.0,
            recall_l1: 0.0,
            recall_top_k_indices: vec![],
            recall_top_k_decoded: "".into(),
            elapsed_ms: 0,
        };
        let payload = build_compressed_payload(&original, &fp, &partitioned);
        let messages = payload["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        let content = messages[0]["content"].as_str().unwrap();
        assert!(content.starts_with("[AXIOM-TTT-CONTEXT-FINGERPRINT"));
        assert!(content.contains("summarise the diff please"));
    }

    #[test]
    fn build_compressed_payload_appends_user_turn_when_only_heavy_messages() {
        // All messages are heavy → stripped → no surviving messages.
        let big = (0..400).map(|i| format!("tok{i}")).collect::<Vec<_>>().join(" ");
        let original = json!({
            "model": "claude-opus-4-7",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": big.clone()}],
        });
        let partitioned = partition_messages(original["messages"].as_array().unwrap(), 50, ws);
        assert!(partitioned.surviving.is_empty());
        let fp = MemoryFingerprint {
            schema: "axiom-ttt-context-fingerprint/v1".into(),
            session_id: "sess-q".into(),
            context_tokens_processed: 400,
            n_layers: 1,
            d_model: 4,
            state_hash: "sha256:abcd".into(),
            layer_frobenius_norms: vec![1.0],
            recall_norm: 1.0,
            recall_l1: 1.0,
            recall_top_k_indices: vec![],
            recall_top_k_decoded: "".into(),
            elapsed_ms: 1,
        };
        let payload = build_compressed_payload(&original, &fp, &partitioned);
        let messages = payload["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        let content = messages[0]["content"].as_str().unwrap();
        assert!(content.contains("AXIOM-TTT-CONTEXT-FINGERPRINT"));
        assert!(!content.contains("tok399")); // raw heavy text was stripped
    }
}
