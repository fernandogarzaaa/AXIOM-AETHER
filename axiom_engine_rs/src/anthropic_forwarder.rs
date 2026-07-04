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
//! All HTTP is `reqwest` async (non-blocking). `forward_messages_json`
//! buffers and JSON-parses the upstream body (non-streaming Messages calls).
//! `forward_messages_raw` buffers the body verbatim — including a
//! `text/event-stream` response produced by a `stream:true` request — and
//! relays it untouched; that is what the streaming client path uses.

use std::time::Duration;

use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};

use crate::context_compressor::MemoryFingerprint;

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Auth / relay headers captured from the inbound client request.
///
/// This is what lets the proxy serve BOTH credential models without ever
/// holding a long-lived secret of its own:
///   * **API-key clients** send `x-api-key` (or the proxy injects its own).
///   * **Subscription clients** (Claude Pro/Max via Claude Code) authenticate
///     with an OAuth bearer token in `Authorization` plus an `anthropic-beta`
///     `oauth-*` flag. We relay those verbatim.
#[derive(Clone, Debug, Default)]
pub struct ClientAuth {
    pub authorization: Option<String>,
    pub x_api_key: Option<String>,
    pub anthropic_version: Option<String>,
    pub anthropic_beta: Option<String>,
}

/// Active outbound bridge. Cheap to clone (Arc-internal `reqwest::Client`).
#[derive(Clone)]
pub struct AnthropicForwarder {
    /// Optional proxy-owned key. `None` in auth-passthrough mode, where the
    /// proxy relies entirely on the inbound client's own credentials.
    api_key: Option<String>,
    base_url: String,
    client: Client,
}

impl AnthropicForwarder {
    pub fn new(api_key: Option<String>, base_url: Option<String>) -> Self {
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

    /// Env-driven activation. Always returns `Some(forwarder)` so the
    /// compression path is usable in two modes:
    ///   * **API-key mode** — `ANTHROPIC_API_KEY` is set; the proxy injects it
    ///     as `x-api-key` whenever the client supplies no auth of its own.
    ///   * **Auth-passthrough mode** — no env key (e.g. a Claude *subscription*
    ///     that authenticates via OAuth). The proxy relays the client's own
    ///     `Authorization` / `x-api-key` headers upstream verbatim.
    ///
    /// The compression-mode flag is checked separately by
    /// [`CompressorConfig::from_env`].
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty());
        let base_url = std::env::var("ANTHROPIC_BASE_URL").ok();
        Some(Self::new(api_key, base_url))
    }

    /// Whether the proxy holds its own API key (vs. pure auth-passthrough).
    pub fn has_own_key(&self) -> bool {
        self.api_key.is_some()
    }

    /// POST to `/v1/messages` on the real Anthropic API.
    ///
    /// Auth precedence (so OAuth subscriptions and API keys both work):
    ///   1. Client `Authorization` bearer (subscription/OAuth) — relayed
    ///      verbatim; `x-api-key` is deliberately NOT sent alongside it.
    ///   2. Client `x-api-key` — relayed verbatim.
    ///   3. Proxy's own `ANTHROPIC_API_KEY` — injected as `x-api-key`.
    ///
    /// Returns the raw JSON response when the upstream call succeeds.
    pub async fn forward_messages_json(
        &self,
        payload: &Value,
        auth: &ClientAuth,
    ) -> Result<Value, ForwarderError> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let mut request = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .header(
                "anthropic-version",
                auth.anthropic_version
                    .as_deref()
                    .unwrap_or(ANTHROPIC_VERSION),
            );

        // Relay beta feature flags. The Claude Code subscription/OAuth path
        // requires its `oauth-*` beta flag to reach the upstream untouched.
        if let Some(beta) = auth.anthropic_beta.as_deref() {
            request = request.header("anthropic-beta", beta);
        }

        if let Some(authz) = auth.authorization.as_deref() {
            request = request.header("authorization", authz);
        } else if let Some(key) = auth.x_api_key.as_deref() {
            request = request.header("x-api-key", key);
        } else if let Some(key) = self.api_key.as_deref() {
            request = request.header("x-api-key", key);
        } else {
            return Err(ForwarderError::MissingAuth);
        }

        let response = request
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
            return Err(ForwarderError::Upstream {
                status: status.as_u16(),
                body,
            });
        }
        serde_json::from_str(&body).map_err(|e| ForwarderError::Decode(e.to_string()))
    }

    /// Raw passthrough forward: returns the upstream status, content-type, and
    /// body verbatim, WITHOUT JSON-parsing. Required for streaming
    /// (`stream:true`) responses, whose `text/event-stream` body is not a single
    /// JSON value and must reach the client untouched — parsing it is what
    /// produced the `decode error: expected value at line 1 column 1` 502s. Only
    /// transport failures (network / missing auth) surface as errors; every HTTP
    /// status returns normally so the caller can relay it (and its body) verbatim.
    pub async fn forward_messages_raw(
        &self,
        payload: &Value,
        auth: &ClientAuth,
    ) -> Result<RawForwardedResponse, ForwarderError> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let mut request = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .header(
                "anthropic-version",
                auth.anthropic_version
                    .as_deref()
                    .unwrap_or(ANTHROPIC_VERSION),
            );

        if let Some(beta) = auth.anthropic_beta.as_deref() {
            request = request.header("anthropic-beta", beta);
        }

        if let Some(authz) = auth.authorization.as_deref() {
            request = request.header("authorization", authz);
        } else if let Some(key) = auth.x_api_key.as_deref() {
            request = request.header("x-api-key", key);
        } else if let Some(key) = self.api_key.as_deref() {
            request = request.header("x-api-key", key);
        } else {
            return Err(ForwarderError::MissingAuth);
        }

        let response = request
            .json(payload)
            .send()
            .await
            .map_err(|e| ForwarderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let body = response
            .text()
            .await
            .map_err(|e| ForwarderError::Network(format!("body read failed: {e}")))?;
        Ok(RawForwardedResponse {
            status,
            content_type,
            body,
        })
    }
}

/// Verbatim upstream `/v1/messages` response used by the streaming passthrough
/// path. The body is NOT parsed, so it carries a JSON object, an SSE event
/// stream, or an error page unchanged.
pub struct RawForwardedResponse {
    pub status: u16,
    pub content_type: Option<String>,
    pub body: String,
}

#[derive(Debug)]
pub enum ForwarderError {
    Network(String),
    Decode(String),
    Upstream {
        status: u16,
        body: String,
    },
    /// Neither the client nor the proxy supplied any usable credential.
    MissingAuth,
}

impl std::fmt::Display for ForwarderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForwarderError::Network(m) => write!(f, "network error: {m}"),
            ForwarderError::Decode(m) => write!(f, "decode error: {m}"),
            ForwarderError::Upstream { status, body } => {
                write!(f, "upstream {status}: {body}")
            }
            ForwarderError::MissingAuth => write!(
                f,
                "no upstream credential: client sent no Authorization/x-api-key \
                 and the proxy has no ANTHROPIC_API_KEY configured"
            ),
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
                let text_opt = block
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
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

    // Claude-readable compression: ship a compact structural skeleton of the
    // heavy context (signatures kept, bodies dropped) instead of the opaque
    // neural fingerprint. Axiom's TTT capability is unaffected — the session
    // already absorbed the context (adapt_session); the drift signal
    // (recall_norm + state_hash) rides along as digest attributes. Falls back to
    // the neural schema marker only when there is no heavy text to skeletonize.
    let heavy_text = partitioned
        .heavy_context
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let original_tokens: usize = partitioned
        .heavy_context
        .iter()
        .map(|c| c.token_count)
        .sum();
    let fingerprint_block = if heavy_text.trim().is_empty() {
        // No heavy text to skeletonize → a compact "absorbed locally" marker.
        // We deliberately do NOT forward the verbose neural fingerprint
        // (recall_top_k_indices / layer norms): those vocab ids are noise to a
        // different model and only waste tokens. The drift signal lives
        // server-side; only a short provenance header rides the wire.
        opaque_fingerprint_block(fingerprint)
    } else {
        let digest = crate::skeleton::build_digest(
            &heavy_text,
            &fingerprint.session_id,
            original_tokens,
            fingerprint.recall_norm,
            &fingerprint.state_hash,
            3, // doc-line cap tuned to ~80% reduction
        );
        if digest.contains("kind=\"structural-skeleton\"") {
            structural_fingerprint_block(fingerprint, &digest)
        } else {
            opaque_fingerprint_block(fingerprint)
        }
    };

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

/// Compact marker for heavy context that was absorbed locally but has no
/// extractable code structure (prose / minified / binary-ish).
///
/// The verbose neural fingerprint (`recall_top_k_indices`, layer Frobenius
/// norms, vocab-id decode instructions) is intentionally NOT forwarded: those
/// fields are meaningless to a *different* upstream model and only burn tokens
/// while degrading answers (see the honesty footnote in `context_compressor`).
/// We keep only a short, readable provenance header; the TTT drift signal stays
/// server-side where the session and metrics consume it.
fn opaque_fingerprint_block(fingerprint: &MemoryFingerprint) -> String {
    format!(
        "<axiom_context_fingerprint session_id=\"{session}\" tokens_compressed=\"{tokens}\" \
schema=\"{schema}\" mode=\"absorbed\">\n\
state_hash={hash}\n\
raw_context=elided\n\
note=Heavy context was absorbed locally via online TTT; it had no extractable \
code structure, so no lossy neural pointer is forwarded (it would be noise to \
this model). Answer from the surrounding turns; the original text is retained \
server-side for this session.\n\
</axiom_context_fingerprint>",
        session = fingerprint.session_id,
        tokens = fingerprint.context_tokens_processed,
        schema = fingerprint.schema,
        hash = fingerprint.state_hash,
    )
}

/// Readable compression block for code: the structural skeleton (signatures
/// kept, bodies elided) plus a short provenance header. As with the marker
/// above, the opaque neural fields are dropped from the wire — the skeleton is
/// the thing Claude can actually read, and dropped bodies are recoverable via
/// `POST /v1/expand`.
fn structural_fingerprint_block(fingerprint: &MemoryFingerprint, digest: &str) -> String {
    format!(
        "<axiom_context_fingerprint session_id=\"{session}\" tokens_compressed=\"{tokens}\" \
schema=\"{schema}\" mode=\"structural-digest\">\n\
state_hash={hash}\n\
{digest}\n\
note=Structural skeleton of locally-absorbed context: declaration signatures \
kept, bodies elided. Request any dropped body with POST /v1/expand \
{{\"session_id\",\"symbol\"}}.\n\
</axiom_context_fingerprint>",
        session = fingerprint.session_id,
        tokens = fingerprint.context_tokens_processed,
        schema = fingerprint.schema,
        hash = fingerprint.state_hash,
    )
}

/// Prepend `block` to the last `user`-role message of a built payload, in
/// place. Used to inject the immunity advisory after the compressed payload is
/// assembled. No-op if there is no user message (the advisory is best-effort).
pub fn prepend_block_to_last_user_turn(payload: &mut Value, block: &str) {
    let Some(messages) = payload.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    if let Some(msg) = messages
        .iter_mut()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
    {
        prepend_to_user_content(msg, block);
    }
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
        let big_text = (0..300)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
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
            schema: "axiom-ttt-context-fingerprint/v2".into(),
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
        assert!(content.starts_with("<axiom_context_fingerprint "));
        assert!(content.contains("</axiom_context_fingerprint>"));
        assert!(content.contains("summarise the diff please"));
    }

    #[test]
    fn build_compressed_payload_appends_user_turn_when_only_heavy_messages() {
        // All messages are heavy → stripped → no surviving messages.
        let big = (0..400)
            .map(|i| format!("tok{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let original = json!({
            "model": "claude-opus-4-7",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": big.clone()}],
        });
        let partitioned = partition_messages(original["messages"].as_array().unwrap(), 50, ws);
        assert!(partitioned.surviving.is_empty());
        let fp = MemoryFingerprint {
            schema: "axiom-ttt-context-fingerprint/v2".into(),
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
        assert!(content.starts_with("<axiom_context_fingerprint "));
        assert!(content.contains("tokens_compressed=\"400\""));
        assert!(content.contains("state_hash=sha256:abcd"));
        assert!(content.contains("raw_context=elided"));
        assert!(!content.contains("tok399"));
        assert!(content.len() < big.len()); // raw heavy text was compressed
        // #2: the opaque neural noise must NOT reach the wire.
        assert!(!content.contains("recall_top_k_indices"));
        assert!(!content.contains("layer_frobenius_norms"));
        assert!(!content.contains("associative_recall_l1"));
    }

    #[test]
    fn build_compressed_payload_wraps_structural_digest_for_code() {
        let code = r#"
use std::collections::HashMap;
pub fn run() -> usize {
    let mut values = HashMap::new();
    values.insert("secret_body", 1usize);
    values.len()
}
"#;
        let original = json!({
            "model": "claude-opus-4-7",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": code}],
        });
        let partitioned = partition_messages(original["messages"].as_array().unwrap(), 10, ws);
        assert!(partitioned.surviving.is_empty());
        let fp = MemoryFingerprint {
            schema: "axiom-ttt-context-fingerprint/v2".into(),
            session_id: "sess-code".into(),
            context_tokens_processed: 32,
            n_layers: 1,
            d_model: 4,
            state_hash: "sha256:c0de".into(),
            layer_frobenius_norms: vec![1.0],
            recall_norm: 1.0,
            recall_l1: 1.0,
            recall_top_k_indices: vec![1, 2],
            recall_top_k_decoded: "".into(),
            elapsed_ms: 1,
        };
        let payload = build_compressed_payload(&original, &fp, &partitioned);
        let content = payload["messages"][0]["content"].as_str().unwrap();
        assert!(content.starts_with("<axiom_context_fingerprint "));
        assert!(content.contains("mode=\"structural-digest\""));
        assert!(content.contains("<axiom_context_digest "));
        assert!(content.contains("fn run"));
        assert!(content.contains("usize"));
        assert!(!content.contains("secret_body"));
        // #2: even with top-k indices present on the fingerprint, the structural
        // block must not forward them — only the readable skeleton goes upstream.
        assert!(!content.contains("recall_top_k_indices"));
    }
}
