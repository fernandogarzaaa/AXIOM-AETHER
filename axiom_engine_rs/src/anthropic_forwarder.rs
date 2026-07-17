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
//! `forward_messages_stream` returns the raw `reqwest::Response` on a
//! no-total-timeout client so a `stream:true` `text/event-stream` body is
//! relayed to the client incrementally — never JSON-parsed (that caused
//! `decode error: expected value at line 1 column 1`) nor capped by the
//! buffered client's request timeout on long generations.

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
    /// No total-timeout client for `stream: true`. Streaming generations can
    /// legitimately exceed the buffered client's request deadline; disconnects
    /// and upstream errors still terminate the response stream.
    streaming_client: Client,
}

impl AnthropicForwarder {
    pub fn new(api_key: Option<String>, base_url: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("reqwest async client should construct");
        // Streaming responses can legitimately exceed two minutes, so this
        // client has no total timeout; disconnects and upstream errors still
        // terminate its response stream.
        let streaming_client = Client::builder()
            .build()
            .expect("reqwest streaming client should construct");
        Self {
            api_key,
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            client,
            streaming_client,
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
        let response = self
            .build_request(&self.client, auth)?
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

    /// Build the `/v1/messages` request with shared headers and the
    /// security-sensitive auth precedence (Authorization → x-api-key → proxy
    /// key → `MissingAuth`), so that order lives in exactly one place and can
    /// not drift between the buffered and streaming paths.
    fn build_request(
        &self,
        client: &Client,
        auth: &ClientAuth,
    ) -> Result<reqwest::RequestBuilder, ForwarderError> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let mut request = client
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
        Ok(request)
    }

    /// Streaming passthrough: send on the no-total-timeout streaming client and
    /// return the raw `reqwest::Response` so the caller can relay its
    /// `bytes_stream()` straight to the client. Required for `stream: true`
    /// (Claude Code): the `text/event-stream` body must flow through
    /// incrementally and must never be JSON-parsed (that caused `decode error:
    /// expected value at line 1 column 1`) nor capped by the buffered client's
    /// total request timeout on long generations.
    pub async fn forward_messages_stream(
        &self,
        payload: &Value,
        auth: &ClientAuth,
    ) -> Result<reqwest::Response, ForwarderError> {
        self.build_request(&self.streaming_client, auth)?
            .json(payload)
            .send()
            .await
            .map_err(|e| ForwarderError::Network(e.to_string()))
    }
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

        // A `role: "system"` entry in the messages array (distinct from the
        // top-level `system` field) is positionally significant: Anthropic
        // requires it to either precede an assistant message or end the
        // array, EXCEPT the directive-only form (`content: []` plus an
        // `output_config`), which is valid at any position. Extracting its
        // content as "heavy" or dropping it for having empty content -- both
        // of which this function does for every other role -- removes it
        // from the array and can leave a DIFFERENT system message no longer
        // immediately before an assistant turn / at the end, producing a 400
        // from Anthropic. System messages are therefore always passed
        // through byte-identical: never a compression candidate.
        if role == "system" {
            surviving.push(raw.clone());
            continue;
        }

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

/// Client harnesses (e.g. Claude Code) inject deterministic instructional
/// boilerplate into the first user turn wrapped in `<system-reminder>...
/// </system-reminder>`. That text is not user-authored conversation
/// history: it's the same on every session. Sweeping it into
/// `heavy_context` and re-presenting it to the model via the fingerprint's
/// `decode_instructions` as "prior heavy context" relevant to "the user's
/// intent" is a false claim on a first turn (there is no prior context
/// yet) — text a well-calibrated model correctly refuses to trust,
/// surfacing as spurious prompt-injection warnings. Reminder spans are
/// therefore excluded from the threshold check and extraction entirely;
/// they pass through in the surviving payload untouched. Genuine heavy
/// content sitting alongside a reminder in the same block (e.g. a large
/// paste after the reminder) is unaffected — only the reminder text itself
/// is carved out before the token count is taken.
fn split_system_reminders(text: &str) -> (String, Vec<String>) {
    const OPEN: &str = "<system-reminder>";
    const CLOSE: &str = "</system-reminder>";
    if !text.contains(OPEN) {
        return (text.to_string(), Vec::new());
    }
    let mut remainder = String::with_capacity(text.len());
    let mut reminders = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        remainder.push_str(&rest[..start]);
        let after_open = &rest[start..];
        match after_open.find(CLOSE) {
            Some(end_rel) => {
                let end = end_rel + CLOSE.len();
                reminders.push(after_open[..end].to_string());
                rest = &after_open[end..];
            }
            None => {
                // Unterminated tag: keep the rest as reminder content rather
                // than risk compressing a truncated instruction block.
                reminders.push(after_open.to_string());
                rest = "";
            }
        }
    }
    remainder.push_str(rest);
    (remainder, reminders)
}

fn split_content(
    role: &str,
    content: &Value,
    threshold_tokens: usize,
    token_counter: &impl Fn(&str) -> usize,
) -> (Value, Vec<ExtractedContent>) {
    match content {
        Value::String(text) => {
            let (remainder, reminders) = split_system_reminders(text);
            if reminders.is_empty() {
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
            } else {
                let count = token_counter(&remainder);
                if count >= threshold_tokens {
                    (
                        Value::String(reminders.join("\n\n")),
                        vec![ExtractedContent {
                            role: role.to_string(),
                            text: remainder,
                            token_count: count,
                        }],
                    )
                } else {
                    (Value::String(text.clone()), Vec::new())
                }
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
                        let (remainder, reminders) = split_system_reminders(&text);
                        let count = token_counter(&remainder);
                        if count >= threshold_tokens {
                            extracted.push(ExtractedContent {
                                role: role.to_string(),
                                text: remainder,
                                token_count: count,
                            });
                            if !reminders.is_empty() {
                                let mut kept_block = block.clone();
                                kept_block["text"] = Value::String(reminders.join("\n\n"));
                                kept.push(kept_block);
                            }
                            continue;
                        }
                    }
                }
                // `tool_result` is the dominant heavy-content shape in real
                // Claude Code traffic (Read/Grep/Bash output) -- distinct
                // from `text` blocks, which are mostly short user prose. It
                // was previously invisible to this threshold check entirely,
                // so the S0 heavy-extraction pipeline never fired against
                // real tool-use transcripts. `content` here is a nested
                // string-or-block-array, so it's flattened the same way
                // `rebase::tool_result_text` flattens it for S1/P2 digestion.
                if block_type == "tool_result" {
                    if let Some(text) = crate::rebase::tool_result_text(block) {
                        let (remainder, reminders) = split_system_reminders(&text);
                        let count = token_counter(&remainder);
                        if count >= threshold_tokens {
                            extracted.push(ExtractedContent {
                                role: role.to_string(),
                                text: remainder,
                                token_count: count,
                            });
                            let mut kept_block = block.clone();
                            // Never leave `content` empty: a `tool_result` is
                            // required 1:1 pairing with a preceding
                            // `tool_use` id, so (unlike a `text` block) it
                            // can't just be dropped from the array, and an
                            // empty string risks upstream validation
                            // rejecting the block outright. Leave a marker
                            // instead of the stripped text.
                            const ABSORBED: &str =
                                "[axiom-ttt: heavy tool_result absorbed into session fingerprint]";
                            let replacement = if reminders.is_empty() {
                                Value::String(ABSORBED.to_string())
                            } else {
                                Value::String(format!("{ABSORBED}\n\n{}", reminders.join("\n\n")))
                            };
                            if let Some(obj) = kept_block.as_object_mut() {
                                obj.insert("content".to_string(), replacement);
                            }
                            kept.push(kept_block);
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
        None if !partitioned.heavy_context.is_empty() => {
            // All surviving messages were extracted as heavy: synthesize a
            // carrier turn for the fingerprint, since there is genuine
            // compressed content that needs a home.
            messages.push(json!({
                "role": "user",
                "content": fingerprint_block,
            }));
        }
        None => {
            // Nothing was extracted AND nothing survived: the mutable tail
            // this function was given was already empty (e.g. the entire
            // conversation is the frozen/cached prefix, with nothing new this
            // turn). There is no fingerprint worth reporting, so `messages`
            // stays empty rather than gaining a phantom "absorbed 0 tokens"
            // turn. The caller splices this onto the untouched frozen prefix,
            // so an empty result here reconstitutes the original array
            // exactly. Synthesizing a turn here was observed live to corrupt
            // an already-valid frozen prefix ending in a positionally
            // significant role:"system" message into an invalid one
            // (2026-07-17: Claude Code 400 "role 'system' must precede an
            // assistant message or end the array").
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
    fn partition_passes_directive_only_system_message_through_untouched() {
        // A role:"system" message with content:[] (Anthropic's directive-only
        // form) must never be dropped as "entirely empty" -- removing it from
        // the array can leave a DIFFERENT system message no longer
        // immediately before an assistant turn, producing a 400 upstream.
        let messages = vec![
            json!({"role": "system", "content": [], "output_config": {"effort": "low"}}),
            json!({"role": "assistant", "content": "ok"}),
            json!({"role": "user", "content": "hello"}),
        ];
        let part = partition_messages(&messages, 100, ws);
        assert_eq!(part.surviving.len(), 3, "the directive-only system message is kept");
        assert_eq!(part.surviving[0]["role"], "system");
        assert_eq!(part.surviving[0]["content"], json!([]));
        assert_eq!(part.surviving[0]["output_config"], json!({"effort": "low"}));
    }

    #[test]
    fn partition_never_extracts_heavy_content_from_a_system_message() {
        // Even a large, genuinely heavy system-role message must pass through
        // byte-identical: it is positionally significant and never a
        // compression candidate.
        let big_text = (0..400).map(|i| format!("tok{i}")).collect::<Vec<_>>().join(" ");
        let messages = vec![
            json!({"role": "system", "content": big_text.clone()}),
            json!({"role": "assistant", "content": "ok"}),
        ];
        let part = partition_messages(&messages, 100, ws);
        assert!(part.heavy_context.is_empty(), "system content is never extracted");
        assert_eq!(part.surviving.len(), 2);
        assert_eq!(part.surviving[0]["content"], json!(big_text));
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
    fn partition_excludes_system_reminder_from_heavy_extraction() {
        // Regression test: a large `<system-reminder>` block (client-injected
        // boilerplate, e.g. CLAUDE.md/env context) alongside a short real
        // question must NOT be swept into heavy_context. Before this fix the
        // combined text crossed the threshold, so Axiom compressed the
        // reminder and re-presented it as "prior heavy context" bearing on
        // "the user's intent" -- a false claim on a first turn that Claude
        // Code correctly flagged as prompt injection.
        let reminder_body = (0..300)
            .map(|i| format!("envfact{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let text = format!(
            "<system-reminder>{reminder_body}</system-reminder>\nwhat does this repo do?"
        );
        let messages = vec![json!({"role": "user", "content": text.clone()})];
        let part = partition_messages(&messages, 100, ws);
        assert!(
            part.heavy_context.is_empty(),
            "system-reminder content must never be extracted as heavy context"
        );
        assert_eq!(part.surviving.len(), 1);
        assert_eq!(part.surviving[0]["content"], text);
    }

    #[test]
    fn partition_still_extracts_genuine_heavy_text_next_to_a_reminder() {
        // A real oversized paste alongside a reminder still gets compressed;
        // only the reminder span itself is excluded from the threshold check.
        let reminder = "<system-reminder>short boilerplate</system-reminder>";
        let big_paste = (0..300)
            .map(|i| format!("logline{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let text = format!("{reminder}\n{big_paste}");
        let messages = vec![json!({"role": "user", "content": text})];
        let part = partition_messages(&messages, 100, ws);
        assert_eq!(part.heavy_context.len(), 1);
        assert_eq!(part.heavy_context[0].text.trim(), big_paste);
        // The reminder itself survives, untouched, in the outbound payload.
        assert_eq!(part.surviving[0]["content"], reminder);
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
    fn build_compressed_payload_stays_empty_when_input_was_already_empty() {
        // Nothing to partition (an empty mutable tail -- e.g. the whole
        // conversation is the frozen prefix, nothing new this turn): no
        // synthetic carrier turn should appear. The caller splices this
        // result onto the untouched frozen prefix, so a spurious turn here
        // would corrupt an already-valid array (2026-07-17 regression: a
        // frozen [user, system] prefix, valid because system ends the array,
        // became [user, system, <synthetic user>] and Anthropic rejected it).
        let original = json!({
            "model": "claude-sonnet-5",
            "max_tokens": 64,
            "messages": [],
        });
        let partitioned = partition_messages(&[], 50, ws);
        assert!(partitioned.surviving.is_empty());
        assert!(partitioned.heavy_context.is_empty());
        let fp = MemoryFingerprint {
            schema: "axiom-ttt-context-fingerprint/v2".into(),
            session_id: "sess-empty".into(),
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
        assert!(messages.is_empty(), "no phantom carrier turn when there was nothing to compress");
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

    #[test]
    fn partition_extracts_heavy_tool_result_string_content() {
        // The dominant heavy-content shape in real Claude Code traffic:
        // Read/Grep/Bash output riding in a `tool_result` block, not a
        // `text` block. Before this fix `split_content`'s Array branch only
        // ever inspected `block_type == "text"`, so this never fired.
        let big = (0..400).map(|i| format!("line{i}")).collect::<Vec<_>>().join(" ");
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": big.clone()}
            ]
        })];
        let part = partition_messages(&messages, 100, ws);
        assert_eq!(part.heavy_context.len(), 1, "heavy tool_result must be extracted");
        assert_eq!(part.heavy_context[0].text, big);
        let blocks = part.surviving[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1, "the tool_result block itself is never dropped");
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "t1", "tool_use pairing is preserved");
        assert_ne!(blocks[0]["content"], json!(big), "raw text removed from the wire payload");
        assert!(
            blocks[0]["content"].as_str().unwrap().contains("absorbed"),
            "a marker is left so the block is never blank"
        );
    }

    #[test]
    fn partition_extracts_heavy_tool_result_array_content() {
        // tool_result content can also be an array of {type:"text", text}
        // parts (e.g. stdout + stderr) -- must flatten the same way
        // rebase::tool_result_text does for S1/P2 digestion.
        let stdout = (0..250).map(|i| format!("out{i}")).collect::<Vec<_>>().join(" ");
        let stderr = (0..250).map(|i| format!("err{i}")).collect::<Vec<_>>().join(" ");
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": [
                    {"type": "text", "text": stdout},
                    {"type": "text", "text": stderr},
                ]}
            ]
        })];
        let part = partition_messages(&messages, 100, ws);
        assert_eq!(part.heavy_context.len(), 1);
        assert!(part.heavy_context[0].text.contains("out0"));
        assert!(part.heavy_context[0].text.contains("err0"));
    }

    #[test]
    fn partition_leaves_light_tool_results_alone() {
        let messages = vec![json!({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t3", "content": "exit 0"}]
        })];
        let part = partition_messages(&messages, 100, ws);
        assert!(part.heavy_context.is_empty());
        assert_eq!(part.surviving[0]["content"][0]["content"], "exit 0");
    }

    #[test]
    fn partition_never_extracts_heavy_tool_result_from_a_system_message() {
        // Same positional-safety contract as text content: system messages
        // are never a compression candidate regardless of block type.
        let big = (0..400).map(|i| format!("tok{i}")).collect::<Vec<_>>().join(" ");
        let messages = vec![
            json!({"role": "system", "content": [
                {"type": "tool_result", "tool_use_id": "t4", "content": big.clone()}
            ]}),
            json!({"role": "assistant", "content": "ok"}),
        ];
        let part = partition_messages(&messages, 100, ws);
        assert!(part.heavy_context.is_empty());
        assert_eq!(part.surviving[0]["content"][0]["content"], json!(big));
    }

    #[test]
    fn partition_extracts_heavy_tool_result_alongside_a_light_text_block() {
        // A realistic Claude Code turn: a short assistant-facing note plus a
        // heavy tool_result in the same message. Only the tool_result block
        // is extracted; the message survives with both blocks present.
        let big = (0..400).map(|i| format!("row{i}")).collect::<Vec<_>>().join(" ");
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "continue"},
                {"type": "tool_result", "tool_use_id": "t5", "content": big.clone()},
            ]
        })];
        let part = partition_messages(&messages, 100, ws);
        assert_eq!(part.heavy_context.len(), 1);
        assert_eq!(part.heavy_context[0].text, big);
        let blocks = part.surviving[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2, "both blocks remain, light text untouched");
        assert_eq!(blocks[0]["text"], "continue");
        assert_eq!(blocks[1]["type"], "tool_result");
    }
}
