//! Local SLM routing through Ollama.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::mesh_router::MeshModelSelector;

const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MODELS: [&str; 4] = ["phi4:3.8b", "deepseek-r1:8b", "llama3.3:8b", "llama3.1:8b"];
const DEFAULT_NUM_CTX: u64 = 4096;

/// How long one streamed generation may run. The non-stream dispatch caps
/// the whole request at `config.timeout_ms` (2 s by default), which a token
/// stream outlives by design, so the streaming request carries its own
/// generous cap instead. Client disconnect still aborts the request
/// immediately, via the dropped channel receiver.
const STREAM_REQUEST_TIMEOUT: Duration = Duration::from_secs(1800);

/// Bound of the streaming channel between the Ollama producer task and the
/// SSE consumer. This is the backpressure mechanism: a slow consumer blocks
/// the producer on `send` once 16 deltas are queued, instead of the
/// producer accumulating the whole generation in memory.
pub const SWARM_STREAM_CHANNEL_CAPACITY: usize = 16;

#[derive(Clone)]
pub struct SwarmRouter {
    config: SwarmRouterConfig,
    client: Client,
    /// `Some` when `AXIOM_MESH_ROUTING=1` — see `mesh_router` module docs
    /// for why this is opt-in rather than the new default.
    mesh_selector: Option<Arc<MeshModelSelector>>,
}

#[derive(Debug, Clone)]
pub struct SwarmRouterConfig {
    pub enabled: bool,
    pub base_url: String,
    pub model_candidates: Vec<String>,
    pub num_ctx: u64,
    pub timeout_ms: u64,
    /// Route model selection through `axiom_core`'s online-learned mesh
    /// instead of the static first-available-candidate order. Off by
    /// default: see `mesh_router` module docs.
    pub mesh_routing: bool,
}

#[derive(Debug)]
pub enum SwarmRouteError {
    Disabled,
    Health(String),
    NoModel {
        requested: Vec<String>,
        available: Vec<String>,
    },
    Upstream(String),
    Decode(String),
}

impl std::fmt::Display for SwarmRouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SwarmRouteError::Disabled => write!(f, "local swarm router disabled"),
            SwarmRouteError::Health(message) => write!(f, "ollama health check failed: {message}"),
            SwarmRouteError::NoModel {
                requested,
                available,
            } => write!(
                f,
                "no requested local model available; requested={requested:?} available={available:?}"
            ),
            SwarmRouteError::Upstream(message) => write!(f, "ollama chat failed: {message}"),
            SwarmRouteError::Decode(message) => write!(f, "ollama response decode failed: {message}"),
        }
    }
}

impl std::error::Error for SwarmRouteError {}

#[derive(Debug, Clone)]
pub struct SwarmChatResult {
    pub model: String,
    pub content: String,
}

/// A live token stream from one Ollama `/api/chat` dispatch.
///
/// The producer task posts with `"stream": true`, parses NDJSON delta lines
/// as they arrive, and sends each non-empty delta through the bounded
/// channel. The receiver is the SSE body: dropping it (client disconnect)
/// makes the producer's next `send` fail, which drops the upstream response
/// and aborts the Ollama request. The channel closing without an error item
/// means the stream completed cleanly (the final `done` line arrived).
pub struct SwarmStream {
    pub model: String,
    pub receiver: tokio::sync::mpsc::Receiver<Result<String, SwarmRouteError>>,
}

#[derive(Debug, Deserialize)]
struct OllamaTags {
    #[serde(default)]
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: Option<OllamaMessage>,
    response: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

impl SwarmRouter {
    pub fn from_env() -> Option<Self> {
        let config = SwarmRouterConfig::from_env();
        if !config.enabled {
            return None;
        }
        Some(Self::new(config))
    }

    pub fn new(config: SwarmRouterConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .expect("reqwest async client should construct");
        let mesh_selector = config
            .mesh_routing
            .then(|| Arc::new(MeshModelSelector::new(&config.model_candidates)));
        Self {
            config,
            client,
            mesh_selector,
        }
    }

    pub fn config(&self) -> &SwarmRouterConfig {
        &self.config
    }

    pub async fn route_chat_payload(
        &self,
        payload: &Value,
    ) -> Result<SwarmChatResult, SwarmRouteError> {
        if !self.config.enabled {
            return Err(SwarmRouteError::Disabled);
        }

        let model = self.select_model_for_dispatch().await?;

        // Timed and outcome-recorded as one unit so the mesh selector (when
        // enabled) learns from exactly what happened dispatching to this
        // model — a pre-selection failure (no model available at all)
        // never reaches here, since there's no model to credit or blame.
        let started = Instant::now();
        let result = self.dispatch_to_model(&model, payload).await;
        if let Some(selector) = &self.mesh_selector {
            selector.record_outcome(&model, result.is_ok(), started.elapsed().as_millis() as u64);
        }
        let content = result?;
        Ok(SwarmChatResult { model, content })
    }

    /// Streaming dispatch: POST `/api/chat` with `"stream": true` and return
    /// immediately with the channel the producer task feeds. The caller
    /// (the SSE body) receives deltas as Ollama emits them; this function
    /// never waits for the full generation.
    pub async fn route_chat_stream(&self, payload: &Value) -> Result<SwarmStream, SwarmRouteError> {
        if !self.config.enabled {
            return Err(SwarmRouteError::Disabled);
        }
        let model = self.select_model_for_dispatch().await?;
        let (tx, rx) = tokio::sync::mpsc::channel(SWARM_STREAM_CHANNEL_CAPACITY);
        let client = self.client.clone();
        let base_url = self.config.base_url.clone();
        let num_ctx = self.config.num_ctx;
        let messages = translate_messages(payload);
        let mesh_selector = self.mesh_selector.clone();
        let stream_model = model.clone();
        tokio::spawn(async move {
            let started = Instant::now();
            let outcome =
                run_stream_dispatch(&client, &base_url, &stream_model, num_ctx, messages, tx).await;
            // A cancellation says nothing about the model: the client
            // went away, so recording it as a failure would train the
            // selector away from healthy models. Completed and Failed
            // both feed the selector unchanged.
            if let (Some(selector), false) =
                (&mesh_selector, matches!(outcome, StreamOutcome::Cancelled))
            {
                selector.record_outcome(
                    &stream_model,
                    matches!(outcome, StreamOutcome::Completed),
                    started.elapsed().as_millis() as u64,
                );
            }
        });
        Ok(SwarmStream {
            model,
            receiver: rx,
        })
    }

    async fn select_model_for_dispatch(&self) -> Result<String, SwarmRouteError> {
        let available = self.available_models().await?;
        match &self.mesh_selector {
            Some(selector) => selector.select(&available),
            None => select_model(&self.config.model_candidates, &available),
        }
        .ok_or_else(|| SwarmRouteError::NoModel {
            requested: self.config.model_candidates.clone(),
            available,
        })
    }

    async fn dispatch_to_model(
        &self,
        model: &str,
        payload: &Value,
    ) -> Result<String, SwarmRouteError> {
        let messages = translate_messages(payload);
        let request = json!({
            "model": model,
            "stream": false,
            "messages": messages,
            "options": {
                "num_ctx": self.config.num_ctx
            }
        });
        let url = format!("{}/api/chat", self.config.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|e| SwarmRouteError::Upstream(e.to_string()))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| SwarmRouteError::Upstream(format!("body read failed: {e}")))?;
        if !status.is_success() {
            return Err(SwarmRouteError::Upstream(format!(
                "ollama status {}: {}",
                status.as_u16(),
                body
            )));
        }
        let decoded: OllamaChatResponse =
            serde_json::from_str(&body).map_err(|e| SwarmRouteError::Decode(e.to_string()))?;
        Ok(decoded
            .message
            .map(|m| m.content)
            .or(decoded.response)
            .unwrap_or_default())
    }

    async fn available_models(&self) -> Result<Vec<String>, SwarmRouteError> {
        let url = format!("{}/api/tags", self.config.base_url.trim_end_matches('/'));
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| SwarmRouteError::Health(e.to_string()))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| SwarmRouteError::Health(format!("body read failed: {e}")))?;
        if !status.is_success() {
            return Err(SwarmRouteError::Health(format!(
                "ollama tags status {}: {}",
                status.as_u16(),
                body
            )));
        }
        parse_tags_models(&body).map_err(SwarmRouteError::Decode)
    }
}

/// Outcome of one streamed dispatch, for mesh-selector learning.
/// `Completed` and `Failed` feed `record_outcome`; `Cancelled` must not:
/// a client disconnect (dropped receiver, failed send, or `tx.closed()`
/// winning the race) says nothing about the model, and recording it as a
/// failure would train the selector away from healthy models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamOutcome {
    Completed,
    Failed,
    Cancelled,
}

/// Producer half of [`SwarmStream`]: streams one Ollama `/api/chat`
/// response, sending each content delta through the bounded channel as it
/// arrives. Returns [`StreamOutcome::Completed`] only when the upstream
/// stream ended with a `done` line. The channel is the backpressure mechanism: `send` blocks
/// while it is full, so a slow consumer stalls this task instead of letting
/// it accumulate the whole generation. A dropped receiver (client
/// disconnect) makes `send` fail, and this task then returns
/// [`StreamOutcome::Cancelled`], dropping the response and aborting the
/// upstream request. The disconnect is also observed while waiting for
/// upstream data: the dispatch and every body read race against
/// `tx.closed()`, so an abandoned request does not keep consuming
/// inference resources for up to the request timeout. A stream that ends
/// without a `done` line is an upstream error, not a clean stop: the SSE
/// layer must not bless a truncated answer with `[DONE]`.
async fn run_stream_dispatch(
    client: &Client,
    base_url: &str,
    model: &str,
    num_ctx: u64,
    messages: Vec<Value>,
    tx: tokio::sync::mpsc::Sender<Result<String, SwarmRouteError>>,
) -> StreamOutcome {
    let request = json!({
        "model": model,
        "stream": true,
        "messages": messages,
        "options": {
            "num_ctx": num_ctx
        }
    });
    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));
    // First-token latency runs from dispatch, not from headers: a slow
    // model load before the first token counts.
    let started = Instant::now();
    let response = tokio::select! {
        result = client
            .post(url)
            .json(&request)
            .timeout(STREAM_REQUEST_TIMEOUT)
            .send() => {
            match result {
                Ok(response) => response,
                Err(e) => {
                    let _ = tx.send(Err(SwarmRouteError::Upstream(e.to_string()))).await;
                    return StreamOutcome::Failed;
                }
            }
        }
        _ = tx.closed() => return StreamOutcome::Cancelled,
    };
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let _ = tx
            .send(Err(SwarmRouteError::Upstream(format!(
                "ollama status {}: {}",
                status.as_u16(),
                body
            ))))
            .await;
        return StreamOutcome::Failed;
    }
    let mut byte_stream = response.bytes_stream();
    let mut assembler = OllamaLineAssembler::new();
    let mut first_token = true;
    let mut done = false;
    while !done {
        let chunk = tokio::select! {
            next = byte_stream.next() => match next {
                Some(Ok(bytes)) => bytes,
                Some(Err(e)) => {
                    let _ = tx
                        .send(Err(SwarmRouteError::Upstream(format!(
                            "body read failed: {e}"
                        ))))
                        .await;
                    return StreamOutcome::Failed;
                }
                None => break,
            },
            _ = tx.closed() => return StreamOutcome::Cancelled,
        };
        let lines = match assembler.push(&chunk) {
            Ok(lines) => lines,
            Err(e) => {
                let _ = tx.send(Err(e)).await;
                return StreamOutcome::Failed;
            }
        };
        for line in lines {
            match parse_ollama_stream_line(&line) {
                Ok(parsed) => {
                    if !parsed.delta.is_empty() {
                        if first_token {
                            crate::metrics::observe_first_token_latency(
                                started.elapsed().as_secs_f64(),
                            );
                            first_token = false;
                        }
                        if tx.send(Ok(parsed.delta)).await.is_err() {
                            return StreamOutcome::Cancelled;
                        }
                    }
                    if parsed.done {
                        done = true;
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return StreamOutcome::Failed;
                }
            }
        }
    }
    if !done {
        // Trailing bytes with no newline terminator: one last parse attempt.
        // A stream that ends without a `done` line is incomplete, not a
        // clean stop: report it so the SSE layer ends without `[DONE]`
        // instead of blessing a truncated answer.
        let remainder = assembler.remainder();
        if let Ok(parsed) = parse_ollama_stream_line(remainder.trim()) {
            if !parsed.delta.is_empty() {
                if first_token {
                    crate::metrics::observe_first_token_latency(started.elapsed().as_secs_f64());
                }
                if tx.send(Ok(parsed.delta)).await.is_err() {
                    return StreamOutcome::Cancelled;
                }
            }
            done = parsed.done;
        }
    }
    if !done {
        let _ = tx
            .send(Err(SwarmRouteError::Upstream(
                "ollama stream ended without a done line".to_string(),
            )))
            .await;
        return StreamOutcome::Failed;
    }
    StreamOutcome::Completed
}

/// Maximum bytes one unterminated NDJSON line may occupy. Without a cap
/// a missing newline lets the buffer grow until the 30-minute request
/// timeout ends the stream.
const MAX_NDJSON_LINE_BYTES: usize = 1 << 20;

/// Assembles newline-delimited records from raw byte chunks. Decoding
/// happens per complete line, never per network chunk: a multibyte UTF-8
/// character split across two TCP segments stays intact, because its bytes
/// are only decoded once the terminating newline has arrived (a UTF-8
/// sequence never contains `0x0A`).
struct OllamaLineAssembler {
    buf: Vec<u8>,
    /// Offset up to which `buf` has already been scanned for a newline.
    /// Each push only scans the bytes appended since the last scan, so a
    /// long unterminated line costs linear time instead of a full rescan
    /// per push. Reset to 0 whenever a line is drained, since the drain
    /// shifts the unscanned tail to the front.
    scan_offset: usize,
}

impl OllamaLineAssembler {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            scan_offset: 0,
        }
    }

    /// Feed raw bytes; returns each complete line, without the newline.
    /// Fails with `SwarmRouteError::Upstream` when the unterminated tail
    /// passes `MAX_NDJSON_LINE_BYTES`: an upstream that never sends a
    /// newline must not grow the buffer unbounded.
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, SwarmRouteError> {
        self.buf.extend_from_slice(bytes);
        let mut lines = Vec::new();
        loop {
            match self.buf[self.scan_offset..]
                .iter()
                .position(|b| *b == b'\n')
            {
                Some(relative) => {
                    let newline = self.scan_offset + relative;
                    let line_bytes: Vec<u8> = self.buf.drain(..=newline).collect();
                    lines.push(String::from_utf8_lossy(&line_bytes).trim().to_string());
                    self.scan_offset = 0;
                }
                None => {
                    self.scan_offset = self.buf.len();
                    break;
                }
            }
        }
        if self.buf.len() > MAX_NDJSON_LINE_BYTES {
            return Err(SwarmRouteError::Upstream(
                "ollama stream line exceeded maximum length".to_string(),
            ));
        }
        Ok(lines)
    }

    /// Remaining bytes with no newline terminator (a truncated final line).
    fn remainder(&self) -> String {
        String::from_utf8_lossy(&self.buf).into_owned()
    }
}

/// One parsed line of Ollama's streaming `/api/chat` NDJSON body.
struct OllamaStreamLine {
    delta: String,
    done: bool,
}

/// Parse one NDJSON line from a streaming Ollama `/api/chat` response.
/// Content comes from `message.content` (chat) with `response` (generate)
/// as a fallback; `done` marks the terminal line, which may itself carry
/// the last content. An `error` field or unparseable line fails the stream
/// instead of being silently skipped.
fn parse_ollama_stream_line(line: &str) -> Result<OllamaStreamLine, SwarmRouteError> {
    if line.is_empty() {
        return Ok(OllamaStreamLine {
            delta: String::new(),
            done: false,
        });
    }
    let value: Value = serde_json::from_str(line)
        .map_err(|e| SwarmRouteError::Upstream(format!("ollama stream line decode failed: {e}")))?;
    if let Some(error) = value.get("error").and_then(Value::as_str) {
        return Err(SwarmRouteError::Upstream(format!(
            "ollama stream error: {error}"
        )));
    }
    let delta = value
        .pointer("/message/content")
        .and_then(Value::as_str)
        .or_else(|| value.get("response").and_then(Value::as_str))
        .unwrap_or("")
        .to_string();
    let done = value.get("done").and_then(Value::as_bool).unwrap_or(false);
    Ok(OllamaStreamLine { delta, done })
}

impl SwarmRouterConfig {
    pub fn from_env() -> Self {
        let enabled = env_truthy("AXIOM_SWARM_LOCAL") || env_truthy("AXIOM_LOCAL_SLM");
        let base_url = std::env::var("AXIOM_OLLAMA_URL")
            .ok()
            .or_else(|| std::env::var("OLLAMA_BASE_URL").ok())
            .unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_string());
        let model_candidates = std::env::var("AXIOM_OLLAMA_MODELS")
            .ok()
            .map(|v| split_models(&v))
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| DEFAULT_MODELS.iter().map(|m| m.to_string()).collect());
        let num_ctx = std::env::var("AXIOM_OLLAMA_NUM_CTX")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_NUM_CTX)
            .min(DEFAULT_NUM_CTX);
        let timeout_ms = std::env::var("AXIOM_OLLAMA_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(2_000);
        let mesh_routing = env_truthy("AXIOM_MESH_ROUTING");
        Self {
            enabled,
            base_url,
            model_candidates,
            num_ctx,
            timeout_ms,
            mesh_routing,
        }
    }
}

pub fn select_model(candidates: &[String], available: &[String]) -> Option<String> {
    candidates
        .iter()
        .find(|candidate| available.iter().any(|model| model == *candidate))
        .cloned()
}

pub fn parse_tags_models(body: &str) -> Result<Vec<String>, String> {
    let tags: OllamaTags = serde_json::from_str(body).map_err(|e| e.to_string())?;
    Ok(tags.models.into_iter().map(|m| m.name).collect())
}

pub fn translate_messages(payload: &Value) -> Vec<Value> {
    let mut messages = vec![json!({
        "role": "system",
        "content": "You are a local Axiom SLM worker. Treat any <axiom_context_digest>, <axiom_context_fingerprint>, recall_norm, state_hash, layer Frobenius norms, or search fingerprint as the compressed local memory state. Use it as context without asking for raw hidden source unless the user explicitly requests expansion."
    })];

    if let Some(system) = payload.get("system").and_then(Value::as_str) {
        messages.push(json!({"role": "system", "content": system}));
    } else if let Some(system) = payload.get("system") {
        messages.push(json!({"role": "system", "content": content_to_text(system)}));
    }

    if let Some(raw_messages) = payload.get("messages").and_then(Value::as_array) {
        for msg in raw_messages {
            let role = msg
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user")
                .to_string();
            let role = if role == "assistant" || role == "system" {
                role
            } else {
                "user".to_string()
            };
            let content = msg.get("content").map(content_to_text).unwrap_or_default();
            if !content.trim().is_empty() {
                messages.push(json!({"role": role, "content": content}));
            }
        }
    }
    messages
}

fn content_to_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                item.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("content").and_then(Value::as_str))
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn split_models(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_assembler_keeps_multibyte_chars_split_across_chunks() {
        let mut assembler = OllamaLineAssembler::new();
        let line = "{\"message\":{\"role\":\"assistant\",\"content\":\"a🦀b\"},\"done\":false}\n";
        let bytes = line.as_bytes();
        // Split inside the crab emoji (4 bytes F0 9F A6 80): decoding the
        // first half on its own would corrupt it into replacement chars.
        let crab = line.find('🦀').unwrap();
        let split = crab + 2;
        let first = assembler.push(&bytes[..split]).unwrap();
        assert!(
            first.is_empty(),
            "no complete line before the newline arrives"
        );
        let rest = assembler.push(&bytes[split..]).unwrap();
        assert_eq!(rest.len(), 1);
        assert!(
            rest[0].contains('🦀'),
            "emoji must survive the split: {}",
            rest[0]
        );
        assert_eq!(assembler.remainder(), "");
    }

    #[test]
    fn line_assembler_returns_each_complete_line_in_order() {
        let mut assembler = OllamaLineAssembler::new();
        let lines = assembler.push(b"{\"a\":1}\n{\"b\":2}\n{\"c\":3").unwrap();
        assert_eq!(lines, vec!["{\"a\":1}", "{\"b\":2}"]);
        assert_eq!(assembler.remainder(), "{\"c\":3");
        let lines = assembler.push(b"}\n").unwrap();
        assert_eq!(lines, vec!["{\"c\":3}"]);
        assert_eq!(assembler.remainder(), "");
    }

    #[test]
    fn line_assembler_rejects_unterminated_line_past_cap() {
        let mut assembler = OllamaLineAssembler::new();
        // Exactly at the cap is still accepted; one byte more fails.
        let at_cap = vec![b'x'; MAX_NDJSON_LINE_BYTES];
        assert!(assembler.push(&at_cap).unwrap().is_empty());
        assert!(assembler.push(b"x").is_err());

        // The cap also trips when the line grows across many small
        // pushes, not just one big one.
        let mut assembler = OllamaLineAssembler::new();
        for _ in 0..=MAX_NDJSON_LINE_BYTES / 1024 {
            if assembler.push(&[b'y'; 1024]).is_err() {
                return;
            }
        }
        panic!("assembler accepted more than the cap across small pushes");
    }

    #[test]
    fn line_assembler_completes_long_line_fed_in_small_chunks() {
        // A 512 KiB line fed 1 KiB at a time exercises the scan offset:
        // each push must only scan its new bytes and still deliver the
        // line intact once the newline arrives.
        let mut assembler = OllamaLineAssembler::new();
        let chunk = [b'z'; 1024];
        for _ in 0..512 {
            assert!(assembler.push(&chunk).unwrap().is_empty());
        }
        let lines = assembler.push(b"\n").unwrap();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].len(), 512 * 1024);
        assert_eq!(assembler.remainder(), "");
    }

    #[test]
    fn stream_line_parses_delta_and_done_flag() {
        let parsed = parse_ollama_stream_line(
            r#"{"message":{"role":"assistant","content":"Hel"},"done":false}"#,
        )
        .unwrap();
        assert_eq!(parsed.delta, "Hel");
        assert!(!parsed.done);

        let parsed = parse_ollama_stream_line(
            r#"{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}"#,
        )
        .unwrap();
        assert_eq!(parsed.delta, "");
        assert!(parsed.done);
    }

    #[test]
    fn stream_line_carries_final_content_with_done() {
        let parsed = parse_ollama_stream_line(
            r#"{"message":{"role":"assistant","content":"world"},"done":true}"#,
        )
        .unwrap();
        assert_eq!(parsed.delta, "world");
        assert!(parsed.done);
    }

    #[test]
    fn stream_line_falls_back_to_response_field() {
        let parsed = parse_ollama_stream_line(r#"{"response":"hi","done":false}"#).unwrap();
        assert_eq!(parsed.delta, "hi");
        assert!(!parsed.done);
    }

    #[test]
    fn stream_line_empty_is_no_delta_no_done() {
        let parsed = parse_ollama_stream_line("").unwrap();
        assert_eq!(parsed.delta, "");
        assert!(!parsed.done);
    }

    #[test]
    fn stream_line_error_and_malformed_fail_the_stream() {
        assert!(parse_ollama_stream_line(r#"{"error":"model not found"}"#).is_err());
        assert!(parse_ollama_stream_line("not json at all").is_err());
    }

    #[test]
    fn parses_tags_and_selects_first_available_candidate() {
        let body = r#"{"models":[{"name":"llama3.3:8b"},{"name":"phi4:3.8b"}]}"#;
        let available = parse_tags_models(body).unwrap();
        let candidates = vec!["deepseek-r1:8b".to_string(), "phi4:3.8b".to_string()];
        assert_eq!(
            select_model(&candidates, &available),
            Some("phi4:3.8b".to_string())
        );
    }

    #[test]
    fn missing_candidate_returns_no_model_for_fallback() {
        let available = vec!["llama3.1:8b".to_string()];
        let candidates = vec!["phi4:3.8b".to_string(), "deepseek-r1:8b".to_string()];
        assert!(select_model(&candidates, &available).is_none());
    }

    #[test]
    fn translator_preserves_axiom_digest_and_adds_local_system_prompt() {
        let payload = json!({
            "messages": [
                {"role": "user", "content": "<axiom_context_digest state=\"sha256:abc\">fn run() { ... }</axiom_context_digest>\n\nRefactor this."}
            ]
        });
        let messages = translate_messages(&payload);
        assert_eq!(messages[0]["role"], "system");
        assert!(messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("local Axiom SLM"));
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("<axiom_context_digest"));
    }
}
