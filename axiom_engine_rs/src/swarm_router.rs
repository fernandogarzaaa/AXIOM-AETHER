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
            let completed =
                run_stream_dispatch(&client, &base_url, &stream_model, num_ctx, messages, tx).await;
            if let Some(selector) = &mesh_selector {
                selector.record_outcome(
                    &stream_model,
                    completed,
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

/// Producer half of [`SwarmStream`]: streams one Ollama `/api/chat`
/// response, sending each content delta through the bounded channel as it
/// arrives. Returns `true` only when the upstream stream ended with a
/// `done` line. The channel is the backpressure mechanism: `send` blocks
/// while it is full, so a slow consumer stalls this task instead of letting
/// it accumulate the whole generation. A dropped receiver (client
/// disconnect) makes `send` fail, and this task then returns, dropping the
/// response and aborting the upstream request.
async fn run_stream_dispatch(
    client: &Client,
    base_url: &str,
    model: &str,
    num_ctx: u64,
    messages: Vec<Value>,
    tx: tokio::sync::mpsc::Sender<Result<String, SwarmRouteError>>,
) -> bool {
    let request = json!({
        "model": model,
        "stream": true,
        "messages": messages,
        "options": {
            "num_ctx": num_ctx
        }
    });
    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));
    let response = match client
        .post(url)
        .json(&request)
        .timeout(STREAM_REQUEST_TIMEOUT)
        .send()
        .await
    {
        Ok(response) => response,
        Err(e) => {
            let _ = tx.send(Err(SwarmRouteError::Upstream(e.to_string()))).await;
            return false;
        }
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
        return false;
    }
    let mut byte_stream = response.bytes_stream();
    let mut line_buf = String::new();
    let mut first_token = true;
    let started = Instant::now();
    let mut done = false;
    while let Some(chunk) = byte_stream.next().await {
        let chunk = match chunk {
            Ok(bytes) => bytes,
            Err(e) => {
                let _ = tx
                    .send(Err(SwarmRouteError::Upstream(format!(
                        "body read failed: {e}"
                    ))))
                    .await;
                return false;
            }
        };
        line_buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(newline) = line_buf.find('\n') {
            let line: String = line_buf.drain(..=newline).collect();
            match parse_ollama_stream_line(line.trim()) {
                Ok(parsed) => {
                    if !parsed.delta.is_empty() {
                        if first_token {
                            crate::metrics::observe_first_token_latency(
                                started.elapsed().as_secs_f64(),
                            );
                            first_token = false;
                        }
                        if tx.send(Ok(parsed.delta)).await.is_err() {
                            return false;
                        }
                    }
                    if parsed.done {
                        done = true;
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return false;
                }
            }
        }
        if done {
            break;
        }
    }
    if !done {
        // A trailing partial line with no newline terminator: parse it once.
        // A stream that ends without a `done` line is incomplete; the
        // channel closing without a done line tells the SSE layer that.
        if let Ok(parsed) = parse_ollama_stream_line(line_buf.trim()) {
            if !parsed.delta.is_empty() {
                let _ = tx.send(Ok(parsed.delta)).await;
            }
            done = parsed.done;
        }
    }
    done
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
