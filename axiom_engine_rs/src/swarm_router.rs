//! Local SLM routing through Ollama.

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MODELS: [&str; 4] = ["phi4:3.8b", "deepseek-r1:8b", "llama3.3:8b", "llama3.1:8b"];
const DEFAULT_NUM_CTX: u64 = 4096;

#[derive(Clone)]
pub struct SwarmRouter {
    config: SwarmRouterConfig,
    client: Client,
}

#[derive(Debug, Clone)]
pub struct SwarmRouterConfig {
    pub enabled: bool,
    pub base_url: String,
    pub model_candidates: Vec<String>,
    pub num_ctx: u64,
    pub timeout_ms: u64,
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
        Self { config, client }
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

        let available = self.available_models().await?;
        let model = select_model(&self.config.model_candidates, &available).ok_or_else(|| {
            SwarmRouteError::NoModel {
                requested: self.config.model_candidates.clone(),
                available: available.clone(),
            }
        })?;

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
        let content = decoded
            .message
            .map(|m| m.content)
            .or(decoded.response)
            .unwrap_or_default();
        Ok(SwarmChatResult { model, content })
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
        Self {
            enabled,
            base_url,
            model_candidates,
            num_ctx,
            timeout_ms,
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
