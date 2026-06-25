//! Outbound bridge to OpenAI-compatible Chat Completions APIs used by the
//! context-compression pipeline.
//!
//! This mirrors the Anthropic bridge for Codex/OpenAI clients: when active
//! compression is enabled, `/v1/chat/completions` payloads are compacted by
//! Axiom first, then forwarded to the configured upstream with client auth
//! relayed or a proxy-owned key injected.

use std::time::Duration;

use reqwest::Client;
use serde_json::Value;

const DEFAULT_BASE_URL: &str = "https://api.openai.com";
/// Default endpoint for a local [OpenDrop](https://github.com/fernandogarzaaa/OpenDrop)
/// server (OpenAI-compatible). Used when `AXIOM_BACKEND=opendrop` and no explicit
/// base URL is set, so `opendrop run <model>` + `AXIOM_BACKEND=opendrop` "just
/// works" against a local model with zero per-token cost.
const OPENDROP_DEFAULT_BASE_URL: &str = "http://127.0.0.1:11434/v1";

/// Resolve the effective OpenAI-compatible base URL from env signals. An explicit
/// `OPENAI_BASE_URL`/`OPENAI_API_BASE` always wins; otherwise `AXIOM_BACKEND=opendrop`
/// selects the local OpenDrop endpoint; otherwise `None` (caller falls back to the
/// real OpenAI default). Pure so it is unit-testable without touching process env.
fn resolve_base_url(
    backend: Option<&str>,
    openai_base_url: Option<String>,
    openai_api_base: Option<String>,
) -> Option<String> {
    if let Some(explicit) = openai_base_url.or(openai_api_base) {
        return Some(explicit);
    }
    if backend.map(|b| b.eq_ignore_ascii_case("opendrop")).unwrap_or(false) {
        return Some(OPENDROP_DEFAULT_BASE_URL.to_string());
    }
    None
}

/// Auth / relay headers captured from the inbound OpenAI-compatible request.
#[derive(Clone, Debug, Default)]
pub struct OpenAiClientAuth {
    pub authorization: Option<String>,
}

/// Active outbound bridge. Cheap to clone (`reqwest::Client` is Arc-backed).
#[derive(Clone)]
pub struct OpenAiForwarder {
    api_key: Option<String>,
    base_url: String,
    client: Client,
}

impl OpenAiForwarder {
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

    /// Env-driven activation. Always returns `Some(forwarder)` so the proxy
    /// works in both API-key mode and auth-passthrough mode.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty());
        let base_url = resolve_base_url(
            std::env::var("AXIOM_BACKEND").ok().as_deref(),
            std::env::var("OPENAI_BASE_URL").ok(),
            std::env::var("OPENAI_API_BASE").ok(),
        );
        Some(Self::new(api_key, base_url))
    }

    /// Whether the proxy holds its own OpenAI API key.
    pub fn has_own_key(&self) -> bool {
        self.api_key.is_some()
    }

    /// POST to `/v1/chat/completions` on the configured OpenAI-compatible API.
    ///
    /// Auth precedence:
    /// 1. Client `Authorization` header, relayed verbatim.
    /// 2. Proxy-owned `OPENAI_API_KEY`, injected as bearer auth.
    pub async fn forward_chat_completions_text(
        &self,
        payload: &Value,
        auth: &OpenAiClientAuth,
    ) -> Result<OpenAiForwardedResponse, OpenAiForwarderError> {
        let url = format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        );
        let mut request = self
            .client
            .post(&url)
            .header("content-type", "application/json");

        if let Some(authz) = auth.authorization.as_deref() {
            request = request.header("authorization", authz);
        } else if let Some(key) = self.api_key.as_deref() {
            request = request.header("authorization", bearer_value(key));
        } else {
            return Err(OpenAiForwarderError::MissingAuth);
        }

        let response = request
            .json(payload)
            .send()
            .await
            .map_err(|e| OpenAiForwarderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let body = response
            .text()
            .await
            .map_err(|e| OpenAiForwarderError::Network(format!("body read failed: {e}")))?;

        Ok(OpenAiForwardedResponse {
            status,
            content_type,
            body,
        })
    }
}

fn bearer_value(key: &str) -> String {
    if key
        .get(..7)
        .map(|prefix| prefix.eq_ignore_ascii_case("bearer "))
        .unwrap_or(false)
    {
        key.to_string()
    } else {
        format!("Bearer {key}")
    }
}

pub struct OpenAiForwardedResponse {
    pub status: u16,
    pub content_type: Option<String>,
    pub body: String,
}

#[derive(Debug)]
pub enum OpenAiForwarderError {
    Network(String),
    MissingAuth,
}

impl std::fmt::Display for OpenAiForwarderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenAiForwarderError::Network(m) => write!(f, "network error: {m}"),
            OpenAiForwarderError::MissingAuth => write!(
                f,
                "no upstream credential: client sent no Authorization header \
                 and the proxy has no OPENAI_API_KEY configured"
            ),
        }
    }
}

impl std::error::Error for OpenAiForwarderError {}

#[cfg(test)]
mod tests {
    use super::{bearer_value, resolve_base_url, OPENDROP_DEFAULT_BASE_URL};

    #[test]
    fn bearer_value_does_not_double_prefix() {
        assert_eq!(bearer_value("sk-test"), "Bearer sk-test");
        assert_eq!(bearer_value("Bearer sk-test"), "Bearer sk-test");
        assert_eq!(bearer_value("bearer sk-test"), "bearer sk-test");
    }

    #[test]
    fn opendrop_backend_defaults_to_local_endpoint() {
        assert_eq!(
            resolve_base_url(Some("opendrop"), None, None).as_deref(),
            Some(OPENDROP_DEFAULT_BASE_URL)
        );
        // Case-insensitive.
        assert_eq!(
            resolve_base_url(Some("OpenDrop"), None, None).as_deref(),
            Some(OPENDROP_DEFAULT_BASE_URL)
        );
    }

    #[test]
    fn explicit_base_url_always_wins() {
        assert_eq!(
            resolve_base_url(Some("opendrop"), Some("http://x:1/v1".into()), None).as_deref(),
            Some("http://x:1/v1")
        );
        assert_eq!(
            resolve_base_url(None, None, Some("http://y:2/v1".into())).as_deref(),
            Some("http://y:2/v1")
        );
    }

    #[test]
    fn no_signal_falls_back_to_default() {
        assert_eq!(resolve_base_url(None, None, None), None);
        assert_eq!(resolve_base_url(Some("openai"), None, None), None);
    }
}
