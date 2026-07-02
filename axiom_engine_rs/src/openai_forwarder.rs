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
///
/// This remains a bare host for consistency with the default OpenAI base.
const OPENDROP_DEFAULT_BASE_URL: &str = "http://127.0.0.1:11434";

/// Responses endpoint on the ChatGPT backend. Codex authenticated with a
/// ChatGPT subscription (OAuth) can only call this backend — its token has no
/// `api.responses.write` scope on the platform API (api.openai.com).
const CHATGPT_CODEX_RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";

fn api_root(base: &str) -> &str {
    let base = base.trim_end_matches('/');
    base.strip_suffix("/v1").unwrap_or(base)
}

fn chat_completions_url(base: &str) -> String {
    format!("{}/v1/chat/completions", api_root(base))
}

fn responses_url(base: &str) -> String {
    format!("{}/v1/responses", api_root(base))
}

/// Pick the Responses upstream for a specific request.
///
/// Precedence:
/// 1. `AXIOM_OPENAI_RESPONSES_UPSTREAM` env var — full URL, used verbatim.
/// 2. Client sent `chatgpt-account-id` (Codex in ChatGPT-subscription mode)
///    — route to the ChatGPT backend, the only upstream its token works on.
/// 3. Default platform API: `{base}/v1/responses`.
fn responses_upstream_url(base: &str, auth: &OpenAiClientAuth) -> String {
    responses_upstream_url_with(
        std::env::var("AXIOM_OPENAI_RESPONSES_UPSTREAM").ok().as_deref(),
        base,
        auth,
    )
}

/// Pure core of [`responses_upstream_url`] — env override injected as a
/// parameter so routing is unit-testable without process-global mutation.
fn responses_upstream_url_with(
    override_url: Option<&str>,
    base: &str,
    auth: &OpenAiClientAuth,
) -> String {
    if let Some(explicit) = override_url {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    let has_chatgpt_account = auth
        .extra_headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("chatgpt-account-id"));
    if has_chatgpt_account {
        return CHATGPT_CODEX_RESPONSES_URL.to_string();
    }
    responses_url(base)
}

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
    /// Additional client headers relayed verbatim upstream (lowercase names).
    /// Codex in ChatGPT-subscription mode needs `chatgpt-account-id`,
    /// `openai-beta`, `session_id`, and `originator` to reach the ChatGPT
    /// backend; hop-by-hop and body-encoding headers must never be included.
    pub extra_headers: Vec<(String, String)>,
}

/// Active outbound bridge. Cheap to clone (`reqwest::Client` is Arc-backed).
#[derive(Clone)]
pub struct OpenAiForwarder {
    api_key: Option<String>,
    base_url: String,
    client: Client,
    streaming_client: Client,
}

impl OpenAiForwarder {
    pub fn new(api_key: Option<String>, base_url: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("reqwest async client should construct");
        // Streaming agent runs can legitimately exceed two minutes. This
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
        let url = chat_completions_url(&self.base_url);
        self.forward_json_text(url, payload, auth).await
    }

    /// POST to `/v1/responses` without translating the request or response.
    /// Native passthrough preserves function calls, reasoning items, and SSE
    /// event types that do not have lossless Chat Completions equivalents.
    pub async fn forward_responses(
        &self,
        payload: &Value,
        auth: &OpenAiClientAuth,
    ) -> Result<reqwest::Response, OpenAiForwarderError> {
        let url = responses_upstream_url(&self.base_url, auth);
        let client = if payload.get("stream").and_then(Value::as_bool) == Some(true) {
            &self.streaming_client
        } else {
            &self.client
        };
        self.send_json_with(client, url, payload, auth).await
    }

    async fn forward_json_text(
        &self,
        url: String,
        payload: &Value,
        auth: &OpenAiClientAuth,
    ) -> Result<OpenAiForwardedResponse, OpenAiForwarderError> {
        let response = self
            .send_json_with(&self.client, url, payload, auth)
            .await?;

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

    async fn send_json_with(
        &self,
        client: &Client,
        url: String,
        payload: &Value,
        auth: &OpenAiClientAuth,
    ) -> Result<reqwest::Response, OpenAiForwarderError> {
        let mut request = client.post(url).header("content-type", "application/json");

        if let Some(authz) = auth.authorization.as_deref() {
            request = request.header("authorization", authz);
        } else if let Some(key) = self.api_key.as_deref() {
            request = request.header("authorization", bearer_value(key));
        } else {
            return Err(OpenAiForwarderError::MissingAuth);
        }

        for (name, value) in &auth.extra_headers {
            if name.eq_ignore_ascii_case("authorization") {
                continue; // handled above
            }
            request = request.header(name.as_str(), value.as_str());
        }

        request
            .json(payload)
            .send()
            .await
            .map_err(|e| OpenAiForwarderError::Network(e.to_string()))
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
    use super::{
        bearer_value, chat_completions_url, resolve_base_url, responses_url,
        OPENDROP_DEFAULT_BASE_URL,
    };

    #[test]
    fn opendrop_default_base_has_no_v1_suffix() {
        // The forwarder appends `/v1/chat/completions`; the base must NOT carry
        // its own `/v1` or the final URL doubles it.
        assert!(
            !OPENDROP_DEFAULT_BASE_URL.ends_with("/v1"),
            "opendrop base must be a bare host, got {OPENDROP_DEFAULT_BASE_URL}"
        );
        assert_eq!(
            chat_completions_url(OPENDROP_DEFAULT_BASE_URL),
            "http://127.0.0.1:11434/v1/chat/completions"
        );
    }

    #[test]
    fn chat_completions_url_appends_v1_path_once() {
        assert_eq!(
            chat_completions_url("https://api.openai.com"),
            "https://api.openai.com/v1/chat/completions"
        );
        // Trailing slash is trimmed (no double slash).
        assert_eq!(
            chat_completions_url("https://api.openai.com/"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://api.openai.com/v1/"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn chatgpt_account_header_routes_to_chatgpt_backend() {
        use super::{responses_upstream_url_with, OpenAiClientAuth, CHATGPT_CODEX_RESPONSES_URL};
        let auth = OpenAiClientAuth {
            authorization: Some("Bearer tok".into()),
            extra_headers: vec![("chatgpt-account-id".into(), "acct-123".into())],
        };
        assert_eq!(
            responses_upstream_url_with(None, "https://api.openai.com", &auth),
            CHATGPT_CODEX_RESPONSES_URL
        );
        // Case-insensitive header match.
        let auth_upper = OpenAiClientAuth {
            authorization: None,
            extra_headers: vec![("ChatGPT-Account-Id".into(), "acct-123".into())],
        };
        assert_eq!(
            responses_upstream_url_with(None, "https://api.openai.com", &auth_upper),
            CHATGPT_CODEX_RESPONSES_URL
        );
    }

    #[test]
    fn no_chatgpt_header_uses_platform_default() {
        use super::{responses_upstream_url_with, OpenAiClientAuth};
        let auth = OpenAiClientAuth::default();
        assert_eq!(
            responses_upstream_url_with(None, "https://api.openai.com", &auth),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn explicit_override_beats_header_routing() {
        use super::{responses_upstream_url_with, OpenAiClientAuth};
        let auth = OpenAiClientAuth {
            authorization: None,
            extra_headers: vec![("chatgpt-account-id".into(), "acct-123".into())],
        };
        assert_eq!(
            responses_upstream_url_with(Some("http://127.0.0.1:9999/x"), "https://api.openai.com", &auth),
            "http://127.0.0.1:9999/x"
        );
        // Blank override falls through to header routing.
        assert_eq!(
            responses_upstream_url_with(Some("  "), "https://api.openai.com", &auth),
            super::CHATGPT_CODEX_RESPONSES_URL
        );
    }

    #[test]
    fn responses_url_appends_v1_path_once() {
        assert_eq!(
            responses_url("https://api.openai.com/"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            responses_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/responses"
        );
    }

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
