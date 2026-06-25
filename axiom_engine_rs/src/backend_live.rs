//! Live backends for the multi-provider [`crate::backend_router::Router`].
//!
//! Bridges the router's synchronous [`ChatBackend`] trait to Axiom's real
//! generators: the local TTT pipeline, the (sync) Claude backend, and an
//! OpenAI-compatible endpoint over blocking HTTP (so GPT — or a local OpenDrop
//! server — is a first-class router provider). [`router_from_env`] assembles a
//! router when `AXIOM_BACKEND=router`, registering whichever providers are
//! configured.

use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::backend_router::{BackendError, ChatBackend, Provider, RoutePolicy, Router};
use crate::claude_backend::ClaudeBackend;
use crate::inference::InferencePipeline;

/// Local TTT pipeline as a router backend (always-available fallback).
struct LocalPipelineBackend {
    pipeline: Arc<Mutex<InferencePipeline>>,
    max_tokens: usize,
}

impl ChatBackend for LocalPipelineBackend {
    fn complete(&self, prompt: &str) -> Result<String, BackendError> {
        let pipeline = self
            .pipeline
            .lock()
            .map_err(|_| BackendError::Upstream("pipeline lock poisoned".into()))?;
        pipeline
            .generate(prompt, self.max_tokens)
            .map_err(|e| BackendError::Upstream(format!("local generation failed: {e}")))
    }
}

/// Anthropic (Claude) as a router backend, wrapping the existing sync backend.
struct ClaudeChatBackend {
    backend: Arc<Option<ClaudeBackend>>,
    max_tokens: usize,
}

impl ChatBackend for ClaudeChatBackend {
    fn complete(&self, prompt: &str) -> Result<String, BackendError> {
        match self.backend.as_ref() {
            Some(b) => b
                .generate(prompt, self.max_tokens)
                .map_err(BackendError::Upstream),
            None => Err(BackendError::NotConfigured),
        }
    }
}

/// OpenAI-compatible endpoint (cloud OpenAI or a local OpenDrop/Ollama/vLLM
/// server) over blocking HTTP — consistent with `run_generation`'s existing
/// blocking-Claude call path.
struct OpenAiBlockingBackend {
    client: reqwest::blocking::Client,
    api_key: Option<String>,
    base_url: String,
    model: String,
    max_tokens: usize,
}

impl ChatBackend for OpenAiBlockingBackend {
    fn complete(&self, prompt: &str) -> Result<String, BackendError> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body = json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": prompt }],
            "max_tokens": self.max_tokens,
        });
        let mut req = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .map_err(|e| BackendError::Upstream(format!("openai request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(BackendError::Upstream(format!("openai {status}: {text}")));
        }
        let v: serde_json::Value = resp
            .json()
            .map_err(|e| BackendError::Upstream(format!("openai bad json: {e}")))?;
        v.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| BackendError::Upstream("openai response missing content".into()))
    }
}

/// Resolve the OpenAI-compatible base URL for the router's OpenAI provider.
fn openai_base_url() -> String {
    std::env::var("OPENAI_BASE_URL")
        .ok()
        .or_else(|| std::env::var("OPENAI_API_BASE").ok())
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
}

/// Build a [`Router`] from the environment when `AXIOM_BACKEND=router`.
///
/// Registers the local pipeline (always), Claude (if a backend is configured),
/// and OpenAI (if `OPENAI_API_KEY` is set or a custom `OPENAI_BASE_URL` points at
/// a local server). Returns `None` unless router mode is selected, so callers
/// fall back to their default single-backend path.
pub fn router_from_env(
    pipeline: Arc<Mutex<InferencePipeline>>,
    claude: Arc<Option<ClaudeBackend>>,
    max_tokens: usize,
) -> Option<Router> {
    let backend = std::env::var("AXIOM_BACKEND").unwrap_or_default();
    if !backend.eq_ignore_ascii_case("router") {
        return None;
    }

    let mut router = Router::new(RoutePolicy::default()).with(
        Provider::Local,
        Box::new(LocalPipelineBackend {
            pipeline,
            max_tokens,
        }),
    );

    if claude.is_some() {
        router = router.with(
            Provider::Anthropic,
            Box::new(ClaudeChatBackend {
                backend: claude,
                max_tokens,
            }),
        );
    }

    let openai_key = std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.trim().is_empty());
    let has_custom_base =
        std::env::var("OPENAI_BASE_URL").is_ok() || std::env::var("OPENAI_API_BASE").is_ok();
    if openai_key.is_some() || has_custom_base {
        let model = std::env::var("AXIOM_OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
        router = router.with(
            Provider::OpenAi,
            Box::new(OpenAiBlockingBackend {
                client: reqwest::blocking::Client::new(),
                api_key: openai_key,
                base_url: openai_base_url(),
                model,
                max_tokens,
            }),
        );
    }

    Some(router)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_base_url_defaults_to_openai() {
        // With no env override the default is the real OpenAI v1 base. (We don't
        // mutate process env here to stay parallel-safe; just assert the default
        // branch shape.)
        let url = openai_base_url();
        assert!(url.contains("/v1"), "base url should include /v1: {url}");
    }
}
