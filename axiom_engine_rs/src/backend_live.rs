//! Live backends for the multi-provider [`crate::backend_router::Router`].
//!
//! Bridges the router's synchronous [`ChatBackend`] trait to Axiom's real
//! generators: the local TTT pipeline, the (sync) Claude backend, and an
//! OpenAI-compatible endpoint over blocking HTTP (so GPT — or a local OpenDrop
//! server — is a first-class router provider). [`router_from_env`] assembles a
//! router when `AXIOM_BACKEND` selects `router`, `openai`, or `opendrop`,
//! registering whichever providers are configured.
//!
//! **Blocking-client caveat:** the OpenAI adapter and Claude backend build
//! `reqwest::blocking` clients, which spin up their own runtime and *panic* if
//! constructed on a thread already inside a Tokio runtime. Callers in the async
//! server therefore build the router inside `tokio::task::spawn_blocking` (see
//! `server.rs`).

use std::sync::{Arc, RwLock};

use serde_json::json;

use crate::backend_router::{BackendError, ChatBackend, Provider, RoutePolicy, Router};
use crate::claude_backend::{ClaudeBackend, DEFAULT_CLAUDE_MODEL};
use crate::inference::InferencePipeline;

/// Default OpenDrop endpoint for the router's OpenAI-compatible provider. The
/// blocking adapter appends `/chat/completions`, so this base includes `/v1`
/// (cf. the forwarder path in `openai_forwarder.rs`, which appends the full
/// `/v1/chat/completions` and therefore uses a bare host).
const OPENDROP_ROUTER_BASE_URL: &str = "http://127.0.0.1:11434/v1";

/// Local TTT pipeline as a router backend (always-available fallback).
struct LocalPipelineBackend {
    pipeline: Arc<RwLock<InferencePipeline>>,
}

impl ChatBackend for LocalPipelineBackend {
    fn complete(&self, prompt: &str, max_tokens: usize) -> Result<String, BackendError> {
        let pipeline = self
            .pipeline
            .read()
            .map_err(|_| BackendError::Upstream("pipeline lock poisoned".into()))?;
        pipeline
            .generate(prompt, max_tokens)
            .map_err(|e| BackendError::Upstream(format!("local generation failed: {e}")))
    }
}

/// Anthropic (Claude) as a router backend, wrapping the existing sync backend.
struct ClaudeChatBackend {
    backend: ClaudeBackend,
}

impl ChatBackend for ClaudeChatBackend {
    fn complete(&self, prompt: &str, max_tokens: usize) -> Result<String, BackendError> {
        self.backend
            .generate(prompt, max_tokens)
            .map_err(BackendError::Upstream)
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
}

impl ChatBackend for OpenAiBlockingBackend {
    fn complete(&self, prompt: &str, max_tokens: usize) -> Result<String, BackendError> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body = json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": prompt }],
            "max_tokens": max_tokens,
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

/// Which backend mode is selected, if any of the router-capable ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouterMode {
    /// Multi-provider routing (code→Claude, else→GPT) with failover.
    Router,
    /// Single OpenAI-compatible endpoint (cloud or local), local fallback.
    OpenAi,
    /// OpenDrop preset: OpenAI-compatible at the local default base.
    OpenDrop,
}

fn router_mode_from(backend: &str) -> Option<RouterMode> {
    match backend.to_ascii_lowercase().as_str() {
        "router" => Some(RouterMode::Router),
        "openai" => Some(RouterMode::OpenAi),
        "opendrop" => Some(RouterMode::OpenDrop),
        _ => None,
    }
}

/// Resolve the OpenAI-compatible base URL for a given mode.
fn openai_base_url_for(mode: RouterMode) -> String {
    if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
        return url;
    }
    if let Ok(url) = std::env::var("OPENAI_API_BASE") {
        return url;
    }
    match mode {
        RouterMode::OpenDrop => OPENDROP_ROUTER_BASE_URL.to_string(),
        RouterMode::Router | RouterMode::OpenAi => "https://api.openai.com/v1".to_string(),
    }
}

/// Build a Claude backend straight from `ANTHROPIC_API_KEY` (router mode does
/// not require `AXIOM_BACKEND=claude`, which is what `ClaudeBackend::from_env`
/// gates on). Returns `None` when no key is configured.
fn claude_from_key() -> Option<ClaudeBackend> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty())?;
    let model =
        std::env::var("AXIOM_CLAUDE_MODEL").unwrap_or_else(|_| DEFAULT_CLAUDE_MODEL.to_string());
    let default_system = std::env::var("AXIOM_CLAUDE_SYSTEM").ok();
    // Note: a custom `ANTHROPIC_BASE_URL` is only honored on the
    // `AXIOM_BACKEND=claude` path (`ClaudeBackend::from_env`), which has the
    // base-url setter; the router uses the default Anthropic base.
    Some(ClaudeBackend::new(model, api_key, default_system))
}

/// Build a [`Router`] from the environment for the router-capable backends
/// (`router`, `openai`, `opendrop`).
///
/// * `router` registers the local pipeline (always), Claude (if
///   `ANTHROPIC_API_KEY` is set), and OpenAI (if `OPENAI_API_KEY` is set or a
///   custom `OPENAI_BASE_URL`/`OPENAI_API_BASE` points at a local server).
/// * `openai` / `opendrop` register a single OpenAI-compatible provider plus the
///   local pipeline as fallback, with a policy that routes every task to it.
///
/// Returns `None` for any other backend so callers fall back to their default
/// single-backend path.
///
/// **Must be called off the Tokio runtime** (see module docs): it constructs
/// blocking `reqwest` clients.
pub fn router_from_env(pipeline: Arc<RwLock<InferencePipeline>>) -> Option<Router> {
    let backend = std::env::var("AXIOM_BACKEND").unwrap_or_default();
    let mode = router_mode_from(&backend)?;

    let consensus = std::env::var("AXIOM_ROUTER_CONSENSUS")
        .ok()
        .map(|v| !matches!(v.trim(), "0" | "false" | "off" | ""))
        .unwrap_or(false);

    match mode {
        RouterMode::Router => {
            let policy = RoutePolicy {
                consensus,
                ..RoutePolicy::default()
            };
            let mut router = Router::new(policy).with(
                Provider::Local,
                Box::new(LocalPipelineBackend { pipeline }),
            );

            if let Some(claude) = claude_from_key() {
                router = router.with(
                    Provider::Anthropic,
                    Box::new(ClaudeChatBackend { backend: claude }),
                );
            }

            let openai_key = std::env::var("OPENAI_API_KEY")
                .ok()
                .filter(|k| !k.trim().is_empty());
            let has_custom_base = std::env::var("OPENAI_BASE_URL").is_ok()
                || std::env::var("OPENAI_API_BASE").is_ok();
            if openai_key.is_some() || has_custom_base {
                let model =
                    std::env::var("AXIOM_OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
                router = router.with(
                    Provider::OpenAi,
                    Box::new(OpenAiBlockingBackend {
                        client: reqwest::blocking::Client::new(),
                        api_key: openai_key,
                        base_url: openai_base_url_for(mode),
                        model,
                    }),
                );
            }

            Some(router)
        }
        RouterMode::OpenAi | RouterMode::OpenDrop => {
            // Route every task to the single OpenAI-compatible provider, with
            // the local pipeline as deterministic last-resort fallback.
            let policy = RoutePolicy {
                code_repair: Provider::OpenAi,
                reasoning: Provider::OpenAi,
                general: Provider::OpenAi,
                failover: vec![Provider::OpenAi, Provider::Local],
                consensus,
            };
            let openai_key = std::env::var("OPENAI_API_KEY")
                .ok()
                .filter(|k| !k.trim().is_empty());
            let model =
                std::env::var("AXIOM_OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
            let router = Router::new(policy)
                .with(
                    Provider::OpenAi,
                    Box::new(OpenAiBlockingBackend {
                        client: reqwest::blocking::Client::new(),
                        api_key: openai_key,
                        base_url: openai_base_url_for(mode),
                        model,
                    }),
                )
                .with(
                    Provider::Local,
                    Box::new(LocalPipelineBackend { pipeline }),
                );
            Some(router)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_mode_parsing() {
        assert_eq!(router_mode_from("router"), Some(RouterMode::Router));
        assert_eq!(router_mode_from("OpenAI"), Some(RouterMode::OpenAi));
        assert_eq!(router_mode_from("opendrop"), Some(RouterMode::OpenDrop));
        assert_eq!(router_mode_from("bootstrap"), None);
        assert_eq!(router_mode_from("claude"), None);
    }

    #[test]
    fn consensus_env_var_parsing() {
        let truthy = ["1", "true", "yes", "on"];
        let falsy = ["0", "false", "off", ""];
        for v in truthy {
            let enabled = !matches!(v.trim(), "0" | "false" | "off" | "");
            assert!(enabled, "expected {v:?} to enable consensus");
        }
        for v in falsy {
            let enabled = !matches!(v.trim(), "0" | "false" | "off" | "");
            assert!(!enabled, "expected {v:?} to disable consensus");
        }
    }

    #[test]
    fn opendrop_base_url_defaults_local() {
        // Only meaningful when no env override is present; assert the mode-mapped
        // default rather than mutating process env (parallel-safe).
        if std::env::var("OPENAI_BASE_URL").is_err()
            && std::env::var("OPENAI_API_BASE").is_err()
        {
            assert_eq!(
                openai_base_url_for(RouterMode::OpenDrop),
                OPENDROP_ROUTER_BASE_URL
            );
            assert!(openai_base_url_for(RouterMode::OpenAi).contains("api.openai.com"));
        }
    }
}
