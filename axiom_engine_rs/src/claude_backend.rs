//! Anthropic Messages API client used as an optional generation backend.
//!
//! When configured (via [`ClaudeBackend::from_env`] or constructed directly
//! and installed on [`crate::server::AppState`]), the server routes
//! generation requests through Anthropic's hosted Claude API instead of
//! the local Axiom-TTT pipeline. The OpenAI and Anthropic HTTP surfaces
//! continue to respond in their native shapes — only the underlying
//! generator changes.
//!
//! Caveat: with this backend active, `/v1/adapt` and the per-layer
//! W̃ lifecycle become no-ops with respect to actual generation because
//! Claude is a remote frozen model. Sessions still exist so the wire
//! contract holds, but adaptation no longer influences output.

use std::time::Duration;

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

pub const DEFAULT_CLAUDE_MODEL: &str = "claude-haiku-4-5-20251001";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ClaudeBackend {
    model: String,
    api_key: String,
    base_url: String,
    default_system: Option<String>,
    client: Client,
}

impl ClaudeBackend {
    pub fn new(model: String, api_key: String, default_system: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("reqwest blocking client should construct");
        Self {
            model,
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            default_system,
            client,
        }
    }

    /// Construct a backend from environment variables, returning `None`
    /// when ``AXIOM_BACKEND`` is not set to ``claude``.
    pub fn from_env() -> Option<Self> {
        if std::env::var("AXIOM_BACKEND")
            .map(|v| v.to_lowercase())
            .ok()
            .as_deref()
            != Some("claude")
        {
            return None;
        }
        let api_key = std::env::var("ANTHROPIC_API_KEY").ok()?;
        let model =
            std::env::var("AXIOM_CLAUDE_MODEL").unwrap_or_else(|_| DEFAULT_CLAUDE_MODEL.to_string());
        let default_system = std::env::var("AXIOM_CLAUDE_SYSTEM").ok();
        let mut backend = Self::new(model, api_key, default_system);
        if let Ok(base) = std::env::var("ANTHROPIC_BASE_URL") {
            backend.base_url = base;
        }
        Some(backend)
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Bare prompt → response text.
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, String> {
        self.generate_chat(
            &[ChatTurn {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens,
            None,
        )
    }

    /// Multi-turn chat → response text. Non-user/assistant roles are
    /// folded into the system field.
    pub fn generate_chat(
        &self,
        turns: &[ChatTurn],
        max_tokens: usize,
        system: Option<String>,
    ) -> Result<String, String> {
        let (messages, folded_system) = normalize_turns(turns);
        let effective_system = system
            .or(folded_system)
            .or_else(|| self.default_system.clone());

        let body = AnthropicRequest {
            model: &self.model,
            max_tokens,
            messages: &messages,
            system: effective_system.as_deref(),
        };

        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| format!("anthropic request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .unwrap_or_else(|_| "<no body>".to_string());
            return Err(format!("anthropic API error {status}: {text}"));
        }

        let parsed: AnthropicResponse = response
            .json()
            .map_err(|e| format!("anthropic response decode failed: {e}"))?;

        Ok(parsed
            .content
            .into_iter()
            .filter(|block| block.block_type == "text")
            .map(|block| block.text)
            .collect::<Vec<_>>()
            .concat())
    }
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: usize,
    messages: &'a [AnthropicChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
}

#[derive(Serialize)]
struct AnthropicChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    content: Vec<AnthropicContentBlock>,
}

#[derive(Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

/// Split system turns out and coerce the chat sequence to start with `user`
/// (Anthropic requires alternating user/assistant beginning with user).
fn normalize_turns(turns: &[ChatTurn]) -> (Vec<AnthropicChatMessage>, Option<String>) {
    let mut system_parts: Vec<String> = Vec::new();
    let mut chat: Vec<AnthropicChatMessage> = Vec::new();
    for turn in turns {
        match turn.role.as_str() {
            "system" => system_parts.push(turn.content.clone()),
            "user" | "assistant" => chat.push(AnthropicChatMessage {
                role: turn.role.clone(),
                content: turn.content.clone(),
            }),
            other => system_parts.push(format!("[{other}] {}", turn.content)),
        }
    }

    if chat.is_empty() {
        chat.push(AnthropicChatMessage {
            role: "user".to_string(),
            content: String::new(),
        });
    } else if chat[0].role != "user" {
        chat.insert(
            0,
            AnthropicChatMessage {
                role: "user".to_string(),
                content: String::new(),
            },
        );
    }

    let folded = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (chat, folded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_extracts_system_messages() {
        let turns = vec![
            ChatTurn {
                role: "system".into(),
                content: "be terse".into(),
            },
            ChatTurn {
                role: "user".into(),
                content: "hi".into(),
            },
            ChatTurn {
                role: "assistant".into(),
                content: "hello".into(),
            },
            ChatTurn {
                role: "user".into(),
                content: "again".into(),
            },
        ];
        let (chat, system) = normalize_turns(&turns);
        assert_eq!(system.as_deref(), Some("be terse"));
        assert_eq!(chat.len(), 3);
        assert_eq!(chat[0].role, "user");
        assert_eq!(chat[1].role, "assistant");
        assert_eq!(chat[2].role, "user");
    }

    #[test]
    fn normalize_forces_user_first_turn() {
        let turns = vec![ChatTurn {
            role: "assistant".into(),
            content: "ok".into(),
        }];
        let (chat, _) = normalize_turns(&turns);
        assert_eq!(chat[0].role, "user");
        assert_eq!(chat[1].role, "assistant");
    }

    #[test]
    fn from_env_disabled_when_backend_unset() {
        std::env::remove_var("AXIOM_BACKEND");
        assert!(ClaudeBackend::from_env().is_none());
    }
}
