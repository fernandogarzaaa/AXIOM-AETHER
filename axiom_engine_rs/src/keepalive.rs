//! S6 (CVM cost stack): actuarial keepalive pings.
//!
//! ~15% of inter-turn gaps exceed Anthropic's 5-minute cache TTL, forcing a
//! full cache re-write (1.25x on ~160K tokens ~ $0.60) where a cheap
//! cache-read ping (0.1x, ~$0.05) would have kept it warm. This module is
//! that ping. It is gated behind `AXIOM_KEEPALIVE=1` (default 0, NEVER
//! auto-flipped by any other step in this plan) because it autonomously
//! replays the client's own auth headers to Anthropic on a timer -- a
//! security decision only the operator can make.
//!
//! See docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S6.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use crate::anthropic_forwarder::{AnthropicForwarder, ClientAuth, ForwarderError};
use crate::belief::BetaBelief;

/// Anthropic's prompt-cache TTL (5 minutes, in seconds).
pub const CACHE_TTL_SECONDS: u64 = 300;

/// Holds a session's relay auth headers **in memory only, for the life of
/// the process**. Deliberately implements neither `Serialize` nor a
/// value-revealing `Debug` -- these are live credentials and must never be
/// logged, persisted, or accidentally printed. Compare to
/// `anthropic_forwarder::ClientAuth`, which is fine to `derive(Debug)`
/// because it only ever lives for the duration of a single request; this
/// type is the one held long-term by the keepalive timer, so it needs the
/// stronger guarantee.
#[derive(Clone)]
pub struct HeldHeaders {
    authorization: Option<String>,
    x_api_key: Option<String>,
    anthropic_version: Option<String>,
    anthropic_beta: Option<String>,
}

impl HeldHeaders {
    pub fn from_client_auth(auth: &ClientAuth) -> Self {
        Self {
            authorization: auth.authorization.clone(),
            x_api_key: auth.x_api_key.clone(),
            anthropic_version: auth.anthropic_version.clone(),
            anthropic_beta: auth.anthropic_beta.clone(),
        }
    }

    fn to_client_auth(&self) -> ClientAuth {
        ClientAuth {
            authorization: self.authorization.clone(),
            x_api_key: self.x_api_key.clone(),
            anthropic_version: self.anthropic_version.clone(),
            anthropic_beta: self.anthropic_beta.clone(),
        }
    }
}

/// Redacted -- secret fields never appear in their actual value, even in
/// debug output. Non-secret fields (protocol version / beta flag) are shown
/// as-is since they carry no credential material.
impl std::fmt::Debug for HeldHeaders {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeldHeaders")
            .field(
                "authorization",
                &self.authorization.as_ref().map(|_| "<redacted>"),
            )
            .field("x_api_key", &self.x_api_key.as_ref().map(|_| "<redacted>"))
            .field("anthropic_version", &self.anthropic_version)
            .field("anthropic_beta", &self.anthropic_beta)
            .finish()
    }
}

/// Number of pings a session will attempt over `horizon_seconds`, spaced
/// roughly one per cache TTL.
pub fn pings_planned(horizon_seconds: u64) -> usize {
    ((horizon_seconds as f64) / (CACHE_TTL_SECONDS as f64)).ceil() as usize
}

/// The actuarial gate.
///
/// Derivation: let `P` be the cached prefix's token count (it cancels out
/// of the comparison below, so it never actually appears in the
/// implementation). If we do NOT ping and a follow-up request arrives after
/// the cache has already expired, that request pays a full cache-write
/// re-price of `1.25 * P`. If we DO ping, each of the `pings_remaining`
/// pings we'd still send pays a guaranteed cache-read price of `0.1 * P`
/// (refreshing the TTL), regardless of whether a follow-up ever shows up.
/// `belief.mean()` is this session's current estimate of the probability a
/// follow-up request lands within the keepalive horizon. Ping only while
/// the expected saving from staying warm exceeds the guaranteed cost of the
/// remaining pings:
///
/// ```text
/// belief.mean() * 1.25 * P > 0.1 * pings_remaining * P
///   =>  belief.mean() * 1.25 > 0.1 * pings_remaining      (P cancels)
/// ```
pub fn should_ping(belief: &BetaBelief, pings_remaining: usize) -> bool {
    belief.mean() * 1.25 > 0.1 * pings_remaining as f32
}

/// Build the keepalive ping body from the last real payload actually sent
/// upstream for this session, per the blueprint's exact recipe:
/// - keep `model`, `tools`, `system` untouched
/// - truncate `messages` to the frozen prefix (everything at or before the
///   last `cache_control` breakpoint -- reusing S1's exact definition of
///   that boundary), dropping everything after it
/// - append one placeholder user message `{"role":"user","content":"."}`
/// - remove `stream`, `thinking`, `tool_choice`, `metadata`, `session_id`
/// - set `max_tokens` to `max_tokens_override` (0, or 1 on retry)
pub fn build_ping_body(last_request: &Value, max_tokens_override: u64) -> Value {
    let mut ping = last_request.clone();
    if let Some(obj) = ping.as_object_mut() {
        obj.remove("stream");
        obj.remove("thinking");
        obj.remove("tool_choice");
        obj.remove("metadata");
        obj.remove("session_id");
    }

    let messages = last_request
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let frozen_len = crate::cache_safety::frozen_prefix_len(last_request, &messages);
    let mut truncated: Vec<Value> = messages[..frozen_len.min(messages.len())].to_vec();
    truncated.push(serde_json::json!({"role": "user", "content": "."}));
    ping["messages"] = Value::Array(truncated);
    ping["max_tokens"] = Value::from(max_tokens_override);
    ping
}

/// What happened when a single ping attempt was made.
#[derive(Debug, Clone, PartialEq)]
pub enum PingOutcome {
    Sent,
    /// The API rejected the credential outright (anti-replay OAuth or a
    /// revoked/expired token). Never retried -- would just spam an auth
    /// endpoint.
    DisabledUnauthorized,
    /// Some other 4xx. Disable keepalive for this session and stop.
    DisabledOtherError(u16),
    /// Transient network/decode failure -- worth retrying next interval,
    /// not a reason to disable the session.
    NetworkError(String),
}

/// Result of one `send_ping` call.
#[derive(Debug, Clone, PartialEq)]
pub struct PingResult {
    pub outcome: PingOutcome,
    /// The `max_tokens` value that actually worked (carries the session's
    /// remembered override forward for the next ping).
    pub max_tokens_used: u64,
    /// Counterfactual $ saved by this ping (a cache-read re-price instead
    /// of a full cache-write re-price), derived from the ping response's
    /// own real `usage` block when available. Always an *estimate* -- the
    /// counterfactual is "what a full rewrite next turn would have cost",
    /// which never actually happens if the ping worked.
    pub estimated_usd_saved: f64,
}

/// Send one keepalive ping. Handles the documented `max_tokens` ambiguity:
/// tries `max_tokens_override` first; if the API 400s specifically about
/// `max_tokens`, retries once with `max_tokens: 1` (never retries any other
/// failure).
pub async fn send_ping(
    forwarder: &AnthropicForwarder,
    headers: &HeldHeaders,
    last_request: &Value,
    max_tokens_override: u64,
) -> PingResult {
    let auth = headers.to_client_auth();
    let model = last_request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let body = build_ping_body(last_request, max_tokens_override);
    match forwarder.forward_messages_json(&body, &auth).await {
        Ok(resp) => estimate_and_wrap(&resp, &model, max_tokens_override),
        Err(ForwarderError::Upstream { status: 400, body: err_body })
            if max_tokens_override == 0 && err_body.to_lowercase().contains("max_tokens") =>
        {
            let retry_body = build_ping_body(last_request, 1);
            match forwarder.forward_messages_json(&retry_body, &auth).await {
                Ok(resp) => estimate_and_wrap(&resp, &model, 1),
                Err(e) => outcome_from_error(e, max_tokens_override),
            }
        }
        Err(e) => outcome_from_error(e, max_tokens_override),
    }
}

fn outcome_from_error(e: ForwarderError, max_tokens_used: u64) -> PingResult {
    let outcome = match e {
        ForwarderError::Upstream { status: 401, .. } => PingOutcome::DisabledUnauthorized,
        ForwarderError::Upstream { status, .. } => PingOutcome::DisabledOtherError(status),
        other => PingOutcome::NetworkError(other.to_string()),
    };
    PingResult {
        outcome,
        max_tokens_used,
        estimated_usd_saved: 0.0,
    }
}

fn estimate_and_wrap(resp: &Value, model: &str, max_tokens_used: u64) -> PingResult {
    let response_model = resp
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(model);
    let estimated_usd_saved = resp
        .get("usage")
        .and_then(|usage| crate::cost_ledger::turn_cost(response_model, usage))
        .map(|tc| {
            let (prices, _) = crate::cost_ledger::PriceTable::for_model(response_model);
            (tc.uncached_equivalent_usd(&prices) - tc.usd).max(0.0)
        })
        .unwrap_or(0.0);
    PingResult {
        outcome: PingOutcome::Sent,
        max_tokens_used,
        estimated_usd_saved,
    }
}

/// Per-session keepalive bookkeeping.
struct SessionKeepalive {
    headers: HeldHeaders,
    last_request: Value,
    belief: BetaBelief,
    pings_sent: usize,
    max_tokens_override: u64,
    disabled: bool,
    timer: Option<tokio::task::JoinHandle<()>>,
}

/// Owns all sessions' keepalive state and (when enabled) the background
/// timers that ping them. A no-op (spawns nothing, stores nothing) unless
/// constructed with `enabled: true` -- see [`KeepaliveManager::from_env`].
#[derive(Clone)]
pub struct KeepaliveManager {
    enabled: bool,
    horizon_seconds: u64,
    sessions: Arc<Mutex<HashMap<String, SessionKeepalive>>>,
    awareness: crate::session_awareness::AwarenessStore,
}

impl KeepaliveManager {
    /// Always-off manager: the safe default for any `AppState` that doesn't
    /// explicitly opt in via `from_env`/`with_keepalive_manager`.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            horizon_seconds: 0,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            awareness: crate::session_awareness::AwarenessStore::default(),
        }
    }

    /// Env-driven construction. `AXIOM_KEEPALIVE=1` opts in (default: off);
    /// `AXIOM_KEEPALIVE_HORIZON_S` overrides the ping horizon (default
    /// 1800s = 30 min). Prints the required security boot banner when
    /// enabled -- never silent about replaying the user's own credentials.
    /// `awareness` is the same store `AppState.awareness` uses, so
    /// successful pings feed S0's cost ledger for the session they warmed.
    pub fn from_env(awareness: crate::session_awareness::AwarenessStore) -> Self {
        let enabled = std::env::var("AXIOM_KEEPALIVE").as_deref() == Ok("1");
        let horizon_seconds = std::env::var("AXIOM_KEEPALIVE_HORIZON_S")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1800);
        if enabled {
            println!(
                "[axiom] keepalive ON — re-uses your API credentials for cost-saving cache \
                 pings (AXIOM_KEEPALIVE=1, horizon={horizon_seconds}s)"
            );
        }
        Self {
            enabled,
            horizon_seconds,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            awareness,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Number of sessions currently tracked. `0` always when disabled --
    /// the "disabled ⇒ zero timers" contract.
    pub fn active_session_count(&self) -> usize {
        self.sessions.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// Record a real (non-ping) request for `session_id`. A no-op unless
    /// the manager is enabled -- `record_activity` on a disabled manager
    /// never stores state and never spawns a timer.
    ///
    /// If this session already had a ping cycle underway (`pings_sent >
    /// 0`), a real request arriving is genuine evidence the gap got
    /// bridged, reinforcing the belief; a real request arriving before any
    /// ping was ever due (ordinary chat cadence, no gap) carries no signal
    /// either way and does not update the belief.
    pub fn record_activity(
        &self,
        session_id: &str,
        headers: HeldHeaders,
        last_request: Value,
        forwarder: AnthropicForwarder,
    ) {
        if !self.enabled {
            return;
        }
        let mut belief = BetaBelief::default();
        if let Ok(mut map) = self.sessions.lock() {
            if let Some(prior) = map.remove(session_id) {
                if let Some(task) = prior.timer {
                    task.abort();
                }
                belief = prior.belief;
                if prior.pings_sent > 0 {
                    belief.reinforce();
                }
            }
            map.insert(
                session_id.to_string(),
                SessionKeepalive {
                    headers,
                    last_request,
                    belief,
                    pings_sent: 0,
                    max_tokens_override: 0,
                    disabled: false,
                    timer: None,
                },
            );
        }
        self.spawn_timer(session_id.to_string(), forwarder);
    }

    fn spawn_timer(&self, session_id: String, forwarder: AnthropicForwarder) {
        let sessions = self.sessions.clone();
        let horizon = self.horizon_seconds;
        let awareness = self.awareness.clone();
        let task_session_id = session_id.clone();
        let handle = tokio::spawn(async move {
            let session_id = task_session_id;
            let planned = pings_planned(horizon);
            let sleep_secs = CACHE_TTL_SECONDS.saturating_sub(30);
            for _ in 0..planned {
                tokio::time::sleep(Duration::from_secs(sleep_secs)).await;

                let snapshot = {
                    let Ok(map) = sessions.lock() else { return };
                    match map.get(&session_id) {
                        Some(s) if !s.disabled => Some((
                            s.belief,
                            s.pings_sent,
                            s.max_tokens_override,
                            s.headers.clone(),
                            s.last_request.clone(),
                        )),
                        _ => None,
                    }
                };
                let Some((belief, pings_sent, max_tokens, headers, last_request)) = snapshot
                else {
                    return;
                };
                let remaining = planned.saturating_sub(pings_sent);
                if remaining == 0 || !should_ping(&belief, remaining) {
                    return;
                }

                let result = send_ping(&forwarder, &headers, &last_request, max_tokens).await;
                let Ok(mut map) = sessions.lock() else { return };
                let Some(s) = map.get_mut(&session_id) else {
                    return;
                };
                match result.outcome {
                    PingOutcome::Sent => {
                        s.pings_sent += 1;
                        s.max_tokens_override = result.max_tokens_used;
                        awareness
                            .get_or_create(&session_id)
                            .record_keepalive_ping(result.estimated_usd_saved);
                    }
                    PingOutcome::DisabledUnauthorized => {
                        s.disabled = true;
                        eprintln!(
                            "[axiom-keepalive] session={session_id} got 401 (anti-replay \
                             OAuth) — disabling keepalive for this session"
                        );
                        return;
                    }
                    PingOutcome::DisabledOtherError(status) => {
                        s.disabled = true;
                        eprintln!(
                            "[axiom-keepalive] session={session_id} ping failed with {status} \
                             — disabling keepalive for this session"
                        );
                        return;
                    }
                    PingOutcome::NetworkError(ref e) => {
                        eprintln!(
                            "[axiom-keepalive] session={session_id} ping network error \
                             (will retry next interval): {e}"
                        );
                    }
                }
            }
            // Horizon fully elapsed with no intervening real request --
            // genuine evidence this session actually ended.
            if let Ok(mut map) = sessions.lock() {
                if let Some(s) = map.get_mut(&session_id) {
                    s.belief.penalize();
                }
            }
        });
        if let Ok(mut map) = self.sessions.lock() {
            if let Some(s) = map.get_mut(&session_id) {
                s.timer = Some(handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- should_ping / pings_planned -------------------------------------

    #[test]
    fn pings_planned_rounds_up_to_full_ttl_intervals() {
        assert_eq!(pings_planned(1800), 6); // 1800 / 300 = 6 exactly
        assert_eq!(pings_planned(1801), 7); // rounds up
        assert_eq!(pings_planned(300), 1);
        assert_eq!(pings_planned(0), 0);
    }

    #[test]
    fn should_ping_true_under_high_confidence_low_remaining() {
        let mut belief = BetaBelief::uniform();
        for _ in 0..10 {
            belief.reinforce();
        }
        // mean ~ 11/12 ~ 0.917; 0.917 * 1.25 ~ 1.146 > 0.1 * 1 = 0.1
        assert!(should_ping(&belief, 1));
    }

    #[test]
    fn should_ping_false_under_low_confidence_high_remaining() {
        let mut belief = BetaBelief::uniform();
        for _ in 0..10 {
            belief.penalize();
        }
        // mean ~ 1/12 ~ 0.083; 0.083 * 1.25 ~ 0.104 < 0.1 * 6 = 0.6
        assert!(!should_ping(&belief, 6));
    }

    #[test]
    fn should_ping_uniform_prior_favors_pinging_when_few_remain() {
        let belief = BetaBelief::uniform(); // mean 0.5
        // 0.5 * 1.25 = 0.625 > 0.1 * 1 = 0.1
        assert!(should_ping(&belief, 1));
        // 0.5 * 1.25 = 0.625 < 0.1 * 10 = 1.0
        assert!(!should_ping(&belief, 10));
    }

    #[test]
    fn belief_reinforce_and_penalize_move_the_mean_as_expected() {
        let mut belief = BetaBelief::uniform();
        assert_eq!(belief.mean(), 0.5);
        belief.reinforce();
        assert!(belief.mean() > 0.5);
        let mut belief2 = BetaBelief::uniform();
        belief2.penalize();
        assert!(belief2.mean() < 0.5);
    }

    // ---- build_ping_body ---------------------------------------------------

    #[test]
    fn build_ping_body_truncates_to_frozen_prefix_and_appends_placeholder() {
        let last_request = json!({
            "model": "claude-sonnet-5",
            "max_tokens": 512,
            "stream": true,
            "session_id": "sess-1",
            "system": "you are helpful",
            "tools": [{"name": "read_file"}],
            "messages": [
                {"role": "user", "content": "turn zero"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "turn one",
                     "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": "turn two, after the breakpoint"},
            ],
        });
        let ping = build_ping_body(&last_request, 0);
        assert_eq!(ping["model"], json!("claude-sonnet-5"));
        assert_eq!(ping["system"], json!("you are helpful"));
        assert_eq!(ping["tools"], json!([{"name": "read_file"}]));
        assert_eq!(ping["max_tokens"], json!(0));
        assert!(ping.get("stream").is_none());
        assert!(ping.get("session_id").is_none());

        let messages = ping["messages"].as_array().unwrap();
        // frozen prefix (indices 0..=1) + placeholder = 3 messages; "turn
        // two" (after the breakpoint) must be dropped.
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["content"], json!("turn zero"));
        assert_eq!(messages[2], json!({"role": "user", "content": "."}));
        assert!(!ping.to_string().contains("turn two"));
    }

    #[test]
    fn build_ping_body_strips_stream_thinking_tool_choice_metadata() {
        let last_request = json!({
            "model": "claude-sonnet-5",
            "max_tokens": 512,
            "stream": true,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tool_choice": {"type": "auto"},
            "metadata": {"user_id": "u-1"},
            "messages": [{"role": "user", "content": "hi"}],
        });
        let ping = build_ping_body(&last_request, 1);
        assert!(ping.get("stream").is_none());
        assert!(ping.get("thinking").is_none());
        assert!(ping.get("tool_choice").is_none());
        assert!(ping.get("metadata").is_none());
        assert_eq!(ping["max_tokens"], json!(1));
    }

    // ---- HeldHeaders redaction ----------------------------------------------

    #[test]
    fn held_headers_debug_never_reveals_the_secret_values() {
        let auth = ClientAuth {
            authorization: Some("Bearer super-secret-token".to_string()),
            x_api_key: Some("sk-ant-super-secret-key".to_string()),
            anthropic_version: Some("2023-06-01".to_string()),
            anthropic_beta: None,
        };
        let held = HeldHeaders::from_client_auth(&auth);
        let debug_output = format!("{held:?}");
        assert!(!debug_output.contains("super-secret-token"));
        assert!(!debug_output.contains("sk-ant-super-secret-key"));
        assert!(debug_output.contains("<redacted>"));
    }

    /// Source-level guard (the blueprint's own "(grep test)"): `HeldHeaders`
    /// must not derive `Debug` (which would print real values) and this
    /// file must never mention `Serialize` at all.
    #[test]
    fn held_headers_source_has_no_derived_debug_or_any_serialize() {
        // Scan only the non-test portion of the file -- the test module
        // itself legitimately discusses "Serialize" in prose (this very
        // assertion message, for instance).
        let full_source = include_str!("keepalive.rs");
        let test_mod_pos = full_source
            .find("#[cfg(test)]")
            .expect("this file must have a #[cfg(test)] module");
        let source = &full_source[..test_mod_pos];

        let struct_pos = source
            .find("pub struct HeldHeaders")
            .expect("HeldHeaders struct must exist");
        let preceding = source[..struct_pos].trim_end();
        assert!(
            preceding.ends_with("#[derive(Clone)]"),
            "HeldHeaders's only derive must be #[derive(Clone)] (no Debug -- would leak secrets)"
        );
        for needle in ["derive(Serialize", "use serde::Serialize", "impl Serialize for", "Serializable for"] {
            assert!(
                !source.contains(needle),
                "keepalive.rs's production code must never import, derive, or implement \
                 Serialize anywhere (HeldHeaders credentials must never be serializable); \
                 found forbidden pattern {needle:?}"
            );
        }
    }

    // ---- KeepaliveManager: disabled state -----------------------------------

    #[test]
    fn disabled_manager_spawns_no_timers_and_stores_no_sessions() {
        let manager = KeepaliveManager::disabled();
        assert!(!manager.is_enabled());
        let auth = ClientAuth::default();
        let forwarder = AnthropicForwarder::new(None, Some("http://127.0.0.1:1".to_string()));
        manager.record_activity(
            "some-session",
            HeldHeaders::from_client_auth(&auth),
            json!({"model": "claude-sonnet-5", "messages": []}),
            forwarder,
        );
        assert_eq!(manager.active_session_count(), 0);
    }

    #[test]
    fn from_env_without_the_flag_set_is_disabled() {
        // Guard against test pollution: only assert when the var is
        // genuinely absent (this crate's test binaries run in parallel and
        // no other test in this file touches this var).
        if std::env::var("AXIOM_KEEPALIVE").is_err() {
            let manager = KeepaliveManager::from_env(crate::session_awareness::AwarenessStore::default());
            assert!(!manager.is_enabled());
        }
    }
}
