//! Integration coverage for S6: keepalive ping shape against a mock
//! Anthropic upstream. Calls `send_ping` directly (bypassing the real
//! multi-minute timer/sleep loop, which is impractical to exercise in a
//! test) -- this is the same function the real timer calls once it decides
//! to ping, so the wire shape asserted here is exactly what production
//! sends.

use axiom_engine::anthropic_forwarder::{AnthropicForwarder, ClientAuth};
use axiom_engine::keepalive::{send_ping, HeldHeaders, PingOutcome};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct Capture {
    requests: Arc<Mutex<Vec<Value>>>,
    /// When `Some`, the handler returns this status/body instead of a
    /// normal 200 -- lets a test simulate 400 (max_tokens retry), 401, or
    /// another 4xx.
    respond_with: Arc<Mutex<Option<(u16, Value)>>>,
}

async fn start_mock_upstream(respond_with: Option<(u16, Value)>) -> (String, Capture) {
    async fn handler(State(capture): State<Capture>, Json(body): Json<Value>) -> Response {
        capture.requests.lock().unwrap().push(body);
        if let Some((status, resp_body)) = capture.respond_with.lock().unwrap().clone() {
            return (
                StatusCode::from_u16(status).unwrap(),
                Json(resp_body),
            )
                .into_response();
        }
        Json(json!({
            "id": "msg_ping",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 0,
                "cache_read_input_tokens": 80000,
                "cache_creation_input_tokens": 0,
                "output_tokens": 1
            }
        }))
        .into_response()
    }
    let capture = Capture::default();
    *capture.respond_with.lock().unwrap() = respond_with;
    let app = Router::new()
        .route("/v1/messages", post(handler))
        .with_state(capture.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), capture)
}

fn sample_last_request() -> Value {
    json!({
        "model": "claude-sonnet-5",
        "max_tokens": 512,
        "stream": true,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tool_choice": {"type": "auto"},
        "metadata": {"user_id": "u-1"},
        "session_id": "sess-keepalive",
        "system": "you are a helpful assistant",
        "tools": [{"name": "read_file"}],
        "messages": [
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "reply one",
                 "cache_control": {"type": "ephemeral"}}
            ]},
            {"role": "user", "content": "second turn, after the breakpoint"},
        ],
    })
}

fn fake_headers() -> HeldHeaders {
    HeldHeaders::from_client_auth(&ClientAuth {
        authorization: Some("Bearer test-oauth-token".to_string()),
        x_api_key: None,
        anthropic_version: Some("2023-06-01".to_string()),
        anthropic_beta: Some("oauth-2025-04-20".to_string()),
    })
}

#[tokio::test]
async fn ping_shape_has_max_tokens_zero_and_no_stream_thinking_tool_choice() {
    let (upstream, capture) = start_mock_upstream(None).await;
    let forwarder = AnthropicForwarder::new(None, Some(upstream));

    let result = send_ping(&forwarder, &fake_headers(), &sample_last_request(), 0).await;
    assert_eq!(result.outcome, PingOutcome::Sent);
    assert_eq!(result.max_tokens_used, 0);

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    let sent = &captured[0];
    assert_eq!(sent["max_tokens"], json!(0));
    assert!(sent.get("stream").is_none());
    assert!(sent.get("thinking").is_none());
    assert!(sent.get("tool_choice").is_none());
    assert!(sent.get("metadata").is_none());
    assert!(sent.get("session_id").is_none());

    // Frozen-prefix truncation: "second turn" (after the cache_control
    // breakpoint) must not appear; the placeholder message must.
    let sent_str = sent.to_string();
    assert!(!sent_str.contains("second turn"));
    let messages = sent["messages"].as_array().unwrap();
    assert_eq!(messages.last().unwrap(), &json!({"role": "user", "content": "."}));
}

#[tokio::test]
async fn ping_estimates_usd_saved_from_the_real_usage_response() {
    let (upstream, _capture) = start_mock_upstream(None).await;
    let forwarder = AnthropicForwarder::new(None, Some(upstream));

    let result = send_ping(&forwarder, &fake_headers(), &sample_last_request(), 0).await;
    assert_eq!(result.outcome, PingOutcome::Sent);
    // 80,000 cache-read tokens at the Sonnet price table: a real, positive
    // savings estimate must come back (the exact cents aren't asserted --
    // that's cost_ledger's own test surface -- only that this wiring
    // actually produces a number greater than zero).
    assert!(result.estimated_usd_saved > 0.0);
}

#[tokio::test]
async fn ping_retries_once_with_max_tokens_one_on_a_max_tokens_400() {
    let (upstream, capture) = start_mock_upstream(Some((
        400,
        json!({"type": "error", "error": {"type": "invalid_request_error",
               "message": "max_tokens: 0 is not supported for this model"}}),
    )))
    .await;
    let forwarder = AnthropicForwarder::new(None, Some(upstream));

    // The mock always returns the same 400 (it doesn't distinguish the
    // retry), so this proves the retry-with-1 attempt actually happens by
    // checking two requests were sent, the second with max_tokens: 1.
    let _ = send_ping(&forwarder, &fake_headers(), &sample_last_request(), 0).await;

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 2, "must retry exactly once on a max_tokens 400");
    assert_eq!(captured[0]["max_tokens"], json!(0));
    assert_eq!(captured[1]["max_tokens"], json!(1));
}

#[tokio::test]
async fn ping_does_not_retry_on_a_400_unrelated_to_max_tokens() {
    let (upstream, capture) = start_mock_upstream(Some((
        400,
        json!({"type": "error", "error": {"type": "invalid_request_error",
               "message": "messages: at least one message is required"}}),
    )))
    .await;
    let forwarder = AnthropicForwarder::new(None, Some(upstream));

    let result = send_ping(&forwarder, &fake_headers(), &sample_last_request(), 0).await;
    assert_eq!(result.outcome, PingOutcome::DisabledOtherError(400));

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 1, "must not retry a non-max_tokens 400");
}

#[tokio::test]
async fn ping_401_is_disabled_unauthorized_never_retried() {
    let (upstream, capture) = start_mock_upstream(Some((
        401,
        json!({"type": "error", "error": {"type": "authentication_error", "message": "invalid credential"}}),
    )))
    .await;
    let forwarder = AnthropicForwarder::new(None, Some(upstream));

    let result = send_ping(&forwarder, &fake_headers(), &sample_last_request(), 0).await;
    assert_eq!(result.outcome, PingOutcome::DisabledUnauthorized);

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 1, "a 401 must never be retried");
}

#[tokio::test]
async fn ping_other_4xx_disables_with_the_status_code() {
    let (upstream, _capture) = start_mock_upstream(Some((
        429,
        json!({"type": "error", "error": {"type": "rate_limit_error", "message": "rate limited"}}),
    )))
    .await;
    let forwarder = AnthropicForwarder::new(None, Some(upstream));

    let result = send_ping(&forwarder, &fake_headers(), &sample_last_request(), 0).await;
    assert_eq!(result.outcome, PingOutcome::DisabledOtherError(429));
}
