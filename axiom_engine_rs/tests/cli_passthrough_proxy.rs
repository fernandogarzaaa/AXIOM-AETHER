//! Integration coverage for the hybrid CLI integration (docs/CLI-INTEGRATION.md):
//! a vendor CLI keeps its own login and points its base URL at the Axiom
//! proxy, which compresses the traffic while holding **no credential of its
//! own**.
//!
//! Two properties, each proven against a capturing fake upstream that records
//! the exact headers and body it receives:
//!
//! (a) **Auth pass-through.** The client's own `Authorization` / `x-api-key` /
//!     `anthropic-version` / `anthropic-beta` (Anthropic shape) and
//!     `Authorization` / `OpenAI-Organization` / `OpenAI-Project` (OpenAI
//!     shapes) reach upstream byte-identical; the proxy adds no key of its
//!     own; and with `AXIOM_API_KEY` unset (the loopback default) no
//!     `X-Axiom-Key` is required.
//!
//! (b) **Cache safety.** Everything at or before a client `cache_control`
//!     breakpoint — the `system` prompt, `tools`, and every frozen leading
//!     message — is forwarded byte-identical, so the vendor's prompt cache
//!     keeps hitting.
//!
//! Harness copied from tests/cache_safety_proxy.rs (capturing upstream + tiny
//! CPU pipeline + `tower::ServiceExt::oneshot`); extended to capture headers.

use axiom_engine::anthropic_forwarder::AnthropicForwarder;
use axiom_engine::config::AxiomConfig;
use axiom_engine::context_compressor::CompressorConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::openai_forwarder::OpenAiForwarder;
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::extract::State;
use axum::http::{HeaderMap, Method, Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use candle_core::Device;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

/// One captured upstream request: lowercase header name → value, plus body.
#[derive(Clone, Debug)]
struct Captured {
    headers: BTreeMap<String, String>,
    body: Value,
}

#[derive(Clone, Default)]
struct Capture {
    requests: Arc<Mutex<Vec<Captured>>>,
}

impl Capture {
    fn take_single(&self) -> Captured {
        let reqs = self.requests.lock().unwrap();
        assert_eq!(reqs.len(), 1, "expected exactly one upstream call");
        reqs[0].clone()
    }
}

fn tiny_pipeline() -> InferencePipeline {
    let config = AxiomConfig {
        d_model: 16,
        n_layers: 1,
        vocab_size: 64,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    };
    InferencePipeline::new(config, Device::Cpu).expect("pipeline init")
}

fn header_map(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .map(|(k, v)| {
            (
                k.as_str().to_ascii_lowercase(),
                v.to_str().unwrap_or("<non-utf8>").to_string(),
            )
        })
        .collect()
}

/// Fake upstream speaking all three vendor shapes on one listener. Every
/// handler records the full header set and body so tests can assert on what
/// actually left the proxy.
async fn start_capturing_upstream() -> (String, Capture, tokio::task::JoinHandle<()>) {
    async fn messages(
        State(capture): State<Capture>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        capture.requests.lock().unwrap().push(Captured {
            headers: header_map(&headers),
            body,
        });
        Json(json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }))
    }
    async fn chat_completions(
        State(capture): State<Capture>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        capture.requests.lock().unwrap().push(Captured {
            headers: header_map(&headers),
            body,
        });
        Json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5.5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
        }))
    }
    async fn responses(
        State(capture): State<Capture>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        capture.requests.lock().unwrap().push(Captured {
            headers: header_map(&headers),
            body,
        });
        Json(json!({
            "id": "resp_test",
            "object": "response",
            "status": "completed",
            "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}]
        }))
    }
    let capture = Capture::default();
    let app = Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
        .with_state(capture.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), capture, task)
}

/// Proxy state exactly as `scripts/start_axiom_for_cli.sh` launches it: both
/// forwarders in auth-passthrough mode (`None` key), compression on, and no
/// `AXIOM_API_KEY` (so `state.api_key` stays `None` — the loopback default).
async fn build_state(upstream: String) -> AppState {
    let pipeline = tokio::task::spawn_blocking(tiny_pipeline).await.unwrap();
    AppState::new(pipeline, "axiom-ttt-test".to_string())
        .with_anthropic_forwarder(Some(AnthropicForwarder::new(None, Some(upstream.clone()))))
        .with_openai_forwarder(Some(OpenAiForwarder::new(None, Some(upstream))))
        .with_compressor_config(CompressorConfig {
            // Low threshold so short test strings count as "heavy" and the
            // compression path (not a bypass) is what forwards the request.
            heavy_message_threshold_tokens: 5,
            recall_top_k: 8,
            enabled: true,
        })
}

async fn post_json(app: &Router, path: &str, headers: &[(&str, &str)], body: Value) -> StatusCode {
    let mut req = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header("content-type", "application/json");
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let req = req.body(Body::from(body.to_string())).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let _ = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    status
}

const HEAVY_TEXT: &str =
    "This paragraph is deliberately long padded content well past a five token \
     heavy threshold so the compression path engages and rewrites the mutable tail of the \
     conversation repeated words repeated words repeated words repeated words";

/// Claude Code with a Pro/Max subscription authenticates with an OAuth bearer
/// token plus an `anthropic-beta` flag — no `x-api-key` anywhere. These are
/// test strings shaped like the real headers, not credentials.
const OAUTH_BEARER: &str = "Bearer sk-ant-oat01-TEST-TOKEN-not-a-real-credential";
const OAUTH_BETA: &str = "oauth-2025-04-20,claude-code-20250219";
const ANTHROPIC_VERSION: &str = "2023-06-01";

// ---------------------------------------------------------------------------
// (a) auth pass-through
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_oauth_headers_reach_upstream_untouched_and_no_proxy_key_is_added() {
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    assert!(
        state.api_key.is_none(),
        "loopback default: no AXIOM_API_KEY configured"
    );
    let app = create_router(state);

    let status = post_json(
        &app,
        "/v1/messages",
        &[
            ("authorization", OAUTH_BEARER),
            ("anthropic-beta", OAUTH_BETA),
            ("anthropic-version", ANTHROPIC_VERSION),
            // Note: deliberately NO X-Axiom-Key header.
        ],
        json!({
            "model": "claude-sonnet-5",
            "max_tokens": 16,
            "session_id": "cli-passthrough-oauth",
            "messages": [
                {"role": "user", "content": HEAVY_TEXT},
                {"role": "user", "content": "summarize"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "no Axiom key required on loopback");

    let sent = capture.take_single();
    assert_eq!(
        sent.headers.get("authorization").map(String::as_str),
        Some(OAUTH_BEARER)
    );
    assert_eq!(
        sent.headers.get("anthropic-beta").map(String::as_str),
        Some(OAUTH_BETA)
    );
    assert_eq!(
        sent.headers.get("anthropic-version").map(String::as_str),
        Some(ANTHROPIC_VERSION)
    );
    assert!(
        !sent.headers.contains_key("x-api-key"),
        "proxy must not add an x-api-key next to the client's bearer token: {:?}",
        sent.headers
    );
    assert!(
        !sent.headers.contains_key("x-axiom-key"),
        "Axiom's own data-plane header must never leak upstream"
    );
}

#[tokio::test]
async fn anthropic_api_key_client_header_is_relayed_verbatim() {
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let app = create_router(build_state(upstream).await);

    let client_key = "sk-ant-api03-TEST-CLIENT-KEY-not-real";
    let status = post_json(
        &app,
        "/v1/messages",
        &[
            ("x-api-key", client_key),
            ("anthropic-version", ANTHROPIC_VERSION),
        ],
        json!({
            "model": "claude-sonnet-5",
            "max_tokens": 16,
            "session_id": "cli-passthrough-apikey",
            "messages": [{"role": "user", "content": HEAVY_TEXT}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let sent = capture.take_single();
    assert_eq!(
        sent.headers.get("x-api-key").map(String::as_str),
        Some(client_key)
    );
    assert!(!sent.headers.contains_key("authorization"));
}

#[tokio::test]
async fn anthropic_request_without_any_client_credential_fails_closed() {
    // The proxy holds no key: a client that sends none gets an upstream
    // error from the proxy itself, and nothing is forwarded — the proxy never
    // invents a credential.
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let app = create_router(build_state(upstream).await);
    let status = post_json(
        &app,
        "/v1/messages",
        &[],
        json!({
            "model": "claude-sonnet-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": HEAVY_TEXT}]
        }),
    )
    .await;
    assert!(
        status.is_client_error() || status.is_server_error(),
        "expected an error status, got {status}"
    );
    assert!(capture.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn openai_chat_completions_relays_bearer_and_organization_headers() {
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let app = create_router(build_state(upstream).await);

    let bearer = "Bearer sk-TEST-OPENAI-KEY-not-real";
    let status = post_json(
        &app,
        "/v1/chat/completions",
        &[
            ("authorization", bearer),
            ("OpenAI-Organization", "org-test-123"),
            ("OpenAI-Project", "proj_test_456"),
        ],
        json!({
            "model": "gpt-5.5",
            "session_id": "cli-passthrough-chat",
            "messages": [
                {"role": "user", "content": HEAVY_TEXT},
                {"role": "user", "content": "summarize"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let sent = capture.take_single();
    assert_eq!(
        sent.headers.get("authorization").map(String::as_str),
        Some(bearer)
    );
    assert_eq!(
        sent.headers.get("openai-organization").map(String::as_str),
        Some("org-test-123")
    );
    assert_eq!(
        sent.headers.get("openai-project").map(String::as_str),
        Some("proj_test_456")
    );
    assert!(!sent.headers.contains_key("x-axiom-key"));
}

#[tokio::test]
async fn openai_responses_relays_bearer_and_organization_headers() {
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let app = create_router(build_state(upstream).await);

    let bearer = "Bearer sk-TEST-OPENAI-KEY-not-real";
    let status = post_json(
        &app,
        "/v1/responses",
        &[
            ("authorization", bearer),
            ("OpenAI-Organization", "org-test-123"),
            ("OpenAI-Project", "proj_test_456"),
        ],
        json!({
            "model": "gpt-5.5",
            "instructions": "Be concise.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": HEAVY_TEXT}]}],
            "stream": false,
            "store": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let sent = capture.take_single();
    assert_eq!(
        sent.headers.get("authorization").map(String::as_str),
        Some(bearer)
    );
    assert_eq!(
        sent.headers.get("openai-organization").map(String::as_str),
        Some("org-test-123")
    );
    assert_eq!(
        sent.headers.get("openai-project").map(String::as_str),
        Some("proj_test_456")
    );
}

#[tokio::test]
async fn axiom_key_is_only_required_when_the_operator_sets_one() {
    // Same request, two proxies: the loopback default accepts it without an
    // X-Axiom-Key; a proxy with AXIOM_API_KEY configured rejects it. This is
    // the behaviour the CLI launcher relies on (it never exports an Axiom key
    // into the CLI's shell).
    let (upstream, _capture, _task) = start_capturing_upstream().await;
    let body = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": HEAVY_TEXT}]
    });
    let headers = [
        ("authorization", OAUTH_BEARER),
        ("anthropic-beta", OAUTH_BETA),
    ];

    let open = create_router(build_state(upstream.clone()).await);
    assert_eq!(
        post_json(&open, "/v1/messages", &headers, body.clone()).await,
        StatusCode::OK
    );

    let guarded = create_router(
        build_state(upstream)
            .await
            .with_api_key(Some("operator-secret".into())),
    );
    assert_eq!(
        post_json(&guarded, "/v1/messages", &headers, body).await,
        StatusCode::UNAUTHORIZED
    );
}

// ---------------------------------------------------------------------------
// (b) cache safety
// ---------------------------------------------------------------------------

#[tokio::test]
async fn content_at_or_before_a_cache_control_breakpoint_is_forwarded_byte_identical() {
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let app = create_router(build_state(upstream).await);

    // Claude Code's real request shape: a cached system prompt, a cached tool
    // list, several leading turns ending in a `cache_control` breakpoint, then
    // a fresh heavy mutable tail.
    let system = json!([
        {"type": "text", "text": "You are Claude Code, a coding assistant.",
         "cache_control": {"type": "ephemeral"}}
    ]);
    let tools = json!([
        {"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}},
        {"name": "Edit", "description": "Edit a file", "input_schema": {"type": "object"},
         "cache_control": {"type": "ephemeral"}}
    ]);
    let frozen_messages = json!([
        {"role": "user", "content": "short user turn zero"},
        {"role": "assistant", "content": "short assistant reply one"},
        {"role": "user", "content": HEAVY_TEXT},
        {"role": "assistant", "content": [
            {"type": "text", "text": "short assistant reply three",
             "cache_control": {"type": "ephemeral"}}
        ]},
    ]);
    let mut messages = frozen_messages.as_array().unwrap().clone();
    messages.push(json!({"role": "user", "content": "short user turn four"}));
    messages.push(json!({"role": "user", "content": HEAVY_TEXT}));

    let status = post_json(
        &app,
        "/v1/messages",
        &[
            ("authorization", OAUTH_BEARER),
            ("anthropic-beta", OAUTH_BETA),
            ("anthropic-version", ANTHROPIC_VERSION),
        ],
        json!({
            "model": "claude-sonnet-5",
            "max_tokens": 16,
            "session_id": "cli-passthrough-cache-safety",
            "system": system,
            "tools": tools,
            "messages": messages,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let sent = capture.take_single();
    // Byte-identical: compare canonical serialisations of each frozen part.
    assert_eq!(
        serde_json::to_string(&sent.body["system"]).unwrap(),
        serde_json::to_string(&system).unwrap(),
        "system prompt (cached prefix) must be forwarded byte-identical"
    );
    assert_eq!(
        serde_json::to_string(&sent.body["tools"]).unwrap(),
        serde_json::to_string(&tools).unwrap(),
        "tool list (cached prefix) must be forwarded byte-identical"
    );
    let sent_messages = sent.body["messages"].as_array().unwrap();
    let frozen = frozen_messages.as_array().unwrap();
    assert!(
        sent_messages.len() >= frozen.len(),
        "frozen prefix must survive: sent {} messages, frozen prefix is {}",
        sent_messages.len(),
        frozen.len()
    );
    for (i, (sent_m, orig_m)) in sent_messages.iter().zip(frozen.iter()).enumerate() {
        assert_eq!(
            serde_json::to_string(sent_m).unwrap(),
            serde_json::to_string(orig_m).unwrap(),
            "messages[{i}] is at or before the cache_control breakpoint and must be byte-identical"
        );
    }
    // And the mutable tail WAS compressed (so this is the compression path,
    // not a passthrough bypass): the heavy text appears exactly once — the
    // frozen copy at index 2 — not a second time for the tail copy.
    let sent_json = serde_json::to_string(sent_messages).unwrap();
    assert_eq!(
        sent_json.matches(HEAVY_TEXT).count(),
        1,
        "mutable-tail heavy content should have been compressed away"
    );
}

#[tokio::test]
async fn openai_chat_cache_shaped_prefix_is_forwarded_byte_identical() {
    // OpenAI-style clients have no explicit breakpoint marker; their prompt
    // cache is an automatic exact-prefix match. The proxy's partitioning must
    // therefore leave the system message — the start of every cached prefix —
    // byte-identical, and only touch heavy content.
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let app = create_router(build_state(upstream).await);

    let system = json!({"role": "system", "content": "You are Codex, a coding agent."});
    let status = post_json(
        &app,
        "/v1/chat/completions",
        &[("authorization", "Bearer sk-TEST-OPENAI-KEY-not-real")],
        json!({
            "model": "gpt-5.5",
            "session_id": "cli-passthrough-chat-prefix",
            "messages": [
                system,
                {"role": "user", "content": HEAVY_TEXT},
                {"role": "user", "content": "summarize"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let sent = capture.take_single();
    let sent_messages = sent.body["messages"].as_array().unwrap();
    assert_eq!(
        serde_json::to_string(&sent_messages[0]).unwrap(),
        serde_json::to_string(&system).unwrap(),
        "system message (cached prefix start) must be byte-identical"
    );
}
