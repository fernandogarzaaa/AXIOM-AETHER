//! Integration coverage for S4: prefix-diet gating in the `/v1/messages`
//! compression path, and the `GET /v1/prefix-diet/report/:session_id`
//! debug endpoint.

use axiom_engine::anthropic_forwarder::AnthropicForwarder;
use axiom_engine::config::AxiomConfig;
use axiom_engine::context_compressor::CompressorConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::extract::State;
use axum::http::{Method, Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use candle_core::Device;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::Mutex as AsyncMutex;
use tower::ServiceExt;

/// Same rationale as `cache_safety_proxy.rs`: `AXIOM_PREFIX_DEDUP` is a
/// process-global env var read inside the proxy's request path, so tests
/// that set/unset it must not overlap with concurrent tests in this binary.
fn env_lock() -> &'static AsyncMutex<()> {
    static LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| AsyncMutex::new(()))
}

#[derive(Clone, Default)]
struct Capture {
    requests: Arc<Mutex<Vec<Value>>>,
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

async fn start_capturing_upstream() -> (String, Capture, tokio::task::JoinHandle<()>) {
    async fn handler(State(capture): State<Capture>, Json(body): Json<Value>) -> Json<Value> {
        capture.requests.lock().unwrap().push(body);
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
    let capture = Capture::default();
    let app = Router::new()
        .route("/v1/messages", post(handler))
        .with_state(capture.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), capture, task)
}

async fn build_state(upstream: String) -> AppState {
    let pipeline = tokio::task::spawn_blocking(tiny_pipeline).await.unwrap();
    AppState::new(pipeline, "axiom-ttt-test".to_string())
        .with_anthropic_forwarder(Some(AnthropicForwarder::new(None, Some(upstream))))
        .with_compressor_config(CompressorConfig {
            heavy_message_threshold_tokens: 5,
            recall_top_k: 8,
            enabled: true,
        })
}

async fn post_messages(app: &Router, body: Value) -> StatusCode {
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .header("x-api-key", "sk-ant-test-key")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let _ = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    status
}

async fn get_report(app: &Router, session_id: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(Method::GET)
        .uri(format!("/v1/prefix-diet/report/{session_id}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

/// A repeated system-prompt block, well over the 400-byte dedup threshold,
/// duplicated the way real Claude Code traffic sometimes duplicates
/// CLAUDE.md/rules content.
fn repeated_system_block() -> String {
    "Always create new objects, never mutate existing ones. ".repeat(8)
}

fn cache_bearing_body(session_id: &str, system: Value) -> Value {
    json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": session_id,
        "system": system,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
            ]},
        ],
    })
}

#[tokio::test]
async fn dedup_applies_when_enabled_and_request_uses_cache() {
    let _guard = env_lock().lock().await;
    std::env::set_var("AXIOM_PREFIX_DEDUP", "1");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let block = repeated_system_block();
    let system = json!(format!("{block}\n\nunique middle\n\n{block}"));
    let session_id = "prefix-diet-enabled";
    let status = post_messages(&app, cache_bearing_body(session_id, system)).await;
    std::env::remove_var("AXIOM_PREFIX_DEDUP");
    assert_eq!(status, StatusCode::OK);

    {
        let captured = capture.requests.lock().unwrap();
        let sent_system = captured[0]["system"].as_str().unwrap();
        assert_eq!(sent_system.matches(&block).count(), 1);
        assert!(sent_system.contains("[AXIOM-DEDUP:"));
    }

    let (report_status, report) = get_report(&app, session_id).await;
    assert_eq!(report_status, StatusCode::OK);
    assert_eq!(report["blocks_deduped"], json!(1));
    assert!(report["dedup_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn dedup_does_not_apply_when_flag_is_off() {
    let _guard = env_lock().lock().await;
    // Ensure the flag is unset (default off) for this test's duration.
    std::env::remove_var("AXIOM_PREFIX_DEDUP");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let block = repeated_system_block();
    let system = json!(format!("{block}\n\nunique middle\n\n{block}"));
    let session_id = "prefix-diet-disabled";
    let status = post_messages(&app, cache_bearing_body(session_id, system)).await;
    assert_eq!(status, StatusCode::OK);

    {
        let captured = capture.requests.lock().unwrap();
        let sent_system = captured[0]["system"].as_str().unwrap();
        assert_eq!(sent_system.matches(&block).count(), 2, "unchanged: both copies must survive");
        assert!(!sent_system.contains("[AXIOM-DEDUP:"));
    }

    let (report_status, _) = get_report(&app, session_id).await;
    assert_eq!(report_status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn dedup_does_not_apply_when_request_does_not_use_cache() {
    let _guard = env_lock().lock().await;
    std::env::set_var("AXIOM_PREFIX_DEDUP", "1");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let block = repeated_system_block();
    let system = json!(format!("{block}\n\nunique middle\n\n{block}"));
    // No cache_control anywhere -- request_uses_cache(body) is false.
    let body = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": "prefix-diet-no-cache",
        "system": system,
        "messages": [{"role": "user", "content": "hello"}],
    });
    let status = post_messages(&app, body).await;
    std::env::remove_var("AXIOM_PREFIX_DEDUP");
    assert_eq!(status, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    let sent_system = captured[0]["system"].as_str().unwrap();
    assert_eq!(sent_system.matches(&block).count(), 2, "no cache -> no dedup");
}
