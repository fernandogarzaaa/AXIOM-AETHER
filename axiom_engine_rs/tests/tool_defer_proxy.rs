//! Integration coverage for P1 (PSS): tool deferral in the `/v1/messages`
//! compression path.

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

/// `AXIOM_TOOL_DEFER` is a process-global env var read in the request path;
/// serialize the tests that set it (same rationale as cache_safety_proxy.rs).
fn env_lock() -> &'static AsyncMutex<()> {
    static LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| AsyncMutex::new(()))
}

struct EnvVarGuard(&'static str);
impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        std::env::remove_var(self.0);
    }
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
            "id": "msg_test", "type": "message", "role": "assistant",
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

fn ten_tools() -> Value {
    let mut tools = vec![json!({"name":"Read"}), json!({"name":"WebFetch"})];
    for i in 0..8 {
        tools.push(json!({"name": format!("ObscureTool{i}")}));
    }
    Value::Array(tools)
}

/// A request whose recent history invokes `WebFetch` (so it joins the working
/// set alongside the core tools); the 8 ObscureTool* are unused.
fn body_with_tools(session_id: &str) -> Value {
    json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": session_id,
        "tools": ten_tools(),
        "messages": [
            {"role":"assistant","content":[
                {"type":"tool_use","name":"WebFetch","id":"t1","input":{}}]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":"fetched"}]},
        ],
    })
}

#[tokio::test]
async fn defer_marks_unused_tools_when_enabled() {
    let _guard = env_lock().lock().await;
    std::env::set_var("AXIOM_TOOL_DEFER", "on");
    // This suite's newest turn is a small, clean tool_result -- exactly what
    // L-B (default ON since the 2026-07-16 flip) short-circuits. Pin it off so
    // the request actually reaches the mock upstream under test.
    std::env::set_var("AXIOM_LOCAL_TRIVIAL", "off");
    let _cleanup = EnvVarGuard("AXIOM_TOOL_DEFER");
    let _cleanup2 = EnvVarGuard("AXIOM_LOCAL_TRIVIAL");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let status = post_messages(&app, body_with_tools("tool-defer-on")).await;
    assert_eq!(status, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    let sent_tools = captured[0]["tools"].as_array().unwrap();
    // Count/order preserved: still 10 tools.
    assert_eq!(sent_tools.len(), 10);
    // The 8 ObscureTool* are deferred; Read (core) + WebFetch (recent) are not.
    let deferred = sent_tools
        .iter()
        .filter(|t| t.get("defer_loading") == Some(&json!(true)))
        .count();
    assert_eq!(deferred, 8, "8 unused tools deferred");
    let read = sent_tools.iter().find(|t| t["name"] == json!("Read")).unwrap();
    assert!(read.get("defer_loading").is_none(), "core tool not deferred");
    let wf = sent_tools.iter().find(|t| t["name"] == json!("WebFetch")).unwrap();
    assert!(wf.get("defer_loading").is_none(), "recently-used tool not deferred");
}

#[tokio::test]
async fn defer_output_is_byte_stable_across_two_turns() {
    let _guard = env_lock().lock().await;
    std::env::set_var("AXIOM_TOOL_DEFER", "on");
    std::env::set_var("AXIOM_LOCAL_TRIVIAL", "off"); // see note in the first test
    let _cleanup = EnvVarGuard("AXIOM_TOOL_DEFER");
    let _cleanup2 = EnvVarGuard("AXIOM_LOCAL_TRIVIAL");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let body = body_with_tools("tool-defer-stable");
    assert_eq!(post_messages(&app, body.clone()).await, StatusCode::OK);
    assert_eq!(post_messages(&app, body).await, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    let t1 = serde_json::to_string(&captured[0]["tools"]).unwrap();
    let t2 = serde_json::to_string(&captured[1]["tools"]).unwrap();
    assert_eq!(t1, t2, "identical input must yield byte-identical tools[] (cache-safe)");
}

#[tokio::test]
async fn defer_off_passes_tools_through_unchanged() {
    let _guard = env_lock().lock().await;
    // Default is ON since the 2026-07-16 flip -- opting out takes an explicit
    // "off" now.
    std::env::set_var("AXIOM_TOOL_DEFER", "off");
    std::env::set_var("AXIOM_LOCAL_TRIVIAL", "off"); // see note in the first test
    let _cleanup = EnvVarGuard("AXIOM_TOOL_DEFER");
    let _cleanup2 = EnvVarGuard("AXIOM_LOCAL_TRIVIAL");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let status = post_messages(&app, body_with_tools("tool-defer-off")).await;
    assert_eq!(status, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    let sent_tools = captured[0]["tools"].as_array().unwrap();
    assert_eq!(sent_tools.len(), 10);
    assert!(
        sent_tools.iter().all(|t| t.get("defer_loading").is_none()),
        "flag off -> no tool is deferred"
    );
}
