//! Integration coverage for P3 (PSS): L-B local trivial-turn short-circuit in
//! the `/v1/messages` compression path.

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

/// `AXIOM_LOCAL_TRIVIAL` / `AXIOM_DRIFT_THRESHOLD` are process-global env vars
/// read in the request path; serialize the tests that set them.
fn env_lock() -> &'static AsyncMutex<()> {
    static LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| AsyncMutex::new(()))
}

/// Restores an env var to its prior value (or unset) on drop, so a test that
/// sets `AXIOM_DRIFT_THRESHOLD` / `AXIOM_LOCAL_TRIVIAL` cannot leak that value
/// into a later test or the ambient environment.
struct EnvVarGuard {
    key: &'static str,
    prior: Option<String>,
}
impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prior = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key, prior }
    }
    fn unset(key: &'static str) -> Self {
        let prior = std::env::var(key).ok();
        std::env::remove_var(key);
        Self { key, prior }
    }
}
impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.prior {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
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
            "id": "msg_upstream", "type": "message", "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "from upstream"}],
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
            heavy_message_threshold_tokens: 100_000,
            recall_top_k: 8,
            enabled: true,
        })
}

async fn post_messages(app: &Router, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .header("x-api-key", "sk-ant-test-key")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let val = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, val)
}

/// Like `post_messages` but returns the raw `(status, content_type, body_text)`
/// so an SSE (`text/event-stream`) response can be inspected without JSON parse.
async fn post_messages_raw(app: &Router, body: Value) -> (StatusCode, String, String) {
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .header("x-api-key", "sk-ant-test-key")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, ct, String::from_utf8_lossy(&bytes).to_string())
}

/// A newest turn that is a clean, small `tool_result` (a genuine mechanical
/// ack). Multi-token so `mean_surprisal` returns a finite value.
fn clean_mechanical(session_id: &str) -> Value {
    json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": session_id,
        "messages": [
            {"role":"assistant","content":[
                {"type":"tool_use","name":"Bash","id":"t1","input":{}}]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":"ok, exit 0 done"}]},
        ],
    })
}

/// Same shape but the `tool_result` carries an error signature -- must never be
/// short-circuited.
fn error_bearing(session_id: &str) -> Value {
    json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": session_id,
        "messages": [
            {"role":"assistant","content":[
                {"type":"tool_use","name":"Bash","id":"t1","input":{}}]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":"Error: build failed"}]},
        ],
    })
}

#[tokio::test]
async fn clean_mechanical_turn_is_answered_locally_without_upstream() {
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "on");
    // Huge gate: any finite surprisal passes, so triviality reduces to the
    // deterministic structural checks (tool_result-only, error-free, small).
    let _c2 = EnvVarGuard::set("AXIOM_DRIFT_THRESHOLD", "1000000");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let (status, body) = post_messages(&app, clean_mechanical("lb-clean")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model"], json!("axiom-local"), "answered locally");
    assert_eq!(body["stop_reason"], json!("end_turn"));
    assert_eq!(
        capture.requests.lock().unwrap().len(),
        0,
        "a trivial turn must never reach upstream"
    );
}

#[tokio::test]
async fn streaming_trivial_turn_gets_a_local_sse_stream() {
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "on");
    let _c2 = EnvVarGuard::set("AXIOM_DRIFT_THRESHOLD", "1000000");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let mut body = clean_mechanical("lb-stream");
    body["stream"] = json!(true);
    let (status, content_type, text) = post_messages_raw(&app, body).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type.contains("text/event-stream"),
        "a streaming client must get SSE, not JSON (got {content_type})"
    );
    assert!(text.contains("event: message_start"), "valid SSE event sequence");
    assert!(text.contains("event: message_stop"));
    assert!(text.contains("axiom-local"), "answered locally");
    assert_eq!(
        capture.requests.lock().unwrap().len(),
        0,
        "a streaming trivial turn must never reach upstream"
    );
}

#[tokio::test]
async fn error_bearing_turn_is_forwarded_upstream() {
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "on");
    let _c2 = EnvVarGuard::set("AXIOM_DRIFT_THRESHOLD", "1000000");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let (status, body) = post_messages(&app, error_bearing("lb-error")).await;
    assert_eq!(status, StatusCode::OK);
    assert_ne!(body["model"], json!("axiom-local"), "error turn is not local");
    assert_eq!(
        capture.requests.lock().unwrap().len(),
        1,
        "an error-bearing turn must be forwarded upstream"
    );
}

#[tokio::test]
async fn flag_off_always_forwards() {
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::unset("AXIOM_LOCAL_TRIVIAL"); // default off

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let (status, body) = post_messages(&app, clean_mechanical("lb-off")).await;
    assert_eq!(status, StatusCode::OK);
    assert_ne!(body["model"], json!("axiom-local"), "flag off -> never local");
    assert_eq!(
        capture.requests.lock().unwrap().len(),
        1,
        "flag off -> even a trivial turn is forwarded"
    );
}
