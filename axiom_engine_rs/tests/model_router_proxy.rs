//! Integration coverage for P4 (PSS): R1 high-tier-gated model routing in the
//! `/v1/messages` compression path.

use axiom_engine::anthropic_forwarder::AnthropicForwarder;
use axiom_engine::config::AxiomConfig;
use axiom_engine::context_compressor::CompressorConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::extract::State;
use axum::http::{Method, Request, StatusCode};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use candle_core::Device;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::Mutex as AsyncMutex;
use tower::ServiceExt;

fn env_lock() -> &'static AsyncMutex<()> {
    static LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| AsyncMutex::new(()))
}

/// Saves/restores the prior env value on drop so a test cannot leak
/// `AXIOM_MODEL_ROUTE` into a later test or the ambient environment.
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
    /// When true, the mock returns 400 for a request whose model is Haiku (the
    /// routed target), forcing the proxy's fallback-to-original retry.
    reject_haiku: bool,
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

async fn start_upstream(reject_haiku: bool) -> (String, Capture, tokio::task::JoinHandle<()>) {
    async fn handler(State(cap): State<Capture>, Json(body): Json<Value>) -> axum::response::Response {
        let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();
        cap.requests.lock().unwrap().push(body);
        if cap.reject_haiku && model == "claude-haiku-4-5" {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"type":"error","error":{"type":"invalid_request_error","message":"nope"}})),
            )
                .into_response();
        }
        Json(json!({
            "id": "msg_up", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }))
        .into_response()
    }
    let capture = Capture { reject_haiku, ..Default::default() };
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

/// A mechanical follow-up turn on `model`: the newest turn is a clean, error-free
/// `tool_result` (a pure feedback turn), which is what R1 downgrades.
fn mechanical_turn(model: &str, session_id: &str) -> Value {
    json!({
        "model": model,
        "max_tokens": 16,
        "session_id": session_id,
        "messages": [
            {"role":"assistant","content":[
                {"type":"tool_use","name":"Bash","id":"t1","input":{}}]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":"ok done"}]},
        ],
    })
}

#[tokio::test]
async fn auto_mode_downgrades_a_mechanical_opus_turn_to_haiku() {
    let _guard = env_lock().lock().await;
    let _c = EnvVarGuard::set("AXIOM_MODEL_ROUTE", "auto");
    // mechanical_turn's newest message is a small, clean tool_result -- L-B
    // (default ON since the 2026-07-16 flip) would answer it locally before
    // routing ever runs. Pin it off so the routed request reaches the mock.
    let _c2 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "off");

    let (upstream, capture, _task) = start_upstream(false).await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    assert_eq!(
        post_messages(&app, mechanical_turn("claude-opus-4-8", "r-opus")).await,
        StatusCode::OK
    );
    let reqs = capture.requests.lock().unwrap();
    assert_eq!(reqs.len(), 1);
    assert_eq!(
        reqs[0]["model"], json!("claude-haiku-4-5"),
        "a mechanical Opus turn is downgraded to Haiku in auto mode"
    );
}

#[tokio::test]
async fn auto_mode_leaves_sonnet_untouched() {
    let _guard = env_lock().lock().await;
    let _c = EnvVarGuard::set("AXIOM_MODEL_ROUTE", "auto");
    // mechanical_turn's newest message is a small, clean tool_result -- L-B
    // (default ON since the 2026-07-16 flip) would answer it locally before
    // routing ever runs. Pin it off so the routed request reaches the mock.
    let _c2 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "off");

    let (upstream, capture, _task) = start_upstream(false).await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    assert_eq!(
        post_messages(&app, mechanical_turn("claude-sonnet-5", "r-sonnet")).await,
        StatusCode::OK
    );
    let reqs = capture.requests.lock().unwrap();
    assert_eq!(reqs.len(), 1);
    assert_eq!(
        reqs[0]["model"], json!("claude-sonnet-5"),
        "Sonnet is not a high tier -> auto mode leaves it alone"
    );
}

#[tokio::test]
async fn routed_turn_that_4xxs_falls_back_once_to_the_original_model() {
    let _guard = env_lock().lock().await;
    let _c = EnvVarGuard::set("AXIOM_MODEL_ROUTE", "auto");
    // mechanical_turn's newest message is a small, clean tool_result -- L-B
    // (default ON since the 2026-07-16 flip) would answer it locally before
    // routing ever runs. Pin it off so the routed request reaches the mock.
    let _c2 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "off");

    // The mock 400s the routed Haiku attempt, forcing a single fallback retry
    // with the original Opus model.
    let (upstream, capture, _task) = start_upstream(true).await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    assert_eq!(
        post_messages(&app, mechanical_turn("claude-opus-4-8", "r-fallback")).await,
        StatusCode::OK
    );
    let reqs = capture.requests.lock().unwrap();
    assert_eq!(reqs.len(), 2, "one routed attempt + exactly one fallback retry");
    assert_eq!(reqs[0]["model"], json!("claude-haiku-4-5"), "first attempt is routed");
    assert_eq!(reqs[1]["model"], json!("claude-opus-4-8"), "fallback uses the original model");
}
