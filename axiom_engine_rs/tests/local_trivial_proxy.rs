//! Integration coverage for P3 (PSS): L-B local trivial-turn short-circuit in
//! the `/v1/messages` compression path.

use axiom_engine::anthropic_forwarder::AnthropicForwarder;
use axiom_engine::config::AxiomConfig;
use axiom_engine::context_compressor::CompressorConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::server::{create_router, run_server, AppState, CheckpointStatus};
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

/// `AXIOM_DRIFT_THRESHOLD` is a process-global env var read in the request
/// path, and `AXIOM_LOCAL_TRIVIAL` one read at startup; serialize the tests
/// that set them.
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
    #[allow(dead_code)] // kept for symmetry with `set`; not every test needs it
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

/// A server as it comes up by default: the short-circuit not asked for, and
/// (as in every test here) no verified checkpoint behind the tiny model.
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

/// A server whose operator opted in to the short-circuit and whose checkpoint
/// verified at startup: the only configuration in which it may run.
async fn build_opted_in_state(upstream: String) -> AppState {
    build_state(upstream)
        .await
        .with_checkpoint_gate(false, CheckpointStatus::Verified)
        .with_local_trivial(true)
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
    // Huge gate: any finite surprisal passes, so triviality reduces to the
    // deterministic structural checks (tool_result-only, error-free, small).
    let _c2 = EnvVarGuard::set("AXIOM_DRIFT_THRESHOLD", "1000000");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_opted_in_state(upstream).await;
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
    let _c2 = EnvVarGuard::set("AXIOM_DRIFT_THRESHOLD", "1000000");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_opted_in_state(upstream).await;
    let app = create_router(state);

    let mut body = clean_mechanical("lb-stream");
    body["stream"] = json!(true);
    let (status, content_type, text) = post_messages_raw(&app, body).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type.contains("text/event-stream"),
        "a streaming client must get SSE, not JSON (got {content_type})"
    );
    assert!(
        text.contains("event: message_start"),
        "valid SSE event sequence"
    );
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
    let _c2 = EnvVarGuard::set("AXIOM_DRIFT_THRESHOLD", "1000000");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_opted_in_state(upstream).await;
    let app = create_router(state);

    let (status, body) = post_messages(&app, error_bearing("lb-error")).await;
    assert_eq!(status, StatusCode::OK);
    assert_ne!(
        body["model"],
        json!("axiom-local"),
        "error turn is not local"
    );
    assert_eq!(
        capture.requests.lock().unwrap().len(),
        1,
        "an error-bearing turn must be forwarded upstream"
    );
}

#[tokio::test]
async fn not_opted_in_always_forwards_even_with_a_verified_checkpoint() {
    let _guard = env_lock().lock().await;
    // A gate so wide every finite surprisal passes: only the opt-in is left
    // to stop the short-circuit.
    let _c1 = EnvVarGuard::set("AXIOM_DRIFT_THRESHOLD", "1000000");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream)
        .await
        .with_checkpoint_gate(false, CheckpointStatus::Verified);
    let app = create_router(state);

    let (status, body) = post_messages(&app, clean_mechanical("lb-off")).await;
    assert_eq!(status, StatusCode::OK);
    assert_ne!(
        body["model"],
        json!("axiom-local"),
        "not opted in -> never local"
    );
    assert_eq!(
        capture.requests.lock().unwrap().len(),
        1,
        "without the opt-in every turn is forwarded"
    );
}

#[tokio::test]
async fn opted_in_without_a_verified_checkpoint_still_forwards() {
    // The opt-in alone must not be enough: with an absent, unverified or
    // mismatched checkpoint the surprisal gate is computed by weights nobody
    // vouched for, so the turn goes to the model.
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::set("AXIOM_DRIFT_THRESHOLD", "1000000");

    for status in [
        CheckpointStatus::Absent,
        CheckpointStatus::Unverified,
        CheckpointStatus::Mismatched,
    ] {
        let (upstream, capture, _task) = start_capturing_upstream().await;
        let state = build_state(upstream)
            .await
            .with_local_trivial(true)
            .with_checkpoint_gate(false, status);
        assert!(!state.local_trivial_active(), "{status:?}");
        let app = create_router(state);

        let (http, body) = post_messages(&app, clean_mechanical("lb-unverified")).await;
        assert_eq!(http, StatusCode::OK);
        assert_ne!(
            body["model"],
            json!("axiom-local"),
            "checkpoint {status:?}: never local"
        );
        assert_eq!(
            capture.requests.lock().unwrap().len(),
            1,
            "checkpoint {status:?}: the turn must reach the model"
        );
    }
}

#[tokio::test]
async fn the_server_refuses_to_start_with_the_flag_on_and_no_verified_checkpoint() {
    // The startup guard, end to end: AXIOM_LOCAL_TRIVIAL=on with a checkpoint
    // that is not there. `run_server` must return the error before it binds
    // (a server that started would never return from this call).
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "on");
    let _c2 = EnvVarGuard::unset("AXIOM_CHECKPOINT_SHA256");

    let missing = std::env::temp_dir().join("axiom_local_trivial_no_such_checkpoint.safetensors");
    let _ = std::fs::remove_file(&missing);
    let config = AxiomConfig {
        d_model: 16,
        n_layers: 1,
        vocab_size: 64,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    };
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(120),
        run_server(
            "127.0.0.1",
            0,
            config,
            missing.to_str().unwrap(),
            Device::Cpu,
        ),
    )
    .await
    .expect("run_server must refuse to start, not serve");
    let err = result.expect_err("starting with the flag on and no checkpoint is an error");
    let message = err.to_string();
    assert!(message.contains("AXIOM_LOCAL_TRIVIAL"), "{message}");
    assert!(
        message.contains("absent"),
        "names the checkpoint state: {message}"
    );
}

#[tokio::test]
async fn the_server_refuses_to_start_when_the_checkpoints_tokenizer_is_not_in_use() {
    // A verified checkpoint vouches for the weights only. This one's sidecar
    // says it was trained with a BPE tokenizer, and AXIOM_TOKENIZER is unset,
    // so the pipeline falls back to the hash tokenizer: the model's surprisal
    // is noise, and the short-circuit must not end turns on it.
    use sha2::{Digest, Sha256};

    let _guard = env_lock().lock().await;
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("model.safetensors");
    // A loadable checkpoint (the loader takes the tensors it recognises and
    // keeps its init for the rest), pinned by hash so that it is Verified.
    let tensor = candle_core::Tensor::zeros((1,), candle_core::DType::F32, &Device::Cpu).unwrap();
    candle_core::safetensors::save(
        &std::collections::HashMap::from([("unused".to_string(), tensor)]),
        &checkpoint,
    )
    .unwrap();
    let digest = format!("{:x}", Sha256::digest(std::fs::read(&checkpoint).unwrap()));
    std::fs::write(
        dir.path().join("model.meta.json"),
        r#"{"schema_version":1,"d_model":16,"n_layers":1,"vocab_size":64,"lr_inner":0.001,"norm_eps":1e-6,"val_ce":3.0,"tokenizer":"checkpoints/axiom_bpe.json"}"#,
    )
    .unwrap();

    let _c1 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "on");
    let _c2 = EnvVarGuard::set("AXIOM_CHECKPOINT_SHA256", &digest);
    let _c3 = EnvVarGuard::unset("AXIOM_TOKENIZER");
    let _c4 = EnvVarGuard::unset("AXIOM_TOKENIZER_SHA256");

    let config = AxiomConfig {
        d_model: 16,
        n_layers: 1,
        vocab_size: 64,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    };
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        run_server(
            "127.0.0.1",
            0,
            config,
            checkpoint.to_str().unwrap(),
            Device::Cpu,
        ),
    )
    .await
    .expect("run_server must refuse to start, not serve");
    let err = result.expect_err("the checkpoint's tokenizer is not in use: an error");
    let message = err.to_string();
    assert!(message.contains("AXIOM_LOCAL_TRIVIAL"), "{message}");
    assert!(
        message.contains("axiom_bpe.json") && message.contains("AXIOM_TOKENIZER"),
        "names the tokenizer the checkpoint needs: {message}"
    );
}

/// The proxy must not end an agent's turn on its own under default settings.
/// With `AXIOM_LOCAL_TRIVIAL` unset and nothing else configured, a small clean
/// `tool_result` (a build finished, a file listing) has to reach the model:
/// the agent loop is waiting for the model's next step, and a locally invented
/// "Acknowledged" with `stop_reason: end_turn` stops it mid-task.
#[tokio::test]
async fn default_settings_forward_a_small_clean_tool_result_to_the_model() {
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::unset("AXIOM_LOCAL_TRIVIAL");
    let _c2 = EnvVarGuard::unset("AXIOM_DRIFT_THRESHOLD");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let (status, body) = post_messages(&app, clean_mechanical("lb-default")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        capture.requests.lock().unwrap().len(),
        1,
        "with default settings the turn must reach the model"
    );
    assert_eq!(
        body["content"][0]["text"],
        json!("from upstream"),
        "the client gets the model's answer, not a local acknowledgement"
    );
    assert_ne!(body["model"], json!("axiom-local"));
}

/// Same for a streaming client, which is how agent CLIs talk to the proxy.
#[tokio::test]
async fn default_settings_forward_a_streaming_tool_result_to_the_model() {
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::unset("AXIOM_LOCAL_TRIVIAL");
    let _c2 = EnvVarGuard::unset("AXIOM_DRIFT_THRESHOLD");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let mut body = clean_mechanical("lb-default-stream");
    body["stream"] = json!(true);
    let (_status, _content_type, text) = post_messages_raw(&app, body).await;
    assert_eq!(
        capture.requests.lock().unwrap().len(),
        1,
        "with default settings a streaming turn must reach the model"
    );
    assert!(
        !text.contains("axiom-local") && !text.contains("answered locally"),
        "no local acknowledgement may be sent: {text}"
    );
}
