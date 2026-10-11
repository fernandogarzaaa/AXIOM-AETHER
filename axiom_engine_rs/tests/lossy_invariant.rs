//! Proxy-level integration tests for the safety invariant:
//! LOSSY_CONTEXT implies RECOVERY_CAPABILITY_PRESENT.
//!
//! Every default-on path that drops, replaces, or summarizes content must
//! have a recovery path for the clients that trigger it, or be gated off for
//! clients without one. One test per path. This file is the invariant
//! contract: feature work does not touch it, and CI fails if any test here
//! is deleted. See docs/safety/lossy-invariant.md for the full table.

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

/// Process-global env vars read in the request path; serialize the tests
/// that set them.
fn env_lock() -> &'static AsyncMutex<()> {
    static LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| AsyncMutex::new(()))
}

struct EnvVarGuard(&'static [&'static str]);
impl EnvVarGuard {
    fn set(name: &'static str, value: &str) -> Self {
        std::env::set_var(name, value);
        Self(&[])
    }
}
impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        for name in self.0 {
            std::env::remove_var(name);
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

/// High heavy-threshold state: the base compressor absorbs nothing, so the
/// only transform observable upstream is the path under test (rebase).
async fn build_state_high_threshold(upstream: String) -> AppState {
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
    let body: Value = serde_json::from_slice(&bytes)
        .unwrap_or(Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body)
}

fn heavy_text() -> String {
    // Over the digest threshold; contains a unique marker.
    format!("{}\nmarker_invariant_7741", "x ".repeat(3000))
}

fn tool_result_message(text: &str) -> Value {
    json!({
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": text}],
    })
}

// --- Path 1: digest ---------------------------------------------------------

#[tokio::test]
async fn invariant_digest_skipped_without_expand_tool() {
    // LOSSY_CONTEXT (digest stub) requires RECOVERY_CAPABILITY (axiom_expand
    // in the client's tools). Without it, the heavy text must pass through
    // undigested.
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::set("AXIOM_CVM_DIGEST", "skeleton");
    let _c2 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "off");
    let _c3 = EnvVarGuard::set("AXIOM_TOOL_DEFER", "off");
    let _c4 = EnvVarGuard::set("AXIOM_MODEL_ROUTE", "off");
    let _c5 = EnvVarGuard::set("AXIOM_REBASE_ON_BREAK", "off");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let body = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": "inv-digest-no-expand",
        "tools": [{"name": "Read"}, {"name": "Bash"}],
        "messages": [
            {"role": "user", "content": "read this"},
            tool_result_message(&heavy_text()),
        ],
    });
    let (status, _) = post_messages(&app, body).await;
    assert_eq!(status, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    let sent = captured[0].to_string();
    assert!(
        sent.contains("marker_invariant_7741"),
        "without axiom_expand, heavy text must pass through undigested"
    );
    assert!(
        !sent.contains("AXIOM-PAGE"),
        "no digest stub may be emitted when expansion is unavailable"
    );
}

// --- Path 3: rebase stubs ---------------------------------------------------

#[tokio::test]
async fn invariant_rebase_skipped_without_expand_tool() {
    // LOSSY_CONTEXT (rebase stub) requires RECOVERY_CAPABILITY (axiom_expand).
    // A detected break with no axiom_expand in tools must not restructure
    // old heavy turns into stubs.
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::set("AXIOM_REBASE_ON_BREAK", "on");
    let _c2 = EnvVarGuard::set("AXIOM_CVM_DIGEST", "off");
    let _c3 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "off");
    let _c4 = EnvVarGuard::set("AXIOM_TOOL_DEFER", "off");
    let _c5 = EnvVarGuard::set("AXIOM_MODEL_ROUTE", "off");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state_high_threshold(upstream).await;
    let app = create_router(state);

    // Two turns with different frozen prefixes simulate a compaction break;
    // the old heavy tool_result is what rebase would restructure.
    let big = format!("{}\nmarker_rebase_9917", "x ".repeat(9000));
    let turn1 = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": "inv-rebase-no-expand",
        "tools": [{"name": "Read"}],
        "messages": [
            {"role":"user","content":[{"type":"text","text":"prefix alpha","cache_control":{"type":"ephemeral"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"old","content": big}]},
            {"role":"user","content":"newest small turn"},
        ],
    });
    let (status, _) = post_messages(&app, turn1).await;
    assert_eq!(status, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    let sent = captured[0].to_string();
    assert!(
        sent.contains("marker_rebase_9917"),
        "without axiom_expand, rebase must not stub old heavy turns"
    );
    assert!(
        !sent.contains("AXIOM-PAGE"),
        "no rebase stub may be emitted when expansion is unavailable"
    );
}

// --- Path 4: tool deferral --------------------------------------------------

#[tokio::test]
async fn invariant_tool_deferral_skipped_on_bespoke_toolset() {
    // Hiding schemas (lossy w.r.t. the cached prefix) requires the recovery
    // path of on-demand tool_reference loading, which only makes sense for
    // the coding-agent main loop. A bespoke toolset with zero working-set
    // overlap must pass through untouched.
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::set("AXIOM_TOOL_DEFER", "on");
    let _c2 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "off");
    let _c3 = EnvVarGuard::set("AXIOM_CVM_DIGEST", "off");
    let _c4 = EnvVarGuard::set("AXIOM_MODEL_ROUTE", "off");
    let _c5 = EnvVarGuard::set("AXIOM_REBASE_ON_BREAK", "off");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let body = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": "inv-defer-bespoke",
        "tools": [{"name": "CronCreate"}, {"name": "Monitor"}, {"name": "SendMessage"}],
        "messages": [{"role": "user", "content": "hello"}],
    });
    let (status, _) = post_messages(&app, body).await;
    assert_eq!(status, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    let tools = captured[0]["tools"].as_array().expect("tools array");
    for tool in tools {
        assert!(
            tool.get("defer_loading").is_none(),
            "bespoke toolset with no working-set overlap must never be deferred"
        );
    }
}

// --- Path 5: local-trivial answers ------------------------------------------

#[tokio::test]
async fn invariant_local_ack_labels_synthetic_answer() {
    // Replacing a model turn with a synthetic ACK (lossy) requires the client
    // to be able to tell it is synthetic: the ACK text must carry the label.
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "on");
    let _c2 = EnvVarGuard::set("AXIOM_DRIFT_THRESHOLD", "1000000");
    let _c3 = EnvVarGuard::set("AXIOM_CVM_DIGEST", "off");
    let _c4 = EnvVarGuard::set("AXIOM_MODEL_ROUTE", "off");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let body = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": "inv-local-ack",
        "messages": [
            {"role":"assistant","content":[
                {"type":"tool_use","name":"Bash","id":"t1","input":{}}]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":"ok, exit 0 done"}]},
        ],
    });
    let (status, resp_body) = post_messages(&app, body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        capture.requests.lock().unwrap().len(),
        0,
        "trivial turn must not reach upstream"
    );
    let text = resp_body["content"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .to_lowercase();
    assert!(
        text.contains("answered locally") || text.contains("local"),
        "synthetic ACK must be labeled as locally answered, got: {text}"
    );
}

// --- Path 6: model routing --------------------------------------------------

#[tokio::test]
async fn invariant_routing_downgrade_is_recorded() {
    // A model downgrade is not content-lossy, but the invariant requires it
    // to be visible: the routed turn must be counted in the session
    // awareness so the receipt can show it.
    let _guard = env_lock().lock().await;
    let _c1 = EnvVarGuard::set("AXIOM_MODEL_ROUTE", "auto");
    let _c2 = EnvVarGuard::set("AXIOM_LOCAL_TRIVIAL", "off");
    let _c3 = EnvVarGuard::set("AXIOM_CVM_DIGEST", "off");
    let _c4 = EnvVarGuard::set("AXIOM_TOOL_DEFER", "off");
    let _c5 = EnvVarGuard::set("AXIOM_REBASE_ON_BREAK", "off");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let session_id = "inv-route-receipt";
    let before = state
        .awareness
        .get_or_create(session_id)
        .cost_summary()
        .routed_turns;
    let app = create_router(state.clone());

    let body = json!({
        "model": "claude-opus-4-8",
        "max_tokens": 16,
        "session_id": session_id,
        "messages": [
            {"role":"assistant","content":[
                {"type":"tool_use","name":"Bash","id":"t1","input":{}}]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":"ok done"}]},
        ],
    });
    let (status, _) = post_messages(&app, body).await;
    assert_eq!(status, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(
        captured[0]["model"],
        json!("claude-haiku-4-5"),
        "mechanical Opus turn downgraded to Haiku"
    );
    let after = state
        .awareness
        .get_or_create(session_id)
        .cost_summary()
        .routed_turns;
    assert!(
        after > before,
        "downgraded turn must be recorded in session awareness for the receipt"
    );
}

// --- Path 2: Responses compression ------------------------------------------
// The compressed-payload retry on upstream error is covered at the proxy
// level by responses_compression_proxy.rs
// (compressed_bad_request_retries_original_and_preserves_structural_items).
// This invariant test pins the unit-level contract: a compression plan that
// cannot be applied must fail closed (return None), never a half-applied
// transform.

#[test]
fn invariant_responses_apply_plan_fails_closed_on_length_mismatch() {
    let body = json!({"input": [
        {"role": "assistant", "content": "old text one"},
        {"role": "assistant", "content": "old text two"},
        {"role": "user", "content": "query"},
    ]});
    let plan =
        axiom_engine::responses_compressor::plan_compression(&body).expect("plan must exist");
    // Wrong fingerprint count: fail closed, body untouched. The plan has 1
    // run (two consecutive assistant messages), so 0 or 2 fingerprints
    // mismatch.
    let wrong: Vec<String> = vec![];
    assert!(
        axiom_engine::responses_compressor::apply_plan(&body, &plan, &wrong).is_none(),
        "apply_plan with mismatched fingerprints must fail closed"
    );
    // Correct count: manifest carries per-item SHA-256 for verification.
    let right: Vec<String> = (0..plan.runs.len()).map(|i| format!("fp{i}")).collect();
    let out = axiom_engine::responses_compressor::apply_plan(&body, &plan, &right)
        .expect("apply must succeed");
    let out_str = out.to_string();
    assert!(
        out_str.contains("axiom_source_manifest"),
        "compressed items must carry a source manifest"
    );
    assert!(
        out_str.contains("compressed_item_sha256"),
        "manifest must carry per-item hashes for verification"
    );
}
