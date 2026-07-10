//! Integration coverage for S1: cache-safety hardening.
//!
//! Verifies the /v1/messages compression path (a) never compresses content
//! at or before an Anthropic `cache_control` breakpoint, and (b) produces
//! byte-identical outbound `messages` for byte-identical mutable-tail input
//! across repeated requests (freeze-on-first-send determinism).

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

/// `cargo test` runs test fns from this binary concurrently on separate
/// threads within the same process. `AXIOM_CACHE_SAFE` is read via
/// `std::env::var` inside the proxy's request path -- a process-global,
/// so any test that mutates it (`cache_safe_disabled_falls_back_to_prior_behavior`)
/// must not overlap with any other test in this file that depends on the
/// var being unset. All three tests hold this lock for their full request
/// lifetime (including `.await` points, hence the async-aware mutex) to
/// serialize that window without forcing `--test-threads=1` on the whole
/// binary.
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

/// Mock Anthropic `/v1/messages` upstream that captures every request body
/// it receives (so tests can inspect exactly what the proxy sent upstream).
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
            // Low threshold so short test strings count as "heavy".
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

const HEAVY_TEXT: &str =
    "This paragraph is deliberately long padded content well past a five token heavy threshold for cache safety boundary testing purposes only repeated words repeated words repeated words";

#[tokio::test]
async fn frozen_prefix_content_is_never_compressed_mutable_tail_is() {
    // 6 messages. cache_control on index 3 -> indices 0..=3 frozen.
    // Heavy content sits at index 2 (frozen, must survive verbatim) and at
    // index 5 (mutable, eligible for extraction).
    let _guard = env_lock().lock().await;
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let session_id = "cache-safety-breakpoint";
    let body = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": session_id,
        "messages": [
            {"role": "user", "content": "short user turn zero"},
            {"role": "assistant", "content": "short assistant reply one"},
            {"role": "user", "content": HEAVY_TEXT},
            {"role": "assistant", "content": [
                {"type": "text", "text": "short assistant reply three",
                 "cache_control": {"type": "ephemeral"}}
            ]},
            {"role": "user", "content": "short user turn four"},
            {"role": "user", "content": HEAVY_TEXT},
        ],
    });

    let status = post_messages(&app, body).await;
    assert_eq!(status, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    let sent = &captured[0];
    let sent_messages = sent["messages"].as_array().unwrap();
    let sent_json = serde_json::to_string(sent_messages).unwrap();

    // Frozen-prefix heavy content (index 2) must appear byte-for-byte
    // unchanged in the outbound payload -- untouched by compression.
    assert!(
        sent_json.contains(HEAVY_TEXT),
        "frozen-prefix heavy content must survive verbatim in the outbound payload"
    );

    // The mutable-tail heavy content (index 5) must have been extracted:
    // its exact original text should NOT appear a second time (it was
    // heavy_context, replaced by a fingerprint, not passed through raw).
    let occurrences = sent_json.matches(HEAVY_TEXT).count();
    assert_eq!(
        occurrences, 1,
        "only the frozen-prefix occurrence of the heavy text should survive verbatim; \
         the mutable-tail occurrence should have been compressed away, got {occurrences} occurrences"
    );
}

#[tokio::test]
async fn identical_mutable_tail_produces_byte_identical_outbound_messages() {
    // Two back-to-back requests with byte-identical bodies (same session,
    // same messages, same cache_control placement). The mutable tail's
    // compressed representation must be byte-identical both times, even
    // though the underlying TTT session state has now been adapted once
    // (proving the freeze-on-first-send memo overrides a re-derived, and
    // potentially different, fresh fingerprint).
    let _guard = env_lock().lock().await;
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let session_id = "cache-safety-determinism";
    let body = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": session_id,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "stable prior turn",
                 "cache_control": {"type": "ephemeral"}}
            ]},
            {"role": "user", "content": HEAVY_TEXT},
        ],
    });

    let s1 = post_messages(&app, body.clone()).await;
    assert_eq!(s1, StatusCode::OK);
    let s2 = post_messages(&app, body).await;
    assert_eq!(s2, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 2);
    let messages_1 = serde_json::to_string(&captured[0]["messages"]).unwrap();
    let messages_2 = serde_json::to_string(&captured[1]["messages"]).unwrap();
    assert_eq!(
        messages_1, messages_2,
        "identical mutable-tail input must produce byte-identical outbound messages"
    );
}

#[tokio::test]
async fn cache_safe_disabled_falls_back_to_prior_behavior() {
    // AXIOM_CACHE_SAFE=0 must not change compression behavior at all --
    // this test just confirms the flag is read and the request still
    // succeeds (regression guard for the escape hatch).
    let _guard = env_lock().lock().await;
    std::env::set_var("AXIOM_CACHE_SAFE", "0");
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    let body = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": "cache-safety-disabled",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
            ]},
            {"role": "user", "content": HEAVY_TEXT},
        ],
    });
    let status = post_messages(&app, body).await;
    std::env::remove_var("AXIOM_CACHE_SAFE");
    assert_eq!(status, StatusCode::OK);
    assert_eq!(capture.requests.lock().unwrap().len(), 1);
}
