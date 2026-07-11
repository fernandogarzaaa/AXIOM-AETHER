//! Integration coverage for S3: digest admission control in the
//! `/v1/messages` compression path.

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
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::Mutex as AsyncMutex;
use tower::ServiceExt;

/// `AXIOM_CVM_DIGEST`/`AXIOM_CVM_DIGEST_THRESHOLD_TOKENS` are process-global
/// env vars read inside the proxy's request path -- same rationale as
/// `cache_safety_proxy.rs`'s `env_lock`.
fn env_lock() -> &'static AsyncMutex<()> {
    static LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| AsyncMutex::new(()))
}

struct EnvVarGuard(&'static [&'static str]);
impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        for var in self.0 {
            std::env::remove_var(var);
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

fn cvm_tempdir(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "axiom-digest-proxy-test-{tag}-{}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

async fn build_state(upstream: String, cvm_dir: &PathBuf) -> AppState {
    let pipeline = tokio::task::spawn_blocking(tiny_pipeline).await.unwrap();
    let cvm_store = axiom_engine::cvm_store::CvmStore::open(cvm_dir).expect("cvm store open");
    AppState::new(pipeline, "axiom-ttt-test".to_string())
        .with_anthropic_forwarder(Some(AnthropicForwarder::new(None, Some(upstream))))
        .with_compressor_config(CompressorConfig {
            heavy_message_threshold_tokens: 5,
            recall_top_k: 8,
            enabled: true,
        })
        .with_cvm_store(cvm_store)
}

async fn post_json(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", "sk-ant-test-key")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

/// A synthetic, well-over-4000-token `tool_result` body -- realistic
/// "heavy file read" content, code-shaped so `SkeletonDigestor` has
/// signatures to keep.
fn heavy_tool_result_text() -> String {
    (0..1200)
        .map(|i| format!("pub fn generated_function_{i}(x: i32) -> i32 {{ x + {i} }}\n"))
        .collect::<Vec<_>>()
        .join("")
}

fn tool_result_message(text: &str) -> Value {
    json!({
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": text},
        ],
    })
}

#[tokio::test]
async fn digest_replaces_heavy_tool_result_with_stub_and_original_is_expandable() {
    let _guard = env_lock().lock().await;
    std::env::set_var("AXIOM_CVM_DIGEST", "skeleton");
    let _cleanup = EnvVarGuard(&["AXIOM_CVM_DIGEST", "AXIOM_CVM_DIGEST_THRESHOLD_TOKENS"]);

    let cvm_dir = cvm_tempdir("basic");
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream, &cvm_dir).await;
    let app = create_router(state);

    let heavy = heavy_tool_result_text();
    let session_id = "digest-basic";
    let body = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": session_id,
        "messages": [
            {"role": "user", "content": "please read this file"},
            tool_result_message(&heavy),
        ],
    });
    let status = post_json(&app, "/v1/messages", body).await.0;
    assert_eq!(status, StatusCode::OK);

    let sent = {
        let captured = capture.requests.lock().unwrap();
        assert_eq!(captured.len(), 1);
        captured[0].clone()
    };
    let sent_str = sent.to_string();
    assert!(
        !sent_str.contains("generated_function_1199"),
        "the original heavy text must not reach upstream verbatim"
    );
    assert!(sent_str.contains("AXIOM-PAGE"), "a stub line must be present");
    assert!(
        sent_str.contains("AXIOM-PAGE-END"),
        "the expand trailer must be present"
    );

    // Extract the page id from the stub and confirm it's expandable back
    // to the exact original text.
    let stub_start = sent_str.find("AXIOM-PAGE ").unwrap() + "AXIOM-PAGE ".len();
    let page_id = &sent_str[stub_start..stub_start + 16];
    let (expand_status, expand_body) = post_json(
        &app,
        "/v1/expand",
        json!({"session_id": session_id, "symbol": page_id}),
    )
    .await;
    assert_eq!(expand_status, StatusCode::OK);
    assert_eq!(expand_body["body"], json!(heavy));

    let _ = fs::remove_dir_all(&cvm_dir);
}

#[tokio::test]
async fn digest_flag_off_passes_bytes_through_unchanged() {
    let _guard = env_lock().lock().await;
    // AXIOM_CVM_DIGEST now defaults to "skeleton" (S5 passed 2026-07-11);
    // "off" is an explicit opt-out, no longer the implicit unset state.
    std::env::set_var("AXIOM_CVM_DIGEST", "off");
    let _cleanup = EnvVarGuard(&["AXIOM_CVM_DIGEST"]);

    let cvm_dir = cvm_tempdir("flag-off");
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream, &cvm_dir).await;
    let app = create_router(state);

    let heavy = heavy_tool_result_text();
    let body = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": "digest-flag-off",
        "messages": [
            {"role": "user", "content": "please read this file"},
            tool_result_message(&heavy),
        ],
    });
    let status = post_json(&app, "/v1/messages", body).await.0;
    assert_eq!(status, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    let sent_str = captured[0].to_string();
    assert!(
        sent_str.contains("generated_function_1199"),
        "flags off -> the original heavy tool_result must pass through unchanged"
    );
    assert!(!sent_str.contains("AXIOM-PAGE"));

    let _ = fs::remove_dir_all(&cvm_dir);
}

#[tokio::test]
async fn digest_default_unset_env_var_digests_since_skeleton_is_now_the_default() {
    let _guard = env_lock().lock().await;
    // No explicit AXIOM_CVM_DIGEST set -- must fall back to the new
    // "skeleton" default (S5 passed 2026-07-11), not "off".
    std::env::remove_var("AXIOM_CVM_DIGEST");
    let _cleanup = EnvVarGuard(&["AXIOM_CVM_DIGEST"]);

    let cvm_dir = cvm_tempdir("default-unset");
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream, &cvm_dir).await;
    let app = create_router(state);

    let heavy = heavy_tool_result_text();
    let body = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": "digest-default-unset",
        "messages": [
            {"role": "user", "content": "please read this file"},
            tool_result_message(&heavy),
        ],
    });
    let status = post_json(&app, "/v1/messages", body).await.0;
    assert_eq!(status, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    let sent_str = captured[0].to_string();
    assert!(
        !sent_str.contains("generated_function_1199"),
        "default (unset AXIOM_CVM_DIGEST) must digest, not pass the original through"
    );
    assert!(sent_str.contains("AXIOM-PAGE"));

    let _ = fs::remove_dir_all(&cvm_dir);
}

#[tokio::test]
async fn digest_only_touches_the_newest_turn_when_cache_control_present() {
    let _guard = env_lock().lock().await;
    std::env::set_var("AXIOM_CVM_DIGEST", "skeleton");
    let _cleanup = EnvVarGuard(&["AXIOM_CVM_DIGEST", "AXIOM_CVM_DIGEST_THRESHOLD_TOKENS"]);

    let cvm_dir = cvm_tempdir("newest-turn");
    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream, &cvm_dir).await;
    let app = create_router(state);

    let heavy_old = heavy_tool_result_text();
    let heavy_new = heavy_tool_result_text().replace("generated_function", "newer_function");

    // Old heavy tool_result sits BEFORE a cache_control breakpoint (frozen
    // prefix); the new one is the newest turn, after the breakpoint.
    let body = json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": "digest-newest-turn",
        "messages": [
            {"role": "user", "content": "first turn"},
            tool_result_message(&heavy_old),
            {"role": "assistant", "content": [
                {"type": "text", "text": "ack",
                 "cache_control": {"type": "ephemeral"}}
            ]},
            tool_result_message(&heavy_new),
        ],
    });
    let status = post_json(&app, "/v1/messages", body).await.0;
    assert_eq!(status, StatusCode::OK);

    let captured = capture.requests.lock().unwrap();
    let sent_str = captured[0].to_string();
    assert!(
        sent_str.contains("generated_function_1199"),
        "frozen-prefix tool_result (before cache_control) must survive verbatim, undigested"
    );
    assert!(
        !sent_str.contains("newer_function_1199"),
        "newest-turn tool_result (after cache_control) must have been digested"
    );

    let _ = fs::remove_dir_all(&cvm_dir);
}
