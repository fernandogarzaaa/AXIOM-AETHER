//! Integration coverage for P2 (PSS): R2 free-window rebasing + R3 adaptive
//! cache TTL in the `/v1/messages` compression path.

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

/// `AXIOM_REBASE_ON_BREAK` / `AXIOM_ADAPTIVE_TTL` are process-global env vars
/// read in the request path; serialize the tests that set them.
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

/// A high heavy-threshold state: the base compressor absorbs nothing, so the
/// only transform observable upstream is P2's own rebase rewrite. Rebase uses
/// its OWN `DEFAULT_DIGEST_THRESHOLD_TOKENS` (4000), independent of this.
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

/// A transcript with a `cache_control`-marked frozen prefix (`prefix_text`
/// varies to simulate a compaction break), an OLD heavy `tool_result` in the
/// mutable tail, and a small newest turn. The old heavy block is what R2
/// restructures into a page stub when a break is detected.
fn body_with_old_heavy(session_id: &str, prefix_text: &str) -> Value {
    let big = "x ".repeat(9000); // 9000 tokens, over rebase's 4000 threshold
    json!({
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "session_id": session_id,
        "messages": [
            {"role":"user","content":[
                {"type":"text","text": prefix_text,
                 "cache_control":{"type":"ephemeral"}}]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"old","content": big}]},
            {"role":"user","content":"newest small turn"},
        ],
    })
}

#[tokio::test]
async fn rebase_restructures_old_heavy_on_a_detected_break() {
    let _guard = env_lock().lock().await;
    std::env::set_var("AXIOM_REBASE_ON_BREAK", "on");
    let _cleanup = EnvVarGuard("AXIOM_REBASE_ON_BREAK");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    // Turn 1 seeds the frozen-prefix hash -> no break, no rebase.
    assert_eq!(
        post_messages(&app, body_with_old_heavy("rebase-brk", "prefix v1")).await,
        StatusCode::OK
    );
    // Turn 2 changes the frozen prefix (a compaction) -> break -> rebase fires.
    assert_eq!(
        post_messages(&app, body_with_old_heavy("rebase-brk", "prefix v2 recompacted")).await,
        StatusCode::OK
    );

    let captured = capture.requests.lock().unwrap();
    let t1 = captured[0].to_string();
    let t2 = captured[1].to_string();
    // Turn 1: no break -> the old heavy block is forwarded raw (no page stub).
    assert!(!t1.contains("AXIOM-PAGE"), "first turn is not a break -> no rebase");
    // Turn 2: break -> the old heavy block arrives as a page stub, and its raw
    // 9000-token body no longer travels upstream.
    assert!(t2.contains("AXIOM-PAGE"), "break turn -> old heavy digested to a page");
    assert!(!t2.contains(&"x ".repeat(9000)), "raw old-heavy body removed on the break turn");
    // The newest turn is preserved verbatim on the break turn.
    assert!(t2.contains("newest small turn"), "newest turn untouched");
}

#[tokio::test]
async fn rebase_off_leaves_the_transcript_untouched() {
    let _guard = env_lock().lock().await;
    std::env::remove_var("AXIOM_REBASE_ON_BREAK"); // default off
    let _cleanup = EnvVarGuard("AXIOM_REBASE_ON_BREAK");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    // Two turns with a changed prefix (a break would be detected if enabled).
    assert_eq!(
        post_messages(&app, body_with_old_heavy("rebase-off", "prefix v1")).await,
        StatusCode::OK
    );
    assert_eq!(
        post_messages(&app, body_with_old_heavy("rebase-off", "prefix v2")).await,
        StatusCode::OK
    );

    let captured = capture.requests.lock().unwrap();
    // Flag off -> no page stubs anywhere, even across the prefix change.
    assert!(
        !captured.iter().any(|r| r.to_string().contains("AXIOM-PAGE")),
        "flag off -> transcript is never rebased"
    );
}

#[tokio::test]
async fn adaptive_ttl_does_not_fire_without_long_gaps() {
    let _guard = env_lock().lock().await;
    std::env::set_var("AXIOM_ADAPTIVE_TTL", "on");
    let _cleanup = EnvVarGuard("AXIOM_ADAPTIVE_TTL");

    let (upstream, capture, _task) = start_capturing_upstream().await;
    let state = build_state(upstream).await;
    let app = create_router(state);

    // Rapid back-to-back turns: the inter-turn gap is milliseconds, far under
    // the 240s window, so the long-gap count never reaches the threshold and
    // no `"ttl":"1h"` is added. (The positive path is covered by rebase.rs
    // unit tests: next_long_gap_count + choose_ttl + set_newest_cache_ttl.)
    for _ in 0..4 {
        assert_eq!(
            post_messages(&app, body_with_old_heavy("ttl-nogap", "steady prefix")).await,
            StatusCode::OK
        );
    }

    let captured = capture.requests.lock().unwrap();
    assert!(
        !captured.iter().any(|r| r.to_string().contains("\"ttl\":\"1h\"")),
        "no long gaps -> adaptive TTL stays at the default 5-minute window"
    );
}
