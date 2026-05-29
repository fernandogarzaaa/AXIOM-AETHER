//! High-concurrency / tensor-memory stress test for the active-compression
//! pipeline.
//!
//! Fires 128 concurrent multi-tenant sessions at the Axum `/v1/messages`
//! endpoint. Every task pushes its own heavy context window (>500 whitespace
//! tokens) under a unique `session_id`, forcing the DashMap-backed session
//! store + `spawn_blocking` pool to absorb a parallel burst of `forward_native`
//! gradient loops.
//!
//! Asserts that under the spike the server:
//!   * answers every request with 200 OK (no dropped connections / deadlock),
//!   * isolates each session's fast weights (distinct `state_hash` per session),
//!   * strips the raw heavy text from every outbound payload, and
//!   * cleans up all detached session memory when the store is cleared.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

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
use tokio::sync::Mutex;
use tower::ServiceExt;

const CONCURRENCY: usize = 128;
const HEAVY_TOKENS: usize = 600; // > 500, comfortably over the threshold

type Captured = Arc<Mutex<Vec<Value>>>;

async fn start_mock_anthropic() -> (SocketAddr, Captured) {
    let captured: Captured = Arc::new(Mutex::new(Vec::new()));
    let mock_state = captured.clone();
    let app = Router::new()
        .route("/v1/messages", post(mock_handler))
        .with_state(mock_state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    (addr, captured)
}

async fn mock_handler(State(captured): State<Captured>, Json(body): Json<Value>) -> Json<Value> {
    captured.lock().await.push(body);
    Json(json!({
        "id": "msg_stress",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": "claude-mock",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
}

fn build_pipeline() -> InferencePipeline {
    let cfg = AxiomConfig {
        d_model: 16,
        n_layers: 1,
        vocab_size: 64,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    };
    InferencePipeline::new(cfg, Device::Cpu).expect("tiny stress pipeline must build")
}

/// Pull the `state_hash=...` line out of a captured upstream payload's
/// fingerprint block.
fn extract_state_hash(body: &Value) -> Option<String> {
    let combined: String = body["messages"]
        .as_array()?
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .concat();
    combined
        .lines()
        .find_map(|l| l.strip_prefix("state_hash=").map(str::to_string))
}

async fn stats(app: &Router) -> Value {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/ttt/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn high_concurrency_multitenant_sessions_stay_isolated_and_clean() {
    let (mock_addr, captured) = start_mock_anthropic().await;

    let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
    let forwarder =
        AnthropicForwarder::new(Some("test-key".to_string()), Some(format!("http://{mock_addr}")));
    let cfg = CompressorConfig {
        enabled: true,
        heavy_message_threshold_tokens: 400,
        recall_top_k: 8,
    };
    let state = AppState::new(pipeline, "axiom-stress".to_string())
        .with_anthropic_forwarder(Some(forwarder))
        .with_compressor_config(cfg);
    let app = create_router(state);

    // Cold store.
    let cold = stats(&app).await;
    assert_eq!(cold["active_sessions"], 0);
    assert_eq!(cold["compression_active"], true);

    // ---- Fire CONCURRENCY independent sessions in parallel ----------------
    let mut handles = Vec::with_capacity(CONCURRENCY);
    for i in 0..CONCURRENCY {
        let app = app.clone();
        handles.push(tokio::spawn(async move {
            // Unique heavy window per session so each W̃ diverges.
            let heavy: String = (0..HEAVY_TOKENS)
                .map(|j| format!("s{i}_tok{j}"))
                .collect::<Vec<_>>()
                .join(" ");
            let body = json!({
                "model": "claude-stress",
                "max_tokens": 8,
                "messages": [
                    {"role": "user", "content": heavy},
                    {"role": "user", "content": format!("summarise session {i}")}
                ],
                "session_id": format!("stress-{i}")
            });
            let req = Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let resp = app.oneshot(req).await.expect("request must not be dropped");
            resp.status()
        }));
    }

    // Every request must finish with 200 OK — no dropped connections, no
    // deadlock, no thread starvation hang.
    let mut ok = 0usize;
    for h in handles {
        let status = h.await.expect("task must not panic");
        assert_eq!(status, StatusCode::OK);
        ok += 1;
    }
    assert_eq!(ok, CONCURRENCY);

    // ---- Upstream payload assertions --------------------------------------
    let payloads = captured.lock().await.clone();
    assert_eq!(
        payloads.len(),
        CONCURRENCY,
        "mock must have received exactly one upstream call per session"
    );

    let mut hashes = std::collections::HashSet::new();
    for body in &payloads {
        assert!(body.get("session_id").is_none(), "session_id must be stripped");
        let combined: String = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m.get("content").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .concat();
        assert!(
            combined.contains("<axiom_context_fingerprint "),
            "every payload must carry the fingerprint block"
        );
        // The last raw heavy token must never have leaked upstream.
        assert!(
            !combined.contains(&format!("tok{}", HEAVY_TOKENS - 1)),
            "raw heavy text must be stripped from every payload"
        );
        if let Some(h) = extract_state_hash(body) {
            hashes.insert(h);
        }
    }
    // Per-session isolation: distinct contexts ⇒ distinct adapted W̃ hashes.
    assert_eq!(
        hashes.len(),
        CONCURRENCY,
        "each session must produce a distinct state_hash (no cross-tenant bleed)"
    );

    // The store should now hold exactly CONCURRENCY live sessions.
    let hot = stats(&app).await;
    assert_eq!(hot["active_sessions"], CONCURRENCY as u64);

    // ---- Cleanup: clear the store and confirm memory is released ----------
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::DELETE)
                .uri("/v1/ttt/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let cleared: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cleared["cleared"], true);

    let after = stats(&app).await;
    assert_eq!(
        after["active_sessions"], 0,
        "all detached session fast-weight tensors must be freed on clear"
    );
}
