//! End-to-end test of the active-compression pipeline.
//!
//! Spins up an in-process mock Anthropic endpoint, configures the
//! Axiom server with compression enabled and the forwarder pointed at
//! the mock, then sends a `/v1/messages` request with one heavy + one
//! light user turn. Asserts that the upstream payload received by the
//! mock has:
//!   * the raw heavy text stripped,
//!   * an <axiom_context_fingerprint> block prepended,
//!   * the original user query preserved.

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

type Captured = Arc<Mutex<Vec<Value>>>;

async fn start_mock_anthropic() -> (SocketAddr, Captured) {
    let captured: Captured = Arc::new(Mutex::new(Vec::new()));
    let mock_state = captured.clone();

    let app = Router::new()
        .route("/v1/messages", post(mock_messages_handler))
        .with_state(mock_state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    // Give axum's serve loop a tick to start polling the listener.
    tokio::time::sleep(Duration::from_millis(20)).await;
    (addr, captured)
}

async fn mock_messages_handler(
    State(captured): State<Captured>,
    Json(body): Json<Value>,
) -> Json<Value> {
    captured.lock().await.push(body);
    Json(json!({
        "id": "msg_mock_e2e",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "mocked reply"}],
        "model": "claude-mock",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 2}
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
    InferencePipeline::new(cfg, Device::Cpu).expect("tiny pipeline must build")
}

#[tokio::test]
async fn compression_strips_heavy_context_and_forwards_lean_payload() {
    let (mock_addr, captured) = start_mock_anthropic().await;

    let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
    let forwarder = AnthropicForwarder::new(
        "test-key".to_string(),
        Some(format!("http://{mock_addr}")),
    );
    let compressor_config = CompressorConfig {
        enabled: true,
        heavy_message_threshold_tokens: 50,
        recall_top_k: 8,
    };
    let state = AppState::new(pipeline, "axiom-e2e".to_string())
        .with_anthropic_forwarder(Some(forwarder))
        .with_compressor_config(compressor_config);
    let app = create_router(state);

    let heavy_text: String = (0..200)
        .map(|i| format!("code{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let req_body = json!({
        "model": "claude-opus-4-7",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": heavy_text.clone()},
            {"role": "user", "content": "summarise that codebase in one sentence"}
        ],
        "session_id": "e2e-session-1"
    });

    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(req_body.to_string()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp_bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let resp_json: Value = serde_json::from_slice(&resp_bytes).unwrap();
    // The proxy returns whatever the upstream returned verbatim.
    assert_eq!(resp_json["id"], "msg_mock_e2e");
    assert_eq!(resp_json["model"], "claude-mock");

    let received = captured.lock().await.clone();
    assert_eq!(received.len(), 1, "mock must receive exactly one upstream call");
    let upstream = &received[0];

    // The session_id Axiom-extension must not reach the upstream payload.
    assert!(upstream.get("session_id").is_none());

    let upstream_msgs = upstream["messages"].as_array().unwrap();
    let combined: String = upstream_msgs
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_str).map(str::to_string))
        .collect::<Vec<_>>()
        .concat();

    assert!(
        !combined.contains("code199"),
        "heavy raw text must NOT reach the upstream API"
    );
    assert!(
        combined.contains("<axiom_context_fingerprint "),
        "fingerprint block must be present in upstream payload"
    );
    assert!(
        combined.contains("</axiom_context_fingerprint>"),
        "fingerprint block must be well-formed (closing tag present)"
    );
    assert!(
        combined.contains("summarise that codebase in one sentence"),
        "user query must survive into upstream payload"
    );
    assert!(
        combined.contains("tokens_compressed="),
        "fingerprint must report ingested-token count"
    );
}

#[tokio::test]
async fn compression_off_uses_local_pipeline_path() {
    // When compression is disabled, /v1/messages must NOT call out to
    // the (configured but unused) forwarder — the local pipeline path
    // services the request.
    let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
    let state = AppState::new(pipeline, "axiom-local".to_string());
    let app = create_router(state);

    let req_body = json!({
        "model": "axiom-local",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "hi"}],
    });
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(req_body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert!(body["id"].as_str().unwrap().starts_with("msg_"));
}

#[tokio::test]
async fn ttt_session_state_persists_across_calls() {
    // Two compressed calls sharing a session_id should produce two
    // distinct state_hash values (W̃ keeps moving), confirming that
    // the DashMap-backed session store carries state across requests.
    let (mock_addr, captured) = start_mock_anthropic().await;

    let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
    let forwarder = AnthropicForwarder::new(
        "test-key".to_string(),
        Some(format!("http://{mock_addr}")),
    );
    let cfg = CompressorConfig {
        enabled: true,
        heavy_message_threshold_tokens: 30,
        recall_top_k: 4,
    };
    let state = AppState::new(pipeline, "axiom-stateful".to_string())
        .with_anthropic_forwarder(Some(forwarder))
        .with_compressor_config(cfg);
    let app = create_router(state);

    for cycle in 0..2 {
        let heavy: String = (0..100)
            .map(|i| format!("payload_{cycle}_{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let req_body = json!({
            "model": "claude-stateful",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": heavy},
                {"role": "user", "content": format!("query for cycle {cycle}")}
            ],
            "session_id": "persistent-session"
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(req_body.to_string()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "cycle {cycle} must succeed");
    }

    let received = captured.lock().await.clone();
    assert_eq!(received.len(), 2);
    let extract_hash = |body: &Value| {
        let combined: String = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m.get("content").and_then(Value::as_str).map(str::to_string))
            .collect::<Vec<_>>()
            .concat();
        // Pull the line `state_hash=...` out of the fingerprint block.
        for line in combined.lines() {
            if let Some(rest) = line.strip_prefix("state_hash=") {
                return rest.to_string();
            }
        }
        String::new()
    };
    let hash_1 = extract_hash(&received[0]);
    let hash_2 = extract_hash(&received[1]);
    assert!(hash_1.starts_with("sha256:"));
    assert!(hash_2.starts_with("sha256:"));
    assert_ne!(
        hash_1, hash_2,
        "session state must mutate between cycles — W̃ is supposed to keep moving"
    );
}

#[tokio::test]
async fn ttt_session_admin_endpoints_reflect_live_state() {
    let (mock_addr, _captured) = start_mock_anthropic().await;
    let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
    let forwarder = AnthropicForwarder::new(
        "test-key".to_string(),
        Some(format!("http://{mock_addr}")),
    );
    let cfg = CompressorConfig {
        enabled: true,
        heavy_message_threshold_tokens: 20,
        recall_top_k: 4,
    };
    let state = AppState::new(pipeline, "axiom-admin".to_string())
        .with_anthropic_forwarder(Some(forwarder))
        .with_compressor_config(cfg);
    let app = create_router(state);

    // Cold stats
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
    let stats: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(stats["active_sessions"], 0);
    assert_eq!(stats["compression_active"], true);

    // Drive one compressed call.
    let heavy: String = (0..50).map(|i| format!("x{i}")).collect::<Vec<_>>().join(" ");
    let req_body = json!({
        "max_tokens": 4,
        "messages": [
            {"role": "user", "content": heavy},
            {"role": "user", "content": "go"}
        ],
        "session_id": "admin-sess"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Stats should now show 1 active session.
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
    let stats: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(stats["active_sessions"], 1);

    // Delete it.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::DELETE)
                .uri("/v1/ttt/sessions/admin-sess")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["removed"], true);
}
