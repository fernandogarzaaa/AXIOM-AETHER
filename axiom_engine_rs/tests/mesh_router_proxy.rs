//! End-to-end coverage for `SwarmRouter` wired to `MeshModelSelector`
//! (`AXIOM_MESH_ROUTING=1`) against a real local mock Ollama server —
//! the closest thing to a live-Ollama integration test achievable without
//! actually running Ollama. Mirrors the mock-upstream pattern already used
//! in `tests/model_router_proxy.rs`.

use axiom_engine::swarm_router::{SwarmRouteError, SwarmRouter, SwarmRouterConfig};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

/// Tracks which models were dispatched to, in order, and which ones the
/// mock should fail for this test.
#[derive(Clone, Default)]
struct MockOllama {
    dispatched: Arc<Mutex<Vec<String>>>,
    fail_models: Arc<Vec<String>>,
}

async fn tags_handler() -> impl IntoResponse {
    Json(json!({
        "models": [
            {"name": "phi4:3.8b"},
            {"name": "deepseek-r1:8b"},
        ]
    }))
}

async fn chat_handler(State(mock): State<MockOllama>, Json(body): Json<Value>) -> axum::response::Response {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    mock.dispatched.lock().unwrap().push(model.clone());
    if mock.fail_models.contains(&model) {
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "mock model unavailable").into_response();
    }
    Json(json!({"message": {"role": "assistant", "content": format!("ok from {model}")}})).into_response()
}

async fn start_mock(fail_models: Vec<String>) -> (String, MockOllama) {
    let mock = MockOllama { dispatched: Arc::new(Mutex::new(Vec::new())), fail_models: Arc::new(fail_models) };
    let app = Router::new()
        .route("/api/tags", get(tags_handler))
        .route("/api/chat", post(chat_handler))
        .with_state(mock.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), mock)
}

fn config(base_url: String, mesh_routing: bool) -> SwarmRouterConfig {
    SwarmRouterConfig {
        enabled: true,
        base_url,
        model_candidates: vec!["phi4:3.8b".to_string(), "deepseek-r1:8b".to_string()],
        num_ctx: 4096,
        timeout_ms: 2_000,
        mesh_routing,
    }
}

#[tokio::test]
async fn naive_and_mesh_routing_both_pick_top_priority_when_healthy() {
    let (base_url, mock) = start_mock(vec![]).await;
    let router = SwarmRouter::new(config(base_url, true));

    let result = router.route_chat_payload(&json!({"messages": [{"role": "user", "content": "hi"}]})).await;
    assert!(result.is_ok(), "expected success, got {result:?}");
    assert_eq!(result.unwrap().model, "phi4:3.8b");
    assert_eq!(*mock.dispatched.lock().unwrap(), vec!["phi4:3.8b".to_string()]);
}

#[tokio::test]
async fn mesh_routing_learns_around_a_failing_top_priority_model() {
    // phi4 is top priority (per candidate order) but always 500s; deepseek
    // always succeeds. Over repeated calls, mesh routing must learn to
    // stop sending traffic to phi4 -- the naive selector never could.
    let (base_url, mock) = start_mock(vec!["phi4:3.8b".to_string()]).await;
    let router = SwarmRouter::new(config(base_url, true));

    for _ in 0..20 {
        let _ = router.route_chat_payload(&json!({"messages": [{"role": "user", "content": "hi"}]})).await;
    }

    let dispatched = mock.dispatched.lock().unwrap();
    let phi4_count = dispatched.iter().filter(|m| *m == "phi4:3.8b").count();
    let deepseek_count = dispatched.iter().filter(|m| *m == "deepseek-r1:8b").count();
    assert_eq!(phi4_count + deepseek_count, 20);
    // The first call always tries phi4 (no learned signal yet, matching
    // the naive selector's default); one recorded failure should be
    // enough to hand every subsequent call to deepseek. This bound is
    // deliberately tight — a bug here previously took 8-13 failed calls
    // to recover from (see record_outcome's doc comment), and a loose
    // bound would hide a regression back to that behavior.
    assert!(
        deepseek_count >= 18,
        "expected mesh routing to recover within ~1-2 calls, got phi4={phi4_count} deepseek={deepseek_count}"
    );
}

#[tokio::test]
async fn naive_routing_keeps_retrying_the_failing_top_priority_model_forever() {
    // Same failing setup, but mesh_routing off -- this pins the *old*
    // behavior as a regression guard: the static selector has no memory,
    // so it must keep sending every single call to the broken model.
    let (base_url, mock) = start_mock(vec!["phi4:3.8b".to_string()]).await;
    let router = SwarmRouter::new(config(base_url, false));

    for _ in 0..10 {
        let result = router.route_chat_payload(&json!({"messages": [{"role": "user", "content": "hi"}]})).await;
        assert!(matches!(result, Err(SwarmRouteError::Upstream(_))));
    }
    let dispatched = mock.dispatched.lock().unwrap();
    assert_eq!(*dispatched, vec!["phi4:3.8b".to_string(); 10]);
}

#[tokio::test]
async fn mesh_routing_off_by_default_matches_naive_selection() {
    let (base_url, mock) = start_mock(vec![]).await;
    // mesh_routing left false, as SwarmRouterConfig::from_env defaults to
    // when AXIOM_MESH_ROUTING is unset.
    let router = SwarmRouter::new(config(base_url, false));

    let result = router.route_chat_payload(&json!({"messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(result.unwrap().model, "phi4:3.8b");
    assert_eq!(*mock.dispatched.lock().unwrap(), vec!["phi4:3.8b".to_string()]);
}
