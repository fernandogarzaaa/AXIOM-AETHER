//! Integration coverage for S2: the CVM L2 store's `POST /v1/expand`
//! wiring and session-drop cleanup.

use axiom_engine::config::AxiomConfig;
use axiom_engine::cvm_store::CvmStore;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use candle_core::Device;
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use tower::ServiceExt;

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

/// Unique temp dir per test (cleaned up on drop) so parallel test runs never
/// share a CvmStore root.
struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn tempdir(tag: &str) -> TempDir {
    let path = std::env::temp_dir().join(format!(
        "axiom-cvm-store-proxy-test-{tag}-{}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    TempDir(path)
}

async fn build_state(cvm_root: &TempDir) -> AppState {
    let pipeline = tokio::task::spawn_blocking(tiny_pipeline).await.unwrap();
    let cvm_store = CvmStore::open(&cvm_root.0).expect("cvm store open");
    AppState::new(pipeline, "axiom-ttt-test".to_string()).with_cvm_store(cvm_store)
}

async fn post_json(app: &axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

#[tokio::test]
async fn expand_by_page_id_returns_the_stored_page_through_the_http_route() {
    let dir = tempdir("expand");
    let state = build_state(&dir).await;
    let session_id = "cvm-expand-session";
    let original = "the full original tool result text, well past a stub's 120-char snippet, \
                     with enough content that a real digest would have replaced it";
    let page_id = state
        .cvm_store
        .put(session_id, "tool_result", original)
        .expect("put");

    let app = create_router(state);
    let (status, body) = post_json(
        &app,
        "/v1/expand",
        json!({"session_id": session_id, "symbol": page_id}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["found"], json!(true));
    assert_eq!(body["body"], json!(original));
    assert_eq!(body["session_id"], json!(session_id));
    assert_eq!(body["symbol"], json!(page_id));
}

#[tokio::test]
async fn expand_by_unknown_page_id_falls_through_to_not_found() {
    let dir = tempdir("expand-miss");
    let state = build_state(&dir).await;
    let app = create_router(state);

    // Well-formed page-id shape (16 hex chars) but never stored, and no
    // skeleton source stored for the session either -- falls through to the
    // pre-existing "no stored source" 404 shape.
    let (status, body) = post_json(
        &app,
        "/v1/expand",
        json!({"session_id": "no-such-session", "symbol": "0011223344556677"}),
    )
    .await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["session_id"], json!("no-such-session"));
}

#[tokio::test]
async fn deleting_a_session_removes_its_cvm_store_file() {
    let dir = tempdir("cleanup");
    let state = build_state(&dir).await;

    // Create a real session via the API so DELETE /v1/sessions/{id} has a
    // legitimate id to act on.
    let app = create_router(state.clone());
    let (status, created) = post_json(&app, "/v1/sessions", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    let session_id = created["session_id"].as_str().unwrap().to_string();

    // Seed a CVM page under that same session id.
    state
        .cvm_store
        .put(&session_id, "output", "will be cleaned up on session drop")
        .expect("put");
    let page_id = CvmStore::page_id_for("will be cleaned up on session drop");
    assert!(state.cvm_store.get(&session_id, &page_id).is_some());

    let req = Request::builder()
        .method(Method::DELETE)
        .uri(format!("/v1/sessions/{session_id}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    assert_eq!(state.cvm_store.get(&session_id, &page_id), None);
}
