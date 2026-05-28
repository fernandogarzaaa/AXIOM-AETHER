//! Integration coverage for the new Anthropic `/v1/messages` surface.
//!
//! Verifies:
//! * String content round-trips through the local pipeline path.
//! * Block-list content is flattened correctly before generation.
//! * Response envelope matches the Anthropic Messages shape.

use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use candle_core::Device;
use serde_json::{json, Value};
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

async fn post_json(app: &axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, parsed)
}

#[tokio::test]
async fn messages_string_content_round_trip() {
    let pipeline = tokio::task::spawn_blocking(tiny_pipeline).await.unwrap();
    let state = AppState::new(pipeline, "axiom-ttt-test".to_string());
    let app = create_router(state);

    let (status, body) = post_json(
        &app,
        "/v1/messages",
        json!({
            "model": "axiom-ttt-test",
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "hello"}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["content"][0]["type"], "text");
    assert!(body["content"][0]["text"].is_string());
    assert!(body["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(body["stop_reason"], "end_turn");
}

#[tokio::test]
async fn messages_block_content_round_trip() {
    let pipeline = tokio::task::spawn_blocking(tiny_pipeline).await.unwrap();
    let state = AppState::new(pipeline, "axiom-ttt-test".to_string());
    let app = create_router(state);

    let (status, body) = post_json(
        &app,
        "/v1/messages",
        json!({
            "max_tokens": 4,
            "system": "be brief",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "alpha "},
                    {"type": "text", "text": "beta"},
                ],
            }],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(body["content"][0]["text"].is_string());
    let usage = &body["usage"];
    assert!(usage["input_tokens"].as_u64().unwrap() >= 2); // "alpha beta"
}

#[tokio::test]
async fn messages_uses_default_model_when_unspecified() {
    let pipeline = tokio::task::spawn_blocking(tiny_pipeline).await.unwrap();
    let state = AppState::new(pipeline, "axiom-ttt-default".to_string());
    let app = create_router(state);

    let (status, body) = post_json(
        &app,
        "/v1/messages",
        json!({
            "max_tokens": 2,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model"], "axiom-ttt-default");
}
