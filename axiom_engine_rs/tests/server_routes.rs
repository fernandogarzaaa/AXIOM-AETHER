use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use candle_core::Device;
use tower::ServiceExt;

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

async fn test_app() -> axum::Router {
    let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
    create_router(AppState::new(pipeline, "axiom-routes-test".to_string()))
}

fn chat_body(stream: bool) -> String {
    let mut msg = serde_json::Map::new();
    msg.insert("role".into(), serde_json::Value::String("user".into()));
    msg.insert("content".into(), serde_json::Value::String("hi".into()));
    let mut body = serde_json::Map::new();
    body.insert(
        "messages".into(),
        serde_json::Value::Array(vec![serde_json::Value::Object(msg)]),
    );
    body.insert(
        "max_tokens".into(),
        serde_json::Value::Number(serde_json::Number::from(2)),
    );
    body.insert("stream".into(), serde_json::Value::Bool(stream));
    serde_json::Value::Object(body).to_string()
}

#[tokio::test]
async fn metrics_endpoint_renders_prometheus_text() {
    let app = test_app().await;
    let req = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("axiom_total_tokens_prefilled"));
    assert!(body.contains("axiom_prefill_latency_seconds_bucket"));
}

#[tokio::test]
async fn chat_completion_stream_returns_sse_done_sentinel() {
    let app = test_app().await;
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat_body(true)))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(content_type.contains("text/event-stream"));
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body = std::str::from_utf8(&body).unwrap();
    assert!(body.contains("[DONE]"));
}

#[tokio::test]
async fn chat_completion_returns_200() {
    let app = test_app().await;
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat_body(false)))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn adapt_requires_nonempty_corpus() {
    let app = test_app().await;
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/adapt")
        .header("content-type", "application/json")
        .body(Body::from("{\"corpus\":[]}"))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn delete_unknown_session_returns_deleted_false() {
    let app = test_app().await;
    let req = Request::builder()
        .method(Method::DELETE)
        .uri("/v1/sessions/nonexistent-id")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["deleted"], false);
}

#[tokio::test]
async fn create_session_returns_session_id() {
    let app = test_app().await;
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/sessions")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["session_id"].is_string());
    assert_eq!(json["object"], "session");
}

#[tokio::test]
async fn list_models_returns_200() {
    let app = test_app().await;
    let req = Request::builder()
        .method(Method::GET)
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn verify_grounding_gated_expansion_spends_tokens_only_where_needed() {
    // Build state, store the full source for a session (as the compressor would),
    // then verify a claim the skeleton alone cannot ground.
    let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
    let state = AppState::new(pipeline, "axiom-gate-test".to_string());
    let full_source = "pub fn parse_header(buf: &[u8]) -> Header { read(buf) }\n\
        pub fn checksum(data: &[u8]) -> u32 { computes crc32 polynomial fold over data }";
    state.store_source("gate-1", full_source.to_string());
    let app = create_router(state);

    // Evidence = the lean skeleton (signatures only). The claim's detail lives
    // only in the elided checksum body.
    let body = serde_json::json!({
        "response": "checksum computes the crc32 polynomial fold over the data.",
        "evidence": "pub fn parse_header(buf: &[u8]) -> Header. pub fn checksum(data: &[u8]) -> u32.",
        "session_id": "gate-1",
        "expand": true,
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/verify")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();

    assert_eq!(v["mode"], "grounding_gated_expansion");
    let before = v["grounded_fraction_before"].as_f64().unwrap();
    let after = v["grounded_fraction_after"].as_f64().unwrap();
    assert!(after > before, "expansion must improve grounding ({before} -> {after})");
    assert!(
        v["expanded_symbols"].as_array().unwrap().contains(&serde_json::json!("checksum")),
        "only the referenced symbol should be expanded: {:?}",
        v["expanded_symbols"]
    );
    assert!(v["expansion_bytes"].as_u64().unwrap() > 0);
}
