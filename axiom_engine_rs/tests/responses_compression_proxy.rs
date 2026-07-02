use std::sync::{Arc, Mutex};

use axiom_engine::config::AxiomConfig;
use axiom_engine::context_compressor::CompressorConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::openai_forwarder::OpenAiForwarder;
use axiom_engine::server::{create_router, AppState};
use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use candle_core::Device;
use serde_json::{json, Value};
use tower::ServiceExt;

fn pipeline() -> InferencePipeline {
    InferencePipeline::new(
        AxiomConfig {
            d_model: 16,
            n_layers: 1,
            vocab_size: 64,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        },
        Device::Cpu,
    )
    .unwrap()
}

#[tokio::test]
async fn compressed_bad_request_retries_original_and_preserves_structural_items() {
    std::env::set_var("AXIOM_RESPONSES_COMPRESS", "1");
    let captured = Arc::new(Mutex::new(Vec::<Value>::new()));
    async fn upstream(
        State(captured): State<Arc<Mutex<Vec<Value>>>>,
        Json(body): Json<Value>,
    ) -> (StatusCode, Json<Value>) {
        let mut requests = captured.lock().unwrap();
        requests.push(body);
        if requests.len() == 1 {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({"error":"reject fingerprint"})),
            )
        } else {
            (
                StatusCode::OK,
                Json(json!({"id":"resp_ok","object":"response"})),
            )
        }
    }
    let upstream_app = Router::new()
        .route("/v1/responses", post(upstream))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, upstream_app).await.unwrap() });

    let test_pipeline = tokio::task::spawn_blocking(pipeline).await.unwrap();
    let state = AppState::new(test_pipeline, "test".into())
        .with_openai_forwarder(Some(OpenAiForwarder::new(None, Some(base))))
        .with_compressor_config(CompressorConfig {
            enabled: true,
            heavy_message_threshold_tokens: 1,
            recall_top_k: 8,
        });
    let app = create_router(state);
    let reasoning = json!({"type":"reasoning","id":"r1","encrypted_content":"opaque","summary":[]});
    let original = json!({"model":"gpt-5.5","input":[
        {"role":"assistant","content":"historical assistant answer with useful context"},
        reasoning,
        {"type":"function_call_output","call_id":"c1","output":"unchanged"},
        {"role":"user","content":"continue"}
    ]});
    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer test")
                .body(Body::from(original.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let requests = captured.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[0]["input"][0]["content"]
        .as_str()
        .unwrap()
        .contains("<axiom_context_fingerprint"));
    assert_eq!(requests[0]["input"][1], original["input"][1]);
    assert_eq!(requests[0]["input"][2], original["input"][2]);
    assert_eq!(requests[0]["input"][3], original["input"][3]);
    assert_eq!(requests[1], original);
    task.abort();
    std::env::remove_var("AXIOM_RESPONSES_COMPRESS");
}
