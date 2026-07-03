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
    // Heavy enough that the fingerprint replacement actually SHRINKS the body —
    // the expansion guard forwards tiny payloads untouched by design.
    let heavy_history =
        "historical assistant answer with useful context about the build system ".repeat(80);
    let original = json!({"model":"gpt-5.5","input":[
        {"role":"assistant","content":heavy_history},
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

#[tokio::test]
async fn recording_persists_scrubbed_exchange_when_enabled() {
    let rec_dir = std::env::temp_dir().join(format!(
        "axiom_recproxy_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::env::set_var("AXIOM_SESSION_RECORD", "1");
    std::env::set_var("AXIOM_SESSIONS_DIR", &rec_dir);

    async fn upstream(Json(_body): Json<Value>) -> (StatusCode, Json<Value>) {
        (StatusCode::OK, Json(json!({"id":"resp_rec","object":"response"})))
    }
    let upstream_app = Router::new().route("/v1/responses", post(upstream));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, upstream_app).await.unwrap() });

    let test_pipeline = tokio::task::spawn_blocking(pipeline).await.unwrap();
    let state = AppState::new(test_pipeline, "test".into())
        .with_openai_forwarder(Some(OpenAiForwarder::new(None, Some(base))));
    let app = create_router(state);
    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer sk-secret-abcdefghijklmnop")
                .header("x-axiom-session-id", "rectest")
                .body(Body::from(
                    json!({"model":"gpt-5.5","input":"hello"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // The append happens on the blocking pool; give it a moment.
    let file = rec_dir.join("rectest.jsonl");
    let mut records = Vec::new();
    for _ in 0..50 {
        if file.exists() {
            records = axiom_engine::session_recorder::read_session(&file).unwrap();
            if !records.is_empty() {
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    std::env::remove_var("AXIOM_SESSION_RECORD");
    std::env::remove_var("AXIOM_SESSIONS_DIR");
    let _ = std::fs::remove_dir_all(&rec_dir);
    task.abort();

    assert_eq!(records.len(), 1, "one exchange must be recorded");
    assert_eq!(records[0].endpoint, "/v1/responses");
    assert_eq!(records[0].session_id, "rectest");
    assert_eq!(records[0].response["streamed"], true);
    assert_eq!(records[0].response["status"], 200);
}
