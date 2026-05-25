use axiom_engine::cluster::StateDeltaUpdate;
use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use candle_core::{DType, Device, Tensor};
use serde_json::json;
use tower::ServiceExt;

fn build_pipeline() -> InferencePipeline {
    let config = AxiomConfig {
        d_model: 16,
        n_layers: 1,
        vocab_size: 64,
        lr_inner: 1e-3,
        rms_norm_eps: 1e-6,
    };
    InferencePipeline::new(config, Device::Cpu).expect("pipeline init")
}

async fn create_session_id(app: &axum::Router) -> String {
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/sessions")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    json["session_id"].as_str().unwrap().to_string()
}

fn assert_structured_status(status: StatusCode) {
    assert!(
        matches!(
            status,
            StatusCode::OK
                | StatusCode::ACCEPTED
                | StatusCode::BAD_REQUEST
                | StatusCode::CONFLICT
                | StatusCode::UNPROCESSABLE_ENTITY
                | StatusCode::NOT_FOUND
        ),
        "unexpected unstructured status: {status}"
    );
}

#[tokio::test]
async fn test_production_under_chaos() {
    let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
    let state = AppState::new(pipeline, "axiom-ttt-chaos".to_string());
    let app = create_router(state.clone());

    let base_session = create_session_id(&app).await;

    // 1) Concurrency blast: 50 concurrent request tasks.
    let mut concurrency_tasks = Vec::new();
    for i in 0..50usize {
        let app_cloned = app.clone();
        let sid = base_session.clone();
        concurrency_tasks.push(tokio::spawn(async move {
            if i % 2 == 0 {
                let req = Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages": [{"role":"user","content": format!("blast-{i}")}],
                            "max_tokens": 2,
                            "session_id": sid,
                        })
                        .to_string(),
                    ))
                    .unwrap();
                app_cloned.oneshot(req).await.unwrap().status()
            } else {
                let req = Request::builder()
                    .method(Method::POST)
                    .uri("/v1/adapt")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "corpus": [format!("adapt-blast-{i}")],
                            "steps": 2,
                            "session_id": sid,
                        })
                        .to_string(),
                    ))
                    .unwrap();
                app_cloned.oneshot(req).await.unwrap().status()
            }
        }));
    }
    for task in concurrency_tasks {
        let status = task.await.expect("concurrency task panic");
        assert_structured_status(status);
    }

    // 2) Out-of-order delta spike against deterministic sequence gate.
    let delta = Tensor::ones((16usize, 16usize), DType::F32, &state.device)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let delta_bytes = safetensors::serialize([("tensor", &delta)], &None).unwrap();
    let mut order_statuses = Vec::new();
    for (sequence_version, timestamp) in [(3u64, 300i64), (2, 200), (4, 400), (4, 399)] {
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/cluster/sync")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&StateDeltaUpdate {
                    session_id: base_session.clone(),
                    layer_index: 0,
                    sequence_version,
                    timestamp,
                    delta_bytes: delta_bytes.clone(),
                })
                .unwrap(),
            ))
            .unwrap();
        let status = app.clone().oneshot(req).await.unwrap().status();
        assert_structured_status(status);
        order_statuses.push(status);
    }
    assert!(
        order_statuses.contains(&StatusCode::CONFLICT),
        "expected stale delta conflict status"
    );

    // 3) VRAM starvation loop: create 40 sessions and hammer them to trigger LRU offload/reload.
    let mut many_sessions = Vec::new();
    for _ in 0..40usize {
        many_sessions.push(create_session_id(&app).await);
    }
    for sid in &many_sessions {
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "messages": [{"role":"user","content":"rapid alternating prompt block"}],
                    "max_tokens": 1,
                    "session_id": sid,
                })
                .to_string(),
            ))
            .unwrap();
        let status = app.clone().oneshot(req).await.unwrap().status();
        assert_structured_status(status);
    }

    // 4) Malformed payload attack mix.
    let malformed_cases = vec![
        (
            "/v1/chat/completions",
            json!({"messages":[{"role":"user","content":""}],"max_tokens":1,"session_id":base_session}).to_string(),
        ),
        ("/v1/adapt", json!({"corpus":[]}).to_string()),
        (
            "/v1/chat/completions",
            json!({"messages":[{"role":"user","content":"NaN Inf -Inf 1e9999"}],"max_tokens":1}).to_string(),
        ),
        ("/v1/chat/completions", "not-json".to_string()),
    ];
    for (uri, body) in malformed_cases {
        let req = Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let status = app.clone().oneshot(req).await.unwrap().status();
        assert_structured_status(status);
    }

    let invalid_delta_req = Request::builder()
        .method(Method::POST)
        .uri("/v1/cluster/sync")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&StateDeltaUpdate {
                session_id: many_sessions[0].clone(),
                layer_index: 0,
                sequence_version: 99,
                timestamp: 999,
                delta_bytes: vec![1, 2, 3, 4, 5],
            })
            .unwrap(),
        ))
        .unwrap();
    let invalid_delta_status = app
        .clone()
        .oneshot(invalid_delta_req)
        .await
        .unwrap()
        .status();
    assert_eq!(invalid_delta_status, StatusCode::BAD_REQUEST);

    // Tensor integrity probe: checkpoint several sessions and assert finite values.
    for sid in many_sessions.iter().take(10) {
        let req = Request::builder()
            .method(Method::GET)
            .uri(format!("/v1/sessions/{sid}/checkpoint"))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let checkpoint: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let layers = checkpoint["layers"].as_array().unwrap();
        for layer in layers {
            let data = layer["data"].as_array().unwrap();
            assert!(!data.is_empty(), "layer data should not be empty");
            for value in data {
                let v = value.as_f64().unwrap();
                assert!(v.is_finite(), "non-finite layer checkpoint value detected");
            }
        }
    }
}
