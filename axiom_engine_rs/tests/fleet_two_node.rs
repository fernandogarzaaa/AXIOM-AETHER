use axiom_engine::config::AxiomConfig;
use axiom_engine::dwe::{DweBus, DweFragment, DweLayerDelta, start_dwe_listener};
use axiom_engine::inference::InferencePipeline;
use axiom_engine::server::{AppState, create_router, start_dwe_apply_loop};
use axum::body::{Body, to_bytes};
use axum::http::{Method, Request, StatusCode};
use candle_core::Device;
use serde_json::json;
use tokio::sync::mpsc;
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

async fn test_state(model: &str, bus: DweBus) -> AppState {
    let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
    AppState::new(pipeline, model.to_string()).with_dwe_bus(bus)
}

fn reserve_addr() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr.to_string()
}

async fn wait_for_listener(addr: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while std::time::Instant::now() < deadline {
        if tokio::net::TcpStream::connect(addr).await.is_ok() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("DWE listener did not open at {addr}");
}

async fn wait_for<F>(mut predicate: F)
where
    F: FnMut() -> bool,
{
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while std::time::Instant::now() < deadline {
        if predicate() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("condition did not become true");
}

async fn create_session(app: axum::Router) -> String {
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/sessions")
                .header("content-type", "application/json")
                .body(Body::from(json!({}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
    body["session_id"].as_str().unwrap().to_string()
}

async fn checkpoint(app: axum::Router, session_id: &str) -> serde_json::Value {
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri(format!("/v1/sessions/{session_id}/checkpoint"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap()
}

async fn get_json(app: axum::Router, uri: &str) -> serde_json::Value {
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap()
}

async fn get_text(app: axum::Router, uri: &str) -> String {
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    String::from_utf8(
        to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap()
}

fn fragment(session_id: &str, sequence: u64) -> DweFragment {
    DweFragment {
        schema: "axiom.dwe.v1".into(),
        session_id: session_id.to_string(),
        sequence,
        layers: vec![DweLayerDelta {
            layer_index: 0,
            shape: vec![16, 16],
            values: vec![0.5; 16 * 16],
        }],
        state_hash: format!("state-{sequence}"),
        hmac: None,
    }
}

#[tokio::test]
async fn two_node_dwe_applies_signed_update_and_counts_rejection() {
    let addr = reserve_addr();
    let node_b = test_state("node-b", DweBus::disabled()).await;
    let telemetry_b = node_b.dwe_bus.telemetry_handle();
    let (tx, rx) = mpsc::channel(8);
    start_dwe_apply_loop(
        node_b.clone(),
        rx,
        b"shared-key".to_vec(),
        Some(b"old-key".to_vec()),
        telemetry_b.clone(),
    );
    let listener_addr = addr.clone();
    let listener_telemetry = telemetry_b.clone();
    tokio::spawn(async move {
        let _ = start_dwe_listener(&listener_addr, tx, listener_telemetry).await;
    });
    wait_for_listener(&addr).await;

    let app_b = create_router(node_b.clone());
    let session_id = create_session(app_b.clone()).await;

    let node_a_bus = DweBus::new_with_signing_key(vec![addr.clone()], Some(b"shared-key".to_vec()));
    let good = fragment(&session_id, 1);
    node_a_bus.broadcast(good);

    wait_for(|| node_b.dwe_bus.telemetry().applied_fragments == 1).await;
    let after = checkpoint(app_b.clone(), &session_id).await;
    let data = after["layers"][0]["data"].as_array().unwrap();
    assert_eq!(data[1].as_f64().unwrap(), 0.5);
    assert_eq!(node_b.dwe_bus.telemetry().received_fragments, 1);

    let wrong_key_bus = DweBus::new_with_signing_key(vec![addr], Some(b"wrong-key".to_vec()));
    let wrong = fragment(&session_id, 2);
    wrong_key_bus.broadcast(wrong);
    wait_for(|| node_b.dwe_bus.telemetry().rejected_fragments == 1).await;
    assert_eq!(node_b.dwe_bus.telemetry().applied_fragments, 1);

    let status = get_json(app_b.clone(), "/v1/fleet/status").await;
    assert_eq!(status["dwe"]["applied_fragments"], 1);
    assert_eq!(status["dwe"]["rejected_fragments"], 1);

    let metrics = get_text(app_b, "/metrics").await;
    assert!(metrics.contains("axiom_dwe_sent"));
    assert!(metrics.contains("axiom_dwe_received 2"));
    assert!(metrics.contains("axiom_dwe_applied 1"));
    assert!(metrics.contains("axiom_dwe_rejected 1"));
}
