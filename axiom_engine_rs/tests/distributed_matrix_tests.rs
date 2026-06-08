use axiom_engine::config::AxiomConfig;
use axiom_engine::dwe::{
    apply_fragment, deserialize_fragment, extract_delta_fragment, serialize_fragment,
};
use axiom_engine::inference::InferencePipeline;
use axiom_engine::server::{create_router, AppState};
use axiom_engine::surprisal::{ExactAttentionResidualCache, DEFAULT_SURPRISAL_TAU};
use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use candle_core::{DType, Device, Tensor};
use serde_json::json;
use tower::ServiceExt;
use uuid::Uuid;

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

async fn test_state() -> AppState {
    let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
    AppState::new(pipeline, "axiom-distributed-test".to_string())
}

#[test]
fn dwe_fragment_roundtrips_binary_tensor_delta() {
    let device = Device::Cpu;
    let baseline = vec![Tensor::zeros((2, 2), DType::F32, &device).unwrap()];
    let optimized = vec![Tensor::ones((2, 2), DType::F32, &device).unwrap()];
    let fragment = extract_delta_fragment("dwe-test", 7, &optimized, &baseline).unwrap();
    let encoded = serialize_fragment(&fragment).unwrap();
    assert!(encoded.len() < 4096);
    let decoded = deserialize_fragment(&encoded).unwrap();
    let mut local = baseline;
    apply_fragment(&mut local, &decoded, &device).unwrap();
    assert_eq!(
        local[0].to_vec2::<f32>().unwrap(),
        vec![vec![1.0, 1.0], vec![1.0, 1.0]]
    );
}

#[test]
fn surprisal_cache_isolates_exact_identifier_tokens() {
    let cache = ExactAttentionResidualCache::new(32, DEFAULT_SURPRISAL_TAU);
    let token_ids = vec![1, 2, 3, 4, 5];
    let (_fast, report) = cache.route_tokens(
        "sr-session",
        &token_ids,
        "stable prose schema_key 9F4C2E7A10B34898AABBCCDDEEFF0011",
    );
    assert!(report.exact_residual_tokens >= 1);
    assert!(cache.telemetry().estimated_bytes <= 32 * 1024 * 1024);
    assert!(cache
        .residual_prompt("sr-session", 8)
        .contains("9F4C2E7A10B34898AABBCCDDEEFF0011"));
}

#[tokio::test]
async fn swarm_matrix_state_endpoint_reports_budget_and_queues() {
    let state = test_state().await;
    let app = create_router(state);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/v1/swarm/matrix_state")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(response["matrix"]["within_vram_budget"], true);
    assert_eq!(response["matrix"]["vram_budget_bytes"], 5_200_000_000_u64);
    assert_eq!(response["dwe"]["queued_fragments"], 0);
}

#[tokio::test]
async fn vfs_mount_updates_localized_swarm_domain() {
    let root = std::env::temp_dir().join(format!("axiom-rust-swarm-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("Cargo.toml"), "[package]\nname='fixture'\n").unwrap();
    std::fs::write(root.join("lib.rs"), "pub fn fixture() -> i32 { 1 }\n").unwrap();

    let state = test_state().await;
    let app = create_router(state);
    let body = json!({
        "root": root.to_string_lossy(),
        "session_id": "swarm-vfs-session",
        "warm_paths": ["lib.rs"]
    });
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/hypervisor/mount")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let req = Request::builder()
        .method(Method::GET)
        .uri("/v1/swarm/matrix_state")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(response["matrix"]["active_domain"], "rust_compile");
    assert_eq!(response["matrix"]["within_vram_budget"], true);

    let _ = std::fs::remove_dir_all(root);
}
