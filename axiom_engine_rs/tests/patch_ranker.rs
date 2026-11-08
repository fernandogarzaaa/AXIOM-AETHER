use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::patch_ranker::{FaultContext, PatchRanker};
use axiom_engine::poly_jit::PolyJitRunRequest;
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use candle_core::Device;
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
    AppState::new(pipeline, "axiom-patch-ranker-test".to_string())
}

#[test]
fn ranker_energies_and_selection_are_consistent() {
    let fault = FaultContext {
        stdout: String::new(),
        stderr: "compile error: Axiom Q-TTT multi fault".into(),
        status_code: Some(1),
        source: "Write-Error 'AXIOM_QTTT_MULTI_FAULT'; throw 'bad'; exit 1".into(),
    };
    let result = PatchRanker::default().rank(&fault).unwrap();
    assert_eq!(result.energies_tensor.dims()[0], result.energies.len());
    assert_eq!(result.telemetry.tensor_shape[0], 2);
    assert_eq!(result.telemetry.bond_dimension, 4);
    assert_eq!(result.telemetry.collapsed_branch, Some(1));
    assert!(result.telemetry.entropy_bits <= 1e-6);
    assert!((result.telemetry.collapse_probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!(result
        .selected_patch
        .unwrap()
        .contains("AXIOM_QTTT_FIXTURE_PASS"));
}

#[tokio::test]
async fn patch_ranker_state_endpoint_reports_default_inactive_state() {
    let state = test_state().await;
    let app = create_router(state);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/v1/hypervisor/patch_ranker_state")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(response["ranker"]["total_optimizations"], 0);
    assert_eq!(response["ranker"]["last_state"]["active"], false);
}

#[tokio::test]
async fn legacy_quantum_coherent_state_route_is_removed() {
    // Deprecated in 0.6.0, removed in 0.7.0 as announced: use patch_ranker_state.
    let state = test_state().await;
    let app = create_router(state);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/v1/hypervisor/quantum_coherent_state")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn jit_status_keeps_the_quantum_json_key_for_compatibility() {
    let state = test_state().await;
    let app = create_router(state);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/v1/hypervisor/jit_status")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(response["jit"]["quantum"].is_object(), "{response}");
}

#[tokio::test]
async fn poly_jit_ranker_repairs_multi_fault_script_and_updates_endpoint() {
    if !cfg!(windows) || std::process::Command::new("powershell").output().is_err() {
        return;
    }

    let root = std::env::temp_dir().join(format!("axiom-qttt-test-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let script = root.join("fault.ps1");
    std::fs::write(
        &script,
        "Write-Error 'AXIOM_QTTT_MULTI_FAULT compile marker'\nthrow 'AXIOM_QTTT_MULTI_FAULT runtime marker'\nexit 1\n",
    )
    .unwrap();

    let state = test_state().await;
    let request = PolyJitRunRequest {
        session_id: "ranker-polyjit-session".to_string(),
        command: "powershell".to_string(),
        args: vec![
            "-NoProfile".into(),
            "-ExecutionPolicy".into(),
            "Bypass".into(),
            "-File".into(),
            script.display().to_string(),
        ],
        working_dir: Some(root.display().to_string()),
        source_path: Some(script.display().to_string()),
    };

    let report = state
        .poly_jit
        .run_with_feedback(request, |_diag| async { Ok(()) })
        .await
        .unwrap();
    assert!(report.passed);
    assert!(report.patched);
    assert_eq!(report.attempts, 2);
    assert!(std::fs::read_to_string(&script)
        .unwrap()
        .contains("AXIOM_QTTT_FIXTURE_PASS"));

    let app = create_router(state);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/v1/hypervisor/patch_ranker_state")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(response["ranker"]["total_optimizations"], 1);
    assert_eq!(response["ranker"]["total_collapses"], 1);
    assert_eq!(response["ranker"]["last_state"]["active"], true);
    assert_eq!(response["ranker"]["last_state"]["tensor_shape"][0], 2);
    assert_eq!(response["ranker"]["last_state"]["collapsed_branch"], 1);

    let _ = std::fs::remove_dir_all(root);
}
