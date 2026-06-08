use std::time::Duration;

use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::poly_jit::{PolyJitEngine, PolyJitRunRequest};
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use candle_core::Device;
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
    AppState::new(pipeline, "axiom-hypervisor-test".to_string())
}

#[tokio::test]
async fn vfs_mount_warm_read_populates_ttt_cache() {
    let root = std::env::temp_dir().join(format!("axiom-vfs-test-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("lib.rs"),
        "pub struct HypervisorFixture { value: i32 }\nimpl HypervisorFixture { pub fn value(&self) -> i32 { self.value } }",
    )
    .unwrap();

    let state = test_state().await;
    let app = create_router(state.clone());
    let body = json!({
        "root": root.to_string_lossy(),
        "session_id": "hypervisor-vfs-session",
        "warm_paths": ["lib.rs"]
    });
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/hypervisor/mount")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(response["mount"]["mode"], "user_loopback");
    assert_eq!(response["warmed"].as_array().unwrap().len(), 1);
    assert!(response["warmed"][0]["digest_tokens"].as_u64().unwrap() > 0);
    assert_eq!(state.ttt_sessions.len(), 1);

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn jit_status_endpoint_reports_defaults() {
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
    assert_eq!(response["jit"]["total_runs"], 0);
    assert_eq!(response["vfs"]["mode"], "user_loopback");
}

#[tokio::test]
async fn poly_jit_repairs_faulted_script_and_finishes() {
    if !cfg!(windows) || std::process::Command::new("powershell").output().is_err() {
        return;
    }

    let root = std::env::temp_dir().join(format!("axiom-polyjit-test-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let script = root.join("fault.ps1");
    std::fs::write(
        &script,
        "$source = Get-Content $PSCommandPath -Raw\nif ($source.Contains('AXIOM_POLYJIT_FIXTURE_FAIL')) { Write-Error 'runtime fault'; exit 1 }\nWrite-Output 'AXIOM_POLYJIT_FIXTURE_FAIL'\nexit 0\n",
    )
    .unwrap();

    let engine = PolyJitEngine::new(Duration::from_secs(20));
    let request = PolyJitRunRequest {
        session_id: "polyjit-session".to_string(),
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
    let mut diagnostics = Vec::new();
    let report = engine
        .run_with_feedback(request, |diag| {
            diagnostics.push(diag);
            async { Ok(()) }
        })
        .await
        .unwrap();

    assert!(report.passed);
    assert!(report.patched);
    assert_eq!(report.attempts, 2);
    assert_eq!(diagnostics.len(), 1);
    let patched = std::fs::read_to_string(&script).unwrap();
    assert!(patched.contains("AXIOM_POLYJIT_FIXTURE_PASS"));
    assert_eq!(engine.status().repaired_runs, 1);

    let _ = std::fs::remove_dir_all(root);
}
