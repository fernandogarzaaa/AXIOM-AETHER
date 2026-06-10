//! Swarm immunity: heals learned on one node immunize the whole fleet.

use std::path::PathBuf;

use axiom_engine::config::AxiomConfig;
use axiom_engine::heal_memory::{fingerprint, HealMemory};
use axiom_engine::inference::InferencePipeline;
use axiom_engine::self_heal::{run_supervised, Heal, SupervisorOptions};
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use candle_core::Device;
use serde_json::Value;
use tower::ServiceExt;

fn tiny_pipeline() -> InferencePipeline {
    let cfg = AxiomConfig {
        d_model: 16,
        n_layers: 2,
        vocab_size: 64,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    };
    InferencePipeline::new(cfg, Device::Cpu).expect("tiny pipeline must build")
}

fn unique_tmp(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("axiom_swarm_{tag}_{nanos}"))
}

fn supervise(
    cmd: String,
    args: Vec<String>,
    heal_memory: PathBuf,
) -> axiom_engine::self_heal::RunReport {
    let pipeline = tiny_pipeline();
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || {
            let opts = SupervisorOptions {
                max_restarts: 3,
                heal_memory_path: Some(heal_memory),
                ..SupervisorOptions::default()
            };
            run_supervised(&pipeline, &cmd, &args, &opts).unwrap()
        })
        .unwrap()
        .join()
        .unwrap()
}

#[tokio::test]
async fn immunity_endpoints_export_and_merge() {
    // Node A has learned something.
    let mem_a = unique_tmp("node_a").with_extension("json");
    let fp = fingerprint("prog", &[]);
    let mut a = HealMemory::load(&mem_a);
    a.remember_dirs(&fp, "prog", &[PathBuf::from("/tmp/axiom_swarm_shared_dir")]);
    a.save().unwrap();

    let app_a = create_router(
        AppState::new(
            tokio::task::spawn_blocking(tiny_pipeline).await.unwrap(),
            "node-a".into(),
        )
        .with_heal_memory_path(Some(mem_a.clone())),
    );

    // Export from A.
    let resp = app_a
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/immunity")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let exported = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let exported = String::from_utf8(exported.to_vec()).unwrap();
    assert!(exported.contains(&fp), "export must contain node A's program");

    // Merge into fresh node B.
    let mem_b = unique_tmp("node_b").with_extension("json");
    let app_b = create_router(
        AppState::new(
            tokio::task::spawn_blocking(tiny_pipeline).await.unwrap(),
            "node-b".into(),
        )
        .with_heal_memory_path(Some(mem_b.clone())),
    );
    let resp = app_b
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/immunity/merge")
                .header("content-type", "application/json")
                .body(Body::from(exported))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["programs_added"], 1);

    // Node B's persisted memory now knows the program.
    let b = HealMemory::load(&mem_b);
    assert!(b.record(&fp).is_some());

    let _ = std::fs::remove_file(&mem_a);
    let _ = std::fs::remove_file(&mem_b);
}

#[tokio::test]
async fn immunity_endpoints_disabled_without_path() {
    let app = create_router(AppState::new(
        tokio::task::spawn_blocking(tiny_pipeline).await.unwrap(),
        "no-mem".into(),
    ));
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/immunity")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
fn herd_immunity_end_to_end() {
    // Machine A: the program crashes on a missing dir, heals, succeeds, learns.
    let base = unique_tmp("herd_env");
    let out_dir = base.join("out");
    let script = format!("echo fleet-output > {}", out_dir.join("r.txt").display());
    let args = vec!["-c".to_string(), script];
    let mem_a = unique_tmp("herd_a").with_extension("json");

    let first = supervise("sh".into(), args.clone(), mem_a.clone());
    assert!(first.success);
    assert_eq!(first.heals, vec![Heal::CreatedDirectory(out_dir.clone())]);

    // The fleet syncs: B merges A's exported memory (what /v1/immunity ships).
    let mem_b = unique_tmp("herd_b").with_extension("json");
    let mut b = HealMemory::load(&mem_b);
    b.merge_json(&HealMemory::load(&mem_a).to_json()).unwrap();
    b.save().unwrap();

    // Machine B is a fresh environment — the program has NEVER run here.
    std::fs::remove_dir_all(&base).unwrap();
    assert!(!out_dir.exists());

    let second = supervise("sh".into(), args, mem_b.clone());
    assert!(second.success);
    assert_eq!(
        second.attempts, 1,
        "herd immunity: first-try success on a machine that never saw the failure"
    );
    assert_eq!(second.heals, vec![Heal::Immunized(out_dir)]);
    assert_eq!(second.tokens_absorbed, 0);

    let _ = std::fs::remove_dir_all(&base);
    let _ = std::fs::remove_file(&mem_a);
    let _ = std::fs::remove_file(&mem_b);
}
