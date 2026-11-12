//! A fleet export counts once.
//!
//! `POST /v1/patches/merge` and `POST /v1/immunity/merge` fold a peer's signed
//! export into this node's stores. The signature says who produced the
//! export, not how many times it was delivered, so anyone who has seen one
//! valid export (it is served by an unauthenticated `GET`) can deliver it
//! again. Each delivery used to count as a fresh confirmation: a patch
//! candidate's `verified_count` rose by one per replay, and a heal record's
//! evidence compounded.
//!
//! The worst case needed no peer at all: read a node's own export and post it
//! straight back. Every candidate confirmed itself, the store (and so the next
//! export) changed, and the loop could be repeated without limit.
//!
//! The contract these tests pin:
//! * an export that was merged before changes nothing when delivered again,
//!   and the response says `"already_merged": true`;
//! * a node's own export, read from its `GET` route and posted back to its
//!   merge route, changes nothing, however many round trips are made;
//! * a different export from the peer still merges;
//! * reading an export never rewrites the store: the record of what was
//!   handed out lives in a ledger file beside it, so a polled export route
//!   cannot drop a fix that a solve or a merge saved in between.

use std::path::{Path, PathBuf};

use axiom_engine::config::AxiomConfig;
use axiom_engine::heal_memory::{fingerprint, HealMemory};
use axiom_engine::inference::InferencePipeline;
use axiom_engine::patch_memory::{PatchMemory, RepoScope};
use axiom_engine::provenance::sign_export;
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use axum::Router;
use candle_core::Device;
use serde_json::Value;
use tower::ServiceExt;

const LOOPBACK_HOST: &str = "127.0.0.1:8080";
const FLEET_KEY: &[u8] = b"idempotence-test-fleet-key";

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

/// A fresh per-test state directory holding `heal_memory.json`; the patch
/// store lives next to it as `axiom_patch_memory.json`.
fn state_dir(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("axiom_merge_idem_{tag}_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    let heal = dir.join("heal_memory.json");
    let patches = dir.join("axiom_patch_memory.json");
    (dir, heal, patches)
}

async fn keyed_node(heal: &Path) -> Router {
    create_router(
        AppState::new(
            tokio::task::spawn_blocking(tiny_pipeline).await.unwrap(),
            "merge-idempotence".into(),
        )
        .with_heal_memory_path(Some(heal.to_path_buf()))
        .with_fleet_key(Some(FLEET_KEY.to_vec())),
    )
}

async fn get_export(node: &Router, uri: &str) -> String {
    let resp = node
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri(uri)
                .header("host", LOOPBACK_HOST)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "GET {uri}");
    String::from_utf8(
        to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap()
}

async fn post_merge(node: &Router, uri: &str, export: &str) -> Value {
    let resp = node
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(uri)
                .header("host", LOOPBACK_HOST)
                .header("content-type", "application/json")
                .body(Body::from(export.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    assert_eq!(status, StatusCode::OK, "POST {uri}: {json}");
    json
}

fn scope() -> RepoScope {
    RepoScope::new("path:some-repo", "src/a.rs")
}

/// The single candidate's confirmation count in the store at `patches`.
fn confirmations(patches: &Path) -> u32 {
    PatchMemory::load(patches).candidates("fp")[0].verified_count
}

/// The whole heal record for `fp`, as text, for an exact comparison.
fn record_text(heal: &Path, fp: &str) -> String {
    format!("{:?}", HealMemory::load(heal).record(fp).expect("record"))
}

// --- patches ------------------------------------------------------------------

#[tokio::test]
async fn a_patch_export_delivered_again_changes_nothing() {
    let (dir, heal, patches) = state_dir("patch_replay");
    // This node verified the fix itself, once.
    let mut local = PatchMemory::new();
    local.record_verified("fp", &scope(), "BROKEN\n", "FIXED\n");
    local.save(&patches).unwrap();
    // A peer verified the same fix and exported its store.
    let mut peer = PatchMemory::new();
    peer.record_verified("fp", &scope(), "BROKEN\n", "FIXED\n");
    let export = serde_json::to_string(&peer.export_signed(Some(FLEET_KEY))).unwrap();

    let node = keyed_node(&heal).await;
    let first = post_merge(&node, "/v1/patches/merge", &export).await;
    assert_eq!(first["reinforced"], 1, "{first}");
    assert_eq!(confirmations(&patches), 2, "this node and one peer");

    // The same export, delivered three more times.
    for delivery in 2..=4 {
        let again = post_merge(&node, "/v1/patches/merge", &export).await;
        assert_eq!(
            confirmations(&patches),
            2,
            "delivery {delivery} of one export counted as another confirmation: {again}"
        );
        assert_eq!(again["already_merged"], true, "{again}");
        assert_eq!(again["reinforced"], 0, "{again}");
        assert_eq!(again["new_candidates"], 0, "{again}");
    }

    // A restarted node (fresh router, same store on disk) still knows it.
    let restarted = keyed_node(&heal).await;
    let again = post_merge(&restarted, "/v1/patches/merge", &export).await;
    assert_eq!(confirmations(&patches), 2, "{again}");

    // A different export from the peer is new information and still merges.
    peer.record_verified("fp-later", &scope(), "BROKEN\n", "LATER\n");
    let next = serde_json::to_string(&peer.export_signed(Some(FLEET_KEY))).unwrap();
    let merged = post_merge(&restarted, "/v1/patches/merge", &next).await;
    assert_eq!(merged["already_merged"], false, "{merged}");
    assert_eq!(merged["new_candidates"], 1, "{merged}");

    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn a_nodes_own_patch_export_posted_back_changes_nothing() {
    let (dir, heal, patches) = state_dir("patch_reflect");
    let mut local = PatchMemory::new();
    local.record_verified("fp", &scope(), "BROKEN\n", "FIXED\n");
    local.save(&patches).unwrap();
    let node = keyed_node(&heal).await;

    // Read the export, post it back, read the (possibly changed) export, post
    // that back. No key and no peer is needed for any of it.
    let store_before = std::fs::read(&patches).unwrap();
    let mut exports = Vec::new();
    for round in 1..=4 {
        let export = get_export(&node, "/v1/patches").await;
        assert_eq!(
            std::fs::read(&patches).unwrap(),
            store_before,
            "round {round}: reading the export rewrote the store"
        );
        let answer = post_merge(&node, "/v1/patches/merge", &export).await;
        assert_eq!(
            confirmations(&patches),
            1,
            "round {round}: the node's own export confirmed its own fix: {answer}"
        );
        assert_eq!(answer["already_merged"], true, "{answer}");
        exports.push(export);
    }
    assert!(
        exports.iter().all(|e| e == &exports[0]),
        "the export must not change when nothing was learned"
    );
    // Nothing was written to the store at any point; what was handed out is
    // on record beside it. The store's own bookkeeping is never exported.
    assert_eq!(std::fs::read(&patches).unwrap(), store_before);
    assert!(dir.join("axiom_patch_memory.json.exports.json").is_file());
    assert!(!exports[0].contains("merged_exports"), "{}", exports[0]);

    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn a_merge_that_cannot_be_saved_leaves_no_record_and_a_retry_merges_it_once() {
    let (dir, heal, patches) = state_dir("patch_unsaved");
    let mut peer = PatchMemory::new();
    peer.record_verified("fp", &scope(), "BROKEN\n", "FIXED\n");
    let export = serde_json::to_string(&peer.export_signed(Some(FLEET_KEY))).unwrap();

    // The store cannot be saved: a directory sits where its file goes.
    std::fs::create_dir(&patches).unwrap();
    let node = keyed_node(&heal).await;
    for attempt in 1..=3 {
        let resp = node
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/patches/merge")
                    .header("host", LOOPBACK_HOST)
                    .header("content-type", "application/json")
                    .body(Body::from(export.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "attempt {attempt}"
        );
    }
    // The merge and the record of it are one file. Neither was written, and
    // nothing beside the store says otherwise.
    assert!(!dir.join("axiom_patch_memory.json.exports.json").exists());

    // Once the store can be saved, the export is new, and merges once.
    std::fs::remove_dir(&patches).unwrap();
    let merged = post_merge(&node, "/v1/patches/merge", &export).await;
    assert_eq!(merged["already_merged"], false, "{merged}");
    assert_eq!(merged["new_candidates"], 1, "{merged}");
    assert_eq!(confirmations(&patches), 1);
    let again = post_merge(&node, "/v1/patches/merge", &export).await;
    assert_eq!(again["already_merged"], true, "{again}");
    assert_eq!(confirmations(&patches), 1);

    // The record is in the store file, and only there.
    assert!(std::fs::read_to_string(&patches)
        .unwrap()
        .contains("merged_exports"));
    assert!(!dir.join("axiom_patch_memory.json.exports.json").exists());

    let _ = std::fs::remove_dir_all(dir);
}

// --- immunity -----------------------------------------------------------------

fn seeded_heal_memory(path: &Path, dir: &str) -> (HealMemory, String) {
    let fp = fingerprint("make", &[]);
    let mut mem = HealMemory::load(path);
    mem.remember_dirs(&fp, "make", &[PathBuf::from(dir)]);
    mem.reinforce_immunity(&fp, 1_700_000_000);
    (mem, fp)
}

#[tokio::test]
async fn an_immunity_export_delivered_again_changes_nothing() {
    let (dir, heal, _patches) = state_dir("immunity_replay");
    let (local, fp) = seeded_heal_memory(&heal, "/local/dir");
    local.save().unwrap();
    let (mut peer, _) = seeded_heal_memory(&dir.join("peer.json"), "/peer/dir");
    let export = serde_json::to_string(&sign_export(&peer.to_json(), Some(FLEET_KEY))).unwrap();

    let node = keyed_node(&heal).await;
    let first = post_merge(&node, "/v1/immunity/merge", &export).await;
    assert_eq!(first["programs_merged"], 1, "{first}");
    assert_eq!(first["dirs_added"], 1, "{first}");
    let after_first = record_text(&heal, &fp);

    for delivery in 2..=4 {
        let again = post_merge(&node, "/v1/immunity/merge", &export).await;
        assert_eq!(
            record_text(&heal, &fp),
            after_first,
            "delivery {delivery} of one export changed the record: {again}"
        );
        assert_eq!(again["already_merged"], true, "{again}");
        assert_eq!(again["programs_merged"], 0, "{again}");
    }

    // A different export from the peer still merges.
    peer.remember_dirs(&fp, "make", &[PathBuf::from("/peer/other")]);
    let next = serde_json::to_string(&sign_export(&peer.to_json(), Some(FLEET_KEY))).unwrap();
    let merged = post_merge(&node, "/v1/immunity/merge", &next).await;
    assert_eq!(merged["already_merged"], false, "{merged}");
    assert_eq!(merged["dirs_added"], 1, "{merged}");

    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn a_nodes_own_immunity_export_posted_back_changes_nothing() {
    let (dir, heal, _patches) = state_dir("immunity_reflect");
    let (local, fp) = seeded_heal_memory(&heal, "/local/dir");
    local.save().unwrap();
    let before = record_text(&heal, &fp);
    let store_before = std::fs::read(&heal).unwrap();
    let node = keyed_node(&heal).await;

    let mut exports = Vec::new();
    for round in 1..=4 {
        let export = get_export(&node, "/v1/immunity").await;
        assert_eq!(
            std::fs::read(&heal).unwrap(),
            store_before,
            "round {round}: reading the export rewrote the memory"
        );
        let answer = post_merge(&node, "/v1/immunity/merge", &export).await;
        assert_eq!(
            record_text(&heal, &fp),
            before,
            "round {round}: the node's own export merged into itself: {answer}"
        );
        assert_eq!(answer["already_merged"], true, "{answer}");
        exports.push(export);
    }
    assert!(
        exports.iter().all(|e| e == &exports[0]),
        "the export must not change when nothing was learned"
    );
    assert_eq!(std::fs::read(&heal).unwrap(), store_before);
    assert!(dir.join("heal_memory.json.exports.json").is_file());

    let _ = std::fs::remove_dir_all(dir);
}
