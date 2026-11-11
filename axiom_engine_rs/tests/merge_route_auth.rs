//! Fleet merge routes must authenticate their input.
//!
//! `POST /v1/patches/merge` and `POST /v1/immunity/merge` fold a peer's
//! export into this node's on-disk stores. A stored patch candidate is later
//! written into a repository and executed by `axiom solve` (the verify run),
//! and a stored immunity record is injected into upstream prompts, so an
//! unauthenticated merge is remote code execution / prompt injection.
//!
//! The contract these tests pin:
//! * with no fleet key configured the routes answer 401 and store nothing,
//!   whatever the body (unsigned export, raw payload, `text/plain`);
//! * with a fleet key, only an export carrying a valid HMAC is merged;
//! * a request whose `Host` is not on the allowlist, or that carries an
//!   `Origin` not on the allowlist, is refused before the key is consulted.

use std::path::PathBuf;

use axiom_engine::config::AxiomConfig;
use axiom_engine::heal_memory::{fingerprint, HealMemory};
use axiom_engine::inference::InferencePipeline;
use axiom_engine::patch_memory::{PatchMemory, RepoScope};
use axiom_engine::provenance::{sha256_hex, sign_export};
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use candle_core::Device;
use serde_json::Value;
use tower::ServiceExt;

const LOOPBACK_HOST: &str = "127.0.0.1:8080";

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
    let dir = std::env::temp_dir().join(format!("axiom_merge_auth_{tag}_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    let heal = dir.join("heal_memory.json");
    let patches = dir.join("axiom_patch_memory.json");
    (dir, heal, patches)
}

async fn keyless_state(heal: &std::path::Path) -> AppState {
    AppState::new(
        tokio::task::spawn_blocking(tiny_pipeline).await.unwrap(),
        "merge-auth".into(),
    )
    .with_heal_memory_path(Some(heal.to_path_buf()))
}

/// The attacker's patch store: one candidate for `calc.py` whose content is
/// arbitrary code, exactly the shape `axiom solve` would later write and run.
fn hostile_patch_payload() -> String {
    let mut pm = PatchMemory::new();
    pm.record_verified(
        &fingerprint("python3", &["test_calc.py".to_string()]),
        &RepoScope::new("path:victim-repo", "calc.py"),
        "def average(xs):\n    return sum(xs) / (len(xs) + 1)\n",
        "import os\nos.system('touch /tmp/pwned')\n",
    );
    serde_json::to_string(&pm).unwrap()
}

/// A hostile heal memory: one program whose "established fix" is attacker text.
fn hostile_immunity_payload() -> String {
    let mut mem = HealMemory::load(std::env::temp_dir().join("axiom_merge_auth_never_saved.json"));
    mem.remember_dirs(
        &fingerprint("make", &[]),
        "make",
        &[PathBuf::from("/attacker/controlled")],
    );
    mem.to_json()
}

/// An export with a correct SHA-256 and no HMAC: the hash proves nothing about
/// who sent it, since anyone can compute it.
fn unsigned_export(payload: &str) -> String {
    serde_json::json!({
        "schema": "axiom_signed_export_v1",
        "payload": payload,
        "sha256": sha256_hex(payload.as_bytes()),
    })
    .to_string()
}

fn post(uri: &str, content_type: &str, body: String) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("host", LOOPBACK_HOST)
        .header("content-type", content_type)
        .body(Body::from(body))
        .unwrap()
}

async fn status_and_json(resp: axum::response::Response) -> (StatusCode, Value) {
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

// --- no fleet key configured: nothing is ever merged -------------------------

#[tokio::test]
async fn patches_merge_without_fleet_key_is_401_and_stores_nothing() {
    let (dir, heal, patches) = state_dir("patches_nokey");
    let app = create_router(keyless_state(&heal).await);

    // The exact request a web page can send with no CORS preflight: a
    // `text/plain` POST of an export whose hash is correct but unsigned.
    let resp = app
        .oneshot(post(
            "/v1/patches/merge",
            "text/plain",
            unsigned_export(&hostile_patch_payload()),
        ))
        .await
        .unwrap();
    let (status, body) = status_and_json(resp).await;

    assert_eq!(
        status,
        StatusCode::UNAUTHORIZED,
        "an unsigned export must not merge on a node with no fleet key (body: {body})"
    );
    assert!(
        PatchMemory::load(&patches).is_empty(),
        "a rejected merge must leave the patch store untouched"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn immunity_merge_without_fleet_key_is_401_and_stores_nothing() {
    let (dir, heal, _patches) = state_dir("immunity_nokey");
    let fp = fingerprint("make", &[]);

    // Both body shapes the route used to accept with no key: the hash-only
    // export, and the raw heal-memory JSON ("back-compat").
    for (label, body) in [
        (
            "unsigned export",
            unsigned_export(&hostile_immunity_payload()),
        ),
        ("raw payload", hostile_immunity_payload()),
    ] {
        let app = create_router(keyless_state(&heal).await);
        let resp = app
            .oneshot(post("/v1/immunity/merge", "text/plain", body))
            .await
            .unwrap();
        let (status, json) = status_and_json(resp).await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "{label} must not merge on a node with no fleet key (body: {json})"
        );
        assert!(
            HealMemory::load(&heal).record(&fp).is_none(),
            "{label}: a rejected merge must leave heal memory untouched"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn an_export_signed_with_some_other_key_is_401_on_a_keyless_node() {
    // A node with no key cannot check an HMAC, so a "signed" export is not
    // evidence of anything either.
    let (dir, heal, patches) = state_dir("patches_foreignsig");
    let app = create_router(keyless_state(&heal).await);
    let export = sign_export(&hostile_patch_payload(), Some(b"attacker-chosen-key"));

    let resp = app
        .oneshot(post(
            "/v1/patches/merge",
            "application/json",
            serde_json::to_string(&export).unwrap(),
        ))
        .await
        .unwrap();
    let (status, _) = status_and_json(resp).await;

    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(PatchMemory::load(&patches).is_empty());
    let _ = std::fs::remove_dir_all(dir);
}

// --- fleet key configured: only a valid HMAC merges --------------------------

const FLEET_KEY: &[u8] = b"shared-fleet-secret";

async fn keyed_state(heal: &std::path::Path) -> AppState {
    keyless_state(heal)
        .await
        .with_fleet_key(Some(FLEET_KEY.to_vec()))
}

fn friendly_patch_payload() -> String {
    let mut pm = PatchMemory::new();
    pm.record_verified(
        "fp-friendly",
        &RepoScope::new("path:some-repo", "src/a.rs"),
        "BROKEN\n",
        "FIXED\n",
    );
    serde_json::to_string(&pm).unwrap()
}

#[tokio::test]
async fn patches_merge_with_fleet_key_accepts_only_a_valid_hmac() {
    let (dir, heal, patches) = state_dir("patches_keyed");
    let payload = friendly_patch_payload();

    // Unsigned, wrong key, and tampered-after-signing are all refused.
    let mut tampered = sign_export(&payload, Some(FLEET_KEY));
    tampered.payload = tampered.payload.replace("FIXED", "EVIL!");
    for (label, body) in [
        ("unsigned", unsigned_export(&payload)),
        (
            "wrong key",
            serde_json::to_string(&sign_export(&payload, Some(b"not-the-fleet-key"))).unwrap(),
        ),
        ("tampered", serde_json::to_string(&tampered).unwrap()),
    ] {
        let app = create_router(keyed_state(&heal).await);
        let resp = app
            .oneshot(post("/v1/patches/merge", "application/json", body))
            .await
            .unwrap();
        let (status, json) = status_and_json(resp).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{label}: {json}");
        assert!(
            PatchMemory::load(&patches).is_empty(),
            "{label} stored a candidate"
        );
    }

    // The same payload signed with the fleet key merges.
    let app = create_router(keyed_state(&heal).await);
    let good = serde_json::to_string(&sign_export(&payload, Some(FLEET_KEY))).unwrap();
    let resp = app
        .oneshot(post("/v1/patches/merge", "application/json", good))
        .await
        .unwrap();
    let (status, json) = status_and_json(resp).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["new_candidates"], 1);
    assert_eq!(
        PatchMemory::load(&patches).candidates("fp-friendly").len(),
        1
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn immunity_merge_with_fleet_key_accepts_only_a_valid_hmac() {
    let (dir, heal, _patches) = state_dir("immunity_keyed");
    let fp = fingerprint("make", &[]);
    let payload = hostile_immunity_payload();

    for (label, body) in [
        ("unsigned export", unsigned_export(&payload)),
        ("raw payload", payload.clone()),
        (
            "wrong key",
            serde_json::to_string(&sign_export(&payload, Some(b"not-the-fleet-key"))).unwrap(),
        ),
    ] {
        let app = create_router(keyed_state(&heal).await);
        let resp = app
            .oneshot(post("/v1/immunity/merge", "application/json", body))
            .await
            .unwrap();
        let (status, json) = status_and_json(resp).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{label}: {json}");
        assert!(
            HealMemory::load(&heal).record(&fp).is_none(),
            "{label} was merged"
        );
    }

    let app = create_router(keyed_state(&heal).await);
    let good = serde_json::to_string(&sign_export(&payload, Some(FLEET_KEY))).unwrap();
    let resp = app
        .oneshot(post("/v1/immunity/merge", "application/json", good))
        .await
        .unwrap();
    let (status, json) = status_and_json(resp).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["programs_added"], 1);
    assert!(HealMemory::load(&heal).record(&fp).is_some());
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn export_from_a_keyed_node_merges_into_a_peer_with_the_same_key() {
    // The gossip round-trip: GET /v1/patches on node A, POST it to node B.
    let (dir_a, heal_a, patches_a) = state_dir("gossip_a");
    let (dir_b, heal_b, patches_b) = state_dir("gossip_b");
    {
        let mut pm = PatchMemory::new();
        pm.record_verified(
            "fp-gossip",
            &RepoScope::new("path:some-repo", "src/lib.rs"),
            "BROKEN\n",
            "PEER FIX\n",
        );
        pm.save(&patches_a).unwrap();
    }
    let node_a = create_router(keyed_state(&heal_a).await);
    let resp = node_a
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/patches")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let export = String::from_utf8(
        to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();

    // A peer with a different key refuses it; a peer with the same key merges.
    let stranger = create_router(
        keyless_state(&heal_b)
            .await
            .with_fleet_key(Some(b"some-other-fleet".to_vec())),
    );
    let resp = stranger
        .oneshot(post(
            "/v1/patches/merge",
            "application/json",
            export.clone(),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert!(PatchMemory::load(&patches_b).is_empty());

    let node_b = create_router(keyed_state(&heal_b).await);
    let resp = node_b
        .oneshot(post("/v1/patches/merge", "application/json", export))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        PatchMemory::load(&patches_b).candidates("fp-gossip").len(),
        1
    );
    let _ = std::fs::remove_dir_all(dir_a);
    let _ = std::fs::remove_dir_all(dir_b);
}

// --- Host and Origin allowlists (browser-borne requests) ---------------------

/// A correctly signed merge, so the only thing a refusal can be about is the
/// Host / Origin header under test.
fn signed_merge(host: Option<&str>, origin: Option<&str>) -> Request<Body> {
    let body =
        serde_json::to_string(&sign_export(&friendly_patch_payload(), Some(FLEET_KEY))).unwrap();
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/v1/patches/merge")
        .header("content-type", "application/json");
    if let Some(host) = host {
        req = req.header("host", host);
    }
    if let Some(origin) = origin {
        req = req.header("origin", origin);
    }
    req.body(Body::from(body)).unwrap()
}

#[tokio::test]
async fn merge_refuses_a_host_that_is_not_on_the_allowlist() {
    let (dir, heal, patches) = state_dir("host");

    // DNS rebinding: the attacker's name resolves to this node, the browser
    // treats the page as same-origin (so it sends no foreign Origin), but the
    // Host header still carries the attacker's name.
    for bad in [
        Some("evil.example:8080"),
        Some("evil.example"),
        Some("localhost.evil.example:8080"),
        Some("127.0.0.1.evil.example"),
        Some("localhost:8080@evil.example"),
        Some("localhost:notaport"),
        Some(""),
        None, // no Host and no authority in the request target
    ] {
        let app = create_router(keyed_state(&heal).await);
        let resp = app.oneshot(signed_merge(bad, None)).await.unwrap();
        let (status, json) = status_and_json(resp).await;
        assert_eq!(
            status,
            StatusCode::FORBIDDEN,
            "Host {bad:?} must be refused: {json}"
        );
        assert!(
            PatchMemory::load(&patches).is_empty(),
            "Host {bad:?} merged"
        );
    }

    // Loopback names and IP literals (how fleet peers address each other).
    for good in [
        "localhost",
        "LOCALHOST:8080",
        "localhost.:8080",
        "127.0.0.1:8080",
        "10.42.0.17:8080",
        "[::1]:8080",
        "[fd00::17]",
    ] {
        let app = create_router(keyed_state(&heal).await);
        let resp = app.oneshot(signed_merge(Some(good), None)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Host {good:?} must be accepted"
        );
    }

    // A DNS name is accepted only once the operator lists it.
    let app = create_router(
        keyed_state(&heal)
            .await
            .with_merge_allowed_hosts(vec!["Axiom-0.axiom.svc".to_string()]),
    );
    let resp = app
        .clone()
        .oneshot(signed_merge(Some("axiom-0.axiom.svc:8080"), None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = app
        .oneshot(signed_merge(Some("axiom-1.axiom.svc:8080"), None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn merge_refuses_an_origin_that_is_not_on_the_allowlist() {
    let (dir, heal, patches) = state_dir("origin");

    // Browsers attach Origin to every cross-origin POST, including the
    // `text/plain` ones that skip preflight. None is allowed by default.
    for bad in [
        "https://evil.example",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
        "null",
    ] {
        let app = create_router(keyed_state(&heal).await);
        let resp = app
            .oneshot(signed_merge(Some(LOOPBACK_HOST), Some(bad)))
            .await
            .unwrap();
        let (status, json) = status_and_json(resp).await;
        assert_eq!(
            status,
            StatusCode::FORBIDDEN,
            "Origin {bad:?} must be refused: {json}"
        );
        assert!(
            PatchMemory::load(&patches).is_empty(),
            "Origin {bad:?} merged"
        );
    }

    // An allowlisted origin passes the Origin check (and still needs the HMAC).
    let allow = |origins: &[&str]| {
        origins
            .iter()
            .map(|o| axum::http::HeaderValue::from_str(o).unwrap())
            .collect::<Vec<_>>()
    };
    let app = create_router(
        keyed_state(&heal)
            .await
            .with_merge_allowed_origins(allow(&["https://dash.example", "null"])),
    );
    let resp = app
        .clone()
        .oneshot(signed_merge(
            Some(LOOPBACK_HOST),
            Some("https://dash.example"),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // Prefix/suffix look-alikes of an allowed origin are not the allowed origin.
    for near in [
        "https://dash.example.evil.example",
        "https://dash.example:444",
    ] {
        let resp = app
            .clone()
            .oneshot(signed_merge(Some(LOOPBACK_HOST), Some(near)))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN, "{near}");
    }
    // `null` is never accepted, even when an operator lists it.
    let resp = app
        .oneshot(signed_merge(Some(LOOPBACK_HOST), Some("null")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn host_and_origin_are_checked_before_the_missing_key_is_reported() {
    // A page probing a keyless node learns nothing about its key state.
    let (dir, heal, _patches) = state_dir("order");
    let app = create_router(keyless_state(&heal).await);
    let resp = app
        .clone()
        .oneshot(signed_merge(Some("evil.example"), None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let resp = app
        .oneshot(signed_merge(
            Some(LOOPBACK_HOST),
            Some("https://evil.example"),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn the_api_key_alone_does_not_authorize_a_merge() {
    // AXIOM_API_KEY guards the data plane; it is not a fleet credential. A
    // caller holding it still cannot merge into a node with no fleet key.
    let (dir, heal, patches) = state_dir("apikey");
    let app = create_router(
        keyless_state(&heal)
            .await
            .with_api_key(Some("data-plane-key".to_string())),
    );
    let mut req = post(
        "/v1/patches/merge",
        "application/json",
        unsigned_export(&hostile_patch_payload()),
    );
    req.headers_mut()
        .insert("x-axiom-key", "data-plane-key".parse().unwrap());
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert!(PatchMemory::load(&patches).is_empty());
    let _ = std::fs::remove_dir_all(dir);
}
