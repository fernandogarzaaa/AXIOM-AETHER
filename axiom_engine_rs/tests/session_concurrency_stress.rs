//! Graded session-concurrency stress harness for the `/v1/messages` route.
//!
//! Drives concurrent sessions against the test server at increasing
//! concurrency (1, 4, 16, 32) and asserts bounds at each level:
//!
//!   * every request answers 200 OK with the mock upstream's text, and every
//!     upstream payload carries a fingerprint with the raw heavy text
//!     stripped and a distinct per-session `state_hash` (no wrong answers,
//!     no cross-tenant bleed under load),
//!   * p95 request latency grows at most [`MAX_P95_GROWTH`] over a warm
//!     sequential baseline,
//!   * RSS grows at most [`MAX_RSS_GROWTH_RATIO`] plus
//!     [`MAX_RSS_GROWTH_SLACK_KIB`] across any one level (no unbounded
//!     memory growth),
//!   * the session registry holds exactly N sessions after a level of N,
//!     and 0 after the clear (no leaked sessions).
//!
//! The bounds are RATIOS, not absolute milliseconds or bytes: this sandbox
//! (and CI runners) are not a perf lab, and absolute numbers there are noise
//! from CPU contention. The measured baseline from the runs that calibrated
//! the bounds is printed by the test (`--nocapture`) and recorded in the PR
//! body; re-calibrate the constants if the baseline moves, do not silently
//! widen them.
//!
//! Baseline measured 2026-11-10 on this sandbox (tiny d_model=16 pipeline,
//! in-process `oneshot` load generator, 600-token heavy window per session).
//! First three runs used a single cold sample as the p95 baseline:
//!
//!   level  p95 run1/run2/run3 (ms)   rss_delta run1/run2/run3 (KiB)
//!   1      139.5 / 177.7 / 125.5      +3324 / +3676 / +3792
//!   4      582.1 / 503.9 / 394.7      +9444 / +9544 / +9664
//!   16     705.8 / 867.4 / 900.6      +18468 / +17976 / +20012
//!   32     1575.5 / 1495.1 / 1559.9   +9320 / +9644 / +4316
//!
//! p95 growth over that cold single sample was 8.4x-12.4x. The redesigned
//! harness (warm-up + 4-request sequential baseline) measured on its first
//! green run: warm RSS 15820 KiB, baseline p95 147.6 ms, then level 1
//! p95 117.2 ms (0.8x, +2236 KiB), level 4 p95 342.1 ms (2.3x, +5304 KiB),
//! level 16 p95 753.6 ms (5.1x, +21532 KiB), level 32 p95 1408.1 ms
//! (9.5x, +6832 KiB). RSS after each clear equals RSS before the clear:
//! the allocator retains arenas, which is why no post-clear RSS bound is
//! asserted.
//!
//! What limits the numbers (Explain the Number): the load generator is
//! in-process `oneshot`, so client and server share the tokio runtime and
//! the sandbox's 1.5 shared CPUs; the tiny pipeline's `forward_native`
//! gradient loops run on the `spawn_blocking` pool, so latency at higher
//! concurrency is bounded by pool/CPU contention, not by the network. The
//! correctness assertions (fingerprint present, raw text stripped, distinct
//! hashes, exactly one upstream call per session) rule out the harness
//! measuring skipped or cached work. RSS grows ~12 MiB cold to ~50 MiB peak
//! across a run; that is one-time warm-up (allocator arenas, candle, tokio)
//! plus the live sessions of the current level, not a leak: per-level
//! deltas stay flat (level 32 adds less than level 16) and the registry
//! returns to 0 after every clear. There is deliberately no peak-RSS-vs-
//! cold-start assertion: that ratio measures allocator retention, not
//! leakage, and would fail on warm-up alone.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axiom_engine::anthropic_forwarder::AnthropicForwarder;
use axiom_engine::config::AxiomConfig;
use axiom_engine::context_compressor::CompressorConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::extract::State;
use axum::http::{Method, Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use candle_core::Device;
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tower::ServiceExt;

/// Concurrency levels, ascending.
const LEVELS: &[usize] = &[1, 4, 16, 32];

/// 600 tokens, comfortably over the 400-token heavy threshold, same as
/// `tests/stress.rs`, so the compression path (not the trivial path) is
/// exercised at every level.
const HEAVY_TOKENS: usize = 600;

/// Sequential warm requests used to build the p95 baseline. Four samples
/// keeps the baseline stable without adding much runtime.
const BASELINE_REQUESTS: usize = 4;

/// Max p95 growth over the warm sequential baseline. Measured 8.4x-12.4x
/// against a single cold sample over three sandbox runs; 20x keeps ~1.6x
/// headroom over the worst observed while still catching unbounded growth
/// (deadlock-adjacent serialization, lock pile-up), which shows up as a
/// 50x+ blowup or a hang, not as contention noise.
const MAX_P95_GROWTH: f64 = 20.0;

/// Max RSS growth of one level over the RSS measured just before it:
/// 25% relative plus 16 MiB of slack for allocator retention and the
/// mock-upstream bookkeeping. Session tensors here are tiny (d_model 16,
/// 1 layer), so a real per-session leak shows up as growth far above this.
/// Measured per-level deltas: +3-4 MiB (level 1), +9-10 MiB (level 4),
/// +18-20 MiB (level 16), +4-10 MiB (level 32).
const MAX_RSS_GROWTH_RATIO: f64 = 1.25;
const MAX_RSS_GROWTH_SLACK_KIB: u64 = 16 * 1024;

type Captured = Arc<Mutex<Vec<Value>>>;

async fn start_mock_anthropic() -> (SocketAddr, Captured) {
    let captured: Captured = Arc::new(Mutex::new(Vec::new()));
    let mock_state = captured.clone();
    let app = Router::new()
        .route("/v1/messages", post(mock_handler))
        .with_state(mock_state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    (addr, captured)
}

async fn mock_handler(State(captured): State<Captured>, Json(body): Json<Value>) -> Json<Value> {
    captured.lock().await.push(body);
    Json(json!({
        "id": "msg_graded_stress",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": "claude-mock",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
}

fn build_pipeline() -> InferencePipeline {
    let cfg = AxiomConfig {
        d_model: 16,
        n_layers: 1,
        vocab_size: 64,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    };
    InferencePipeline::new(cfg, Device::Cpu).expect("tiny stress pipeline must build")
}

/// Resident set size in KiB, Linux only (`/proc/self/status` VmRSS).
/// Returns `None` off Linux, where RSS is unmeasured and the RSS bounds
/// are skipped (documented, not silently passed as zero).
fn rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let rest = line.strip_prefix("VmRSS:")?;
        rest.trim()
            .strip_suffix("kB")
            .and_then(|num| num.trim().parse().ok())
    })
}

/// Nearest-rank percentile over a sorted slice of durations.
fn percentile(sorted: &[Duration], p: f64) -> Duration {
    assert!(!sorted.is_empty(), "percentile of an empty sample");
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.max(1).min(sorted.len()) - 1]
}

fn extract_state_hash(body: &Value) -> Option<String> {
    let combined: String = body["messages"]
        .as_array()?
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .concat();
    combined
        .lines()
        .find_map(|l| l.strip_prefix("state_hash=").map(str::to_string))
}

async fn session_count(app: &Router) -> u64 {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/ttt/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let stats: Value = serde_json::from_slice(&bytes).unwrap();
    stats["active_sessions"].as_u64().unwrap()
}

async fn clear_sessions(app: &Router) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::DELETE)
                .uri("/v1/ttt/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let cleared: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        cleared["cleared"], true,
        "session clear must report success"
    );
}

/// One `/v1/messages` request for `session_id` carrying a unique heavy
/// window, returning status, latency, and the raw response body.
async fn run_one(app: &Router, session_id: &str, tag: &str) -> (StatusCode, Duration, Vec<u8>) {
    let heavy: String = (0..HEAVY_TOKENS)
        .map(|j| format!("{tag}_tok{j}"))
        .collect::<Vec<_>>()
        .join(" ");
    let body = json!({
        "model": "claude-graded-stress",
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": heavy},
            {"role": "user", "content": format!("summarise session {session_id}")}
        ],
        "session_id": session_id
    });
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let started = Instant::now();
    let resp = app
        .clone()
        .oneshot(req)
        .await
        .expect("request must not be dropped");
    let latency = started.elapsed();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap()
        .to_vec();
    (status, latency, bytes)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn graded_concurrency_bounds_p95_rss_and_registry() {
    let (mock_addr, captured) = start_mock_anthropic().await;

    let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
    let forwarder = AnthropicForwarder::new(
        Some("test-key".to_string()),
        Some(format!("http://{mock_addr}")),
    );
    let cfg = CompressorConfig {
        enabled: true,
        heavy_message_threshold_tokens: 400,
        recall_top_k: 8,
    };
    let state = AppState::new(pipeline, "axiom-graded-stress".to_string())
        .with_anthropic_forwarder(Some(forwarder))
        .with_compressor_config(cfg);
    let app = create_router(state);

    assert_eq!(session_count(&app).await, 0, "store must start cold");

    // ---- Warm-up: one request, then clear ------------------------------
    // Warms the pipeline, the allocator, and the mock route so the baseline
    // below and the level-1 RSS are warm, not cold-start numbers.
    let (status, _, bytes) = run_one(&app, "graded-warmup", "warmup").await;
    assert_eq!(status, StatusCode::OK, "warm-up request must be 200");
    assert!(String::from_utf8_lossy(&bytes).contains("ok"));
    clear_sessions(&app).await;
    let warm_rss = rss_kib();
    println!("[graded-stress] warm_rss={warm_rss:?}KiB");

    // ---- Warm sequential baseline for the p95 ratio --------------------
    let mut baseline_latencies = Vec::with_capacity(BASELINE_REQUESTS);
    for i in 0..BASELINE_REQUESTS {
        let (status, latency, bytes) =
            run_one(&app, &format!("graded-base-{i}"), &format!("base{i}")).await;
        assert_eq!(status, StatusCode::OK, "baseline request {i} must be 200");
        assert!(String::from_utf8_lossy(&bytes).contains("ok"));
        baseline_latencies.push(latency);
    }
    baseline_latencies.sort_unstable();
    let baseline_p95 = percentile(&baseline_latencies, 0.95);
    println!(
        "[graded-stress] baseline p95={:.1}ms ({} sequential warm requests)",
        baseline_p95.as_secs_f64() * 1000.0,
        BASELINE_REQUESTS
    );
    clear_sessions(&app).await;
    assert_eq!(session_count(&app).await, 0);

    for &level in LEVELS {
        let rss_before = rss_kib();
        let upstream_before = captured.lock().await.len();

        // ---- Fire `level` independent sessions in parallel --------------
        let mut handles = Vec::with_capacity(level);
        for i in 0..level {
            let app = app.clone();
            handles.push(tokio::spawn(async move {
                run_one(
                    &app,
                    &format!("graded-{level}-{i}"),
                    &format!("g{level}_s{i}"),
                )
                .await
            }));
        }

        let mut latencies = Vec::with_capacity(level);
        for h in handles {
            let (status, latency, bytes) = h.await.expect("task must not panic");
            assert_eq!(
                status,
                StatusCode::OK,
                "level {level}: every request must be 200"
            );
            let text = String::from_utf8_lossy(&bytes);
            assert!(
                text.contains("ok"),
                "level {level}: response must carry the mock upstream answer, got: {text}"
            );
            latencies.push(latency);
        }
        assert_eq!(latencies.len(), level);

        let rss_after = rss_kib();
        latencies.sort_unstable();
        let p95 = percentile(&latencies, 0.95);
        let mean = latencies.iter().sum::<Duration>() / level as u32;
        let p95_ms = p95.as_secs_f64() * 1000.0;
        let mean_ms = mean.as_secs_f64() * 1000.0;

        // ---- Correctness of the upstream payloads at this level ---------
        let payloads = captured.lock().await.clone();
        let new_payloads = &payloads[upstream_before..];
        assert_eq!(
            new_payloads.len(),
            level,
            "level {level}: mock must have received exactly one upstream call per session"
        );
        let mut hashes = std::collections::HashSet::new();
        for body in new_payloads {
            assert!(
                body.get("session_id").is_none(),
                "session_id must be stripped"
            );
            let combined: String = body["messages"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|m| m.get("content").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .concat();
            assert!(
                combined.contains("<axiom_context_fingerprint "),
                "level {level}: every payload must carry the fingerprint block"
            );
            assert!(
                !combined.contains(&format!("tok{}", HEAVY_TOKENS - 1)),
                "level {level}: raw heavy text must be stripped from every payload"
            );
            if let Some(h) = extract_state_hash(body) {
                hashes.insert(h);
            }
        }
        assert_eq!(
            hashes.len(),
            level,
            "level {level}: each session must produce a distinct state_hash (no cross-tenant bleed)"
        );

        // ---- Registry: exactly N live sessions, then 0 after clear -----
        assert_eq!(
            session_count(&app).await,
            level as u64,
            "level {level}: registry must hold exactly the sessions just created"
        );

        // ---- Bounds ------------------------------------------------------
        let growth = p95.as_secs_f64() / baseline_p95.as_secs_f64();
        assert!(
            growth <= MAX_P95_GROWTH,
            "level {level}: p95 {p95_ms:.1}ms is {growth:.1}x the warm baseline \
             {:.1}ms (bound {MAX_P95_GROWTH}x): unbounded latency growth under load",
            baseline_p95.as_secs_f64() * 1000.0
        );
        if let (Some(before), Some(after)) = (rss_before, rss_after) {
            let allowed = (before as f64 * MAX_RSS_GROWTH_RATIO) as u64 + MAX_RSS_GROWTH_SLACK_KIB;
            assert!(
                after <= allowed,
                "level {level}: RSS grew {before}KiB -> {after}KiB (allowed {allowed}KiB): \
                 unbounded memory growth under load"
            );
        }

        clear_sessions(&app).await;
        assert_eq!(
            session_count(&app).await,
            0,
            "level {level}: registry must be empty after clear (no leaked sessions)"
        );
        let rss_after_clear = rss_kib();

        // ---- Baseline line (the numbers the PR body records) ------------
        let rss_line = match (rss_before, rss_after) {
            (Some(before), Some(after)) => format!(
                "rss_before={before}KiB rss_after={after}KiB rss_delta={}KiB",
                after as i64 - before as i64
            ),
            _ => "rss=unmeasured (non-Linux)".to_string(),
        };
        println!(
            "[graded-stress] level={level} p95={p95_ms:.1}ms mean={mean_ms:.1}ms \
             p95_growth={growth:.1}x {rss_line} rss_after_clear={rss_after_clear:?}KiB registry={level}"
        );
    }

    println!(
        "[graded-stress] warm_rss={warm_rss:?}KiB baseline_p95={:.1}ms",
        baseline_p95.as_secs_f64() * 1000.0
    );
}
