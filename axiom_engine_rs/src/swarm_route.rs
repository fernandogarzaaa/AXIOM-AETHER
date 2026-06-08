//! Localized high-plasticity swarm routing state.
//!
//! This module tracks which lightweight local checkpoint should be active for
//! the current VFS/domain target and exposes bounded VRAM telemetry. The actual
//! model load remains conservative: hot-swap is represented as metadata plus an
//! async flush boundary so existing Candle sessions are not disturbed.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::dwe::DweTelemetry;
use crate::surprisal::ExactResidualTelemetry;

pub const RTX_2060_SAFE_VRAM_BYTES: u64 = 5_200_000_000;
pub const DEFAULT_FAST_WEIGHT_BYTES: u64 = 96 * 1024 * 1024;
pub const DEFAULT_JIT_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMatrixState {
    pub active_domain: String,
    pub active_checkpoint: Option<String>,
    pub active_d_model: usize,
    pub allocated_vram_bytes: u64,
    pub vram_budget_bytes: u64,
    pub within_vram_budget: bool,
    pub hot_swap_count: u64,
    pub last_swap_ms: u128,
    pub last_updated_unix_secs: u64,
    pub dwe: DweTelemetry,
    pub exact_residual: ExactResidualTelemetry,
}

impl Default for SwarmMatrixState {
    fn default() -> Self {
        Self {
            active_domain: "generic".into(),
            active_checkpoint: None,
            active_d_model: 256,
            allocated_vram_bytes: DEFAULT_FAST_WEIGHT_BYTES + DEFAULT_JIT_WORKSPACE_BYTES,
            vram_budget_bytes: RTX_2060_SAFE_VRAM_BYTES,
            within_vram_budget: true,
            hot_swap_count: 0,
            last_swap_ms: 0,
            last_updated_unix_secs: unix_now(),
            dwe: DweTelemetry::default(),
            exact_residual: ExactResidualTelemetry::default(),
        }
    }
}

#[derive(Clone)]
pub struct LocalSwarmRouteMatrix {
    inner: Arc<Mutex<SwarmMatrixState>>,
}

impl Default for LocalSwarmRouteMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalSwarmRouteMatrix {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(SwarmMatrixState::default())),
        }
    }

    pub fn state(
        &self,
        dwe: DweTelemetry,
        exact_residual: ExactResidualTelemetry,
    ) -> SwarmMatrixState {
        let mut state = self.inner.lock().map(|s| s.clone()).unwrap_or_default();
        state.dwe = dwe;
        state.exact_residual = exact_residual.clone();
        state.allocated_vram_bytes = estimate_allocated_vram(
            state.active_d_model,
            exact_residual.estimated_bytes,
            DEFAULT_JIT_WORKSPACE_BYTES,
        );
        state.within_vram_budget = state.allocated_vram_bytes <= RTX_2060_SAFE_VRAM_BYTES;
        state
    }

    pub async fn observe_vfs_target(&self, target: impl AsRef<Path>) -> SwarmMatrixState {
        let target = target.as_ref();
        let domain = detect_domain(target);
        let checkpoint = checkpoint_for_domain(&domain);
        let d_model = d_model_for_domain(&domain);
        let start = std::time::Instant::now();
        tokio::task::yield_now().await;
        if let Ok(mut state) = self.inner.lock() {
            if state.active_domain != domain
                || state.active_checkpoint.as_deref() != Some(&checkpoint)
            {
                state.hot_swap_count += 1;
                state.last_swap_ms = start.elapsed().as_millis();
            }
            state.active_domain = domain;
            state.active_checkpoint = Some(checkpoint);
            state.active_d_model = d_model;
            state.last_updated_unix_secs = unix_now();
            state.allocated_vram_bytes = estimate_allocated_vram(
                d_model,
                state.exact_residual.estimated_bytes,
                DEFAULT_JIT_WORKSPACE_BYTES,
            );
            state.within_vram_budget = state.allocated_vram_bytes <= RTX_2060_SAFE_VRAM_BYTES;
            return state.clone();
        }
        SwarmMatrixState::default()
    }
}

pub fn detect_domain(path: &Path) -> String {
    let text = path.to_string_lossy().to_ascii_lowercase();
    if text.contains("scrape") || text.contains("search") || text.contains("ingest") {
        "web_ingest".into()
    } else if text.contains("rust") || text.contains("cargo") || text.ends_with(".rs") {
        "rust_compile".into()
    } else if text.contains("python") || text.ends_with(".py") {
        "python_runtime".into()
    } else if text.contains("typescript") || text.ends_with(".ts") || text.ends_with(".js") {
        "web_code".into()
    } else {
        "generic".into()
    }
}

fn checkpoint_for_domain(domain: &str) -> String {
    match domain {
        "rust_compile" => "checkpoints/local/axiom_rust_d256.bin",
        "web_ingest" => "checkpoints/local/axiom_ingest_d256.bin",
        "python_runtime" => "checkpoints/local/axiom_python_d256.bin",
        "web_code" => "checkpoints/local/axiom_web_d256.bin",
        _ => "checkpoints/local/axiom_generic_d256.bin",
    }
    .into()
}

fn d_model_for_domain(domain: &str) -> usize {
    match domain {
        "rust_compile" | "web_ingest" => 256,
        "python_runtime" | "web_code" => 256,
        _ => 256,
    }
}

fn estimate_allocated_vram(
    d_model: usize,
    exact_cache_bytes: u64,
    jit_workspace_bytes: u64,
) -> u64 {
    let layers = 4_u64;
    let fast_weights = (d_model as u64) * (d_model as u64) * layers * 4;
    fast_weights + exact_cache_bytes + jit_workspace_bytes
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn domain_switch_updates_state_and_stays_in_budget() {
        let matrix = LocalSwarmRouteMatrix::new();
        let state = matrix.observe_vfs_target("src/search_ingest.rs").await;
        assert_eq!(state.active_domain, "web_ingest");
        assert!(state.within_vram_budget);
        assert!(state.hot_swap_count >= 1);
    }

    #[test]
    fn detects_rust_compile_domain() {
        assert_eq!(
            detect_domain(Path::new("axiom_engine_rs/Cargo.toml")),
            "rust_compile"
        );
        assert_eq!(detect_domain(Path::new("src/main.rs")), "rust_compile");
    }
}
