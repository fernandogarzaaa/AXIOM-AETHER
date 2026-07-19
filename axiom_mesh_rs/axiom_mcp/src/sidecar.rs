//! The Mini Aether sidecar: the per-worker middleware pipeline.
//!
//! Payloads are passed by `Arc` so the orchestrator, mesh, and sidecars
//! share one buffer — the pipeline only allocates when it actually rewrites
//! content (zero-copy on the pass-through path).

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::compress::{compress_context, CompressionStats};
use crate::filter::TokenScrubber;

/// A payload conditioned for a specific worker node.
#[derive(Debug, Clone)]
pub struct SidecarPayload {
    /// The scrubbed, compressed content the worker receives.
    pub content: Arc<str>,
    /// Compression telemetry for cost accounting upstream.
    pub stats: CompressionStats,
}

/// Static configuration for one sidecar instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarConfig {
    /// Name of the worker backend this sidecar fronts (telemetry only).
    pub worker: String,
    /// Skip the compression pass (e.g. for tool runners that need exact
    /// bytes).
    pub compress: bool,
}

/// Middleware between Axiom Prime and one worker node: deterministic
/// scrub, then context compression.
pub struct MiniAetherSidecar {
    config: SidecarConfig,
    scrubber: TokenScrubber,
}

impl MiniAetherSidecar {
    pub fn new(config: SidecarConfig, scrubber: TokenScrubber) -> Self {
        Self { config, scrubber }
    }

    /// A sidecar with the standard scrub rules and compression on.
    pub fn standard(worker: impl Into<String>) -> Self {
        Self::new(
            SidecarConfig { worker: worker.into(), compress: true },
            TokenScrubber::standard(),
        )
    }

    pub fn worker(&self) -> &str {
        &self.config.worker
    }

    /// Condition an orchestrator payload for this sidecar's worker.
    pub fn condition(&self, raw: &str) -> SidecarPayload {
        let scrubbed = self.scrubber.scrub(raw);
        if self.config.compress {
            let (compressed, stats) = compress_context(&scrubbed);
            SidecarPayload { content: Arc::from(compressed), stats }
        } else {
            let bytes = scrubbed.len();
            let lines = scrubbed.lines().count();
            SidecarPayload {
                content: Arc::from(scrubbed),
                stats: CompressionStats {
                    input_lines: lines,
                    output_lines: lines,
                    input_bytes: bytes,
                    output_bytes: bytes,
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_scrubs_then_compresses() {
        let sidecar = MiniAetherSidecar::standard("claude");
        let payload = sidecar.condition(
            "Sure, here you go!\napi_key=sk-live-123\n<axiom-internal> hop=2\nresidual norm: 0.9",
        );
        let content = payload.content.as_ref();
        assert!(!content.contains("sk-live-123"));
        assert!(!content.contains("axiom-internal"));
        assert!(!content.to_lowercase().contains("sure,"));
        assert!(content.contains("residual norm: 0.9"));
    }

    #[test]
    fn shared_payload_is_cheaply_clonable() {
        let sidecar = MiniAetherSidecar::standard("codex");
        let payload = sidecar.condition("state: x=1");
        let a = Arc::clone(&payload.content);
        let b = payload.content.clone();
        assert!(Arc::ptr_eq(&a, &b));
    }
}
