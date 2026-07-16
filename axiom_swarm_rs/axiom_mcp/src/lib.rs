//! # axiom_mcp — Mini Aether sidecars (the "hands")
//!
//! Every worker node in the mesh sits behind a Mini Aether sidecar. The
//! sidecar is middleware between Axiom Prime (the orchestrator) and the
//! worker backend, and does exactly two deterministic jobs:
//!
//! 1. **Filtering** ([`filter`]) — scrub tokens flowing from Axiom Prime
//!    before they reach the worker: strip secrets-shaped strings, control
//!    characters, and orchestrator-internal markers.
//! 2. **Compression** ([`compress`]) — strip conversational filler and
//!    extract only the state the worker's specific context needs.
//!
//! No mesh logic lives here, on purpose: the KNM (axiom_core) decides
//! *where* a payload goes; the sidecar only conditions *what* arrives.

pub mod compress;
pub mod filter;
pub mod sidecar;

pub use compress::{compress_context, CompressionStats};
pub use filter::{FilterRule, TokenScrubber};
pub use sidecar::{MiniAetherSidecar, SidecarPayload};
