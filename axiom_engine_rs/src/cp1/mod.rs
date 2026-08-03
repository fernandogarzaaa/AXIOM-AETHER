//! CP/1 — the Cognitive Protocol, version 1.
//!
//! AXIOM is the normative owner of CP/1: the specification, JSON Schema, golden
//! fixtures and vendoring manifest live in `protocol/cp1/` at the repository
//! root, and this module is the reference Rust binding for them.
//!
//! # What lives here, and what deliberately does not
//!
//! AXIOM authors exactly one canonical type, [`types::Context`], and emits two
//! events, `ContextCompressed` and `GroundingFailed`. This module therefore
//! ships:
//!
//! - [`canonical`] — the byte-exact encoding every document hashes over. Shared
//!   by all three components and the reason they can agree on a hash at all.
//! - [`types`] — `Context` plus the primitives every document needs.
//! - [`event`] — the closed event set and the sink trait subsystems emit through.
//! - [`envelope`] — signed transport, reusing [`crate::provenance`]'s crypto.
//! - [`conformance`] — the shared suite each binding runs against the corpus.
//!
//! It does **not** ship structs for `Genome`, `Skill`, `Belief` and the rest.
//! Authorship of each canonical type is exclusive (see `protocol/cp1/SPEC.md`
//! section 3), so those are types AXIOM may never mint and does not read; a
//! struct for each would be a dead abstraction requiring perpetual maintenance
//! against a schema it never exercises. Cross-binding agreement on those types
//! is established structurally, over the fixture corpus, by [`conformance`] —
//! which tests the encoding, the thing that actually has to match.
//!
//! # Example
//!
//! ```
//! use axiom_engine::cp1::{canonical, envelope::SignedEnvelope, types::*};
//!
//! let context = Context::assemble(
//!     "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
//!     "answer a question about the organism's goals",
//!     4096,
//!     31_500,
//!     3_780,
//!     vec![ContextSegment {
//!         role: SegmentRole::Memory,
//!         content: "Amendments to goals require an approving fitness result.".into(),
//!         source_id: Some("66666666-6666-4666-8666-666666666666".into()),
//!     }],
//!     Provenance::now(Component::Axiom, "axiom:context/compress"),
//! );
//!
//! assert!(context.grounded);
//! assert_eq!(context.compression_ratio_bp.raw(), 1_200);
//!
//! let wire = SignedEnvelope::seal(&context.seal().unwrap(), None).unwrap();
//! let received = wire.open(None).unwrap();
//! assert!(canonical::verify_seal(&received).unwrap());
//! ```

pub mod canonical;
pub mod conformance;
pub mod envelope;
pub mod event;
pub mod types;

/// The protocol identifier carried by every CP/1 document.
pub const CP: &str = "cp1";

/// The revision of the normative source this binding implements.
///
/// Kept in step with `protocol/cp1/VERSION` by
/// [`tests::version_matches_the_normative_source`].
pub const VERSION: &str = "1.0.0";

pub use canonical::{content_hash, seal, to_canonical, verify_seal, CanonicalError};
pub use envelope::{EnvelopeError, SignedEnvelope};
pub use event::{Event, EventKind, EventSink, PayloadValue, RecordingSink, SubjectType};
pub use types::{
    BasisPoints, Component, Context, ContextSegment, Provenance, SegmentRole, Timestamp,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_the_normative_source() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../protocol/cp1/VERSION");
        let declared = std::fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("cannot read {path}: {err}"));
        assert_eq!(
            declared.trim(),
            VERSION,
            "this binding claims CP/1 {VERSION} but the normative source is at {}",
            declared.trim()
        );
    }
}
