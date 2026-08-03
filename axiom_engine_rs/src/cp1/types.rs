//! CP/1 canonical types that AXIOM authors or consumes directly.
//!
//! AXIOM owns exactly one canonical type — [`Context`] — and reads several
//! others to build it. This module therefore ships typed structs for what AXIOM
//! authors and shared primitives ([`Provenance`], [`BasisPoints`],
//! [`Component`], [`Timestamp`]) that every type needs, rather than mirroring
//! all fourteen documents.
//!
//! That is deliberate. A struct for `Skill` in AXIOM would be a type AXIOM may
//! never mint (SPEC.md section 3 makes authorship exclusive) and never reads —
//! a dead abstraction that would still have to be kept in sync with the schema
//! forever. Conformance over the full corpus is done structurally, on
//! `serde_json::Value`, by [`super::conformance`]: that tests the *encoding*,
//! which is what actually has to agree across bindings.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use super::canonical::{self, CanonicalError};

/// Which repository authored a document. Authorship is exclusive per type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Component {
    Adam,
    Eve,
    Axiom,
}

impl Component {
    pub fn as_str(self) -> &'static str {
        match self {
            Component::Adam => "adam",
            Component::Eve => "eve",
            Component::Axiom => "axiom",
        }
    }
}

/// A ratio in `[0, 1]` carried on the wire as an integer in `[0, 10000]`.
///
/// CP/1 puts no floating point on the wire (SPEC.md section 2.1), so every
/// fractional quantity crosses a boundary through this type. Conversion is the
/// only place rounding happens, and it is explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BasisPoints(u16);

impl BasisPoints {
    pub const ZERO: BasisPoints = BasisPoints(0);
    pub const ONE: BasisPoints = BasisPoints(10_000);

    /// Convert a ratio to basis points, clamping to `[0, 1]` and rounding
    /// half away from zero.
    ///
    /// Clamping rather than erroring is the right call at this boundary: the
    /// inputs are scores from statistical models that can legitimately land a
    /// hair outside `[0, 1]` through accumulated error, and refusing to encode
    /// a `1.0000001` similarity would fail a pipeline over nothing. Values that
    /// are wildly out of range indicate a caller bug, but clamping still
    /// produces a well-formed document that records what was meant.
    pub fn from_ratio(ratio: f64) -> Self {
        if !ratio.is_finite() {
            return BasisPoints::ZERO;
        }
        let scaled = (ratio.clamp(0.0, 1.0) * 10_000.0).round();
        BasisPoints(scaled as u16)
    }

    /// Reject rather than clamp — for callers that would rather fail than
    /// silently record a value they did not mean.
    pub fn try_from_ratio(ratio: f64) -> Result<Self, String> {
        if !ratio.is_finite() {
            return Err(format!("ratio must be finite, got {ratio}"));
        }
        if !(0.0..=1.0).contains(&ratio) {
            return Err(format!("ratio must lie in [0, 1], got {ratio}"));
        }
        Ok(Self::from_ratio(ratio))
    }

    pub const fn raw(self) -> u16 {
        self.0
    }

    pub fn as_ratio(self) -> f64 {
        f64::from(self.0) / 10_000.0
    }
}

/// RFC 3339 UTC with exactly millisecond precision.
///
/// Fixed precision is a hashing requirement, not a style choice: a timestamp
/// one binding renders with microseconds and another with seconds produces two
/// different `content_hash` values for the same document.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(String);

impl Timestamp {
    /// The current instant, truncated to milliseconds.
    pub fn now() -> Self {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            // A clock before the epoch is a misconfigured host, not a
            // recoverable condition for a provenance record; the epoch itself
            // is the only defensible stand-in and is obviously wrong on sight.
            .unwrap_or(0);
        Self::from_unix_millis(millis)
    }

    /// Render a Unix millisecond timestamp in CP/1's fixed format.
    ///
    /// Implemented directly rather than via a date library because this is the
    /// only date arithmetic in the binding, and adding a dependency to a
    /// protocol layer that three repositories must vendor is a cost every one
    /// of them pays.
    pub fn from_unix_millis(millis: i64) -> Self {
        let (days, ms_of_day) = {
            let day = millis.div_euclid(86_400_000);
            let rem = millis.rem_euclid(86_400_000);
            (day, rem)
        };
        let (year, month, dom) = civil_from_days(days);
        let seconds_of_day = ms_of_day / 1000;
        let ms = ms_of_day % 1000;
        Self(format!(
            "{year:04}-{month:02}-{dom:02}T{:02}:{:02}:{:02}.{ms:03}Z",
            seconds_of_day / 3600,
            (seconds_of_day / 60) % 60,
            seconds_of_day % 60,
        ))
    }

    /// Accept a string only if it matches CP/1's fixed shape exactly.
    pub fn parse(value: &str) -> Result<Self, String> {
        let bytes = value.as_bytes();
        let shape_ok = bytes.len() == 24
            && bytes[4] == b'-'
            && bytes[7] == b'-'
            && bytes[10] == b'T'
            && bytes[13] == b':'
            && bytes[16] == b':'
            && bytes[19] == b'.'
            && bytes[23] == b'Z'
            && [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 22]
                .iter()
                .all(|&i| bytes[i].is_ascii_digit());
        if shape_ok {
            Ok(Self(value.to_string()))
        } else {
            Err(format!(
                "expected RFC 3339 UTC with millisecond precision (YYYY-MM-DDTHH:MM:SS.sssZ), got {value:?}"
            ))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Days since the Unix epoch to a civil `(year, month, day)`.
///
/// Howard Hinnant's `civil_from_days`, which is exact for the proleptic
/// Gregorian calendar over the whole range CP/1 can express.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Chain of custody. Mandatory on every CP/1 document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub authored_by: Component,
    pub produced_at: Timestamp,
    pub origin: String,
    pub evidence: Vec<String>,
    pub derived_from: Vec<String>,
    /// Filled in by [`super::canonical::seal`]. Empty until the document is
    /// sealed; a document that crosses a boundary unsealed is malformed.
    #[serde(default)]
    pub content_hash: String,
}

impl Provenance {
    /// A provenance record stamped with the current instant and no hash yet.
    pub fn now(authored_by: Component, origin: impl Into<String>) -> Self {
        Self {
            authored_by,
            produced_at: Timestamp::now(),
            origin: origin.into(),
            evidence: Vec::new(),
            derived_from: Vec::new(),
            content_hash: String::new(),
        }
    }

    pub fn with_evidence(mut self, evidence: impl IntoIterator<Item = String>) -> Self {
        self.evidence.extend(evidence);
        self
    }

    pub fn derived_from(mut self, ids: impl IntoIterator<Item = String>) -> Self {
        self.derived_from.extend(ids);
        self
    }
}

/// Where a context segment came from, which determines how it may be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SegmentRole {
    Identity,
    Goal,
    Memory,
    Belief,
    Observation,
    Instruction,
}

/// One piece of a compressed working set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextSegment {
    pub role: SegmentRole,
    pub content: String,
    /// The canonical document this segment was drawn from. Absent for segments
    /// AXIOM synthesized (a goal restatement, a system instruction); present
    /// for everything quoted from ADAM's memory or beliefs, which is what makes
    /// [`Context::grounded`] checkable rather than asserted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
}

/// A bounded, compressed, grounded working set handed to a model.
///
/// The only canonical type AXIOM authors. It is deliberately distinct from
/// `Memory` and `Belief`: what the organism *knows* is durable and owned by
/// ADAM; what it is *currently attending to* is ephemeral, derived, and owned
/// here. Conflating them is what makes a system unable to explain why it said
/// something — the context is the explanation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Context {
    pub cp: String,
    #[serde(rename = "type")]
    pub doc_type: String,
    pub id: String,
    pub purpose: String,
    pub token_budget: u32,
    pub tokens_before: u32,
    pub tokens_after: u32,
    pub compression_ratio_bp: BasisPoints,
    pub grounded: bool,
    pub grounding_failures: Vec<String>,
    pub memory_ids: Vec<String>,
    pub belief_ids: Vec<String>,
    pub segments: Vec<ContextSegment>,
    pub provenance: Provenance,
}

impl Context {
    /// Assemble a context and compute its derived members.
    ///
    /// `compression_ratio_bp` and `grounded` are computed here rather than
    /// accepted from the caller so they cannot disagree with the segments they
    /// describe. A caller-supplied `grounded: true` on a context with an
    /// unsourced memory segment is exactly the failure this type exists to make
    /// impossible.
    pub fn assemble(
        id: impl Into<String>,
        purpose: impl Into<String>,
        token_budget: u32,
        tokens_before: u32,
        tokens_after: u32,
        segments: Vec<ContextSegment>,
        provenance: Provenance,
    ) -> Self {
        let ratio = if tokens_before == 0 {
            1.0
        } else {
            f64::from(tokens_after) / f64::from(tokens_before)
        };

        // A quoted segment with no source is an ungrounded claim: the model
        // will read it as fact with nothing behind it.
        let grounding_failures: Vec<String> = segments
            .iter()
            .filter(|s| {
                matches!(
                    s.role,
                    SegmentRole::Memory | SegmentRole::Belief | SegmentRole::Observation
                ) && s.source_id.is_none()
            })
            .map(|s| {
                let preview: String = s.content.chars().take(60).collect();
                format!("{:?} segment has no source_id: {preview:?}", s.role)
            })
            .collect();

        let memory_ids = collect_sources(&segments, SegmentRole::Memory);
        let belief_ids = collect_sources(&segments, SegmentRole::Belief);

        Self {
            cp: "cp1".to_string(),
            doc_type: "Context".to_string(),
            id: id.into(),
            purpose: purpose.into(),
            token_budget,
            tokens_before,
            tokens_after,
            compression_ratio_bp: BasisPoints::from_ratio(ratio),
            grounded: grounding_failures.is_empty(),
            grounding_failures,
            memory_ids,
            belief_ids,
            segments,
            provenance,
        }
    }

    /// Serialize to a sealed `serde_json::Value` ready for transport.
    pub fn seal(&self) -> Result<serde_json::Value, CanonicalError> {
        let mut value = serde_json::to_value(self).expect("Context always serializes");
        canonical::seal(&mut value)?;
        Ok(value)
    }
}

fn collect_sources(segments: &[ContextSegment], role: SegmentRole) -> Vec<String> {
    let mut ids: Vec<String> = segments
        .iter()
        .filter(|s| s.role == role)
        .filter_map(|s| s.source_id.clone())
        .collect();
    ids.dedup();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(role: SegmentRole, content: &str, source: Option<&str>) -> ContextSegment {
        ContextSegment {
            role,
            content: content.to_string(),
            source_id: source.map(str::to_string),
        }
    }

    #[test]
    fn basis_points_round_half_away_from_zero() {
        assert_eq!(BasisPoints::from_ratio(0.00005).raw(), 1);
        assert_eq!(BasisPoints::from_ratio(0.5).raw(), 5_000);
        assert_eq!(BasisPoints::from_ratio(1.0).raw(), 10_000);
        assert_eq!(BasisPoints::from_ratio(0.0).raw(), 0);
    }

    #[test]
    fn basis_points_clamp_out_of_range_and_reject_nonfinite() {
        assert_eq!(BasisPoints::from_ratio(1.5).raw(), 10_000);
        assert_eq!(BasisPoints::from_ratio(-0.2).raw(), 0);
        assert_eq!(BasisPoints::from_ratio(f64::NAN).raw(), 0);
        assert!(BasisPoints::try_from_ratio(1.5).is_err());
        assert!(BasisPoints::try_from_ratio(0.25).is_ok());
    }

    #[test]
    fn basis_points_round_trip_through_ratio() {
        for raw in [0u16, 1, 1234, 5000, 9999, 10_000] {
            let bp = BasisPoints::from_ratio(BasisPoints(raw).as_ratio());
            assert_eq!(bp.raw(), raw);
        }
    }

    #[test]
    fn timestamp_renders_known_instants() {
        assert_eq!(
            Timestamp::from_unix_millis(0).as_str(),
            "1970-01-01T00:00:00.000Z"
        );
        // 2026-01-01T00:00:00Z
        assert_eq!(
            Timestamp::from_unix_millis(1_767_225_600_000).as_str(),
            "2026-01-01T00:00:00.000Z"
        );
        // A leap day, to exercise civil_from_days properly.
        assert_eq!(
            Timestamp::from_unix_millis(1_709_164_800_123).as_str(),
            "2024-02-29T00:00:00.123Z"
        );
    }

    #[test]
    fn timestamp_now_has_the_canonical_shape() {
        assert!(Timestamp::parse(Timestamp::now().as_str()).is_ok());
    }

    #[test]
    fn timestamp_parse_rejects_other_precisions() {
        assert!(Timestamp::parse("2026-01-01T00:00:00Z").is_err());
        assert!(Timestamp::parse("2026-01-01T00:00:00.123456Z").is_err());
        assert!(Timestamp::parse("2026-01-01T00:00:00.123+00:00").is_err());
        assert!(Timestamp::parse("2026-01-01T00:00:00.123Z").is_ok());
    }

    #[test]
    fn assemble_computes_compression_ratio() {
        let context = Context::assemble(
            "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            "answer a question",
            4096,
            10_000,
            2_500,
            vec![segment(SegmentRole::Goal, "ship it", None)],
            Provenance::now(Component::Axiom, "axiom:context/compress"),
        );
        assert_eq!(context.compression_ratio_bp.raw(), 2_500);
    }

    #[test]
    fn assemble_flags_an_unsourced_quoted_segment_as_ungrounded() {
        let context = Context::assemble(
            "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            "answer a question",
            4096,
            100,
            50,
            vec![
                segment(SegmentRole::Goal, "ship it", None),
                segment(SegmentRole::Memory, "the build always fails on Tuesdays", None),
            ],
            Provenance::now(Component::Axiom, "axiom:context/compress"),
        );
        assert!(!context.grounded);
        assert_eq!(context.grounding_failures.len(), 1);
        assert!(context.grounding_failures[0].contains("no source_id"));
    }

    #[test]
    fn assemble_is_grounded_when_every_quote_is_sourced() {
        let context = Context::assemble(
            "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            "answer a question",
            4096,
            100,
            50,
            vec![
                segment(SegmentRole::Goal, "ship it", None),
                segment(SegmentRole::Memory, "builds fail on missing deps", Some("66666666-6666-4666-8666-666666666666")),
                segment(SegmentRole::Belief, "tests catch regressions", Some("55555555-5555-4555-8555-555555555555")),
            ],
            Provenance::now(Component::Axiom, "axiom:context/compress"),
        );
        assert!(context.grounded);
        assert!(context.grounding_failures.is_empty());
        assert_eq!(context.memory_ids, vec!["66666666-6666-4666-8666-666666666666"]);
        assert_eq!(context.belief_ids, vec!["55555555-5555-4555-8555-555555555555"]);
    }

    #[test]
    fn assemble_handles_an_empty_input_without_dividing_by_zero() {
        let context = Context::assemble(
            "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            "nothing to compress",
            4096,
            0,
            0,
            vec![],
            Provenance::now(Component::Axiom, "axiom:context/compress"),
        );
        assert_eq!(context.compression_ratio_bp, BasisPoints::ONE);
        assert!(context.grounded);
    }

    #[test]
    fn a_sealed_context_verifies_and_carries_no_floats() {
        let context = Context::assemble(
            "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            "answer a question",
            4096,
            10_000,
            2_500,
            vec![segment(SegmentRole::Goal, "ship it", None)],
            Provenance::now(Component::Axiom, "axiom:context/compress"),
        );
        let sealed = context.seal().unwrap();
        assert!(canonical::verify_seal(&sealed).unwrap());
        // Sealing would have failed on a float; proving the type is wire-safe.
        assert!(canonical::to_canonical(&sealed).is_ok());
    }
}
