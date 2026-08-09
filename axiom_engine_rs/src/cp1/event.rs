//! CP/1 events — the organism's nervous system.
//!
//! Subsystems announce facts; nothing calls anything directly. An event names
//! the canonical document it is about (`subject_id`/`subject_type`) and carries
//! only enough payload to be readable in a log — the full document is fetched
//! by id when detail is needed. Keeping payloads to scalars is what lets an
//! event log stay cheap enough to always be on.
//!
//! The event set is closed. Consumers switch exhaustively over [`EventKind`],
//! so an open enumeration would make every consumer's "unknown event" branch an
//! untested code path; adding an event is a CP/1 version change (SPEC.md
//! section 8).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::canonical::{self, CanonicalError};
use super::types::{Component, Provenance, Timestamp};

/// Every event the organism can emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventKind {
    /// EVE perceived an environment.
    ObservationRecorded,
    /// EVE situated an observation in goal, action and outcome.
    ExperienceCreated,
    /// AXIOM compressed a working set.
    ContextCompressed,
    /// AXIOM could not support a claim from the supplied evidence.
    GroundingFailed,
    /// ADAM distilled experiences into a durable memory.
    MemoryConsolidated,
    /// ADAM formed, reinforced, weakened or retracted a belief.
    BeliefUpdated,
    /// ADAM promoted a skill.
    SkillLearned,
    /// ADAM produced a self-assessment across subsystems.
    ReflectionCompleted,
    /// ADAM proposed a change to genome, skills or beliefs.
    MutationProposed,
    /// EVE finished a deterministic scenario run.
    SimulationCompleted,
    /// EVE scored a mutation against baseline and candidate runs.
    FitnessMeasured,
    /// ADAM applied a proposal that passed governance.
    MutationAccepted,
    /// ADAM refused a proposal.
    MutationRejected,
    /// ADAM appended a new immutable genome version.
    GenomeCommitted,
}

impl EventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::ObservationRecorded => "ObservationRecorded",
            EventKind::ExperienceCreated => "ExperienceCreated",
            EventKind::ContextCompressed => "ContextCompressed",
            EventKind::GroundingFailed => "GroundingFailed",
            EventKind::MemoryConsolidated => "MemoryConsolidated",
            EventKind::BeliefUpdated => "BeliefUpdated",
            EventKind::SkillLearned => "SkillLearned",
            EventKind::ReflectionCompleted => "ReflectionCompleted",
            EventKind::MutationProposed => "MutationProposed",
            EventKind::SimulationCompleted => "SimulationCompleted",
            EventKind::FitnessMeasured => "FitnessMeasured",
            EventKind::MutationAccepted => "MutationAccepted",
            EventKind::MutationRejected => "MutationRejected",
            EventKind::GenomeCommitted => "GenomeCommitted",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "ObservationRecorded" => EventKind::ObservationRecorded,
            "ExperienceCreated" => EventKind::ExperienceCreated,
            "ContextCompressed" => EventKind::ContextCompressed,
            "GroundingFailed" => EventKind::GroundingFailed,
            "MemoryConsolidated" => EventKind::MemoryConsolidated,
            "BeliefUpdated" => EventKind::BeliefUpdated,
            "SkillLearned" => EventKind::SkillLearned,
            "ReflectionCompleted" => EventKind::ReflectionCompleted,
            "MutationProposed" => EventKind::MutationProposed,
            "SimulationCompleted" => EventKind::SimulationCompleted,
            "FitnessMeasured" => EventKind::FitnessMeasured,
            "MutationAccepted" => EventKind::MutationAccepted,
            "MutationRejected" => EventKind::MutationRejected,
            "GenomeCommitted" => EventKind::GenomeCommitted,
            _ => return None,
        })
    }

    /// The component that is permitted to emit this event.
    ///
    /// Ownership of an event follows ownership of the concept it announces, so
    /// this is checkable: an `ObservationRecorded` from ADAM means ADAM minted
    /// an EVE-owned fact, which the boundary should reject rather than record.
    pub fn emitter(self) -> Component {
        match self {
            EventKind::ObservationRecorded
            | EventKind::ExperienceCreated
            | EventKind::SimulationCompleted
            | EventKind::FitnessMeasured => Component::Eve,
            EventKind::ContextCompressed | EventKind::GroundingFailed => Component::Axiom,
            EventKind::MemoryConsolidated
            | EventKind::BeliefUpdated
            | EventKind::SkillLearned
            | EventKind::ReflectionCompleted
            | EventKind::MutationProposed
            | EventKind::MutationAccepted
            | EventKind::MutationRejected
            | EventKind::GenomeCommitted => Component::Adam,
        }
    }

    /// Every event, for exhaustive iteration in tests and catalog rendering.
    pub const ALL: [EventKind; 14] = [
        EventKind::ObservationRecorded,
        EventKind::ExperienceCreated,
        EventKind::ContextCompressed,
        EventKind::GroundingFailed,
        EventKind::MemoryConsolidated,
        EventKind::BeliefUpdated,
        EventKind::SkillLearned,
        EventKind::ReflectionCompleted,
        EventKind::MutationProposed,
        EventKind::SimulationCompleted,
        EventKind::FitnessMeasured,
        EventKind::MutationAccepted,
        EventKind::MutationRejected,
        EventKind::GenomeCommitted,
    ];
}

/// The canonical type an event is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubjectType {
    Identity,
    Genome,
    Capability,
    Belief,
    Memory,
    Skill,
    Mutation,
    Reflection,
    Observation,
    Experience,
    FitnessResult,
    Context,
}

/// A payload member. Scalars only, by design (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PayloadValue {
    Bool(bool),
    Int(i64),
    Text(String),
}

impl From<bool> for PayloadValue {
    fn from(v: bool) -> Self {
        PayloadValue::Bool(v)
    }
}

impl From<i64> for PayloadValue {
    fn from(v: i64) -> Self {
        PayloadValue::Int(v)
    }
}

impl From<u32> for PayloadValue {
    fn from(v: u32) -> Self {
        PayloadValue::Int(i64::from(v))
    }
}

impl From<&str> for PayloadValue {
    fn from(v: &str) -> Self {
        PayloadValue::Text(v.to_string())
    }
}

impl From<String> for PayloadValue {
    fn from(v: String) -> Self {
        PayloadValue::Text(v)
    }
}

/// One announced fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    pub cp: String,
    #[serde(rename = "type")]
    pub kind: EventKind,
    pub id: String,
    pub occurred_at: Timestamp,
    pub actor: Component,
    pub subject_id: String,
    pub subject_type: SubjectType,
    /// Shared by every event of one developmental turn, which is what makes a
    /// full Observe-through-Commit cycle reconstructible from the log.
    pub correlation_id: String,
    /// The event that caused this one, when there was one. Together with
    /// `correlation_id` this makes a turn a tree rather than a bag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub causation_id: Option<String>,
    /// `BTreeMap` rather than `HashMap` so serialization is already key-sorted,
    /// matching canonical form without a second pass.
    pub payload: BTreeMap<String, PayloadValue>,
    pub provenance: Provenance,
}

impl Event {
    /// Build an event, or refuse it when the provenance disagrees with the
    /// kind's owning emitter.
    ///
    /// `actor` is derived from [`EventKind::emitter`] while `provenance` comes
    /// from the caller, and the two carry the same concept. Left unchecked they
    /// can disagree — an `ObservationRecorded` with `actor: "eve"` and
    /// `provenance.authored_by: "adam"` seals cleanly, satisfies the schema
    /// because both are valid components, and passes conformance, which never
    /// compares them. This is the case [`EventKind::emitter`] exists to make
    /// rejectable, so it is rejected here, where both values are known.
    pub fn try_new(
        kind: EventKind,
        subject_id: impl Into<String>,
        subject_type: SubjectType,
        correlation_id: impl Into<String>,
        payload: BTreeMap<String, PayloadValue>,
        provenance: Provenance,
    ) -> Result<Self, EventError> {
        if provenance.authored_by != kind.emitter() {
            return Err(EventError::EmitterMismatch {
                kind,
                expected: kind.emitter(),
                found: provenance.authored_by,
            });
        }
        Ok(Self::new(
            kind,
            subject_id,
            subject_type,
            correlation_id,
            payload,
            provenance,
        ))
    }

    /// Build an event, stamping it with a fresh id and the current instant.
    ///
    /// Prefer [`Event::try_new`] when the provenance comes from a caller rather
    /// than being constructed alongside the event: it rejects a provenance
    /// whose author contradicts the kind's emitter.
    pub fn new(
        kind: EventKind,
        subject_id: impl Into<String>,
        subject_type: SubjectType,
        correlation_id: impl Into<String>,
        payload: BTreeMap<String, PayloadValue>,
        provenance: Provenance,
    ) -> Self {
        Self {
            cp: "cp1".to_string(),
            kind,
            id: uuid::Uuid::new_v4().to_string(),
            occurred_at: Timestamp::now(),
            actor: kind.emitter(),
            subject_id: subject_id.into(),
            subject_type,
            correlation_id: correlation_id.into(),
            causation_id: None,
            payload,
            provenance,
        }
    }

    /// Record which event triggered this one.
    pub fn caused_by(mut self, event_id: impl Into<String>) -> Self {
        self.causation_id = Some(event_id.into());
        self
    }

    /// Serialize to a sealed `serde_json::Value` ready for transport.
    pub fn seal(&self) -> Result<serde_json::Value, CanonicalError> {
        let mut value = serde_json::to_value(self).expect("Event always serializes");
        canonical::seal(&mut value)?;
        Ok(value)
    }
}

/// Why an event could not be constructed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventError {
    /// The supplied provenance was authored by a component that does not own
    /// this event kind.
    EmitterMismatch {
        kind: EventKind,
        expected: Component,
        found: Component,
    },
}

impl std::fmt::Display for EventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventError::EmitterMismatch {
                kind,
                expected,
                found,
            } => write!(
                f,
                "{} may only be emitted by {}, but the provenance is authored by {}",
                kind.as_str(),
                expected.as_str(),
                found.as_str()
            ),
        }
    }
}

impl std::error::Error for EventError {}

/// Anything that accepts emitted events.
///
/// A trait rather than a concrete bus so the emitting subsystem never learns
/// where its events go — which is the property that keeps the event system a
/// nervous system rather than another call graph.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: &Event);
}

/// An `EventSink` that keeps events in memory. Intended for tests and for
/// short-lived processes that report their event log on exit.
#[derive(Debug, Default)]
pub struct RecordingSink {
    events: std::sync::Mutex<Vec<Event>>,
}

impl RecordingSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> Vec<Event> {
        // The lock guards a push and a clone, so the data is never left
        // partially written. Recovering from poisoning keeps one panicking
        // thread from turning every later `emit` into a panic of its own.
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn kinds(&self) -> Vec<EventKind> {
        self.events().iter().map(|e| e.kind).collect()
    }
}

impl EventSink for RecordingSink {
    fn emit(&self, event: &Event) {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(pairs: &[(&str, PayloadValue)]) -> BTreeMap<String, PayloadValue> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn every_event_name_round_trips() {
        for kind in EventKind::ALL {
            assert_eq!(EventKind::parse(kind.as_str()), Some(kind));
        }
    }

    #[test]
    fn event_names_serialize_as_bare_strings() {
        let json = serde_json::to_string(&EventKind::GenomeCommitted).unwrap();
        assert_eq!(json, "\"GenomeCommitted\"");
    }

    #[test]
    fn unknown_event_names_are_rejected() {
        assert_eq!(EventKind::parse("GenomeDeleted"), None);
    }

    #[test]
    fn actor_defaults_to_the_owning_emitter() {
        let event = Event::new(
            EventKind::ContextCompressed,
            "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            SubjectType::Context,
            "0f0f0f0f-0f0f-4f0f-8f0f-0f0f0f0f0f0f",
            payload(&[("tokens_after", PayloadValue::Int(3780))]),
            Provenance::now(Component::Axiom, "axiom:context/compress"),
        );
        assert_eq!(event.actor, Component::Axiom);
        assert_eq!(EventKind::ContextCompressed.emitter(), Component::Axiom);
    }

    #[test]
    fn emitter_assignment_covers_all_three_components() {
        let emitters: std::collections::BTreeSet<&str> = EventKind::ALL
            .iter()
            .map(|k| k.emitter().as_str())
            .collect();
        assert_eq!(
            emitters,
            ["adam", "axiom", "eve"].into_iter().collect::<std::collections::BTreeSet<_>>()
        );
    }

    #[test]
    fn a_sealed_event_verifies_and_omits_absent_causation() {
        let event = Event::new(
            EventKind::GroundingFailed,
            "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            SubjectType::Context,
            "0f0f0f0f-0f0f-4f0f-8f0f-0f0f0f0f0f0f",
            payload(&[("claim", PayloadValue::Text("unsupported".into()))]),
            Provenance::now(Component::Axiom, "axiom:grounding/verify"),
        );
        let sealed = event.seal().unwrap();
        assert!(canonical::verify_seal(&sealed).unwrap());
        assert!(
            sealed.get("causation_id").is_none(),
            "an absent causation must be an absent key, never a null"
        );
    }

    #[test]
    fn causation_is_recorded_when_set() {
        let event = Event::new(
            EventKind::ContextCompressed,
            "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            SubjectType::Context,
            "0f0f0f0f-0f0f-4f0f-8f0f-0f0f0f0f0f0f",
            BTreeMap::new(),
            Provenance::now(Component::Axiom, "axiom:context/compress"),
        )
        .caused_by("1f1f1f1f-1f1f-4f1f-8f1f-1f1f1f1f1f1f");
        let sealed = event.seal().unwrap();
        assert_eq!(
            sealed["causation_id"].as_str(),
            Some("1f1f1f1f-1f1f-4f1f-8f1f-1f1f1f1f1f1f")
        );
    }

    #[test]
    fn try_new_refuses_provenance_that_contradicts_the_owning_emitter() {
        let refused = Event::try_new(
            EventKind::ObservationRecorded,
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            SubjectType::Observation,
            "0f0f0f0f-0f0f-4f0f-8f0f-0f0f0f0f0f0f",
            BTreeMap::new(),
            // ObservationRecorded is EVE's to emit.
            Provenance::now(Component::Adam, "adam:organism"),
        );
        let err = refused.expect_err("an ADAM-authored EVE event must be refused");
        assert!(err.to_string().contains("may only be emitted by eve"));
    }

    #[test]
    fn try_new_accepts_provenance_from_the_owning_emitter() {
        assert!(Event::try_new(
            EventKind::ContextCompressed,
            "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            SubjectType::Context,
            "0f0f0f0f-0f0f-4f0f-8f0f-0f0f0f0f0f0f",
            BTreeMap::new(),
            Provenance::now(Component::Axiom, "axiom:context/compress"),
        )
        .is_ok());
    }

    #[test]
    fn recording_sink_survives_a_poisoned_lock() {
        // A telemetry sink must not convert one thread's panic into a
        // process-wide cascade of panics on every later emit.
        let sink = std::sync::Arc::new(RecordingSink::new());
        let poisoner = std::sync::Arc::clone(&sink);
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.events.lock().unwrap();
            panic!("poison the lock");
        })
        .join();

        sink.emit(&Event::new(
            EventKind::ContextCompressed,
            "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            SubjectType::Context,
            "0f0f0f0f-0f0f-4f0f-8f0f-0f0f0f0f0f0f",
            BTreeMap::new(),
            Provenance::now(Component::Axiom, "axiom:test"),
        ));
        assert_eq!(sink.events().len(), 1);
    }

    #[test]
    fn recording_sink_captures_emission_order() {
        let sink = RecordingSink::new();
        for kind in [EventKind::ContextCompressed, EventKind::GroundingFailed] {
            sink.emit(&Event::new(
                kind,
                "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
                SubjectType::Context,
                "0f0f0f0f-0f0f-4f0f-8f0f-0f0f0f0f0f0f",
                BTreeMap::new(),
                Provenance::now(Component::Axiom, "axiom:test"),
            ));
        }
        assert_eq!(
            sink.kinds(),
            vec![EventKind::ContextCompressed, EventKind::GroundingFailed]
        );
    }
}
