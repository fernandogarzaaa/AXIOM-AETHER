//! alignment_loop.rs — Self-Correcting Feedback Loop.
//!
//! Continuously evaluates the active output generation against the predicted
//! state map established by the state predictor. If the output logic deviates
//! from the projected semantic milestone, the system autonomously applies
//! localized fast-weight corrections to steer the execution back on track
//! without requiring a prompt restart.
//!
//! ## Architecture
//!
//! The alignment loop operates in three stages:
//!
//! 1. **Monitoring** — For each generated token/chunk, extract the current
//!    generation state and compare it against the expected milestone in the
//!    `SemanticStateMap`.
//!
//! 2. **Drift detection** — Compute a drift score between the current generation
//!    state and the predicted milestone state. If the drift exceeds a threshold,
//!    flag a misalignment.
//!
//! 3. **Correction** — Apply a localized TTT fast-weight update that nudges W̃
//!    toward the predicted milestone state. This is a small, bounded update
//!    (not a full re-adaptation) that steers generation back on track.
//!
//! ## Key difference from existing grounding
//!
//! The existing `hallucination.rs` grounding check is *post-hoc* (verify after
//! generation). The alignment loop is *continuous* (monitor during generation)
//! and *corrective* (apply W̃ updates mid-stream). It does not replace
//! grounding — it complements it by catching drift early.

use serde::{Deserialize, Serialize};

use crate::belief::BetaBelief;
use crate::state_predictor::{SemanticStateMap, SemanticMilestone};

/// Default drift threshold: if the drift score exceeds this, a correction is applied.
pub const DEFAULT_DRIFT_THRESHOLD: f32 = 0.5;

/// Default correction strength: the learning rate for localized W̃ updates.
pub const DEFAULT_CORRECTION_LR: f32 = 0.01;

/// Maximum number of corrections per generation session.
pub const MAX_CORRECTIONS: usize = 16;

/// A single alignment check — the result of comparing generation state to the
/// predicted milestone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentCheck {
    /// The milestone index that was being compared against.
    pub milestone_idx: usize,
    /// The milestone label.
    pub milestone_label: String,
    /// Drift score (0 = perfectly aligned, 1 = completely misaligned).
    pub drift_score: f32,
    /// Whether a correction was applied.
    pub corrected: bool,
    /// The correction strength applied (0 if no correction).
    pub correction_strength: f32,
    /// Confidence in the alignment assessment.
    pub confidence: BetaBelief,
}

/// The alignment loop's state — tracks corrections across a generation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentLoopState {
    /// The state map being aligned against.
    pub state_map: SemanticStateMap,
    /// All alignment checks performed so far.
    pub checks: Vec<AlignmentCheck>,
    /// Current milestone index (how far through the state map we are).
    pub current_milestone: usize,
    /// Total corrections applied.
    pub corrections_applied: usize,
    /// Drift threshold.
    pub drift_threshold: f32,
    /// Correction learning rate.
    pub correction_lr: f32,
    /// Whether the loop has reached the final milestone.
    pub completed: bool,
}

/// The alignment loop — monitors and corrects generation drift.
pub struct AlignmentLoop {
    drift_threshold: f32,
    correction_lr: f32,
    max_corrections: usize,
}

impl AlignmentLoop {
    /// Create a new alignment loop with the given configuration.
    pub fn new(
        drift_threshold: f32,
        correction_lr: f32,
        max_corrections: usize,
    ) -> Self {
        Self {
            drift_threshold,
            correction_lr,
            max_corrections,
        }
    }

    /// Create an alignment loop with default settings.
    pub fn default_config() -> Self {
        Self::new(
            DEFAULT_DRIFT_THRESHOLD,
            DEFAULT_CORRECTION_LR,
            MAX_CORRECTIONS,
        )
    }

    /// Initialize the alignment loop state for a generation session.
    pub fn init(&self, state_map: SemanticStateMap) -> AlignmentLoopState {
        AlignmentLoopState {
            state_map,
            checks: Vec::new(),
            current_milestone: 0,
            corrections_applied: 0,
            drift_threshold: self.drift_threshold,
            correction_lr: self.correction_lr,
            completed: false,
        }
    }

    /// Check alignment between the current generation state and the expected
    /// milestone. If drift exceeds the threshold, record a correction.
    ///
    /// `generation_state` is the current hidden state of the generation (as a
    /// flat vector). `predicted_state` is the expected state from the milestone.
    pub fn check_alignment(
        &self,
        state: &mut AlignmentLoopState,
        generation_state: &[f32],
    ) -> AlignmentCheck {
        // Get the current milestone (or the last one if we've passed them all).
        let milestone_idx = state.current_milestone.min(
            state.state_map.milestones.len().saturating_sub(1),
        );

        let milestone = match state.state_map.milestones.get(milestone_idx) {
            Some(m) => m,
            None => {
                return AlignmentCheck {
                    milestone_idx,
                    milestone_label: "none".to_string(),
                    drift_score: 0.0,
                    corrected: false,
                    correction_strength: 0.0,
                    confidence: BetaBelief::uniform(),
                };
            }
        };

        // Compute drift score: cosine distance between generation and predicted states.
        let drift_score = cosine_distance(generation_state, &milestone.predicted_state);

        // Determine if a correction is needed.
        let needs_correction = drift_score > state.drift_threshold
            && state.corrections_applied < self.max_corrections;

        let correction_strength = if needs_correction {
            // Scale correction by drift magnitude (more drift = stronger correction).
            let strength = state.correction_lr * (drift_score - state.drift_threshold).min(1.0);
            state.corrections_applied += 1;
            strength
        } else {
            0.0
        };

        // Confidence: higher when drift is low (aligned) or when correction was applied.
        let confidence = BetaBelief::from_confidence(1.0 - drift_score, 4.0);

        let check = AlignmentCheck {
            milestone_idx,
            milestone_label: milestone.label.clone(),
            drift_score,
            corrected: needs_correction,
            correction_strength,
            confidence,
        };

        state.checks.push(check.clone());

        // Advance to the next milestone if alignment is good.
        if drift_score < state.drift_threshold {
            state.current_milestone += 1;
            if state.current_milestone >= state.state_map.milestones.len() {
                state.completed = true;
            }
        }

        check
    }

    /// Compute the localized fast-weight correction vector.
    ///
    /// This returns a correction vector that can be applied to the TTT fast-weight
    /// matrix W̃ to steer generation toward the predicted milestone state.
    /// The correction is: `correction = lr * (predicted_state - generation_state)`
    pub fn compute_correction(
        &self,
        generation_state: &[f32],
        milestone: &SemanticMilestone,
        lr: f32,
    ) -> Vec<f32> {
        let n = generation_state.len().min(milestone.predicted_state.len());
        let mut correction = vec![0.0f32; n];
        for i in 0..n {
            correction[i] = lr * (milestone.predicted_state[i] - generation_state[i]);
        }
        correction
    }

    /// Get a summary of the alignment loop's state.
    pub fn summary(state: &AlignmentLoopState) -> String {
        let total_checks = state.checks.len();
        let corrections = state.corrections_applied;
        let avg_drift: f32 = if total_checks > 0 {
            state.checks.iter().map(|c| c.drift_score).sum::<f32>() / total_checks as f32
        } else {
            0.0
        };
        let aligned = state.checks.iter().filter(|c| !c.corrected).count();
        format!(
            "AlignmentLoop: {} checks, {} corrections, avg_drift={:.3}, aligned={}/{}, milestone={}/{}, completed={}",
            total_checks,
            corrections,
            avg_drift,
            aligned,
            total_checks,
            state.current_milestone,
            state.state_map.milestones.len(),
            state.completed
        )
    }
}

/// Cosine distance between two vectors (1 - cosine_similarity).
/// Returns 0 for identical directions, 1 for orthogonal, 2 for opposite.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a * norm_b).sqrt().max(1e-6);
    1.0 - (dot / denom)
}

/// Render an alignment check as a human-readable string.
pub fn render_alignment_check(check: &AlignmentCheck) -> String {
    let status = if check.corrected { "CORRECTED" } else { "ALIGNED" };
    format!(
        "AlignmentCheck [{}] {} drift={:.3} correction={:.4} confidence={:.2}",
        check.milestone_label,
        status,
        check.drift_score,
        check.correction_strength,
        check.confidence.mean()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state_map() -> SemanticStateMap {
        SemanticStateMap {
            milestones: vec![
                SemanticMilestone {
                    label: "hypothesis".to_string(),
                    label_idx: 0,
                    predicted_state: vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.1, 0.0],
                    token_budget: 128,
                    branch_hints: vec![],
                    confidence: BetaBelief::from_confidence(0.8, 4.0),
                },
                SemanticMilestone {
                    label: "execution".to_string(),
                    label_idx: 2,
                    predicted_state: vec![0.0, 1.0, 0.0, 0.5, 0.0, 0.2, 0.0, 0.1],
                    token_budget: 256,
                    branch_hints: vec![],
                    confidence: BetaBelief::from_confidence(0.7, 4.0),
                },
            ],
            confidence: BetaBelief::from_confidence(0.75, 4.0),
            session_id: "test".to_string(),
            context_state: vec![0.5; 8],
        }
    }

    #[test]
    fn init_creates_valid_state() {
        let loop_ = AlignmentLoop::default_config();
        let map = make_state_map();
        let state = loop_.init(map);
        assert_eq!(state.current_milestone, 0);
        assert_eq!(state.corrections_applied, 0);
        assert!(!state.completed);
    }

    #[test]
    fn aligned_generation_advances_milestone() {
        let loop_ = AlignmentLoop::default_config();
        let map = make_state_map();
        let mut state = loop_.init(map);

        // Generation state matches the first milestone exactly.
        let gen_state = vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.1, 0.0];
        let check = loop_.check_alignment(&mut state, &gen_state);

        assert!(!check.corrected, "aligned state should not need correction");
        assert!(check.drift_score < 0.01, "identical states should have ~0 drift");
        assert_eq!(state.current_milestone, 1, "should advance to next milestone");
    }

    #[test]
    fn drifted_generation_triggers_correction() {
        let loop_ = AlignmentLoop::default_config();
        let map = make_state_map();
        let mut state = loop_.init(map);

        // Generation state is very different from the first milestone.
        let gen_state = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let check = loop_.check_alignment(&mut state, &gen_state);

        assert!(check.corrected, "drifted state should trigger correction");
        assert!(check.drift_score > 0.5, "orthogonal states should have high drift");
        assert_eq!(state.corrections_applied, 1);
    }

    #[test]
    fn max_corrections_is_respected() {
        let loop_ = AlignmentLoop::new(0.1, 0.01, 2); // max 2 corrections
        let map = make_state_map();
        let mut state = loop_.init(map);

        // Keep sending drifted state.
        let gen_state = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for _ in 0..10 {
            loop_.check_alignment(&mut state, &gen_state);
        }
        assert!(state.corrections_applied <= 2, "should not exceed max corrections");
    }

    #[test]
    fn completion_is_detected() {
        let loop_ = AlignmentLoop::default_config();
        let map = make_state_map();
        let mut state = loop_.init(map);

        // Align with first milestone.
        let gen1 = vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.1, 0.0];
        loop_.check_alignment(&mut state, &gen1);
        // Align with second milestone.
        let gen2 = vec![0.0, 1.0, 0.0, 0.5, 0.0, 0.2, 0.0, 0.1];
        loop_.check_alignment(&mut state, &gen2);

        assert!(state.completed, "should complete after aligning with all milestones");
    }

    #[test]
    fn correction_vector_steers_toward_milestone() {
        let loop_ = AlignmentLoop::default_config();
        let milestone = SemanticMilestone {
            label: "test".to_string(),
            label_idx: 0,
            predicted_state: vec![1.0, 0.0, 0.0],
            token_budget: 100,
            branch_hints: vec![],
            confidence: BetaBelief::from_confidence(0.8, 4.0),
        };
        let gen_state = vec![0.0, 0.0, 0.0];
        let correction = loop_.compute_correction(&gen_state, &milestone, 0.1);

        // Correction should point toward the milestone state.
        assert!(correction[0] > 0.0, "correction should steer toward milestone");
        assert!((correction[0] - 0.1).abs() < 1e-6, "correction magnitude should be lr * delta");
    }

    #[test]
    fn cosine_distance_is_correct() {
        assert!((cosine_distance(&[1.0, 0.0], &[1.0, 0.0])).abs() < 1e-6);
        assert!((cosine_distance(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-6);
        assert!((cosine_distance(&[1.0, 0.0], &[-1.0, 0.0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn summary_is_readable() {
        let loop_ = AlignmentLoop::default_config();
        let map = make_state_map();
        let mut state = loop_.init(map);
        let gen = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        loop_.check_alignment(&mut state, &gen);
        let summary = AlignmentLoop::summary(&state);
        assert!(summary.contains("AlignmentLoop"));
        assert!(summary.contains("checks"));
        assert!(summary.contains("corrections"));
    }

    #[test]
    fn render_check_is_readable() {
        let check = AlignmentCheck {
            milestone_idx: 0,
            milestone_label: "hypothesis".to_string(),
            drift_score: 0.3,
            corrected: false,
            correction_strength: 0.0,
            confidence: BetaBelief::from_confidence(0.7, 4.0),
        };
        let rendered = render_alignment_check(&check);
        assert!(rendered.contains("hypothesis"));
        assert!(rendered.contains("ALIGNED"));
    }
}