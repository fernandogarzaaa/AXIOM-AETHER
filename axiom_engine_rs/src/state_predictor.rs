//! state_predictor.rs — State-Based Look-Ahead Prediction.
//!
//! Forecasts high-level semantic states (hypothesis generation, code execution,
//! verification) rather than immediately predicting the next standard language
//! token. This reduces the immediate computational load by mapping out the
//! cognitive milestones of a task before generating the granular code to fulfill
//! it.
//!
//! ## Architecture
//!
//! The predictor operates on the TTT-adapted fast-weight matrix W̃. After the
//! context has been absorbed (via `adapt_session`), the predictor:
//!
//! 1. Extracts a **context state vector** from the final hidden state of the
//!    adapted session.
//! 2. Projects it through a learned **state prediction head** (Linear(d -> d))
//!    to produce a sequence of predicted semantic milestones.
//! 3. Each milestone carries a label, a predicted hidden state, an estimated
//!    token budget, and optional branch hints for trajectory sampling.
//!
//! The milestone labels are drawn from a fixed vocabulary of cognitive phases
//! that cover the typical reasoning lifecycle:
//!
//! - `hypothesis` — initial hypothesis or plan formation
//! - `decomposition` — breaking the problem into sub-tasks
//! - `execution` — generating code or performing computation
//! - `verification` — checking output against requirements
//! - `refinement` — iterative improvement based on feedback
//! - `synthesis` — combining results into a final answer

use candle_core::{Device, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::belief::BetaBelief;
use crate::config::AxiomConfig;

/// The fixed vocabulary of cognitive milestone labels.
pub const MILESTONE_LABELS: &[&str] = &[
    "hypothesis",
    "decomposition",
    "execution",
    "verification",
    "refinement",
    "synthesis",
];

/// Maximum number of milestones to predict in a single look-ahead.
pub const MAX_MILESTONES: usize = 8;

/// A single predicted cognitive milestone in the state map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMilestone {
    /// Human-readable label from `MILESTONE_LABELS`.
    pub label: String,
    /// Index into `MILESTONE_LABELS`.
    pub label_idx: usize,
    /// Predicted hidden state projection [d_model] (serialized as Vec<f32>).
    pub predicted_state: Vec<f32>,
    /// Estimated token budget to reach this milestone from the previous one.
    pub token_budget: usize,
    /// Optional branch hints for trajectory sampling (e.g., "try approach A",
    /// "try approach B").
    pub branch_hints: Vec<String>,
    /// Confidence in this milestone prediction (Beta belief over viability).
    pub confidence: BetaBelief,
}

/// A complete state map — the sequence of predicted cognitive milestones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticStateMap {
    /// The predicted milestone sequence.
    pub milestones: Vec<SemanticMilestone>,
    /// Overall confidence in the state map.
    pub confidence: BetaBelief,
    /// Session ID this map was generated for.
    pub session_id: String,
    /// The context state vector used to generate this map.
    pub context_state: Vec<f32>,
}

#[allow(clippy::derivable_impls)]
impl Default for SemanticStateMap {
    fn default() -> Self {
        Self {
            milestones: Vec::new(),
            confidence: BetaBelief::default(),
            session_id: String::new(),
            context_state: Vec::new(),
        }
    }
}

/// The state prediction head — a learned projection from context state to
/// milestone predictions.
///
/// Architecture:
/// - `state_proj`: Linear(d_model -> d_model) — projects the context state
/// - `label_proj`: Linear(d_model -> num_labels) — classifies the milestone label
/// - `budget_proj`: Linear(d_model -> 1) — predicts the token budget
///
/// The predictor runs autoregressively: each predicted state feeds back as
/// input to predict the next milestone, up to `MAX_MILESTONES`.
pub struct StatePredictor {
    state_proj: Linear,
    label_proj: Linear,
    budget_proj: Linear,
    d_model: usize,
}

impl StatePredictor {
    /// Construct the prediction head from a VarBuilder (checkpoint-loaded
    /// weights) and the model config.
    pub fn new(vs: VarBuilder, config: &AxiomConfig) -> Result<Self> {
        let d = config.d_model;
        Ok(Self {
            state_proj: candle_nn::linear(d, d, vs.pp("state_proj"))?,
            label_proj: candle_nn::linear(d, MILESTONE_LABELS.len(), vs.pp("label_proj"))?,
            budget_proj: candle_nn::linear(d, 1, vs.pp("budget_proj"))?,
            d_model: d,
        })
    }

    /// Predict a state map from a context state vector.
    ///
    /// `context_state` is the final hidden state from the TTT-adapted session
    /// (shape [d_model]). The predictor runs autoregressively for up to
    /// `max_milestones` steps, producing a sequence of predicted milestones.
    ///
    /// `device` is the compute device (CPU or GPU).
    pub fn predict(
        &self,
        context_state: &Tensor,
        session_id: &str,
        max_milestones: usize,
        _device: &Device,
    ) -> Result<SemanticStateMap> {
        let max_milestones = max_milestones.min(MAX_MILESTONES);
        let mut milestones = Vec::with_capacity(max_milestones);
        let mut current_state = context_state.clone();

        for step in 0..max_milestones {
            // Project the current state to the next milestone state.
            let projected = self.state_proj.forward(&current_state)?;

            // Classify the milestone label.
            let label_logits = self.label_proj.forward(&projected)?;
            let label_probs = candle_nn::ops::softmax(&label_logits, D::Minus1)?;
            let label_idx = argmax(&label_probs)?;
            let label = MILESTONE_LABELS
                .get(label_idx)
                .unwrap_or(&"unknown")
                .to_string();

            // Predict the token budget (clamped to [16, 4096]).
            let budget_logit = self.budget_proj.forward(&projected)?;
            let budget_raw = budget_logit.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;
            let token_budget = (budget_raw.exp().clamp(16.0, 4096.0)) as usize;

            // Extract the predicted state as a vector for serialization.
            let predicted_state_vec = projected
                .squeeze(0)?
                .to_vec1::<f32>()?;

            // Confidence: use the max softmax probability as the point estimate.
            let max_prob = max_value(&label_probs)?;
            let confidence = BetaBelief::from_confidence(max_prob, 4.0);

            // Generate branch hints based on the label and step.
            let branch_hints = generate_branch_hints(&label, step);

            milestones.push(SemanticMilestone {
                label,
                label_idx,
                predicted_state: predicted_state_vec,
                token_budget,
                branch_hints,
                confidence,
            });

            // Feed the projected state back for the next milestone.
            current_state = projected.detach();
        }

        // Overall confidence: mean of milestone confidences.
        let overall = if milestones.is_empty() {
            BetaBelief::uniform()
        } else {
            let mean_alpha: f32 =
                milestones.iter().map(|m| m.confidence.alpha).sum::<f32>() / milestones.len() as f32;
            let mean_beta: f32 =
                milestones.iter().map(|m| m.confidence.beta).sum::<f32>() / milestones.len() as f32;
            BetaBelief {
                alpha: mean_alpha,
                beta: mean_beta,
            }
        };

        // Serialize the context state for the map.
        let context_state_vec = context_state
            .squeeze(0)
            .ok()
            .and_then(|t| t.to_vec1::<f32>().ok())
            .unwrap_or_default();

        Ok(SemanticStateMap {
            milestones,
            confidence: overall,
            session_id: session_id.to_string(),
            context_state: context_state_vec,
        })
    }

    /// Predict a state map from a raw context state vector (convenience
    /// wrapper that constructs the tensor internally).
    pub fn predict_from_vec(
        &self,
        context_state: &[f32],
        session_id: &str,
        max_milestones: usize,
        device: &Device,
    ) -> Result<SemanticStateMap> {
        let tensor = Tensor::from_vec(context_state.to_vec(), (1, self.d_model), device)?;
        self.predict(&tensor, session_id, max_milestones, device)
    }

    /// Return the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

/// Generate branch hints for a milestone based on its label and step index.
fn generate_branch_hints(label: &str, step: usize) -> Vec<String> {
    match label {
        "hypothesis" => vec![
            "direct approach".to_string(),
            "decomposition approach".to_string(),
        ],
        "decomposition" => vec![
            "sequential sub-tasks".to_string(),
            "parallel sub-tasks".to_string(),
        ],
        "execution" => {
            if step < 2 {
                vec!["primary implementation".to_string()]
            } else {
                vec![
                    "primary implementation".to_string(),
                    "fallback implementation".to_string(),
                ]
            }
        }
        "verification" => vec![
            "unit test verification".to_string(),
            "integration test verification".to_string(),
        ],
        "refinement" => vec!["optimize for clarity".to_string()],
        "synthesis" => vec!["combine all results".to_string()],
        _ => vec![],
    }
}

/// Argmax of a 1-D tensor (returns the index of the max element).
fn argmax(t: &Tensor) -> Result<usize> {
    let vec = t.squeeze(0)?.to_vec1::<f32>()?;
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in vec.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx)
}

/// Max value of a tensor (scalar).
fn max_value(t: &Tensor) -> Result<f32> {
    let vec = t.squeeze(0)?.to_vec1::<f32>()?;
    Ok(vec.into_iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
}

/// Render a state map as a human-readable string for logging/debugging.
pub fn render_state_map(map: &SemanticStateMap) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "SemanticStateMap (session={}, confidence={:.2}):\n",
        map.session_id,
        map.confidence.mean()
    ));
    for (i, m) in map.milestones.iter().enumerate() {
        out.push_str(&format!(
            "  [{}] {} (budget={} tokens, confidence={:.2})\n",
            i,
            m.label,
            m.token_budget,
            m.confidence.mean()
        ));
        if !m.branch_hints.is_empty() {
            out.push_str(&format!("      branches: {}\n", m.branch_hints.join(", ")));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};

    fn make_predictor(d_model: usize) -> (StatePredictor, Device) {
        let device = Device::Cpu;
        let config = AxiomConfig {
            d_model,
            n_layers: 2,
            vocab_size: 256,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let predictor = StatePredictor::new(vb.pp("predictor"), &config).unwrap();
        (predictor, device)
    }

    #[test]
    fn predict_produces_milestones() {
        let d = 32;
        let (predictor, device) = make_predictor(d);
        let context = vec![0.5f32; d];
        let map = predictor
            .predict_from_vec(&context, "test-session", 4, &device)
            .unwrap();
        assert!(!map.milestones.is_empty());
        assert!(map.milestones.len() <= 4);
        assert_eq!(map.session_id, "test-session");
    }

    #[test]
    fn milestones_have_valid_labels() {
        let d = 16;
        let (predictor, device) = make_predictor(d);
        let context = vec![0.3f32; d];
        let map = predictor
            .predict_from_vec(&context, "s", 3, &device)
            .unwrap();
        for m in &map.milestones {
            assert!(
                MILESTONE_LABELS.contains(&m.label.as_str()),
                "invalid label: {}",
                m.label
            );
        }
    }

    #[test]
    fn token_budget_is_bounded() {
        let d = 16;
        let (predictor, device) = make_predictor(d);
        let context = vec![1.0f32; d];
        let map = predictor
            .predict_from_vec(&context, "s", 2, &device)
            .unwrap();
        for m in &map.milestones {
            assert!(m.token_budget >= 16, "budget too small: {}", m.token_budget);
            assert!(m.token_budget <= 4096, "budget too large: {}", m.token_budget);
        }
    }

    #[test]
    fn max_milestones_is_respected() {
        let d = 16;
        let (predictor, device) = make_predictor(d);
        let context = vec![0.1f32; d];
        let map = predictor
            .predict_from_vec(&context, "s", 100, &device)
            .unwrap();
        assert!(map.milestones.len() <= MAX_MILESTONES);
    }

    #[test]
    fn render_state_map_is_readable() {
        let map = SemanticStateMap {
            milestones: vec![SemanticMilestone {
                label: "hypothesis".to_string(),
                label_idx: 0,
                predicted_state: vec![0.1; 16],
                token_budget: 128,
                branch_hints: vec!["a".to_string(), "b".to_string()],
                confidence: BetaBelief::from_confidence(0.8, 4.0),
            }],
            confidence: BetaBelief::from_confidence(0.8, 4.0),
            session_id: "test".to_string(),
            context_state: vec![0.5; 16],
        };
        let rendered = render_state_map(&map);
        assert!(rendered.contains("hypothesis"));
        assert!(rendered.contains("128 tokens"));
        assert!(rendered.contains("a, b"));
    }

    #[test]
    fn branch_hints_are_generated() {
        let hints = generate_branch_hints("hypothesis", 0);
        assert_eq!(hints.len(), 2);
        assert!(hints[0].contains("approach"));

        let hints = generate_branch_hints("execution", 3);
        assert!(hints.len() >= 1);
    }

    #[test]
    fn empty_context_produces_valid_map() {
        let d = 8;
        let (predictor, device) = make_predictor(d);
        let context = vec![0.0f32; d];
        let map = predictor
            .predict_from_vec(&context, "empty", 2, &device)
            .unwrap();
        assert!(!map.milestones.is_empty());
        for m in &map.milestones {
            assert!(m.confidence.alpha > 0.0);
            assert!(m.confidence.beta > 0.0);
        }
    }
}