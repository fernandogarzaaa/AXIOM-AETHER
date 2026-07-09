//! predictive_tools.rs - MCP tool definitions for the Predictive Reasoning Engine.
//!
//! This module provides the tool schemas and handler functions for the three
//! new MCP tools that expose the Predictive Reasoning Engine to AI agents:
//!
//! - `axiom_predict_states` - forecast semantic milestones from context
//! - `axiom_sample_trajectories` - fork and score reasoning branches
//! - `axiom_align_generation` - monitor and correct generation drift
//!
//! These are designed to be integrated into `mcp_stdio.rs`'s `tools_list()`
//! and `handle_tools_call()` functions.

use serde_json::{json, Value};

use crate::alignment_loop::AlignmentLoop;
use crate::state_predictor::{SemanticStateMap, StatePredictor, render_state_map};
use crate::trajectory_sampler::{TrajectorySampler, render_trajectory_result};

/// Return the JSON tool definitions for the three predictive tools.
///
/// Call this from `tools_list()` and merge the returned array into the
/// `"tools"` array.
pub fn predictive_tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "axiom_predict_states",
            "description": "Forecast high-level semantic states (hypothesis, execution, verification) from the current context. Returns a state map with predicted cognitive milestones, token budgets, and branch hints. Use before starting a complex task to map out the reasoning trajectory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "context_summary": {
                        "type": "string",
                        "description": "A summary of the current context or task description."
                    },
                    "max_milestones": {
                        "type": "integer",
                        "description": "Maximum number of milestones to predict (default: 4, max: 8)."
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for state tracking (default: 'predictive')."
                    }
                },
                "required": ["context_summary"]
            }
        }),
        json!({
            "name": "axiom_sample_trajectories",
            "description": "Fork the reasoning process into multiple parallel trajectories and score each branch for viability, compute cost, and information gain. Prunes dead ends and selects the most promising trajectory. Use after axiom_predict_states to evaluate reasoning branches.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state_map_json": {
                        "type": "string",
                        "description": "The SemanticStateMap JSON from axiom_predict_states."
                    },
                    "context": {
                        "type": "string",
                        "description": "Context text for manifold encoding (optional)."
                    },
                    "prune_threshold": {
                        "type": "number",
                        "description": "Branches with composite score below this are pruned (default: 0.1)."
                    }
                },
                "required": ["state_map_json"]
            }
        }),
        json!({
            "name": "axiom_align_generation",
            "description": "Check alignment between current generation state and the predicted semantic milestones. Detects logic drift and computes a suggested localized fast-weight correction vector — this vector is returned for the caller to act on, it is NOT applied to the live session's W̃ automatically. Use during generation to monitor reasoning deviations.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state_map_json": {
                        "type": "string",
                        "description": "The SemanticStateMap JSON from axiom_predict_states."
                    },
                    "generation_state": {
                        "type": "string",
                        "description": "JSON array of floats representing the current generation hidden state."
                    },
                    "drift_threshold": {
                        "type": "number",
                        "description": "Drift score threshold for correction (default: 0.5)."
                    },
                    "correction_lr": {
                        "type": "number",
                        "description": "Learning rate for localized W̃ corrections (default: 0.01)."
                    }
                },
                "required": ["state_map_json", "generation_state"]
            }
        }),
    ]
}

/// Handle the `axiom_predict_states` tool call.
///
/// Creates a StatePredictor (always randomly initialized — no trained
/// checkpoint exists yet for this head) and runs prediction over a context
/// state vector.
///
/// `context_override`, when `Some`, is the real session embedding (e.g. from
/// `embed_text` against the live TTT pipeline) and is used verbatim as both
/// the predictor's input and its `d_model`. When `None` (no live pipeline
/// available, e.g. a standalone/offline call), falls back to a deterministic
/// hash of `context_summary` — a proxy, not a real semantic state. The
/// response always reports which path was taken via `state_source`, and
/// `trained: false` so callers never mistake this for a calibrated model.
pub fn handle_predict_states(
    args: &Value,
    device: &candle_core::Device,
    context_override: Option<&[f32]>,
) -> Value {
    let context_summary = args
        .get("context_summary")
        .and_then(Value::as_str)
        .unwrap_or("");
    let max_milestones = args
        .get("max_milestones")
        .and_then(Value::as_u64)
        .unwrap_or(4) as usize;
    let session_id = args
        .get("session_id")
        .and_then(Value::as_str)
        .unwrap_or("predictive")
        .to_string();

    if context_summary.trim().is_empty() {
        return json!({
            "error": "context_summary is required"
        });
    }

    let (context_state, state_source): (Vec<f32>, &str) = match context_override {
        Some(v) if !v.is_empty() => (v.to_vec(), "session_embedding"),
        _ => {
            let d = crate::config::AxiomConfig::runtime_small().d_model;
            (
                summary_to_state_vector(context_summary, d),
                "hashed_summary_fallback",
            )
        }
    };

    // State predictor head is always randomly initialized (no trained
    // checkpoint yet); its d_model must match whatever context vector we have.
    let config = crate::config::AxiomConfig {
        d_model: context_state.len(),
        ..crate::config::AxiomConfig::runtime_small()
    };
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(
        &varmap,
        candle_core::DType::F32,
        device,
    );
    let predictor = match StatePredictor::new(vb.pp("predictor"), &config) {
        Ok(p) => p,
        Err(e) => {
            return json!({
                "error": format!("failed to create state predictor: {e}")
            });
        }
    };

    let map = match predictor.predict_from_vec(&context_state, &session_id, max_milestones, device) {
        Ok(m) => m,
        Err(e) => {
            return json!({
                "error": format!("prediction failed: {e}")
            });
        }
    };

    let rendered = render_state_map(&map);
    let map_json = serde_json::to_string(&map).unwrap_or_default();

    json!({
        "state_map": map_json,
        "rendered": rendered,
        "milestone_count": map.milestones.len(),
        "confidence": map.confidence.mean(),
        "trained": false,
        "state_source": state_source,
        "note": "StatePredictor uses randomly-initialized weights (no trained checkpoint exists yet). Milestone labels, budgets, and confidence values are not yet meaningful predictions."
    })
}

/// Handle the `axiom_sample_trajectories` tool call.
///
/// Deserializes the state map, runs trajectory sampling, and returns the result.
pub fn handle_sample_trajectories(args: &Value) -> Value {
    let state_map_json = args
        .get("state_map_json")
        .and_then(Value::as_str)
        .unwrap_or("");
    let context = args
        .get("context")
        .and_then(Value::as_str)
        .unwrap_or("");
    let prune_threshold = args
        .get("prune_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.1) as f32;

    if state_map_json.is_empty() {
        return json!({
            "error": "state_map_json is required"
        });
    }

    let state_map: SemanticStateMap = match serde_json::from_str(state_map_json) {
        Ok(m) => m,
        Err(e) => {
            return json!({
                "error": format!("failed to parse state map: {e}")
            });
        }
    };

    let sampler = TrajectorySampler::new(
        crate::trajectory_sampler::DEFAULT_NUM_BRANCHES,
        prune_threshold,
    );

    let (result, manifold) = sampler.sample_with_manifold(&state_map, context);
    let rendered = render_trajectory_result(&result);
    let result_json = serde_json::to_string(&result).unwrap_or_default();

    let manifold_info = json!({
        "node_count": manifold.nodes.len(),
        "edge_count": manifold.edges.len(),
    });

    json!({
        "result": result_json,
        "rendered": rendered,
        "surviving_branches": result.branches.len(),
        "total_explored": result.total_explored,
        "pruned_count": result.pruned,
        "manifold": manifold_info
    })
}

/// Handle the `axiom_align_generation` tool call.
///
/// Deserializes the state map and generation state, runs alignment check,
/// and returns the result.
pub fn handle_align_generation(args: &Value) -> Value {
    let state_map_json = args
        .get("state_map_json")
        .and_then(Value::as_str)
        .unwrap_or("");
    let generation_state_json = args
        .get("generation_state")
        .and_then(Value::as_str)
        .unwrap_or("");
    let drift_threshold = args
        .get("drift_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.5) as f32;
    let correction_lr = args
        .get("correction_lr")
        .and_then(Value::as_f64)
        .unwrap_or(0.01) as f32;

    if state_map_json.is_empty() || generation_state_json.is_empty() {
        return json!({
            "error": "state_map_json and generation_state are required"
        });
    }

    let state_map: SemanticStateMap = match serde_json::from_str(state_map_json) {
        Ok(m) => m,
        Err(e) => {
            return json!({
                "error": format!("failed to parse state map: {e}")
            });
        }
    };

    let generation_state: Vec<f32> = match serde_json::from_str(generation_state_json) {
        Ok(v) => v,
        Err(e) => {
            return json!({
                "error": format!("failed to parse generation state: {e}")
            });
        }
    };

    let loop_ = AlignmentLoop::new(drift_threshold, correction_lr, crate::alignment_loop::MAX_CORRECTIONS);
    let mut loop_state = loop_.init(state_map);
    let check = loop_.check_alignment(&mut loop_state, &generation_state);

    let summary = AlignmentLoop::summary(&loop_state);
    let check_json = serde_json::to_string(&check).unwrap_or_default();

    json!({
        "check": check_json,
        "summary": summary,
        "drift_score": check.drift_score,
        "corrected": check.corrected,
        "correction_strength": check.correction_strength,
        "milestone": check.milestone_label,
        "completed": loop_state.completed
    })
}

/// Convert a text summary to a state vector (hash-based proxy for TTT state).
fn summary_to_state_vector(summary: &str, dim: usize) -> Vec<f32> {
    use sha2::{Digest, Sha256};
    let mut result = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut hasher = Sha256::new();
        hasher.update(summary.as_bytes());
        hasher.update((i as u64).to_le_bytes());
        let digest = hasher.finalize();
        let raw = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
        result.push((raw as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definitions_are_valid() {
        let tools = predictive_tool_definitions();
        assert_eq!(tools.len(), 3);
        for tool in &tools {
            assert!(tool["name"].is_string());
            assert!(tool["description"].is_string());
            assert!(tool["inputSchema"]["type"].is_string());
        }
    }

    #[test]
    fn predict_states_returns_result() {
        let args = json!({
            "context_summary": "Write a function to sort an array",
            "max_milestones": 3
        });
        let result = handle_predict_states(&args, &candle_core::Device::Cpu, None);
        assert!(result["state_map"].is_string() || result["error"].is_string());
    }

    #[test]
    fn predict_states_uses_provided_context_and_reports_source() {
        // When a real session embedding is supplied, the predictor must use it
        // (not the hash-of-summary fallback) and say so honestly in the output.
        let args = json!({ "context_summary": "test", "max_milestones": 2 });
        let real_embedding = vec![0.25f32; 16];
        let result = handle_predict_states(&args, &candle_core::Device::Cpu, Some(&real_embedding));
        assert_eq!(result["state_source"], "session_embedding");
        let map: SemanticStateMap =
            serde_json::from_str(result["state_map"].as_str().unwrap()).unwrap();
        assert_eq!(map.context_state.len(), 16);
    }

    #[test]
    fn predict_states_without_context_falls_back_and_marks_untrained() {
        // No trained checkpoint exists yet. Callers must never be told a
        // prediction is trustworthy when it came from randomly-initialized
        // weights over a hash-of-text proxy state.
        let args = json!({ "context_summary": "test", "max_milestones": 2 });
        let result = handle_predict_states(&args, &candle_core::Device::Cpu, None);
        assert_eq!(result["state_source"], "hashed_summary_fallback");
        assert_eq!(result["trained"], false);
    }

    #[test]
    fn sample_trajectories_with_empty_map_returns_error() {
        let args = json!({
            "state_map_json": ""
        });
        let result = handle_sample_trajectories(&args);
        assert!(result["error"].is_string());
    }

    #[test]
    fn align_generation_with_empty_inputs_returns_error() {
        let args = json!({
            "state_map_json": "",
            "generation_state": ""
        });
        let result = handle_align_generation(&args);
        assert!(result["error"].is_string());
    }

    #[test]
    fn summary_to_state_vector_is_deterministic() {
        let v1 = summary_to_state_vector("test", 16);
        let v2 = summary_to_state_vector("test", 16);
        assert_eq!(v1, v2);
        let v3 = summary_to_state_vector("different", 16);
        assert_ne!(v1, v3);
    }
}