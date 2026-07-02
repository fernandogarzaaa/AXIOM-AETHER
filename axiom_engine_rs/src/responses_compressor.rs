//! Conservative planning and transformation for Responses API input.
//!
//! Only old, text-only assistant messages in the safe leading message prefix
//! are eligible. User messages and every structural item remain untouched.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// A maximal run of consecutive eligible assistant items (consecutive
/// original `input` indices). Each run compresses to ONE fingerprint at its
/// first index; because a run is contiguous, no non-selected item ever sits
/// between its members, so replacing it in place cannot reorder the transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssistantRun {
    pub indices: Vec<usize>,
    pub source_hashes: Vec<String>,
    pub context: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponsesCompressionPlan {
    pub runs: Vec<AssistantRun>,
    pub query: String,
}

impl ResponsesCompressionPlan {
    /// All compressed item indices, flattened in ascending order (metrics/logging).
    pub fn item_indices(&self) -> Vec<usize> {
        self.runs.iter().flat_map(|r| r.indices.iter().copied()).collect()
    }

    /// Concatenated context across every run — the total the threshold check
    /// weighs and the total the receipt accounting attributes as absorbed.
    pub fn total_context(&self) -> String {
        self.runs
            .iter()
            .map(|r| r.context.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

pub fn plan_compression(body: &Value) -> Option<ResponsesCompressionPlan> {
    let input = body.get("input")?.as_array()?;
    let latest_user = input
        .iter()
        .enumerate()
        .rev()
        .find(|(_, item)| item.get("role").and_then(Value::as_str) == Some("user"))
        .map(|(index, _)| index)?;
    let first_structural = input
        .iter()
        .position(|item| {
            !matches!(
                item.get("type").and_then(Value::as_str),
                None | Some("message")
            )
        })
        .unwrap_or(input.len());
    let safe_end = latest_user.min(first_structural);

    let selected: Vec<(usize, String)> = input[..safe_end]
        .iter()
        .enumerate()
        .filter(|(_, item)| item.get("role").and_then(Value::as_str) == Some("assistant"))
        .filter_map(|(index, item)| text_content(item.get("content")?).map(|text| (index, text)))
        .filter(|(_, text)| !text.trim().is_empty())
        .collect();
    if selected.is_empty() {
        return None;
    }

    // Group into maximal contiguous runs by consecutive original index.
    let mut runs: Vec<AssistantRun> = Vec::new();
    for (index, text) in selected {
        let hash = format!("{:x}", Sha256::digest(text.as_bytes()));
        match runs.last_mut() {
            Some(run) if run.indices.last().copied() == Some(index - 1) => {
                run.indices.push(index);
                run.source_hashes.push(hash);
                run.context.push_str("\n\n");
                run.context.push_str(&text);
            }
            _ => runs.push(AssistantRun {
                indices: vec![index],
                source_hashes: vec![hash],
                context: text,
            }),
        }
    }

    let query = text_content(input[latest_user].get("content")?).unwrap_or_default();
    Some(ResponsesCompressionPlan { runs, query })
}

/// Replace each run with its fingerprint, in place. Non-selected items keep
/// their exact positions; a contiguous run collapses to one fingerprint at the
/// run's first index. `fingerprints[i]` corresponds to `plan.runs[i]`.
pub fn apply_plan(
    body: &Value,
    plan: &ResponsesCompressionPlan,
    fingerprints: &[String],
) -> Option<Value> {
    if fingerprints.len() != plan.runs.len() {
        return None;
    }
    let mut output = body.clone();
    let input = output.get_mut("input")?.as_array_mut()?;

    use std::collections::{HashMap, HashSet};
    // first-index -> run ordinal; trailing run indices -> dropped (folded into
    // the run's leading fingerprint).
    let mut run_start: HashMap<usize, usize> = HashMap::new();
    let mut dropped: HashSet<usize> = HashSet::new();
    for (ordinal, run) in plan.runs.iter().enumerate() {
        let first = *run.indices.first()?;
        run_start.insert(first, ordinal);
        for &idx in &run.indices[1..] {
            dropped.insert(idx);
        }
    }

    let mut rebuilt: Vec<Value> = Vec::with_capacity(input.len());
    for (idx, item) in input.iter().enumerate() {
        if let Some(&ordinal) = run_start.get(&idx) {
            let run = &plan.runs[ordinal];
            let manifest = run
                .indices
                .iter()
                .zip(&run.source_hashes)
                .map(|(i, h)| format!("{i}:{h}"))
                .collect::<Vec<_>>()
                .join(",");
            rebuilt.push(json!({
                "type": "message",
                "role": "assistant",
                "content": format!(
                    "<axiom_source_manifest compressed_item_sha256=\"{manifest}\" />\n{}",
                    fingerprints[ordinal]
                )
            }));
        } else if !dropped.contains(&idx) {
            rebuilt.push(item.clone());
        }
    }
    *input = rebuilt;
    Some(output)
}

fn text_content(content: &Value) -> Option<String> {
    if let Some(text) = content.as_str() {
        return Some(text.to_string());
    }
    let parts = content.as_array()?;
    let mut text = Vec::with_capacity(parts.len());
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("input_text") | Some("output_text") => {
                text.push(part.get("text")?.as_str()?.to_string())
            }
            _ => return None,
        }
    }
    Some(text.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assistant(text: &str) -> serde_json::Value {
        json!({"type":"message","role":"assistant","content":text})
    }
    fn user(text: &str) -> serde_json::Value {
        json!({"type":"message","role":"user","content":text})
    }

    #[test]
    fn contiguous_assistants_form_one_run() {
        let body = json!({"input":[
            assistant("A0"), assistant("A1"), user("latest")
        ]});
        let plan = plan_compression(&body).unwrap();
        assert_eq!(plan.runs.len(), 1);
        assert_eq!(plan.runs[0].indices, vec![0, 1]);
        assert_eq!(plan.runs[0].context, "A0\n\nA1");
        assert_eq!(plan.query, "latest");
    }

    #[test]
    fn assistants_split_by_user_form_separate_runs() {
        let body = json!({"input":[
            assistant("A0"), user("U1"), assistant("A2"), user("latest")
        ]});
        let plan = plan_compression(&body).unwrap();
        assert_eq!(plan.runs.len(), 2, "non-contiguous assistants are distinct runs");
        assert_eq!(plan.runs[0].indices, vec![0]);
        assert_eq!(plan.runs[1].indices, vec![2]);
        assert_eq!(plan.item_indices(), vec![0, 2]);
    }

    #[test]
    fn no_eligible_assistant_returns_none() {
        assert!(plan_compression(&json!({"input":"hello"})).is_none());
        assert!(plan_compression(&json!({"input":[user("only user")]})).is_none());
    }

    #[test]
    fn apply_preserves_position_of_interleaved_user_message() {
        // Regression for #85: A0, U1, A2 — the user turn between two
        // non-contiguous assistants must NOT move.
        let body = json!({"input":[
            assistant("A0"), user("U1"), assistant("A2"), user("latest")
        ]});
        let plan = plan_compression(&body).unwrap();
        let out = apply_plan(&body, &plan, &["FP0".into(), "FP2".into()]).unwrap();
        let items = out.get("input").unwrap().as_array().unwrap();

        assert_eq!(items.len(), 4, "two runs replaced 1-for-1, nothing collapsed away");
        assert_eq!(items[1], user("U1"), "interleaved user stays at position 1");
        assert_eq!(items[3], user("latest"));
        assert_eq!(items[0]["role"], "assistant");
        assert!(items[0]["content"].as_str().unwrap().contains("FP0"));
        assert!(items[2]["content"].as_str().unwrap().contains("FP2"));
    }

    #[test]
    fn apply_collapses_a_contiguous_run_to_one_fingerprint() {
        let body = json!({"input":[
            assistant("A0"), assistant("A1"), user("latest")
        ]});
        let plan = plan_compression(&body).unwrap();
        let out = apply_plan(&body, &plan, &["FP".into()]).unwrap();
        let items = out.get("input").unwrap().as_array().unwrap();
        assert_eq!(items.len(), 2, "the 2-message run collapses to one fingerprint");
        assert!(items[0]["content"].as_str().unwrap().contains("0:"));
        assert!(items[0]["content"].as_str().unwrap().contains("1:"));
        assert_eq!(items[1], user("latest"));
    }

    #[test]
    fn apply_rejects_fingerprint_count_mismatch() {
        let body = json!({"input":[
            assistant("A0"), user("U1"), assistant("A2"), user("latest")
        ]});
        let plan = plan_compression(&body).unwrap();
        assert!(apply_plan(&body, &plan, &["only-one".into()]).is_none());
    }
}
