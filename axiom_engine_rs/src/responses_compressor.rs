//! Conservative planning and transformation for Responses API input.
//!
//! Only old, text-only assistant messages in the safe leading message prefix
//! are eligible. User messages and every structural item remain untouched.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponsesCompressionPlan {
    pub item_indices: Vec<usize>,
    pub source_hashes: Vec<String>,
    pub context: String,
    pub query: String,
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

    let item_indices = selected.iter().map(|(index, _)| *index).collect();
    let source_hashes = selected
        .iter()
        .map(|(_, text)| format!("{:x}", Sha256::digest(text.as_bytes())))
        .collect();
    let context = selected
        .iter()
        .map(|(_, text)| text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let query = text_content(input[latest_user].get("content")?).unwrap_or_default();

    Some(ResponsesCompressionPlan {
        item_indices,
        source_hashes,
        context,
        query,
    })
}

pub fn apply_plan(
    body: &Value,
    plan: &ResponsesCompressionPlan,
    fingerprint: &str,
) -> Option<Value> {
    let mut output = body.clone();
    let input = output.get_mut("input")?.as_array_mut()?;
    let insertion_index = *plan.item_indices.first()?;
    let manifest = plan
        .item_indices
        .iter()
        .zip(&plan.source_hashes)
        .map(|(index, hash)| format!("{index}:{hash}"))
        .collect::<Vec<_>>()
        .join(",");
    let replacement = json!({
        "type": "message",
        "role": "assistant",
        "content": format!(
            "<axiom_source_manifest compressed_item_sha256=\"{manifest}\" />\n{fingerprint}"
        )
    });

    for index in plan.item_indices.iter().rev() {
        input.remove(*index);
    }
    input.insert(insertion_index, replacement);
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
