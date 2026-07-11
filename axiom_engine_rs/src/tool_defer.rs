//! P1 (Prolonged-Session Stack): tool deferral via Anthropic's native
//! `defer_loading`.
//!
//! Claude Code's ~80K-token request prefix is dominated by `tools[]` schemas
//! (many connected MCP servers), re-read on every turn. Most turns use 2-3
//! tools. Anthropic supports `defer_loading: true` on a tool definition: a
//! deferred tool is NOT rendered into the cached `tools[]` prefix; when the
//! model needs it, tool-search appends its schema as a `tool_reference` block
//! in `messages` (after the breakpoint), so the prefix cache is never broken,
//! even mid-session.
//!
//! This module marks every tool NOT in the recent working set with
//! `defer_loading: true`, keeping the working set always-loaded. It only ever
//! ADDS the flag -- names, order, and count are unchanged -- so the `tools[]`
//! array stays byte-stable turn-over-turn (the binding cache-safety rule)
//! while the cached prefix shrinks. See
//! docs/superpowers/plans/2026-07-11-prolonged-session-stack.md, step P1.

use std::collections::HashSet;

use serde_json::Value;

/// Tools always kept loaded regardless of recent use -- the core edit/inspect
/// loop a coding agent reaches for constantly, so deferring them would just
/// cause immediate reloads.
const CORE: [&str; 6] = ["Read", "Edit", "Write", "Bash", "Glob", "Grep"];

/// The working set for a session: every tool invoked (`tool_use`) in the last
/// `recent_k` turns, unioned with the always-keep [`CORE`] set.
pub fn working_set(messages: &[Value], recent_k: usize) -> HashSet<String> {
    let mut keep: HashSet<String> = CORE.iter().map(|s| s.to_string()).collect();
    let start = messages.len().saturating_sub(recent_k);
    for msg in &messages[start..] {
        collect_tool_use_names(msg.get("content"), &mut keep);
    }
    keep
}

fn collect_tool_use_names(content: Option<&Value>, out: &mut HashSet<String>) {
    let Some(blocks) = content.and_then(Value::as_array) else {
        return;
    };
    for block in blocks {
        if block.get("type").and_then(Value::as_str) == Some("tool_use") {
            if let Some(name) = block.get("name").and_then(Value::as_str) {
                out.insert(name.to_string());
            }
        }
    }
}

/// Mark every tool whose `name` is NOT in `keep` with `defer_loading: true`,
/// preserving order and count (only a boolean flag is added, so the `tools[]`
/// bytes stay stable modulo the flag). Returns `(tools, deferred_count)`.
///
/// A tool that already carries a `cache_control` breakpoint keeps it; a tool
/// with no `name` is passed through untouched (defensive: never corrupt an
/// unexpected shape).
pub fn mark_deferred(tools: &[Value], keep: &HashSet<String>) -> (Vec<Value>, usize) {
    let mut deferred = 0usize;
    let out = tools
        .iter()
        .map(|tool| {
            let name = tool.get("name").and_then(Value::as_str);
            match name {
                Some(n) if !keep.contains(n) => {
                    let mut t = tool.clone();
                    if let Some(obj) = t.as_object_mut() {
                        obj.insert("defer_loading".to_string(), Value::Bool(true));
                    }
                    deferred += 1;
                    t
                }
                _ => tool.clone(),
            }
        })
        .collect();
    (out, deferred)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn working_set_keeps_recently_invoked_tools_plus_core() {
        let messages = vec![json!({"role":"assistant","content":[
            {"type":"tool_use","name":"WebFetch","id":"t1","input":{}}]})];
        let ws = working_set(&messages, 8);
        assert!(ws.contains("WebFetch"), "recently used tool kept");
        assert!(ws.contains("Read"), "core tool always kept");
    }

    #[test]
    fn working_set_only_scans_the_recent_window() {
        // An old tool_use outside the recent_k window is NOT in the set.
        let mut messages = vec![json!({"role":"assistant","content":[
            {"type":"tool_use","name":"AncientTool","id":"a","input":{}}]})];
        for i in 0..10 {
            messages.push(json!({"role":"user","content":format!("turn {i}")}));
        }
        let ws = working_set(&messages, 3);
        assert!(!ws.contains("AncientTool"), "tool outside window dropped");
    }

    #[test]
    fn mark_deferred_sets_flag_on_unused_tools_preserving_order() {
        let tools = vec![
            json!({"name":"Read"}),
            json!({"name":"WebFetch"}),
            json!({"name":"ObscureTool"}),
        ];
        let mut keep = HashSet::new();
        keep.insert("Read".to_string());
        let (out, deferred) = mark_deferred(&tools, &keep);
        assert_eq!(deferred, 2);
        // order + count unchanged (byte-stability rule)
        assert_eq!(out.len(), 3);
        assert_eq!(out[0]["name"], json!("Read"));
        assert_eq!(out[1]["name"], json!("WebFetch"));
        assert_eq!(out[2]["name"], json!("ObscureTool"));
        // kept tool: no defer_loading; unused tools: defer_loading true
        assert!(out[0].get("defer_loading").is_none());
        assert_eq!(out[1]["defer_loading"], json!(true));
        assert_eq!(out[2]["defer_loading"], json!(true));
    }

    #[test]
    fn mark_deferred_is_deterministic_across_calls() {
        // Same inputs -> byte-identical output (cache-safety).
        let tools = vec![json!({"name":"A"}), json!({"name":"B"})];
        let keep: HashSet<String> = ["A".to_string()].into_iter().collect();
        let a = mark_deferred(&tools, &keep).0;
        let b = mark_deferred(&tools, &keep).0;
        assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap());
    }

    #[test]
    fn mark_deferred_passes_through_nameless_tools_untouched() {
        let tools = vec![json!({"description":"no name here"})];
        let (out, deferred) = mark_deferred(&tools, &HashSet::new());
        assert_eq!(deferred, 0);
        assert!(out[0].get("defer_loading").is_none());
    }
}
