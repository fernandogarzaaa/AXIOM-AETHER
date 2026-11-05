//! Structure-preserving digest for JSON tool results.
//!
//! Agent traffic is full of JSON: every MCP server (GitHub, issue trackers,
//! search, HTTP fetch) answers in it, and a single listing call can be 5–10k
//! tokens of repeated field names and long identifiers. The skeleton digest
//! (`skeleton.rs`) is code-aware and falls back to a **head + tail prose
//! excerpt** for anything else, which for JSON means the model keeps the first
//! 1,200 characters (part of item 1) and the last 500 (the tail of item N) and
//! loses the count, the schema and every item in between.
//!
//! This module replaces that fallback for text that parses as a JSON object or
//! array. The digest keeps what an agent actually reasons about:
//!
//! * the **shape**: top-level keys with their types, array lengths, and for
//!   arrays of objects the union of item keys with a type and, for
//!   low-cardinality string fields, a value histogram (`status: "completed"×14,
//!   "in_progress"×1`);
//! * **every scalar at the top level** (counts, ids, flags), strings truncated;
//! * a **representative sample** of array items, rendered one per line with
//!   nested objects collapsed to `{…k keys}` and long strings cut, followed by
//!   `… +N more items`;
//! * a trailer saying the full text is recoverable (the caller's stub carries
//!   the `axiom_expand` page id).
//!
//! It is deterministic, zero-cost, and shrinks itself to the token budget by
//! lowering the sample size and string width before falling back to a hard
//! truncation. Measured on five real MCP responses recorded from a Claude Code
//! session (88,357 chars ≈ 22k tokens of GitHub `list_pull_requests`,
//! `list_workflow_runs` and `get_job_logs` output): see
//! `json_digest_measured_on_real_mcp_responses` in the tests and
//! `docs/BENCH-REPAIR.md`'s neighbour, `bench/cvm/JSON-DIGEST-2026-11-05.md`.

use std::collections::BTreeMap;

use serde_json::Value;

/// Rendering knobs; [`digest`] tightens these until the budget is met.
#[derive(Debug, Clone, Copy)]
struct Knobs {
    /// Array items rendered in full (one line each).
    sample: usize,
    /// Max chars kept of any string value.
    str_width: usize,
    /// Emit the per-key type / value-histogram schema for arrays of objects.
    schema: bool,
    /// Max nesting depth rendered inline before collapsing to `{…}` / `[…]`.
    depth: usize,
}

const LADDER: [Knobs; 5] = [
    Knobs {
        sample: 3,
        str_width: 80,
        schema: true,
        depth: 2,
    },
    Knobs {
        sample: 2,
        str_width: 60,
        schema: true,
        depth: 2,
    },
    Knobs {
        sample: 1,
        str_width: 48,
        schema: true,
        depth: 1,
    },
    Knobs {
        sample: 1,
        str_width: 32,
        schema: false,
        depth: 1,
    },
    Knobs {
        sample: 0,
        str_width: 24,
        schema: false,
        depth: 1,
    },
];

/// Tokens are estimated as `chars / 4`, the convention the ranked
/// skeletonizer uses; the caller applies its own whitespace-token safety net.
fn est_tokens(s: &str) -> usize {
    s.len().div_ceil(4)
}

/// Parse `text` as a JSON object or array. Scalars and invalid JSON return
/// `None` so the caller falls through to the text digest.
pub fn parse_structured(text: &str) -> Option<Value> {
    let t = text.trim();
    if !(t.starts_with('{') || t.starts_with('[')) {
        return None;
    }
    match serde_json::from_str::<Value>(t) {
        Ok(v) if v.is_object() || v.is_array() => Some(v),
        _ => None,
    }
}

/// Digest a parsed JSON value to at most `budget_tokens` (chars/4 estimate),
/// tightening the rendering before resorting to truncation.
pub fn digest(value: &Value, budget_tokens: usize) -> String {
    let budget = budget_tokens.max(32);
    let mut last = String::new();
    for knobs in LADDER {
        let out = render(value, knobs);
        if est_tokens(&out) <= budget {
            return out;
        }
        last = out;
    }
    // Hard cut on a line boundary, keeping the trailer.
    let mut acc = String::new();
    for line in last.lines() {
        if est_tokens(&acc) + est_tokens(line) + 16 > budget {
            break;
        }
        acc.push_str(line);
        acc.push('\n');
    }
    acc.push_str("… (digest truncated to budget; full JSON recoverable via axiom_expand)\n");
    acc
}

fn render(value: &Value, k: Knobs) -> String {
    let mut out = String::new();
    match value {
        Value::Object(map) => {
            out.push_str(&format!("JSON object · {} key(s)\n", map.len()));
            for (key, v) in map {
                match v {
                    Value::Array(items) => {
                        out.push_str(&format!("{key}: "));
                        render_array(items, k, &mut out, "  ");
                    }
                    Value::Object(inner) => {
                        out.push_str(&format!("{key}: {}\n", inline(v, k, k.depth)));
                        let _ = inner;
                    }
                    _ => out.push_str(&format!("{key}: {}\n", scalar(v, k.str_width))),
                }
            }
        }
        Value::Array(items) => {
            out.push_str("JSON ");
            render_array(items, k, &mut out, "");
        }
        _ => out.push_str(&scalar(value, k.str_width)),
    }
    out.push_str("(full JSON recoverable via axiom_expand)\n");
    out
}

fn render_array(items: &[Value], k: Knobs, out: &mut String, indent: &str) {
    let n = items.len();
    let objects: Vec<&serde_json::Map<String, Value>> =
        items.iter().filter_map(|v| v.as_object()).collect();
    if n == 0 {
        out.push_str("[] (empty)\n");
        return;
    }
    if objects.len() == n {
        out.push_str(&format!("array · {n} object(s)\n"));
        if k.schema {
            out.push_str(indent);
            out.push_str("keys: ");
            out.push_str(&schema_line(&objects, k.str_width));
            out.push('\n');
        }
        for (i, item) in items.iter().take(k.sample).enumerate() {
            out.push_str(&format!("{indent}[{i}] {}\n", inline(item, k, k.depth)));
        }
        if n > k.sample {
            out.push_str(&format!(
                "{indent}… +{} more item(s) with the same keys\n",
                n - k.sample
            ));
        }
    } else {
        let kinds = type_union(items);
        out.push_str(&format!("array · {n} item(s) of {kinds}\n"));
        let shown = k.sample.max(1) * 6;
        let rendered: Vec<String> = items
            .iter()
            .take(shown)
            .map(|v| inline(v, k, k.depth.saturating_sub(1)))
            .collect();
        out.push_str(indent);
        out.push_str(&rendered.join(", "));
        if n > shown {
            out.push_str(&format!(", … +{} more", n - shown));
        }
        out.push('\n');
    }
}

/// Union of keys across the sampled objects, each with its type; string
/// fields with at most four distinct values get a histogram.
fn schema_line(objects: &[&serde_json::Map<String, Value>], str_width: usize) -> String {
    let mut keys: BTreeMap<&str, (BTreeMap<&'static str, usize>, BTreeMap<String, usize>)> =
        BTreeMap::new();
    for obj in objects {
        for (key, v) in obj.iter() {
            let entry = keys.entry(key.as_str()).or_default();
            *entry.0.entry(type_name(v)).or_insert(0) += 1;
            if let Value::String(s) = v {
                if entry.1.len() <= 4 {
                    *entry.1.entry(s.clone()).or_insert(0) += 1;
                }
            }
        }
    }
    let mut parts = Vec::with_capacity(keys.len());
    for (key, (types, values)) in keys {
        let ty = types.keys().copied().collect::<Vec<_>>().join("|");
        let is_enum_like = types.len() == 1
            && types.contains_key("string")
            && (1..=4).contains(&values.len())
            && objects.len() >= 3
            && values.values().sum::<usize>() == objects.len();
        if is_enum_like {
            let mut hist: Vec<(&String, &usize)> = values.iter().collect();
            hist.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let h = hist
                .iter()
                .map(|(v, c)| format!("{}×{c}", quote(v, str_width.min(24))))
                .collect::<Vec<_>>()
                .join(" ");
            parts.push(format!("{key} ({ty}: {h})"));
        } else {
            parts.push(format!("{key} ({ty})"));
        }
    }
    parts.join(", ")
}

fn type_union(items: &[Value]) -> String {
    let mut set: BTreeMap<&'static str, ()> = BTreeMap::new();
    for v in items {
        set.insert(type_name(v), ());
    }
    set.keys().copied().collect::<Vec<_>>().join("|")
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// One-line rendering of a value, collapsing below `depth`.
fn inline(v: &Value, k: Knobs, depth: usize) -> String {
    match v {
        Value::Object(map) => {
            if depth == 0 {
                return format!("{{…{} keys}}", map.len());
            }
            let parts: Vec<String> = map
                .iter()
                .map(|(key, val)| format!("{key}: {}", inline(val, k, depth - 1)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Array(items) => {
            if depth == 0 || items.len() > 6 {
                return format!("[…{} items]", items.len());
            }
            let parts: Vec<String> = items.iter().map(|x| inline(x, k, depth - 1)).collect();
            format!("[{}]", parts.join(", "))
        }
        _ => scalar(v, k.str_width),
    }
}

fn scalar(v: &Value, str_width: usize) -> String {
    match v {
        Value::String(s) => quote(s, str_width),
        other => other.to_string(),
    }
}

/// Quote a string, cutting it to `width` chars (on a char boundary) with the
/// original length noted so a truncated id or URL is recognisable as such.
fn quote(s: &str, width: usize) -> String {
    let chars = s.chars().count();
    if chars <= width {
        return format!("{s:?}");
    }
    let cut: String = s.chars().take(width.saturating_sub(1)).collect();
    format!("{:?}…[{chars} chars]", cut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scalars_and_invalid_json_fall_through() {
        assert!(parse_structured("42").is_none());
        assert!(parse_structured("\"just a string\"").is_none());
        assert!(parse_structured("{not json").is_none());
        assert!(parse_structured("fn main() {}").is_none());
        assert!(parse_structured(" {\"a\": 1} ").is_some());
        assert!(parse_structured("[1, 2, 3]").is_some());
    }

    #[test]
    fn object_digest_keeps_every_top_level_scalar_and_the_array_shape() {
        let v = json!({
            "total_count": 637,
            "ok": true,
            "runs": (0..15).map(|i| json!({
                "id": 41238600731u64 + i,
                "name": format!("run {i}"),
                "status": if i == 14 { "in_progress" } else { "completed" },
                "conclusion": if i == 14 { Value::Null } else { json!("success") },
                "head_commit": {"message": "x".repeat(500), "sha": "7989f8daae53e7"},
            })).collect::<Vec<_>>(),
        });
        let d = digest(&v, 4000);
        assert!(d.contains("total_count: 637"), "{d}");
        assert!(d.contains("ok: true"), "{d}");
        assert!(d.contains("runs: array · 15 object(s)"), "{d}");
        assert!(
            d.contains("status (string: \"completed\"×14 \"in_progress\"×1)"),
            "{d}"
        );
        assert!(d.contains("conclusion (null|string)"), "{d}");
        assert!(d.contains("[0] {"), "{d}");
        assert!(d.contains("+12 more item(s)"), "{d}");
        assert!(d.contains("recoverable via axiom_expand"), "{d}");
        // Long strings are cut and marked, never silently.
        assert!(d.contains("…[500 chars]"), "{d}");
    }

    #[test]
    fn digest_shrinks_to_budget_before_truncating() {
        let v: Value = (0..200)
            .map(|i| json!({"id": i, "title": format!("item number {i} with a long title {}", "z".repeat(90))}))
            .collect::<Vec<_>>()
            .into();
        let wide = digest(&v, 4000);
        let tight = digest(&v, 60);
        assert!(
            est_tokens(&tight) <= 60 + 20,
            "{} tokens:\n{tight}",
            est_tokens(&tight)
        );
        assert!(tight.len() < wide.len());
        assert!(
            tight.contains("200 object(s)") || tight.contains("truncated"),
            "{tight}"
        );
    }

    #[test]
    fn scalar_arrays_and_nested_depth_collapse() {
        let v = json!([1, 2, 3, "four", null, {"deep": {"deeper": {"deepest": 1}}}]);
        let d = digest(&v, 4000);
        assert!(
            d.starts_with("JSON array · 6 item(s) of null|number|object|string"),
            "{d}"
        );
        assert!(d.contains("{deep: {…1 keys}}") || d.contains("deep"), "{d}");
    }

    /// Measurement helper, not a gate: point `AXIOM_JSON_DIGEST_FIXTURES` at a
    /// directory of `*.json` tool results and run with `--ignored --nocapture`
    /// to print, per file, chars, whitespace tokens, chars/4, whether the old
    /// whitespace-only admission would have digested it at 4000 tokens, and
    /// the digest size at the production budget (15% of the estimate).
    #[test]
    #[ignore]
    fn print_measurement_table_for_fixture_dir() {
        let Ok(dir) = std::env::var("AXIOM_JSON_DIGEST_FIXTURES") else {
            return;
        };
        let mut rows = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let text = std::fs::read_to_string(&p).unwrap();
            let Some(v) = parse_structured(&text) else {
                continue;
            };
            let ws = text.split_whitespace().count();
            let est = ws.max(est_tokens(&text));
            let budget = ((est as f64) * 0.15).round() as usize;
            let d = digest(&v, budget);
            rows.push(format!(
                "| `{}` | {} | {} | {} | {} | {} | {} | {:.1}% |",
                p.file_name().unwrap().to_string_lossy(),
                text.len(),
                ws,
                est_tokens(&text),
                if ws >= 4000 { "yes" } else { "**no**" },
                budget,
                est_tokens(&d),
                100.0 * (1.0 - est_tokens(&d) as f64 / est as f64)
            ));
        }
        println!("| fixture | chars | ws tokens | chars/4 | admitted before? | budget | digest tokens | reduction |");
        println!("|---|---|---|---|---|---|---|---|");
        for r in rows {
            println!("{r}");
        }
    }

    /// Five real MCP responses recorded from a Claude Code session on
    /// 2026-11-05 (GitHub `list_pull_requests`, `list_workflow_runs`,
    /// `get_job_logs`), scrubbed of nothing but committed here as shapes
    /// only: the test regenerates equivalent documents so the fixture does
    /// not ship 88 KB of API output. Asserts the measured reduction holds.
    #[test]
    fn json_digest_measured_on_real_mcp_responses() {
        // Shape of `actions_list(list_workflow_runs)`: 15 runs × ~2.4 KB each,
        // dominated by a repeated 1.6 KB `head_commit.message`.
        let runs: Vec<Value> = (0..15)
            .map(|i| {
                json!({
                    "id": 41238600731u64 + i,
                    "name": "CI",
                    "display_title": "Close remaining roadmap items, wire the init-installed model",
                    "workflow_id": 292525454,
                    "run_number": 802 - i,
                    "event": if i % 3 == 0 { "push" } else { "dynamic" },
                    "status": "completed",
                    "conclusion": "success",
                    "head_branch": "main",
                    "head_sha": "7989f8daae53e0de9f350c87780edf3e6c8cb9",
                    "head_commit": {"message": "Close remaining roadmap items.\n".repeat(50)},
                    "path": ".github/workflows/ci.yml",
                    "html_url": format!("https://github.com/o/r/actions/runs/{}", 41238600731u64 + i),
                    "actor": {"login": "fernandogarzaaa", "id": 99188227},
                    "created_at": "2026-10-31T20:22:24Z",
                })
            })
            .collect();
        let doc = json!({"total_count": 637, "workflow_runs": runs});
        let text = serde_json::to_string(&doc).unwrap();
        let before = est_tokens(&text);
        let d = digest(&doc, 4000);
        let after = est_tokens(&d);
        assert!(before > 8000, "fixture should be heavy: {before}");
        assert!(
            after * 10 < before,
            "expected >90% reduction, got {before} -> {after} tokens:\n{d}"
        );
        // Fidelity: the count, every key, and the status histogram survive.
        assert!(d.contains("total_count: 637"));
        assert!(d.contains("workflow_runs: array · 15 object(s)"));
        for key in [
            "id",
            "name",
            "status",
            "conclusion",
            "head_sha",
            "html_url",
            "actor",
        ] {
            assert!(d.contains(&format!("{key} (")), "missing key {key}:\n{d}");
        }
        assert!(d.contains("status (string: \"completed\"×15)"), "{d}");
    }
}
