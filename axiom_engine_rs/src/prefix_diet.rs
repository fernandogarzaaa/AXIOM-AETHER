//! S4 (CVM cost stack): prefix diet, lossless dedup tier.
//!
//! Claude Code's fixed system prefix (CLAUDE.md/rules content, plugin/skill
//! catalogs) is paid on every cache write (1.25x) and read (0.1x) forever.
//! Observed on this machine: identical file content is sometimes injected
//! more than once. This module is a pure, stateless, deterministic
//! deduplicator: it finds byte-identical repeated blocks (>= 400 bytes,
//! occurring >= 2 times) within one piece of text and replaces every
//! occurrence after the first with a one-line marker. As a pure function of
//! the input bytes it produces the same output every time -- cache-safe by
//! construction, no session store or determinism trick needed (unlike S1's
//! compression path). See
//! docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S4.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

/// Minimum block size (bytes, after trimming) eligible for dedup. Below this
/// the marker line itself would cost more than it saves.
const MIN_DEDUP_BYTES: usize = 400;

/// Replaces every occurrence of a deduped block after the first.
pub const DEDUP_MARKER: &str = "[AXIOM-DEDUP: identical to an earlier block in this prompt]";

/// Aggregated dedup telemetry for one `diet` (or `diet_system_field`) call.
/// Tokens are approximated by whitespace-word count, matching the same
/// convention `local_messages_path` uses elsewhere in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DietReport {
    pub original_tokens: usize,
    pub dedup_tokens: usize,
    pub blocks_deduped: usize,
}

fn is_markdown_heading(line: &str) -> bool {
    let trimmed = line.trim_start();
    let hashes = trimmed.chars().take_while(|&c| c == '#').count();
    (1..=6).contains(&hashes) && trimmed.as_bytes().get(hashes) == Some(&b' ')
}

/// Split `text` on runs of 2+ newlines ("double-newline" paragraph
/// boundaries). Returns `(segments, separators)` where
/// `separators.len() == segments.len() - 1` and rejoining
/// `segments[0] + separators[0] + segments[1] + ...` reconstructs `text`
/// exactly.
fn split_paragraphs(text: &str) -> (Vec<String>, Vec<String>) {
    let bytes = text.as_bytes();
    let mut segments = Vec::new();
    let mut separators = Vec::new();
    let mut seg_start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\n' {
            let run_start = i;
            let mut j = i;
            while j < bytes.len() && bytes[j] == b'\n' {
                j += 1;
            }
            if j - run_start >= 2 {
                segments.push(text[seg_start..run_start].to_string());
                separators.push(text[run_start..j].to_string());
                seg_start = j;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    segments.push(text[seg_start..].to_string());
    (segments, separators)
}

/// Further split one paragraph segment at interior markdown-heading line
/// boundaries (a heading that isn't already the segment's first line starts
/// a new block). Returns `(sub_segments, sub_separators)` with the same
/// reconstruction contract as [`split_paragraphs`]; separators here are
/// always a single `"\n"` (no blank line existed at these boundaries).
fn split_heading_boundaries(segment: &str) -> (Vec<String>, Vec<String>) {
    let lines: Vec<&str> = segment.split('\n').collect();
    let mut segments = Vec::new();
    let mut separators = Vec::new();
    let mut current = String::new();
    for (idx, line) in lines.iter().enumerate() {
        if idx > 0 {
            // A heading line is always isolated as its own single-line
            // block: a boundary opens both before it and right after it,
            // so content following a heading (even with no blank line in
            // between) is dedup-eligible independently of the heading's
            // own (usually short, rarely-repeated) text.
            let prev_was_heading = is_markdown_heading(lines[idx - 1]);
            if (is_markdown_heading(line) || prev_was_heading) && !current.is_empty() {
                segments.push(std::mem::take(&mut current));
                separators.push("\n".to_string());
            } else {
                current.push('\n');
            }
        }
        current.push_str(line);
    }
    segments.push(current);
    (segments, separators)
}

/// Full block split: paragraph boundaries, then heading boundaries within
/// each paragraph. Same `(blocks, separators)` reconstruction contract.
fn split_blocks(text: &str) -> (Vec<String>, Vec<String>) {
    let (para_segments, para_seps) = split_paragraphs(text);
    let mut all_blocks = Vec::new();
    let mut all_seps = Vec::new();
    for (i, seg) in para_segments.iter().enumerate() {
        let (sub_segments, sub_seps) = split_heading_boundaries(seg);
        for (j, sub) in sub_segments.into_iter().enumerate() {
            if j > 0 {
                all_seps.push(sub_seps[j - 1].clone());
            } else if i > 0 {
                all_seps.push(para_seps[i - 1].clone());
            }
            all_blocks.push(sub);
        }
    }
    (all_blocks, all_seps)
}

/// Deduplicate `text`: blocks >= 400 bytes (after trimming) occurring >= 2
/// times have every occurrence after the first replaced by [`DEDUP_MARKER`].
/// Pure and stateless -- `diet(diet(x)) == diet(x)`, since a deduped block's
/// marker is far under the 400-byte threshold and is never itself
/// re-deduped.
pub fn diet(text: &str) -> String {
    diet_with_report(text).0
}

/// Same as [`diet`] but also returns the count of blocks that were replaced.
pub fn diet_with_report(text: &str) -> (String, usize) {
    let (blocks, seps) = split_blocks(text);

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for b in &blocks {
        let t = b.trim();
        if t.len() >= MIN_DEDUP_BYTES {
            *counts.entry(t).or_insert(0) += 1;
        }
    }

    let mut seen: HashSet<&str> = HashSet::new();
    let mut out_blocks: Vec<&str> = Vec::with_capacity(blocks.len());
    let mut deduped = 0usize;
    for b in &blocks {
        let t = b.trim();
        if t.len() >= MIN_DEDUP_BYTES && counts.get(t).copied().unwrap_or(0) >= 2 {
            if seen.contains(t) {
                out_blocks.push(DEDUP_MARKER);
                deduped += 1;
                continue;
            }
            seen.insert(t);
        }
        out_blocks.push(b.as_str());
    }

    let mut result = String::new();
    for (i, b) in out_blocks.iter().enumerate() {
        if i > 0 {
            result.push_str(&seps[i - 1]);
        }
        result.push_str(b);
    }
    (result, deduped)
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Apply [`diet`] to an Anthropic request's `system` field. Handles both
/// shapes: a plain string (single tier), or an array of content blocks
/// (each may carry its own `cache_control` breakpoint). Per-block
/// independence is deliberate: blocks are never concatenated before
/// dedup'ing, so a repeated block can never be deduped *across* a tier
/// boundary -- only within the same block's own text (the "simplest safe
/// rule" this step ships).
pub fn diet_system_field(system: &Value) -> (Value, DietReport) {
    match system {
        Value::String(s) => {
            let original_tokens = word_count(s);
            let (dieted, blocks_deduped) = diet_with_report(s);
            let dedup_tokens = original_tokens.saturating_sub(word_count(&dieted));
            (
                Value::String(dieted),
                DietReport {
                    original_tokens,
                    dedup_tokens,
                    blocks_deduped,
                },
            )
        }
        Value::Array(blocks) => {
            let mut report = DietReport::default();
            let mut out = Vec::with_capacity(blocks.len());
            for block in blocks {
                let Some(text) = block.get("text").and_then(Value::as_str) else {
                    out.push(block.clone());
                    continue;
                };
                let original_tokens = word_count(text);
                let (dieted, blocks_deduped) = diet_with_report(text);
                let dedup_tokens = original_tokens.saturating_sub(word_count(&dieted));
                report.original_tokens += original_tokens;
                report.dedup_tokens += dedup_tokens;
                report.blocks_deduped += blocks_deduped;
                let mut new_block = block.clone();
                new_block["text"] = Value::String(dieted);
                out.push(new_block);
            }
            (Value::Array(out), report)
        }
        other => (other.clone(), DietReport::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A deterministic, reproducible block of >= `len` bytes made of
    /// whitespace-separated "words" (not one giant unbroken run) so its
    /// approximate token count behaves like real prose rather than
    /// collapsing to a single token.
    fn block(byte: char, len: usize) -> String {
        let word: String = std::iter::repeat(byte).take(4).collect();
        let mut s = String::new();
        while s.len() < len {
            if !s.is_empty() {
                s.push(' ');
            }
            s.push_str(&word);
        }
        s.truncate(len);
        s
    }

    #[test]
    fn diet_is_pure_same_input_same_output() {
        let repeated = block('a', 500);
        let text = format!("{repeated}\n\nunique middle content\n\n{repeated}");
        assert_eq!(diet(&text), diet(&text));
    }

    #[test]
    fn diet_is_idempotent() {
        let repeated = block('b', 500);
        let text = format!("{repeated}\n\nunique middle content\n\n{repeated}\n\n{repeated}");
        let once = diet(&text);
        let twice = diet(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn diet_leaves_short_blocks_and_singletons_untouched() {
        let short_repeated = block('c', 50); // below MIN_DEDUP_BYTES
        let unique = block('d', 500); // long but occurs once
        let text = format!("{short_repeated}\n\n{unique}\n\n{short_repeated}");
        let out = diet(&text);
        assert_eq!(out, text, "nothing here should qualify for dedup");
    }

    #[test]
    fn diet_replaces_all_occurrences_after_the_first() {
        let repeated = block('e', 500);
        let text = format!("{repeated}\n\nunique\n\n{repeated}\n\n{repeated}");
        let (out, count) = diet_with_report(&text);
        assert_eq!(count, 2);
        assert_eq!(out.matches(&repeated).count(), 1);
        assert_eq!(out.matches(DEDUP_MARKER).count(), 2);
    }

    #[test]
    fn diet_splits_on_markdown_heading_boundaries_without_blank_lines() {
        let repeated = block('f', 450);
        // Headings back-to-back with a single newline (no blank line) --
        // must still be recognised as separate blocks.
        let text = format!("# Heading One\n{repeated}\n# Heading Two\n{repeated}");
        let (out, count) = diet_with_report(&text);
        assert_eq!(count, 1);
        assert!(out.contains("# Heading One"));
        assert!(out.contains("# Heading Two"));
        assert_eq!(out.matches(DEDUP_MARKER).count(), 1);
    }

    #[test]
    fn diet_correctness_on_sanitized_claude_code_style_fixture() {
        // A sanitized stand-in for the real observed pattern: a rules block
        // injected twice (e.g. once from a project CLAUDE.md include and
        // once from a duplicated plugin catalog entry), each well over the
        // 400-byte threshold.
        let sentence = "Always create new objects, never mutate existing ones. ";
        let rules_block: String = sentence.repeat(8);
        assert!(rules_block.len() >= MIN_DEDUP_BYTES);
        let text = format!(
            "# Project Instructions\nSome project-specific preamble text.\n\n{rules_block}\n\n\
             # Plugin Catalog\nplugin-a, plugin-b, plugin-c\n\n{rules_block}"
        );
        let (out, count) = diet_with_report(&text);
        assert_eq!(count, 1);
        assert_eq!(out.matches(rules_block.as_str()).count(), 1);
        assert!(out.contains("# Project Instructions"));
        assert!(out.contains("# Plugin Catalog"));
        assert!(out.contains(DEDUP_MARKER));
    }

    #[test]
    fn diet_system_field_string_shape_reports_stats() {
        let repeated = block('g', 500);
        let text = format!("{repeated}\n\nunique\n\n{repeated}");
        let (out, report) = diet_system_field(&json!(text));
        assert_eq!(out.as_str().unwrap(), diet(&text));
        assert_eq!(report.blocks_deduped, 1);
        assert!(report.dedup_tokens > 0);
        assert_eq!(report.original_tokens, word_count(&text));
    }

    #[test]
    fn diet_system_field_never_dedups_across_tier_boundaries() {
        // Same 500-byte block appears once in each of two separate system
        // content blocks (simulating two different cache_control tiers).
        // Within each individual block it occurs only once, so nothing
        // should be deduped -- dedup requires >= 2 occurrences *within the
        // same tier's own text*, never across tiers.
        let repeated = block('h', 500);
        let system = json!([
            {"type": "text", "text": repeated.clone(), "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": repeated.clone()},
        ]);
        let (out, report) = diet_system_field(&system);
        assert_eq!(report.blocks_deduped, 0);
        let arr = out.as_array().unwrap();
        assert_eq!(arr[0]["text"].as_str().unwrap(), repeated);
        assert_eq!(arr[1]["text"].as_str().unwrap(), repeated);
        assert!(!out.to_string().contains(DEDUP_MARKER));
    }

    #[test]
    fn diet_system_field_dedups_within_a_single_tier() {
        let repeated = block('i', 500);
        let text = format!("{repeated}\n\nunique\n\n{repeated}");
        let system = json!([
            {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}},
        ]);
        let (out, report) = diet_system_field(&system);
        assert_eq!(report.blocks_deduped, 1);
        let arr = out.as_array().unwrap();
        assert!(arr[0]["text"].as_str().unwrap().contains(DEDUP_MARKER));
        // cache_control marker on the block itself must survive untouched.
        assert_eq!(arr[0]["cache_control"], json!({"type": "ephemeral"}));
    }

    #[test]
    fn diet_system_field_non_array_non_string_passes_through() {
        let (out, report) = diet_system_field(&Value::Null);
        assert_eq!(out, Value::Null);
        assert_eq!(report, DietReport::default());
    }
}
