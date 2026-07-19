//! Context compression: strip conversational filler, keep state.
//!
//! Workers don't need pleasantries or hedging — they need the residual,
//! the relevant code, and the errors. This pass is rule-based and
//! deterministic; it keeps structural content (code fences, diffs, error
//! lines, paths) and drops filler.

use serde::{Deserialize, Serialize};

/// What the compressor did, for telemetry and cost accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub input_lines: usize,
    pub output_lines: usize,
    pub input_bytes: usize,
    pub output_bytes: usize,
}

impl CompressionStats {
    pub fn ratio(&self) -> f32 {
        if self.input_bytes == 0 {
            return 1.0;
        }
        self.output_bytes as f32 / self.input_bytes as f32
    }
}

/// Filler openers that carry no state. Matched case-insensitively against
/// the start of a line.
const FILLER_PREFIXES: &[&str] = &[
    "sure,",
    "sure!",
    "certainly",
    "of course",
    "great question",
    "i hope this helps",
    "let me know if",
    "as an ai",
    "i'd be happy to",
    "here's what i",
    "thanks for",
    "no problem",
];

/// True when a line carries state a worker node could act on.
fn is_state_line(line: &str, in_code_fence: bool) -> bool {
    if in_code_fence {
        return true;
    }
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lower = trimmed.to_lowercase();
    if FILLER_PREFIXES.iter().any(|p| lower.starts_with(p)) {
        return false;
    }
    true
}

/// Compress a context blob for a worker node. Keeps code fences verbatim,
/// drops blank lines and conversational filler, and collapses runs of
/// whitespace in prose lines.
pub fn compress_context(input: &str) -> (String, CompressionStats) {
    let mut out: Vec<String> = Vec::new();
    let mut in_fence = false;

    for line in input.lines() {
        let fence_toggle = line.trim_start().starts_with("```");
        if fence_toggle {
            out.push(line.to_string());
            in_fence = !in_fence;
            continue;
        }
        if !is_state_line(line, in_fence) {
            continue;
        }
        if in_fence {
            out.push(line.to_string());
        } else {
            out.push(line.split_whitespace().collect::<Vec<_>>().join(" "));
        }
    }

    let output = out.join("\n");
    let stats = CompressionStats {
        input_lines: input.lines().count(),
        output_lines: out.len(),
        input_bytes: input.len(),
        output_bytes: output.len(),
    };
    (output, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_filler_keeps_state() {
        let input = "Sure, I'd be happy to help!\n\nerror[E0308]: mismatched types\n --> src/main.rs:4:5\nLet me know if you need anything else.";
        let (out, stats) = compress_context(input);
        assert_eq!(out, "error[E0308]: mismatched types\n--> src/main.rs:4:5");
        assert!(stats.ratio() < 1.0);
    }

    #[test]
    fn code_fences_survive_verbatim() {
        let input = "Certainly! Here is the fix:\n```rust\nfn main() {\n    let  x  =  1;\n}\n```\nI hope this helps!";
        let (out, _) = compress_context(input);
        assert_eq!(out, "```rust\nfn main() {\n    let  x  =  1;\n}\n```");
    }

    #[test]
    fn is_deterministic_and_idempotent_on_clean_input() {
        let input = "residual norm: 0.42\ntarget: tests green";
        let (once, _) = compress_context(input);
        let (twice, stats) = compress_context(&once);
        assert_eq!(once, twice);
        assert_eq!(stats.ratio(), 1.0);
    }
}
