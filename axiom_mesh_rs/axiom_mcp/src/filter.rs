//! Deterministic token scrubbing.
//!
//! Rules are plain string predicates — no regex engine, no model — so the
//! same input always produces the same output and the filter is auditable
//! line by line.

use serde::{Deserialize, Serialize};

/// A single scrub rule applied to each line of an incoming payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterRule {
    /// Drop any line containing this substring (case-sensitive).
    DropContaining(String),
    /// Redact the value following `key=` / `key:` assignments for this key
    /// (case-insensitive key match), keeping the rest of the line.
    RedactAssignment(String),
}

/// Deterministic scrubber for tokens flowing orchestrator → worker.
#[derive(Debug, Clone, Default)]
pub struct TokenScrubber {
    rules: Vec<FilterRule>,
}

impl TokenScrubber {
    pub fn new(rules: Vec<FilterRule>) -> Self {
        Self { rules }
    }

    /// A sensible default: strip orchestrator-internal markers and redact
    /// common credential-shaped assignments so worker prompts never carry
    /// them.
    pub fn standard() -> Self {
        Self::new(vec![
            FilterRule::DropContaining("<axiom-internal>".into()),
            FilterRule::RedactAssignment("api_key".into()),
            FilterRule::RedactAssignment("token".into()),
            FilterRule::RedactAssignment("password".into()),
            FilterRule::RedactAssignment("secret".into()),
        ])
    }

    pub fn push_rule(&mut self, rule: FilterRule) {
        self.rules.push(rule);
    }

    /// Scrub a payload. Control characters (except `\n` and `\t`) are always
    /// removed; rules then apply per line.
    pub fn scrub(&self, input: &str) -> String {
        let cleaned: String =
            input.chars().filter(|c| !c.is_control() || *c == '\n' || *c == '\t').collect();

        let mut out = Vec::new();
        'line: for line in cleaned.lines() {
            let mut line = line.to_string();
            for rule in &self.rules {
                match rule {
                    FilterRule::DropContaining(s) => {
                        if line.contains(s.as_str()) {
                            continue 'line;
                        }
                    }
                    FilterRule::RedactAssignment(key) => {
                        line = redact_assignment(&line, key);
                    }
                }
            }
            out.push(line);
        }
        out.join("\n")
    }
}

/// Replace the value in `key=value` / `key: value` with `[REDACTED]`,
/// matching the key case-insensitively anywhere in the line.
fn redact_assignment(line: &str, key: &str) -> String {
    let lower = line.to_lowercase();
    let key = key.to_lowercase();
    let mut search_from = 0;
    let mut result = line.to_string();
    while let Some(pos) = lower[search_from..].find(&key) {
        let key_end = search_from + pos + key.len();
        let rest = &line[key_end..];
        let sep_len = rest.find(['=', ':']).filter(|&i| {
            rest[..i].chars().all(|c| c.is_whitespace())
        });
        if let Some(i) = sep_len {
            let value_start = key_end + i + 1;
            // Value runs to the next whitespace or end of line.
            let value = line[value_start..].trim_start();
            let value_offset = line[value_start..].len() - value.len();
            let value_end =
                value.find(char::is_whitespace).map(|e| value_start + value_offset + e).unwrap_or(line.len());
            if value_start + value_offset < value_end {
                result =
                    format!("{}[REDACTED]{}", &line[..value_start + value_offset], &line[value_end..]);
                return result; // one redaction per rule per line is enough
            }
        }
        search_from = key_end;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_internal_marker_lines() {
        let s = TokenScrubber::standard();
        let out = s.scrub("keep me\n<axiom-internal> route=claude\nkeep me too");
        assert_eq!(out, "keep me\nkeep me too");
    }

    #[test]
    fn redacts_credential_assignments() {
        let s = TokenScrubber::standard();
        let out = s.scrub("export API_KEY=sk-abc123 # deploy");
        assert!(out.contains("[REDACTED]"), "got: {out}");
        assert!(!out.contains("sk-abc123"));
        assert!(out.contains("# deploy"));
    }

    #[test]
    fn strips_control_characters() {
        let s = TokenScrubber::default();
        assert_eq!(s.scrub("a\u{1b}[31mb\u{7}c"), "a[31mbc");
    }

    #[test]
    fn is_deterministic() {
        let s = TokenScrubber::standard();
        let input = "TOKEN: xyz\nplain line\n<axiom-internal>x";
        assert_eq!(s.scrub(input), s.scrub(input));
    }
}
