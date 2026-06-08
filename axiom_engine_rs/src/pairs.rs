//! Label-free contrastive pair mining. Each `Pair` is `{anchor, positive}` where
//! anchor is NL-ish (a doc comment / heading / question) and positive is content.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Pair {
    pub anchor: String,
    pub positive: String,
    pub source: String, // "doc" | "md" | "synthetic"
}

/// Mine `(leading doc comment, following declaration body)` pairs from source
/// text. Handles `///`/`//!` (Rust), `#` runs (Python/shell), `//` and `/**`
/// (C-family). A pair is emitted when a comment run of >= `min_comment_chars`
/// is immediately followed by >= 1 non-comment, non-blank line; the positive is
/// the following block up to the next blank line or comment run.
pub fn mine_doc_body(text: &str, min_comment_chars: usize) -> Vec<Pair> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let is_comment = |l: &str| {
        let t = l.trim_start();
        t.starts_with("///")
            || t.starts_with("//!")
            || t.starts_with("//")
            || t.starts_with('#')
            || t.starts_with("* ")
            || t.starts_with("/**")
    };
    let strip = |l: &str| {
        l.trim_start()
            .trim_start_matches("///")
            .trim_start_matches("//!")
            .trim_start_matches("/**")
            .trim_start_matches("//")
            .trim_start_matches('#')
            .trim_start_matches('*')
            .trim()
            .to_string()
    };
    let mut i = 0;
    while i < lines.len() {
        if is_comment(lines[i]) {
            let mut c = String::new();
            while i < lines.len() && is_comment(lines[i]) {
                let s = strip(lines[i]);
                if !s.is_empty() {
                    c.push_str(&s);
                    c.push(' ');
                }
                i += 1;
            }
            let mut body = String::new();
            while i < lines.len() && !lines[i].trim().is_empty() && !is_comment(lines[i]) {
                body.push_str(lines[i]);
                body.push('\n');
                i += 1;
            }
            let c = c.trim().to_string();
            let body = body.trim().to_string();
            if c.len() >= min_comment_chars && !body.is_empty() {
                out.push(Pair {
                    anchor: c,
                    positive: body,
                    source: "doc".into(),
                });
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Mine `(markdown heading, section body)` pairs. Heading lines start with `#`;
/// the positive is everything up to the next heading.
pub fn mine_markdown(text: &str, min_body_chars: usize) -> Vec<Pair> {
    let mut out = Vec::new();
    let mut heading: Option<String> = None;
    let mut body = String::new();
    let flush = |heading: &Option<String>, body: &str, out: &mut Vec<Pair>| {
        if let Some(h) = heading {
            let b = body.trim();
            if b.len() >= min_body_chars {
                out.push(Pair {
                    anchor: h.clone(),
                    positive: b.to_string(),
                    source: "md".into(),
                });
            }
        }
    };
    for line in text.lines() {
        let t = line.trim_start();
        if let Some(h) = t.strip_prefix('#') {
            flush(&heading, &body, &mut out);
            heading = Some(h.trim_start_matches('#').trim().to_string());
            body.clear();
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }
    flush(&heading, &body, &mut out);
    out
}

/// Append pairs to a JSONL file.
pub fn write_pairs_jsonl(path: impl AsRef<Path>, pairs: &[Pair]) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for p in pairs {
        let line = serde_json::to_string(p)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
    }
    Ok(())
}

/// Read all pairs from a JSONL file (corrupt lines skipped).
pub fn read_pairs_jsonl(path: impl AsRef<Path>) -> Vec<Pair> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    BufReader::new(file)
        .lines()
        .map_while(std::io::Result::ok)
        .filter_map(|l| serde_json::from_str::<Pair>(l.trim()).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mines_rust_doc_body() {
        let src = "/// Adds two numbers and returns the sum.\nfn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        let pairs = mine_doc_body(src, 10);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].anchor.contains("Adds two numbers"));
        assert!(pairs[0].positive.contains("fn add"));
        assert_eq!(pairs[0].source, "doc");
    }

    #[test]
    fn skips_short_comments() {
        let src = "// x\nfn f() {}\n";
        assert!(mine_doc_body(src, 10).is_empty());
    }

    #[test]
    fn mines_markdown_heading_body() {
        let md = "# Title\nintro\n## Section A\nThis is a sufficiently long body of section A text.\n## Section B\nmore body text here that is long enough\n";
        let pairs = mine_markdown(md, 10);
        let anchors: Vec<&str> = pairs.iter().map(|p| p.anchor.as_str()).collect();
        assert!(anchors.contains(&"Section A"));
        assert!(anchors.contains(&"Section B"));
    }

    #[test]
    fn pairs_jsonl_roundtrip() {
        let path =
            std::env::temp_dir().join(format!("axiom_pairs_test_{}.jsonl", std::process::id()));
        let p = vec![Pair {
            anchor: "q".into(),
            positive: "a".into(),
            source: "doc".into(),
        }];
        write_pairs_jsonl(&path, &p).unwrap();
        let back = read_pairs_jsonl(&path);
        assert_eq!(back, p);
        let _ = std::fs::remove_file(&path);
    }
}
