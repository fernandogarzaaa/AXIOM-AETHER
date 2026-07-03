//! Corpus promotion/listing for AxiomBench cost replay.

use axiom_engine::session_recorder::{ExchangeRecord, read_session, scrub, sessions_dir};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub fn default_corpus_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/corpus"))
}

fn session_source(input: &str) -> PathBuf {
    let candidate = PathBuf::from(input);
    if candidate.exists() {
        return candidate;
    }
    let file = if input.ends_with(".jsonl") {
        input.to_string()
    } else {
        format!("{input}.jsonl")
    };
    sessions_dir().join(file)
}

fn looks_secret(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.starts_with("bearer ")
        || s.len() >= 20
            && ["sk-", "ghp_", "gho_", "xoxb-", "eyj"]
                .iter()
                .any(|prefix| lower.contains(prefix))
}

fn contains_secret_miss(value: &Value) -> bool {
    match value {
        Value::String(s) => looks_secret(s),
        Value::Array(items) => items.iter().any(contains_secret_miss),
        Value::Object(map) => map.values().any(contains_secret_miss),
        _ => false,
    }
}

fn scrub_record(record: &ExchangeRecord) -> ExchangeRecord {
    ExchangeRecord {
        request: scrub(&record.request),
        response: scrub(&record.response),
        ..record.clone()
    }
}

fn scrubbed_jsonl(records: &[ExchangeRecord]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for record in records {
        let scrubbed = scrub_record(record);
        if contains_secret_miss(&scrubbed.request) || contains_secret_miss(&scrubbed.response) {
            return Err(format!(
                "secret-looking value survived scrub in session {}",
                scrubbed.session_id
            ));
        }
        let line = serde_json::to_string(&scrubbed)
            .map_err(|error| format!("serialize scrubbed record failed: {error}"))?;
        out.extend_from_slice(line.as_bytes());
        out.push(b'\n');
    }
    Ok(out)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn promote_to(source: &Path, corpus_dir: &Path) -> Result<PathBuf, String> {
    let records = read_session(source)
        .map_err(|error| format!("read {} failed: {error}", source.display()))?;
    if records.is_empty() {
        return Err(format!("{} contains no exchange records", source.display()));
    }
    let bytes = scrubbed_jsonl(&records)?;
    let hash = sha256_hex(&bytes);
    std::fs::create_dir_all(corpus_dir)
        .map_err(|error| format!("create {} failed: {error}", corpus_dir.display()))?;
    let path = corpus_dir.join(format!("{}.jsonl", &hash[..16]));
    std::fs::write(&path, bytes)
        .map_err(|error| format!("write {} failed: {error}", path.display()))?;
    Ok(path)
}

pub fn list_corpus(corpus_dir: &Path) -> Result<Vec<(PathBuf, usize, String)>, String> {
    let mut rows = Vec::new();
    let entries = match std::fs::read_dir(corpus_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(rows),
        Err(error) => return Err(format!("read {} failed: {error}", corpus_dir.display())),
    };
    for entry in entries {
        let entry = entry.map_err(|error| format!("read corpus entry failed: {error}"))?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
            continue;
        }
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("read {} failed: {error}", path.display()))?;
        let records = read_session(&path)
            .map_err(|error| format!("parse {} failed: {error}", path.display()))?;
        rows.push((path, records.len(), sha256_hex(&bytes)));
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(rows)
}

pub fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("promote") => {
            let Some(input) = args.get(1) else {
                return Err("usage: axiombench corpus promote <session-id|path>".into());
            };
            let source = session_source(input);
            let path = promote_to(&source, &default_corpus_dir())?;
            let records = read_session(&path)
                .map_err(|error| format!("read promoted {} failed: {error}", path.display()))?;
            let bytes = std::fs::read(&path)
                .map_err(|error| format!("read {} failed: {error}", path.display()))?;
            println!(
                "promoted source={} records={} sha256={} -> {}",
                source.display(),
                records.len(),
                sha256_hex(&bytes),
                path.display()
            );
            Ok(())
        }
        Some("list") => {
            println!("file\trecords\tsha256");
            for (path, records, hash) in list_corpus(&default_corpus_dir())? {
                println!("{}\t{}\t{}", path.display(), records, hash);
            }
            Ok(())
        }
        _ => Err("usage: axiombench corpus <promote|list> ...".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(session_id: &str, request: Value) -> ExchangeRecord {
        ExchangeRecord {
            ts: 1,
            endpoint: "/v1/responses".into(),
            session_id: session_id.into(),
            request,
            response: json!({"status": 200}),
            compressed: false,
        }
    }

    #[test]
    fn promote_and_list_roundtrip() {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        let corpus_dir = temp.path().join("corpus");
        std::fs::create_dir_all(&source_dir).unwrap();
        let source = source_dir.join("source.jsonl");
        let rec = record(
            "s1",
            json!({"input": "hello", "authorization": "Bearer secret"}),
        );
        std::fs::write(
            &source,
            format!("{}\n", serde_json::to_string(&rec).unwrap()),
        )
        .unwrap();

        let out = promote_to(&source, &corpus_dir).unwrap();
        assert!(out.exists());
        let promoted = std::fs::read_to_string(&out).unwrap();
        assert!(promoted.contains("[REDACTED]"));
        let rows = list_corpus(&corpus_dir).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(
            rows.iter()
                .any(|(path, records, _)| path == &out && *records == 1)
        );
    }

    #[test]
    fn promotion_rejects_scrub_miss() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.jsonl");
        let rec = record(
            "s1",
            json!({"input": "token xoxb-abcdefghijklmnopqrstuvwxyz"}),
        );
        std::fs::write(
            &source,
            format!("{}\n", serde_json::to_string(&rec).unwrap()),
        )
        .unwrap();

        let error = promote_to(&source, temp.path()).unwrap_err();
        assert!(error.contains("secret-looking value survived scrub"));
    }
}
