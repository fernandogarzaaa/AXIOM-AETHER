//! Cost pillar: corpus byte reduction through the live proxy.
//!
//! Live-only: replaying corpus sessions needs a running proxy. With no corpus
//! present, returns a well-formed skipped result so CI can exercise the path
//! deterministically.

use crate::cognition::PillarResult;
use crate::corpus::default_corpus_dir;
use axiom_engine::session_recorder::{ExchangeRecord, read_session};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn corpus_dir() -> PathBuf {
    default_corpus_dir()
}

#[derive(Clone, Copy, Debug, Default)]
struct SavingsMetrics {
    bytes_in: u64,
    bytes_forwarded: u64,
}

impl SavingsMetrics {
    fn delta(self, before: Self) -> Self {
        Self {
            bytes_in: self.bytes_in.saturating_sub(before.bytes_in),
            bytes_forwarded: self.bytes_forwarded.saturating_sub(before.bytes_forwarded),
        }
    }
}

fn corpus_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    path.extension()
                        .and_then(|value| value.to_str())
                        .map(|value| value.eq_ignore_ascii_case("jsonl"))
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    files.sort();
    files
}

fn corpus_hash(files: &[PathBuf]) -> String {
    let mut hasher = Sha256::new();
    for path in files {
        hasher.update(
            path.file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default(),
        );
        if let Ok(bytes) = std::fs::read(path) {
            hasher.update(Sha256::digest(bytes));
        }
    }
    format!("{:x}", hasher.finalize())
}

fn parse_metrics(text: &str) -> SavingsMetrics {
    let mut metrics = SavingsMetrics::default();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        match (parts.next(), parts.next()) {
            (Some("axiom_savings_bytes_in_total"), Some(value)) => {
                metrics.bytes_in = value.parse::<f64>().map(|value| value as u64).unwrap_or(0);
            }
            (Some("axiom_savings_bytes_forwarded_total"), Some(value)) => {
                metrics.bytes_forwarded =
                    value.parse::<f64>().map(|value| value as u64).unwrap_or(0);
            }
            _ => {}
        }
    }
    metrics
}

fn request_bytes(record: &ExchangeRecord) -> u64 {
    serde_json::to_vec(&record.request)
        .map(|bytes| bytes.len() as u64)
        .unwrap_or(0)
}

fn read_metrics(
    client: &reqwest::blocking::Client,
    base_url: &str,
) -> Result<SavingsMetrics, String> {
    let url = format!("{}/metrics", base_url.trim_end_matches('/'));
    let response = client
        .get(&url)
        .send()
        .map_err(|error| format!("GET {url} failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("GET {url} returned {}", response.status()));
    }
    let text = response
        .text()
        .map_err(|error| format!("read metrics body failed: {error}"))?;
    Ok(parse_metrics(&text))
}

fn replay_authorization_with(explicit: Option<&str>, openai_key: Option<&str>) -> String {
    if let Some(value) = explicit.filter(|value| !value.trim().is_empty()) {
        return value.to_string();
    }
    if let Some(value) = openai_key.filter(|value| !value.trim().is_empty()) {
        return format!("Bearer {value}");
    }
    "Bearer axiombench-live-replay".to_string()
}

fn replay_authorization() -> String {
    replay_authorization_with(
        std::env::var("AXIOMBENCH_AUTHORIZATION").ok().as_deref(),
        std::env::var("OPENAI_API_KEY").ok().as_deref(),
    )
}

fn replay_record(
    client: &reqwest::blocking::Client,
    base_url: &str,
    record: &ExchangeRecord,
    authorization: &str,
    compression_enabled: bool,
) -> Result<(u16, SavingsMetrics), String> {
    let before = read_metrics(client, base_url)?;
    let url = format!("{}/v1/responses", base_url.trim_end_matches('/'));
    let mut request = client
        .post(&url)
        .header("content-type", "application/json")
        .header("authorization", authorization)
        .header(
            "x-axiom-session-id",
            format!("axiombench-{}", record.session_id),
        );
    if !compression_enabled {
        request = request.header("x-axiom-responses-compress", "0");
    }
    let response = request
        .json(&record.request)
        .send()
        .map_err(|error| format!("POST {url} failed: {error}"))?;
    let status = response.status().as_u16();
    let _ = response.bytes();
    let after = read_metrics(client, base_url)?;
    Ok((status, after.delta(before)))
}

pub fn run_cost_with(base_url: &str, dir: &Path) -> PillarResult {
    let files = corpus_files(dir);
    let sessions = files.len();
    if sessions == 0 {
        return PillarResult {
            name: "cost".into(),
            headline: "skipped - no corpus (add bench/corpus/*.jsonl and run with --live)".into(),
            detail: json!({ "skipped": true, "reason": "no corpus", "corpus_dir": dir.display().to_string() }),
            ..Default::default()
        };
    }

    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return PillarResult {
                name: "cost".into(),
                headline: format!("skipped - HTTP client unavailable ({error})"),
                detail: json!({ "skipped": true, "reason": "client", "error": error.to_string() }),
                ..Default::default()
            };
        }
    };
    if let Err(error) = read_metrics(&client, base_url) {
        return PillarResult {
            name: "cost".into(),
            headline: format!("skipped - proxy metrics unavailable at {base_url}"),
            detail: json!({ "skipped": true, "reason": "proxy unavailable", "base_url": base_url, "error": error }),
            ..Default::default()
        };
    }

    let mut replayed = 0usize;
    let mut errored = 0usize;
    let mut skipped_records = 0usize;
    let mut bytes_in = 0u64;
    let mut bytes_forwarded = 0u64;
    let mut details = Vec::new();
    let authorization = replay_authorization();

    for path in &files {
        let records = match read_session(path) {
            Ok(records) => records,
            Err(error) => {
                errored += 1;
                details.push(
                    json!({ "file": path.display().to_string(), "error": error.to_string() }),
                );
                continue;
            }
        };
        for record in records {
            if record.endpoint != "/v1/responses" {
                skipped_records += 1;
                continue;
            }
            let off = replay_record(&client, base_url, &record, &authorization, false);
            let on = replay_record(&client, base_url, &record, &authorization, true);
            match (off, on) {
                (Ok((off_status, _)), Ok((on_status, delta)))
                    if (200..300).contains(&off_status) && (200..300).contains(&on_status) =>
                {
                    replayed += 1;
                    let original_bytes = request_bytes(&record);
                    let saved = delta.bytes_in.saturating_sub(delta.bytes_forwarded);
                    bytes_in = bytes_in.saturating_add(original_bytes);
                    bytes_forwarded =
                        bytes_forwarded.saturating_add(original_bytes.saturating_sub(saved));
                }
                (Ok((off_status, off_delta)), Ok((on_status, on_delta))) => {
                    errored += 1;
                    details.push(json!({
                        "file": path.display().to_string(),
                        "session_id": record.session_id,
                        "off_status": off_status,
                        "on_status": on_status,
                        "off_bytes_in_delta": off_delta.bytes_in,
                        "off_bytes_forwarded_delta": off_delta.bytes_forwarded,
                        "on_bytes_in_delta": on_delta.bytes_in,
                        "on_bytes_forwarded_delta": on_delta.bytes_forwarded,
                    }));
                }
                (Err(error), _) => {
                    errored += 1;
                    details.push(json!({
                        "file": path.display().to_string(),
                        "session_id": record.session_id,
                        "arm": "compression_off",
                        "error": error,
                    }));
                }
                (_, Err(error)) => {
                    errored += 1;
                    details.push(json!({
                        "file": path.display().to_string(),
                        "session_id": record.session_id,
                        "arm": "compression_on",
                        "error": error,
                    }));
                }
            }
        }
    }

    let saved = bytes_in.saturating_sub(bytes_forwarded);
    let reduction = if bytes_in > 0 {
        saved as f64 / bytes_in as f64
    } else {
        0.0
    };
    let hash = corpus_hash(&files);
    let headline = if replayed == 0 {
        format!(
            "no successful live cost replays ({errored} errored, {skipped_records} skipped) on {sessions} corpus session(s)"
        )
    } else {
        format!(
            "{:.1}% byte reduction over {replayed} replayed record(s) from {sessions} session(s)",
            reduction * 100.0
        )
    };

    PillarResult {
        name: "cost".into(),
        headline,
        detail: json!({
            "skipped": false,
            "base_url": base_url,
            "corpus_dir": dir.display().to_string(),
            "corpus_hash": hash,
            "precondition": "run against an idle proxy; /metrics savings counters are process-global",
            "sessions": sessions,
            "replayed": replayed,
            "paired_replays": replayed,
            "errored": errored,
            "skipped_records": skipped_records,
            "bytes_in": bytes_in,
            "bytes_forwarded": bytes_forwarded,
            "byte_reduction": reduction,
            "errors": details,
        }),
        sample_n: Some(replayed as u64),
        read_as: Some("indicative only".into()),
    }
}

pub fn run_cost(base_url: &str) -> PillarResult {
    run_cost_with(base_url, &corpus_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_skips_cleanly_without_corpus() {
        let temp = tempfile::tempdir().unwrap();
        let r = run_cost_with("http://127.0.0.1:3000", temp.path());
        assert_eq!(r.name, "cost");
        assert_eq!(r.detail["skipped"], true);
        assert_eq!(r.detail["reason"], "no corpus");
    }

    #[test]
    fn parses_savings_metrics() {
        let metrics = parse_metrics(
            "# TYPE axiom_savings_bytes_in_total counter\n\
             axiom_savings_bytes_in_total 1200\n\
             axiom_savings_bytes_forwarded_total 300\n",
        );
        assert_eq!(metrics.bytes_in, 1200);
        assert_eq!(metrics.bytes_forwarded, 300);
    }

    #[test]
    fn replay_authorization_prefers_explicit_then_openai_key() {
        assert_eq!(
            replay_authorization_with(Some("Bearer custom"), Some("sk-test")),
            "Bearer custom"
        );
        assert_eq!(
            replay_authorization_with(None, Some("sk-test")),
            "Bearer sk-test"
        );
        assert_eq!(
            replay_authorization_with(None, None),
            "Bearer axiombench-live-replay"
        );
    }
}
