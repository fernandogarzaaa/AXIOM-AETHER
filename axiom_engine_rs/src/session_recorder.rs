//! Per-session request/response recording (opt-in via AXIOM_SESSION_RECORD).
//!
//! One JSONL file per session under `sessions_dir()`. Secrets are scrubbed at
//! write time; recording failures never block the request path.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeRecord {
    pub ts: u64,
    pub endpoint: String,
    pub session_id: String,
    pub request: Value,
    pub response: Value,
    pub compressed: bool,
}

pub fn recording_enabled() -> bool {
    std::env::var("AXIOM_SESSION_RECORD")
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

pub fn sessions_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("AXIOM_SESSIONS_DIR") {
        if !dir.trim().is_empty() {
            return PathBuf::from(dir);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join(".axiom")
        .join("sessions")
}

/// Replace every value under a secret-named key, and every string that looks
/// like a bearer token / API key, with "[REDACTED]". Conservative by design:
/// a string containing an apparent key is redacted wholesale rather than
/// attempting surgical extraction.
pub fn scrub(value: &Value) -> Value {
    fn looks_secret(s: &str) -> bool {
        let lower = s.to_ascii_lowercase();
        lower.starts_with("bearer ")
            || s.len() >= 20
                && ["sk-", "ghp_", "gho_", "eyj"]
                    .iter()
                    .any(|p| lower.contains(p))
    }
    fn walk(v: &Value) -> Value {
        match v {
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, val)| {
                        let lk = k.to_ascii_lowercase();
                        let redact = [
                            "authorization",
                            "x-api-key",
                            "api_key",
                            "apikey",
                            "token",
                            "secret",
                            "password",
                        ]
                        .contains(&lk.as_str());
                        if redact {
                            (k.clone(), Value::String("[REDACTED]".into()))
                        } else {
                            (k.clone(), walk(val))
                        }
                    })
                    .collect(),
            ),
            Value::Array(items) => Value::Array(items.iter().map(walk).collect()),
            Value::String(s) if looks_secret(s) => Value::String("[REDACTED]".into()),
            other => other.clone(),
        }
    }
    walk(value)
}

fn session_file_name(session_id: &str) -> String {
    let safe: String = session_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{safe}.jsonl")
}

/// Append one scrubbed record; every failure is log-and-continue.
pub fn record_exchange(record: ExchangeRecord) {
    let dir = sessions_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("[record] cannot create {}: {e}", dir.display());
        return;
    }
    let scrubbed = ExchangeRecord {
        request: scrub(&record.request),
        response: scrub(&record.response),
        ..record
    };
    let path = dir.join(session_file_name(&scrubbed.session_id));
    let line = match serde_json::to_string(&scrubbed) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[record] serialize failed: {e}");
            return;
        }
    };
    use std::io::Write;
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{line}") {
                eprintln!("[record] append failed for {}: {e}", path.display());
            }
        }
        Err(e) => eprintln!("[record] open failed for {}: {e}", path.display()),
    }
}

pub fn read_session(path: &Path) -> std::io::Result<Vec<ExchangeRecord>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scrub_redacts_secret_looking_strings_and_auth_keys() {
        let v = json!({
            "headers": {"authorization": "Bearer sk-live-abc123def456ghi789", "accept": "json"},
            "body": {"note": "use key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX please",
                      "ok": "plain text stays"}
        });
        let s = scrub(&v);
        assert_eq!(s["headers"]["authorization"], "[REDACTED]");
        assert_eq!(s["headers"]["accept"], "json");
        assert!(!s["body"]["note"].as_str().unwrap().contains("sk-proj-"));
        assert_eq!(s["body"]["ok"], "plain text stays");
    }

    #[test]
    fn record_and_read_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "axiom_rec_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::env::set_var("AXIOM_SESSIONS_DIR", &dir);
        let rec = ExchangeRecord {
            ts: 1234,
            endpoint: "/v1/responses".into(),
            session_id: "s1".into(),
            request: json!({"input":"hi"}),
            response: json!({"output":"there"}),
            compressed: true,
        };
        record_exchange(rec);
        let file = sessions_dir().join("s1.jsonl");
        let records = read_session(&file).unwrap();
        std::env::remove_var("AXIOM_SESSIONS_DIR");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].endpoint, "/v1/responses");
        assert_eq!(records[0].response["output"], "there");
    }

    #[test]
    fn session_id_is_sanitized_for_filesystem() {
        assert_eq!(session_file_name("a/b\\c:d"), "a_b_c_d.jsonl");
        assert_eq!(session_file_name("normal-id_1"), "normal-id_1.jsonl");
    }
}
