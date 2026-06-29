//! Semantic epistemic-drift validation and local telemetry.
//!
//! This module complements grounding verification. Grounding asks whether
//! claims are supported by supplied evidence; semantic drift asks whether a
//! response abandons the prompt's empirical task for unsupported abstraction.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub const DRIFT_SCHEMA_VERSION: u32 = 1;
pub const JUDGE_PROMPT_VERSION: &str = "epistemic-drift-v1";
const MAX_ANALYSIS_BYTES: usize = 8 * 1024;
const MAX_TRIGGER_PHRASES: usize = 32;
const MAX_TRIGGER_BYTES: usize = 512;
const MAX_JUDGE_RESPONSE_BYTES: usize = 64 * 1024;
const MAX_EVALUATED_TEXT_BYTES: usize = 512 * 1024;
static TELEMETRY_APPEND_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub const JUDGE_SYSTEM_PROMPT: &str = r#"You are an adversarial telemetry validator specializing in epistemic drift detection. Compare the assistant response with the original prompt and any supplied evidence. Detect the exact point where the response abandons the epistemic mode requested by the prompt or introduces unsupported interpretive, philosophical, metaphorical, anthropomorphic, or teleological framing.

Do not treat abstract or philosophical language as drift when the original prompt explicitly requests that mode. This is an evidence-and-intent alignment check, not a style preference. Analyze sentence by sentence. Return one JSON object only, with no markdown or commentary, matching this schema:
{
  "schema_version": 1,
  "empirical_score": 0.0,
  "drift_detected": true,
  "drift_onset_index": 2,
  "primary_trigger_phrases": ["exact phrase"],
  "analysis": "concise rationale",
  "confidence": 0.0
}
Use null for drift_onset_index and [] for primary_trigger_phrases when drift_detected is false. empirical_score and confidence must be numbers from 0.0 to 1.0."#;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DriftValidation {
    pub schema_version: u32,
    pub empirical_score: f32,
    pub drift_detected: bool,
    pub drift_onset_index: Option<usize>,
    pub primary_trigger_phrases: Vec<String>,
    pub analysis: String,
    pub confidence: f32,
}

impl DriftValidation {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != DRIFT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported drift schema version {}",
                self.schema_version
            ));
        }
        validate_unit_interval("empirical_score", self.empirical_score)?;
        validate_unit_interval("confidence", self.confidence)?;
        if self.drift_detected != self.drift_onset_index.is_some() {
            return Err(
                "drift_onset_index must be present exactly when drift_detected is true".into(),
            );
        }
        if !self.drift_detected && !self.primary_trigger_phrases.is_empty() {
            return Err("trigger phrases must be empty when no drift is detected".into());
        }
        if self.primary_trigger_phrases.len() > MAX_TRIGGER_PHRASES {
            return Err(format!(
                "too many trigger phrases (max {MAX_TRIGGER_PHRASES})"
            ));
        }
        if self
            .primary_trigger_phrases
            .iter()
            .any(|phrase| phrase.len() > MAX_TRIGGER_BYTES)
        {
            return Err(format!("trigger phrase exceeds {MAX_TRIGGER_BYTES} bytes"));
        }
        if self.analysis.len() > MAX_ANALYSIS_BYTES {
            return Err(format!("analysis exceeds {MAX_ANALYSIS_BYTES} bytes"));
        }
        Ok(())
    }
}

pub fn parse_judge_response(content: &str) -> Result<DriftValidation, String> {
    let trimmed = content.trim();
    let json_text = if let Some(rest) = trimmed.strip_prefix("```json") {
        rest.strip_suffix("```")
            .ok_or_else(|| "unterminated JSON markdown fence".to_string())?
            .trim()
    } else if let Some(rest) = trimmed.strip_prefix("```") {
        rest.strip_suffix("```")
            .ok_or_else(|| "unterminated markdown fence".to_string())?
            .trim()
    } else {
        trimmed
    };
    let validation: DriftValidation = serde_json::from_str(json_text)
        .map_err(|e| format!("judge output is not strict schema JSON: {e}"))?;
    validation.validate()?;
    Ok(validation)
}

#[derive(Clone)]
pub struct EpistemicJudgeConfig {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    timeout: Duration,
}

impl EpistemicJudgeConfig {
    pub fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
        timeout: Duration,
    ) -> Result<Self, String> {
        let endpoint = normalize_chat_completions_url(&endpoint.into())?;
        let model = model.into();
        if model.trim().is_empty() {
            return Err("epistemic judge model must not be empty".into());
        }
        Ok(Self {
            endpoint,
            model,
            api_key: api_key.filter(|key| !key.trim().is_empty()),
            timeout,
        })
    }

    pub fn from_env() -> Result<Option<Self>, String> {
        let Some(endpoint) = std::env::var("AXIOM_EPISTEMIC_JUDGE_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            return Ok(None);
        };
        let model = std::env::var("AXIOM_EPISTEMIC_JUDGE_MODEL")
            .map_err(|_| "AXIOM_EPISTEMIC_JUDGE_MODEL is required when the judge URL is set")?;
        let api_key = std::env::var("AXIOM_EPISTEMIC_JUDGE_API_KEY").ok();
        let timeout_secs = std::env::var("AXIOM_EPISTEMIC_JUDGE_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(60)
            .clamp(1, 300);
        Self::new(endpoint, model, api_key, Duration::from_secs(timeout_secs)).map(Some)
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

fn normalize_chat_completions_url(raw: &str) -> Result<String, String> {
    let base = raw.trim().trim_end_matches('/');
    if !(base.starts_with("http://") || base.starts_with("https://")) {
        return Err("epistemic judge URL must use http or https".into());
    }
    if base.ends_with("/chat/completions") {
        Ok(base.to_string())
    } else if base.ends_with("/v1") {
        Ok(format!("{base}/chat/completions"))
    } else {
        Ok(format!("{base}/v1/chat/completions"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JudgeOutcome {
    pub validation: DriftValidation,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EpistemicEvaluation {
    pub request_id: String,
    pub judge_model: String,
    pub judge_prompt_version: String,
    pub judge_latency_ms: u64,
    pub telemetry_logged: bool,
    pub report: CombinedEpistemicReport,
}

#[derive(Clone)]
pub struct OpenAiSemanticJudge {
    config: EpistemicJudgeConfig,
    client: Client,
}

impl OpenAiSemanticJudge {
    pub fn new(config: EpistemicJudgeConfig) -> Result<Self, String> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| format!("failed to construct epistemic judge client: {e}"))?;
        Ok(Self { config, client })
    }

    pub fn model(&self) -> &str {
        self.config.model()
    }

    pub async fn evaluate(
        &self,
        prompt: &str,
        response: &str,
        evidence: &str,
    ) -> Result<JudgeOutcome, String> {
        validate_input_size("prompt", prompt)?;
        validate_input_size("response", response)?;
        validate_input_size("evidence", evidence)?;
        let user_payload = json!({
            "original_prompt": prompt,
            "assistant_response": response,
            "evidence": evidence,
        });
        let payload = json!({
            "model": self.config.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload.to_string()}
            ]
        });
        let mut request = self
            .client
            .post(&self.config.endpoint)
            .header("content-type", "application/json")
            .json(&payload);
        if let Some(key) = self.config.api_key.as_deref() {
            request = request.bearer_auth(key);
        }
        let started = Instant::now();
        let upstream = request
            .send()
            .await
            .map_err(|e| format!("epistemic judge request failed: {e}"))?;
        let status = upstream.status();
        let bytes = upstream
            .bytes()
            .await
            .map_err(|e| format!("epistemic judge response read failed: {e}"))?;
        if !status.is_success() {
            return Err(format!("epistemic judge returned HTTP {}", status.as_u16()));
        }
        if bytes.len() > MAX_JUDGE_RESPONSE_BYTES {
            return Err(format!(
                "epistemic judge response exceeds {MAX_JUDGE_RESPONSE_BYTES} bytes"
            ));
        }
        let envelope: Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("epistemic judge envelope is invalid JSON: {e}"))?;
        let content = envelope
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                "epistemic judge response has no choices[0].message.content".to_string()
            })?;
        let validation = parse_judge_response(content)?;
        Ok(JudgeOutcome {
            validation,
            latency_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn evaluate_with_judge(
    judge: &OpenAiSemanticJudge,
    request_id: impl Into<String>,
    prompt: &str,
    response: &str,
    evidence: &str,
    target_model: &str,
) -> Result<EpistemicEvaluation, String> {
    let request_id = request_id.into();
    let grounding = crate::hallucination::verify(response, evidence);
    let outcome = judge.evaluate(prompt, response, evidence).await?;
    let report = combine_epistemic_signals(
        grounding.grounded_fraction,
        grounding.unsupported,
        grounding.unverified,
        Some(outcome.validation.clone()),
    );
    let record = EpistemicTelemetryRecord::new(
        request_id.clone(),
        prompt,
        response,
        target_model,
        judge.model(),
        JUDGE_PROMPT_VERSION,
        outcome.latency_ms,
        grounding.grounded_fraction,
        grounding.unsupported,
        grounding.unverified,
        outcome.validation,
        telemetry_capture_from_env(),
    );
    let telemetry_logged = if let Some(sink) = telemetry_sink_from_env() {
        let result = tokio::task::spawn_blocking(move || sink.append(&record))
            .await
            .map_err(|e| format!("telemetry worker failed: {e}"))?;
        result?;
        true
    } else {
        false
    };
    Ok(EpistemicEvaluation {
        request_id,
        judge_model: judge.model().to_string(),
        judge_prompt_version: JUDGE_PROMPT_VERSION.to_string(),
        judge_latency_ms: outcome.latency_ms,
        telemetry_logged,
        report,
    })
}

/// Schedule semantic validation after generation without delaying the primary
/// response. Returns `true` only when automatic validation was enabled and a
/// judge task was scheduled.
pub fn spawn_automatic_validation(
    prompt: String,
    response: String,
    evidence: String,
    target_model: String,
) -> bool {
    if std::env::var("AXIOM_EPISTEMIC_AUTO").as_deref() != Ok("1") {
        return false;
    }
    tokio::spawn(async move {
        let config = match EpistemicJudgeConfig::from_env() {
            Ok(Some(config)) => config,
            Ok(None) => {
                eprintln!("[epistemic] automatic validation skipped: judge is not configured");
                return;
            }
            Err(error) => {
                eprintln!("[epistemic] automatic validation skipped: {error}");
                return;
            }
        };
        let judge = match OpenAiSemanticJudge::new(config) {
            Ok(judge) => judge,
            Err(error) => {
                eprintln!("[epistemic] automatic validation skipped: {error}");
                return;
            }
        };
        let request_id = uuid::Uuid::new_v4().to_string();
        match evaluate_with_judge(
            &judge,
            request_id,
            &prompt,
            &response,
            &evidence,
            &target_model,
        )
        .await
        {
            Ok(evaluation) => eprintln!(
                "[epistemic] decision={:?} drift={} empirical_score={:.3} telemetry_logged={}",
                evaluation.report.decision,
                evaluation.report.semantic_drift_detected,
                evaluation
                    .report
                    .semantic
                    .as_ref()
                    .map(|result| result.empirical_score)
                    .unwrap_or_default(),
                evaluation.telemetry_logged
            ),
            Err(error) => eprintln!("[epistemic] automatic validation failed: {error}"),
        }
    });
    true
}

fn telemetry_sink_from_env() -> Option<JsonlTelemetrySink> {
    std::env::var("AXIOM_EPISTEMIC_TELEMETRY_PATH")
        .ok()
        .filter(|path| !path.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("AXIOM_MEMORY_DIR")
                .ok()
                .filter(|path| !path.trim().is_empty())
                .map(|path| PathBuf::from(path).join("epistemic_telemetry.jsonl"))
        })
        .map(JsonlTelemetrySink::new)
}

fn telemetry_capture_from_env() -> TelemetryCapture {
    if std::env::var("AXIOM_EPISTEMIC_CAPTURE_TEXT")
        .map(|value| value == "1")
        .unwrap_or(false)
    {
        TelemetryCapture::FullText
    } else {
        TelemetryCapture::HashesOnly
    }
}

fn validate_input_size(field: &str, value: &str) -> Result<(), String> {
    if value.len() > MAX_EVALUATED_TEXT_BYTES {
        Err(format!("{field} exceeds {MAX_EVALUATED_TEXT_BYTES} bytes"))
    } else {
        Ok(())
    }
}

fn validate_unit_interval(field: &str, value: f32) -> Result<(), String> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(format!("{field} must be a finite value from 0.0 to 1.0"))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicDecision {
    Allow,
    Review,
    Block,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CombinedEpistemicReport {
    pub decision: EpistemicDecision,
    pub grounded_fraction: f32,
    pub unsupported_claims: usize,
    pub unverified_claims: usize,
    pub semantic_judge_ran: bool,
    pub semantic_drift_detected: bool,
    pub semantic: Option<DriftValidation>,
    pub reasons: Vec<String>,
}

pub fn combine_epistemic_signals(
    grounded_fraction: f32,
    unsupported_claims: usize,
    unverified_claims: usize,
    semantic: Option<DriftValidation>,
) -> CombinedEpistemicReport {
    let mut reasons = Vec::new();
    let semantic_judge_ran = semantic.is_some();
    let semantic_drift_detected = semantic
        .as_ref()
        .map(|result| result.drift_detected)
        .unwrap_or(false);

    let decision = if unsupported_claims > 0 || semantic_drift_detected {
        if unsupported_claims > 0 {
            reasons.push(format!(
                "{unsupported_claims} unsupported grounding claim(s)"
            ));
        }
        if semantic_drift_detected {
            reasons.push("semantic epistemic drift detected".to_string());
        }
        EpistemicDecision::Block
    } else if unverified_claims > 0 || !semantic_judge_ran {
        if unverified_claims > 0 {
            reasons.push(format!("{unverified_claims} unverified grounding claim(s)"));
        }
        if !semantic_judge_ran {
            reasons.push("semantic judge was not run".to_string());
        }
        EpistemicDecision::Review
    } else {
        reasons.push("grounding and semantic checks passed".to_string());
        EpistemicDecision::Allow
    };

    CombinedEpistemicReport {
        decision,
        grounded_fraction,
        unsupported_claims,
        unverified_claims,
        semantic_judge_ran,
        semantic_drift_detected,
        semantic,
        reasons,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryCapture {
    HashesOnly,
    FullText,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EpistemicTelemetryRecord {
    pub schema_version: u32,
    pub timestamp_ms: u64,
    pub request_id: String,
    pub prompt_sha256: String,
    pub response_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    pub target_model: String,
    pub judge_model: String,
    pub judge_prompt_version: String,
    pub judge_latency_ms: u64,
    pub grounded_fraction: f32,
    pub unsupported_claims: usize,
    pub unverified_claims: usize,
    pub semantic: DriftValidation,
}

impl EpistemicTelemetryRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: impl Into<String>,
        prompt: &str,
        response: &str,
        target_model: impl Into<String>,
        judge_model: impl Into<String>,
        judge_prompt_version: impl Into<String>,
        judge_latency_ms: u64,
        grounded_fraction: f32,
        unsupported_claims: usize,
        unverified_claims: usize,
        semantic: DriftValidation,
        capture: TelemetryCapture,
    ) -> Self {
        let (stored_prompt, stored_response) = match capture {
            TelemetryCapture::HashesOnly => (None, None),
            TelemetryCapture::FullText => (Some(prompt.to_string()), Some(response.to_string())),
        };
        Self {
            schema_version: DRIFT_SCHEMA_VERSION,
            timestamp_ms: unix_time_ms(),
            request_id: request_id.into(),
            prompt_sha256: sha256_text(prompt),
            response_sha256: sha256_text(response),
            prompt: stored_prompt,
            response: stored_response,
            target_model: target_model.into(),
            judge_model: judge_model.into(),
            judge_prompt_version: judge_prompt_version.into(),
            judge_latency_ms,
            grounded_fraction,
            unsupported_claims,
            unverified_claims,
            semantic,
        }
    }
}

pub fn sha256_text(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[derive(Debug, Clone)]
pub struct JsonlTelemetrySink {
    path: PathBuf,
}

impl JsonlTelemetrySink {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn append(&self, record: &EpistemicTelemetryRecord) -> Result<(), String> {
        record.semantic.validate()?;
        let _guard = TELEMETRY_APPEND_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| "telemetry append lock poisoned".to_string())?;
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("telemetry directory creation failed: {e}"))?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("telemetry open failed: {e}"))?;
        let mut encoded = serde_json::to_vec(record)
            .map_err(|e| format!("telemetry serialization failed: {e}"))?;
        encoded.push(b'\n');
        file.write_all(&encoded)
            .map_err(|e| format!("telemetry append failed: {e}"))?;
        file.flush()
            .map_err(|e| format!("telemetry flush failed: {e}"))
    }
}
