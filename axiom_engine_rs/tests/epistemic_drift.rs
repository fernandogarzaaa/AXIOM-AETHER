use std::fs;
use std::time::Duration;

use axiom_engine::epistemic_drift::{
    combine_epistemic_signals, parse_judge_response, sha256_text, DriftValidation,
    EpistemicDecision, EpistemicJudgeConfig, EpistemicTelemetryRecord, JsonlTelemetrySink,
    OpenAiSemanticJudge, TelemetryCapture,
};
use axum::{routing::post, Json, Router};

fn drift_validation() -> DriftValidation {
    DriftValidation {
        schema_version: 1,
        empirical_score: 0.2,
        drift_detected: true,
        drift_onset_index: Some(2),
        primary_trigger_phrases: vec!["existence interprets continuity".to_string()],
        analysis: "The final sentence introduces unsupported teleology.".to_string(),
        confidence: 0.91,
    }
}

#[test]
fn judge_parser_accepts_json_content_and_markdown_fences() {
    let json = serde_json::json!({
        "schema_version": 1,
        "empirical_score": 0.8,
        "drift_detected": false,
        "drift_onset_index": null,
        "primary_trigger_phrases": [],
        "analysis": "The response stays within the requested mechanical explanation.",
        "confidence": 0.9
    })
    .to_string();

    let direct = parse_judge_response(&json).expect("strict JSON should parse");
    let fenced = parse_judge_response(&format!("```json\n{json}\n```"))
        .expect("a single JSON fence should be tolerated");

    assert_eq!(direct, fenced);
    assert!(!direct.drift_detected);
}

#[test]
fn judge_parser_rejects_schema_inconsistency_and_trailing_prose() {
    let inconsistent = r#"{
        "schema_version": 1,
        "empirical_score": 0.9,
        "drift_detected": false,
        "drift_onset_index": 2,
        "primary_trigger_phrases": [],
        "analysis": "inconsistent",
        "confidence": 0.9
    }"#;
    assert!(parse_judge_response(inconsistent).is_err());

    let trailing = format!(
        "{} extra commentary",
        serde_json::to_string(&drift_validation()).unwrap()
    );
    assert!(parse_judge_response(&trailing).is_err());
}

#[tokio::test]
async fn openai_compatible_judge_transport_parses_structured_result() {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            Json(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": serde_json::json!({
                            "schema_version": 1,
                            "empirical_score": 0.15,
                            "drift_detected": true,
                            "drift_onset_index": 1,
                            "primary_trigger_phrases": ["the clock remembers mortality"],
                            "analysis": "The second sentence adds unsupported anthropomorphic meaning.",
                            "confidence": 0.96
                        }).to_string()
                    }
                }]
            }))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("mock listener should bind");
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let config = EpistemicJudgeConfig::new(
        format!("http://{address}"),
        "mock-judge",
        None,
        Duration::from_secs(5),
    )
    .unwrap();
    let judge = OpenAiSemanticJudge::new(config).unwrap();

    let outcome = judge
        .evaluate(
            "Explain how the clock works mechanically.",
            "The escapement advances. The clock remembers mortality.",
            "The escapement releases one gear tooth per oscillation.",
        )
        .await
        .expect("mock judge response should validate");

    assert!(outcome.validation.drift_detected);
    assert_eq!(outcome.validation.drift_onset_index, Some(1));
    server.abort();
}

#[test]
fn validation_rejects_inconsistent_or_unbounded_judge_output() {
    let mut validation = drift_validation();
    assert!(validation.validate().is_ok());

    validation.empirical_score = 1.1;
    assert!(validation.validate().is_err());

    validation = drift_validation();
    validation.confidence = -0.1;
    assert!(validation.validate().is_err());

    validation = drift_validation();
    validation.drift_detected = false;
    assert!(validation.validate().is_err());

    validation.drift_onset_index = None;
    validation.primary_trigger_phrases.clear();
    assert!(validation.validate().is_ok());
}

#[test]
fn combined_decision_escalates_when_semantic_judge_detects_drift() {
    let decision = combine_epistemic_signals(1.0, 0, 0, Some(drift_validation()));

    assert_eq!(decision.decision, EpistemicDecision::Block);
    assert_eq!(decision.grounded_fraction, 1.0);
    assert!(decision.semantic_drift_detected);
    assert!(decision
        .reasons
        .iter()
        .any(|reason| reason.contains("semantic")));
}

#[test]
fn combined_decision_handles_missing_judge_without_claiming_semantic_safety() {
    let decision = combine_epistemic_signals(1.0, 0, 0, None);

    assert_eq!(decision.decision, EpistemicDecision::Review);
    assert!(!decision.semantic_judge_ran);
    assert!(decision
        .reasons
        .iter()
        .any(|reason| reason.contains("not run")));
}

#[test]
fn telemetry_redacts_text_by_default_and_keeps_stable_hashes() {
    let prompt = "Explain the quartz oscillator.";
    let response = "The crystal vibrates at a measurable frequency.";
    let record = EpistemicTelemetryRecord::new(
        "request-1",
        prompt,
        response,
        "target-model",
        "judge-model",
        "epistemic-drift-v1",
        42,
        1.0,
        0,
        0,
        drift_validation(),
        TelemetryCapture::HashesOnly,
    );

    assert_eq!(record.prompt_sha256, sha256_text(prompt));
    assert_eq!(record.response_sha256, sha256_text(response));
    assert_eq!(record.prompt, None);
    assert_eq!(record.response, None);
}

#[test]
fn telemetry_can_capture_text_only_when_explicitly_requested() {
    let record = EpistemicTelemetryRecord::new(
        "request-2",
        "prompt",
        "response",
        "target-model",
        "judge-model",
        "epistemic-drift-v1",
        7,
        0.5,
        1,
        0,
        drift_validation(),
        TelemetryCapture::FullText,
    );

    assert_eq!(record.prompt.as_deref(), Some("prompt"));
    assert_eq!(record.response.as_deref(), Some("response"));
}

#[test]
fn jsonl_sink_appends_one_valid_record_per_line() {
    let dir =
        std::env::temp_dir().join(format!("axiom-epistemic-telemetry-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let path = dir.join("events.jsonl");
    let sink = JsonlTelemetrySink::new(path.clone());
    let record = EpistemicTelemetryRecord::new(
        "request-3",
        "prompt",
        "response",
        "target-model",
        "judge-model",
        "epistemic-drift-v1",
        9,
        0.75,
        0,
        1,
        drift_validation(),
        TelemetryCapture::HashesOnly,
    );

    sink.append(&record).expect("first append should succeed");
    sink.append(&record).expect("second append should succeed");

    let contents = fs::read_to_string(&path).expect("telemetry file should exist");
    let lines: Vec<_> = contents.lines().collect();
    assert_eq!(lines.len(), 2);
    for line in lines {
        let decoded: EpistemicTelemetryRecord =
            serde_json::from_str(line).expect("each line should be valid JSON");
        assert_eq!(decoded.request_id, "request-3");
    }

    fs::remove_dir_all(dir).expect("temporary telemetry should be removable");
}
