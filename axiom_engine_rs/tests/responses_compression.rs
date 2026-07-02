use axiom_engine::responses_compressor::{apply_plan, plan_compression};
use serde_json::{json, Value};

#[test]
fn selects_only_old_text_assistant_messages() {
    let body = json!({
        "instructions": "keep exactly",
        "input": [
            {"role":"user", "content":"old question"},
            {"role":"assistant", "content":"long old answer"},
            {"role":"developer", "content":"policy"},
            {"role":"assistant", "content":[{"type":"input_image","image_url":"data:image/png;base64,x"}]},
            {"role":"user", "content":"current question"}
        ],
        "tools": [{"type":"function", "name":"lookup", "parameters":{"type":"object"}}]
    });
    let plan = plan_compression(&body).expect("one old assistant message is eligible");
    assert_eq!(plan.item_indices(), vec![1]);
    assert_eq!(plan.total_context(), "long old answer");
    assert_eq!(plan.query, "current question");
}

#[test]
fn structural_item_protects_itself_and_everything_after_it() {
    let body = json!({"input":[
        {"role":"assistant", "content":"eligible prefix"},
        {"type":"reasoning", "id":"r1", "summary":[]},
        {"role":"assistant", "content":"must stay"},
        {"type":"function_call", "call_id":"c1", "name":"f", "arguments":"{}"},
        {"type":"function_call_output", "call_id":"c1", "output":"ok"},
        {"role":"user", "content":"continue"}
    ]});
    let plan = plan_compression(&body).unwrap();
    assert_eq!(plan.item_indices(), vec![0]);
}

#[test]
fn transform_preserves_every_unselected_value_exactly() {
    let body = json!({"model":"gpt-5.5", "previous_response_id":"resp_1", "input":[
        {"role":"user", "content":"old question"},
        {"role":"assistant", "content":"old answer"},
        {"type":"reasoning", "id":"r1", "encrypted_content":"opaque", "summary":[]},
        {"type":"function_call_output", "call_id":"c1", "output":"tool output"},
        {"role":"user", "content":"new question"}
    ]});
    let plan = plan_compression(&body).unwrap();
    let output = apply_plan(&body, &plan, "<axiom_context_fingerprint />").unwrap();
    assert_eq!(output["model"], body["model"]);
    assert_eq!(output["previous_response_id"], body["previous_response_id"]);
    assert_eq!(output["input"][0], body["input"][0]);
    assert_eq!(output["input"][2], body["input"][2]);
    assert_eq!(output["input"][3], body["input"][3]);
    assert_eq!(output["input"][4], body["input"][4]);
    assert_eq!(output["input"][1]["role"], "assistant");
    assert!(output["input"][1]["content"]
        .as_str()
        .unwrap()
        .contains("<axiom_context_fingerprint"));
}

#[test]
fn string_input_and_tool_only_input_are_not_compressed() {
    assert!(plan_compression(&json!({"input":"hello"})).is_none());
    assert!(plan_compression(&json!({"input":[
        {"type":"function_call_output", "call_id":"c1", "output":"large"},
        {"role":"user", "content":"continue"}
    ]}))
    .is_none());
}

#[test]
fn selected_source_hashes_are_stable_and_do_not_expose_text() {
    let body = json!({"input":[
        {"role":"assistant", "content":"sensitive historical answer"},
        {"role":"user", "content":"next"}
    ]});
    let plan = plan_compression(&body).unwrap();
    assert_eq!(plan.runs.len(), 1);
    assert_eq!(plan.runs[0].source_hashes.len(), 1);
    assert_eq!(plan.runs[0].source_hashes[0].len(), 64);
    assert!(!plan.runs[0].source_hashes[0].contains("sensitive"));
}

fn _assert_value(_: Value) {}
