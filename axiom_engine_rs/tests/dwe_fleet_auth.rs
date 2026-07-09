use axiom_engine::dwe::{sign_fragment, verify_fragment, DweFragment, DweLayerDelta};

fn frag() -> DweFragment {
    DweFragment {
        schema: "axiom.dwe.v1".into(),
        session_id: "s".into(),
        sequence: 1,
        layers: vec![DweLayerDelta {
            layer_index: 0,
            shape: vec![1],
            values: vec![0.5],
        }],
        state_hash: "h".into(),
        hmac: None,
    }
}

#[test]
fn only_matching_key_and_signed_fragments_verify() {
    let mut f = frag();
    sign_fragment(&mut f, b"shared");
    assert!(verify_fragment(&f, b"shared").is_ok());
    assert!(verify_fragment(&f, b"other").is_err());

    let unsigned = frag();
    assert!(verify_fragment(&unsigned, b"shared").is_err());
}
