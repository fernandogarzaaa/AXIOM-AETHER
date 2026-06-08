use axiom_engine::skeleton::build_digest;

const ANOMALY: &str = include_str!("fixtures/anomaly_target.rs.txt");
const SURGICAL: &str = include_str!("fixtures/surgical_target.rs.txt");

fn digest(source: &str) -> String {
    build_digest(source, "fixture", 4096, 1.0, "sha256:test", 4)
}

#[test]
fn anomaly_fixture_is_compressed_without_compiling_it() {
    let out = digest(ANOMALY);

    assert!(out.contains("kind=\"structural-skeleton\""));
    assert!(out.contains("static mut GLOBAL_COUNTER_XQZ"), "{out}");
    assert!(out.contains("unsafe extern \"C\" fn frobnicate_the_void"));
    assert!(out.contains("fn goto_emulator_3000(seed: i32) -> i32"));
    assert!(!out.contains("wrapping_mul(31)"));
    assert!(!out.contains("manual_memcpy_horror(scratch"));
}

#[test]
fn surgical_fixture_keeps_clean_api_surface() {
    let out = digest(SURGICAL);

    assert!(out.contains("pub struct LayerState"));
    assert!(out.contains("impl LayerState"));
    assert!(out.contains("pub fn ema_merge(a: &Tensor, b: &Tensor, keep: f64) -> Result<Tensor>"));
    assert!(out.contains("pub fn summarise(states: &[LayerState]) -> Result<HashMap<String, f32>>"));
    assert!(!out.contains("Tensor::eye(d_model"));
    assert!(!out.contains("out.insert(format!"));
}
