#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct StateDeltaUpdate {
    pub session_id: String,
    pub layer_index: usize,
    pub sequence_version: u64,
    pub timestamp: i64,
    pub delta_bytes: Vec<u8>,
}
