//! Differential Weight Exchange (DWE).
//!
//! DWE serializes fast-weight deltas as compact bincode fragments and ships
//! them over isolated Tokio TCP tasks. Consumers inflate the fragment and apply
//! tensor additions directly, avoiding text prefill replay.

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

pub const MAX_DWE_FRAGMENT_BYTES: usize = 8 * 1024 * 1024;
pub const DEFAULT_DWE_QUEUE: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DweLayerDelta {
    pub layer_index: usize,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DweFragment {
    pub schema: String,
    pub session_id: String,
    pub sequence: u64,
    pub layers: Vec<DweLayerDelta>,
    pub state_hash: String,
    /// Fleet-key HMAC over the authenticated fields (see `sign_fragment`).
    /// `None` on an unsigned fragment; the listener rejects unsigned input
    /// when a fleet key is configured.
    #[serde(default)]
    pub hmac: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DweTelemetry {
    pub queued_fragments: usize,
    pub sent_fragments: u64,
    pub received_fragments: u64,
    pub applied_fragments: u64,
    pub rejected_fragments: u64,
    pub last_session_id: Option<String>,
    pub last_fragment_bytes: usize,
    pub last_error: Option<String>,
}

#[derive(Clone)]
pub struct DweBus {
    tx: mpsc::Sender<DweFragment>,
    telemetry: Arc<Mutex<DweTelemetry>>,
    signing_key: Option<Vec<u8>>,
}

impl DweBus {
    pub fn new(peers: Vec<String>) -> Self {
        Self::new_with_signing_key(peers, None)
    }

    pub fn new_with_signing_key(peers: Vec<String>, signing_key: Option<Vec<u8>>) -> Self {
        let (tx, rx) = mpsc::channel::<DweFragment>(DEFAULT_DWE_QUEUE);
        let telemetry = Arc::new(Mutex::new(DweTelemetry::default()));
        attach_broadcast_task(peers, rx, telemetry.clone());
        Self {
            tx,
            telemetry,
            signing_key,
        }
    }

    pub fn disabled() -> Self {
        Self::new(Vec::new())
    }

    pub fn from_env() -> Self {
        let peers = std::env::var("AXIOM_DWE_PEERS")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|peer| !peer.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        let signing_key = std::env::var("AXIOM_FLEET_KEY")
            .ok()
            .map(|key| key.trim().as_bytes().to_vec())
            .filter(|key| !key.is_empty());
        Self::new_with_signing_key(peers, signing_key)
    }

    /// Shared telemetry handle for wiring an inbound listener
    /// (`start_dwe_listener`) to the same counters the bus reports.
    pub fn telemetry_handle(&self) -> Arc<Mutex<DweTelemetry>> {
        self.telemetry.clone()
    }

    pub fn telemetry(&self) -> DweTelemetry {
        let mut telemetry = self.telemetry.lock().map(|t| t.clone()).unwrap_or_default();
        telemetry.queued_fragments = self.tx.max_capacity().saturating_sub(self.tx.capacity());
        telemetry
    }

    pub fn broadcast(&self, mut fragment: DweFragment) {
        // Stamp a process-monotonic sequence so a peer's (session, sequence)
        // replay guard never wrongly drops two legitimate updates produced in
        // the same wall-clock second (producers seed `sequence` from seconds).
        // A true replay re-sends identical signed bytes → identical sequence →
        // correctly rejected; distinct updates always differ here.
        static DWE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        fragment.sequence = DWE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Sign every outbound fragment when a fleet key is configured so peers
        // can authenticate it. Signing happens AFTER the sequence is stamped so
        // the HMAC covers the final sequence. Unsigned broadcast is only
        // meaningful for a keyless (single-node) setup, where the listener also
        // refuses to run — so a peer only ever sees signed frames.
        if let Some(key) = &self.signing_key {
            if !key.is_empty() {
                sign_fragment(&mut fragment, key);
            }
        }
        if let Err(err) = self.tx.try_send(fragment) {
            update_telemetry(&self.telemetry, |t| {
                t.last_error = Some(format!("dwe queue full: {err}"));
            });
        }
    }
}

fn attach_broadcast_task(
    peers: Vec<String>,
    rx: mpsc::Receiver<DweFragment>,
    telemetry: Arc<Mutex<DweTelemetry>>,
) {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        if peers.is_empty() {
            std::mem::forget(rx);
        } else {
            update_telemetry(&telemetry, |t| {
                t.last_error = Some("dwe peers configured but no tokio runtime active".into());
            });
        }
        return;
    };
    handle.spawn(broadcast_loop(peers, rx, telemetry));
}

async fn broadcast_loop(
    peers: Vec<String>,
    mut rx: mpsc::Receiver<DweFragment>,
    telemetry: Arc<Mutex<DweTelemetry>>,
) {
    while let Some(fragment) = rx.recv().await {
        if peers.is_empty() {
            continue;
        }
        let Ok(encoded) = serialize_fragment_for_telemetry(&fragment, &telemetry) else {
            continue;
        };
        broadcast_fragment_to_peers(&peers, &fragment, &encoded, &telemetry).await;
    }
}

fn serialize_fragment_for_telemetry(
    fragment: &DweFragment,
    telemetry: &Arc<Mutex<DweTelemetry>>,
) -> Result<Vec<u8>, String> {
    serialize_fragment(fragment).inspect_err(|err| {
        update_telemetry(telemetry, |t| {
            t.last_error = Some(err.clone());
        });
    })
}

async fn broadcast_fragment_to_peers(
    peers: &[String],
    fragment: &DweFragment,
    encoded: &[u8],
    telemetry: &Arc<Mutex<DweTelemetry>>,
) {
    for peer in peers {
        match send_encoded_fragment(peer, encoded).await {
            Ok(()) => record_send_success(telemetry, fragment, encoded.len()),
            Err(err) => update_telemetry(telemetry, |t| {
                t.last_error = Some(format!("dwe send {peer}: {err}"));
            }),
        }
    }
}

fn record_send_success(
    telemetry: &Arc<Mutex<DweTelemetry>>,
    fragment: &DweFragment,
    fragment_bytes: usize,
) {
    update_telemetry(telemetry, |t| {
        t.sent_fragments += 1;
        t.last_session_id = Some(fragment.session_id.clone());
        t.last_fragment_bytes = fragment_bytes;
        t.last_error = None;
    });
}

pub fn extract_delta_fragment(
    session_id: &str,
    sequence: u64,
    optimized: &[Tensor],
    baseline: &[Tensor],
) -> CandleResult<DweFragment> {
    let mut layers = Vec::new();
    for (layer_index, (optimized, baseline)) in optimized.iter().zip(baseline).enumerate() {
        let delta = optimized.sub(baseline)?.to_dtype(DType::F32)?;
        let shape = delta.dims().to_vec();
        let values = delta.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        layers.push(DweLayerDelta {
            layer_index,
            shape,
            values,
        });
    }
    let state_hash = hash_layers(&layers);
    Ok(DweFragment {
        schema: "axiom.dwe.v1".into(),
        session_id: session_id.to_string(),
        sequence,
        layers,
        state_hash,
        hmac: None,
    })
}

pub fn serialize_fragment(fragment: &DweFragment) -> Result<Vec<u8>, String> {
    let encoded = bincode::serialize(fragment).map_err(|e| e.to_string())?;
    if encoded.len() > MAX_DWE_FRAGMENT_BYTES {
        return Err(format!(
            "fragment too large: {} > {}",
            encoded.len(),
            MAX_DWE_FRAGMENT_BYTES
        ));
    }
    Ok(encoded)
}

pub fn deserialize_fragment(bytes: &[u8]) -> Result<DweFragment, String> {
    if bytes.len() > MAX_DWE_FRAGMENT_BYTES {
        return Err(format!(
            "fragment too large: {} > {}",
            bytes.len(),
            MAX_DWE_FRAGMENT_BYTES
        ));
    }
    bincode::deserialize(bytes).map_err(|e| e.to_string())
}

/// Deterministic HMAC preimage over the authenticated fields (everything but
/// the `hmac` itself), so a signature covers schema, session, sequence, every
/// layer delta, and the state hash.
pub fn fragment_preimage(fragment: &DweFragment) -> Vec<u8> {
    let shadow = (
        &fragment.schema,
        &fragment.session_id,
        fragment.sequence,
        &fragment.layers,
        &fragment.state_hash,
    );
    bincode::serialize(&shadow).unwrap_or_default()
}

/// Sign a fragment in place with the fleet key (reuses the provenance HMAC).
pub fn sign_fragment(fragment: &mut DweFragment, key: &[u8]) {
    let mac = crate::provenance::hmac_sha256_hex(key, &fragment_preimage(fragment));
    fragment.hmac = Some(mac);
}

/// Verify a fragment's HMAC against the fleet key (constant-time compare over
/// equal-length hex). Rejects unsigned fragments.
pub fn verify_fragment(fragment: &DweFragment, key: &[u8]) -> Result<(), String> {
    let Some(mac) = fragment.hmac.as_deref() else {
        return Err("fragment is unsigned".into());
    };
    let expected = crate::provenance::hmac_sha256_hex(key, &fragment_preimage(fragment));
    if expected.len() != mac.len() {
        return Err("hmac length mismatch".into());
    }
    let diff = expected
        .bytes()
        .zip(mac.bytes())
        .fold(0u8, |acc, (a, b)| acc | (a ^ b));
    if diff == 0 {
        Ok(())
    } else {
        Err("hmac verification failed".into())
    }
}

pub fn apply_fragment(
    states: &mut [Tensor],
    fragment: &DweFragment,
    device: &Device,
) -> CandleResult<()> {
    for layer in &fragment.layers {
        if layer.layer_index >= states.len() {
            candle_core::bail!("dwe layer {} out of range", layer.layer_index);
        }
        let total: usize = layer.shape.iter().product();
        if total != layer.values.len() {
            candle_core::bail!("dwe layer shape/value mismatch");
        }
        let delta = Tensor::from_vec(layer.values.clone(), (total,), device)?
            .reshape(layer.shape.as_slice())?;
        states[layer.layer_index] = states[layer.layer_index].add(&delta)?;
    }
    Ok(())
}

pub async fn start_dwe_listener(
    addr: &str,
    tx: mpsc::Sender<DweFragment>,
    telemetry: Arc<Mutex<DweTelemetry>>,
) -> Result<(), String> {
    let listener = TcpListener::bind(addr).await.map_err(|e| e.to_string())?;
    loop {
        let (mut socket, _) = listener.accept().await.map_err(|e| e.to_string())?;
        let tx = tx.clone();
        let telemetry = telemetry.clone();
        tokio::spawn(async move {
            let mut len_buf = [0_u8; 4];
            if let Err(err) = socket.read_exact(&mut len_buf).await {
                update_telemetry(&telemetry, |t| t.last_error = Some(err.to_string()));
                return;
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            if len > MAX_DWE_FRAGMENT_BYTES {
                update_telemetry(&telemetry, |t| {
                    t.last_error = Some(format!("incoming fragment too large: {len}"));
                });
                return;
            }
            let mut bytes = vec![0_u8; len];
            if let Err(err) = socket.read_exact(&mut bytes).await {
                update_telemetry(&telemetry, |t| t.last_error = Some(err.to_string()));
                return;
            }
            match deserialize_fragment(&bytes) {
                Ok(fragment) => {
                    update_telemetry(&telemetry, |t| {
                        t.received_fragments += 1;
                        t.last_session_id = Some(fragment.session_id.clone());
                        t.last_fragment_bytes = bytes.len();
                        t.last_error = None;
                    });
                    let _ = tx.send(fragment).await;
                }
                Err(err) => update_telemetry(&telemetry, |t| t.last_error = Some(err)),
            }
        });
    }
}

async fn send_encoded_fragment(peer: &str, encoded: &[u8]) -> Result<(), String> {
    let mut stream = TcpStream::connect(peer).await.map_err(|e| e.to_string())?;
    stream
        .write_all(&(encoded.len() as u32).to_le_bytes())
        .await
        .map_err(|e| e.to_string())?;
    stream.write_all(encoded).await.map_err(|e| e.to_string())
}

fn hash_layers(layers: &[DweLayerDelta]) -> String {
    let mut hasher = Sha256::new();
    for layer in layers {
        hasher.update(layer.layer_index.to_le_bytes());
        for dim in &layer.shape {
            hasher.update(dim.to_le_bytes());
        }
        for value in &layer.values {
            hasher.update(value.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn update_telemetry(telemetry: &Arc<Mutex<DweTelemetry>>, update: impl FnOnce(&mut DweTelemetry)) {
    if let Ok(mut telemetry) = telemetry.lock() {
        update(&mut telemetry);
    }
}

/// Verify during graceful fleet-key rotation. The current key is tried first;
/// `previous_key`, when present, is accepted only as a rotation window fallback.
pub fn verify_fragment_with_rotation(
    fragment: &DweFragment,
    current_key: &[u8],
    previous_key: Option<&[u8]>,
) -> Result<(), String> {
    match verify_fragment(fragment, current_key) {
        Ok(()) => Ok(()),
        Err(current_error) => {
            if previous_key
                .map(|key| verify_fragment(fragment, key).is_ok())
                .unwrap_or(false)
            {
                Ok(())
            } else {
                Err(current_error)
            }
        }
    }
}

pub fn record_applied_fragment(telemetry: &Arc<Mutex<DweTelemetry>>, session_id: &str) {
    update_telemetry(telemetry, |t| {
        t.applied_fragments += 1;
        t.last_session_id = Some(session_id.to_string());
        t.last_error = None;
    });
}

pub fn record_rejected_fragment(
    telemetry: &Arc<Mutex<DweTelemetry>>,
    session_id: &str,
    error: &str,
) {
    update_telemetry(telemetry, |t| {
        t.rejected_fragments += 1;
        t.last_session_id = Some(session_id.to_string());
        t.last_error = Some(error.to_string());
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragment_roundtrips_and_applies_delta() {
        let device = Device::Cpu;
        let base = vec![Tensor::zeros((2, 2), DType::F32, &device).unwrap()];
        let optimized = vec![Tensor::ones((2, 2), DType::F32, &device).unwrap()];
        let fragment = extract_delta_fragment("dwe", 1, &optimized, &base).unwrap();
        let bytes = serialize_fragment(&fragment).unwrap();
        let decoded = deserialize_fragment(&bytes).unwrap();
        let mut local = base;
        apply_fragment(&mut local, &decoded, &device).unwrap();
        assert_eq!(
            local[0].to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 1.0], vec![1.0, 1.0]]
        );
    }

    fn sample_fragment() -> DweFragment {
        DweFragment {
            schema: "axiom.dwe.v1".into(),
            session_id: "s1".into(),
            sequence: 7,
            layers: vec![DweLayerDelta {
                layer_index: 0,
                shape: vec![2],
                values: vec![1.0, 2.0],
            }],
            state_hash: "abc".into(),
            hmac: None,
        }
    }

    #[test]
    fn signed_fragment_verifies_with_same_key() {
        let key = b"fleet-secret";
        let mut f = sample_fragment();
        sign_fragment(&mut f, key);
        assert!(f.hmac.is_some());
        assert!(verify_fragment(&f, key).is_ok());
    }

    #[test]
    fn tampered_fragment_fails_verification() {
        let key = b"fleet-secret";
        let mut f = sample_fragment();
        sign_fragment(&mut f, key);
        f.layers[0].values[0] = 99.0;
        assert!(verify_fragment(&f, key).is_err());
    }

    #[test]
    fn wrong_key_and_missing_hmac_fail() {
        let mut f = sample_fragment();
        sign_fragment(&mut f, b"key-a");
        assert!(verify_fragment(&f, b"key-b").is_err());
        f.hmac = None;
        assert!(verify_fragment(&f, b"key-a").is_err());
    }

    #[test]
    fn rotation_window_accepts_previous_key() {
        let mut f = sample_fragment();
        sign_fragment(&mut f, b"old-key");
        assert!(verify_fragment_with_rotation(&f, b"new-key", Some(b"old-key")).is_ok());
        assert!(verify_fragment_with_rotation(&f, b"new-key", Some(b"wrong")).is_err());
    }
}
