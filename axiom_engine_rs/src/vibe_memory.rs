//! Persistent "vibe memory": an EMA-merged master set of fast-weight (W̃)
//! matrices that accumulates structural codebase patterns across sessions.
//!
//! Each TTT session adapts a per-layer `[d_model, d_model]` fast-weight matrix.
//! Those matrices are normally dropped when the session ends. This module folds
//! them into a long-lived master tensor set using an exponential moving average
//! (EMA):
//!
//! ```text
//!   W_master = decay * W_master + (1 - decay) * W_session
//! ```
//!
//! and serialises the result to `axiom_master_vibe.bin` (safetensors format).
//!
//! Semantics (chosen for production safety):
//! * **Persist-only by default** — committing + saving never changes how new
//!   sessions start. New sessions only start from the master when the caller
//!   explicitly opts in (e.g. `AXIOM_VIBE_PRIME=1`) and uses [`MasterVibe::prime_states`].
//! * **First commit seeds** the master directly from the session (no averaging
//!   against an empty / identity master), so early structure isn't diluted.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result, Tensor};

/// Default decay used by the EMA merge (blueprint specifies 0.9).
pub const DEFAULT_VIBE_DECAY: f64 = 0.9;

/// Default on-disk location for the master vibe tensor set.
pub const DEFAULT_VIBE_PATH: &str = "axiom_master_vibe.bin";

/// Long-lived, EMA-merged master fast-weights.
///
/// Not internally synchronised — wrap in `Arc<Mutex<MasterVibe>>` when shared
/// across request handlers.
pub struct MasterVibe {
    /// One `[d_model, d_model]` master matrix per layer, or `None` until the
    /// first session is committed (or a compatible file is loaded).
    master: Option<Vec<Tensor>>,
    device: Device,
    n_layers: usize,
    d_model: usize,
    path: PathBuf,
    decay: f64,
}

impl MasterVibe {
    /// Per-layer tensor key used inside the safetensors map.
    fn layer_key(i: usize) -> String {
        format!("vibe.layer_{i}")
    }

    /// Load the master vibe from `path` if present, otherwise start empty.
    ///
    /// Shape validation: every loaded layer must be `[d_model, d_model]` and the
    /// layer count must match `n_layers`. An incompatible or corrupt file is
    /// logged and ignored (the engine falls back to an empty master rather than
    /// crashing the running proxy).
    pub fn load_or_init(
        path: impl AsRef<Path>,
        n_layers: usize,
        d_model: usize,
        device: &Device,
        decay: f64,
    ) -> Self {
        let path = path.as_ref().to_path_buf();
        let mut me = Self {
            master: None,
            device: device.clone(),
            n_layers,
            d_model,
            path,
            decay,
        };
        if me.path.exists() {
            match me.try_load() {
                Ok(states) => {
                    eprintln!(
                        "[vibe] loaded master vibe from {} ({} layers, d_model={})",
                        me.path.display(),
                        states.len(),
                        d_model
                    );
                    me.master = Some(states);
                }
                Err(err) => {
                    eprintln!(
                        "[vibe] WARNING: ignoring {} ({err}); starting from an empty master",
                        me.path.display()
                    );
                }
            }
        } else {
            eprintln!(
                "[vibe] no master vibe at {} yet; it will be created on first commit",
                me.path.display()
            );
        }
        me
    }

    /// Convenience constructor reading `decay` from `AXIOM_VIBE_DECAY` (falling
    /// back to [`DEFAULT_VIBE_DECAY`]) and the path from `AXIOM_VIBE_PATH`
    /// (falling back to [`DEFAULT_VIBE_PATH`]).
    pub fn from_env(n_layers: usize, d_model: usize, device: &Device) -> Self {
        let path =
            std::env::var("AXIOM_VIBE_PATH").unwrap_or_else(|_| DEFAULT_VIBE_PATH.to_string());
        let decay = std::env::var("AXIOM_VIBE_DECAY")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|d| (0.0..=1.0).contains(d))
            .unwrap_or(DEFAULT_VIBE_DECAY);
        Self::load_or_init(path, n_layers, d_model, device, decay)
    }

    fn try_load(&self) -> Result<Vec<Tensor>> {
        let map = candle_core::safetensors::load(&self.path, &self.device)?;
        let mut states = Vec::with_capacity(self.n_layers);
        for i in 0..self.n_layers {
            let key = Self::layer_key(i);
            let t = map.get(&key).ok_or_else(|| {
                candle_core::Error::Msg(format!("vibe file missing tensor '{key}'"))
            })?;
            let dims = t.dims();
            if dims != [self.d_model, self.d_model] {
                return Err(candle_core::Error::Msg(format!(
                    "vibe layer {i} shape {dims:?} != [{}, {}]",
                    self.d_model, self.d_model
                )));
            }
            states.push(t.to_dtype(DType::F32)?);
        }
        Ok(states)
    }

    /// Whether a master has been established (a file loaded or a session
    /// committed at least once).
    pub fn is_initialized(&self) -> bool {
        self.master.is_some()
    }

    /// The configured EMA decay.
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Clone the master states for priming a new session. Returns `None` until
    /// the first commit. The caller uses these in place of identity-initialised
    /// W̃ matrices when opt-in priming is enabled.
    pub fn prime_states(&self) -> Option<Vec<Tensor>> {
        self.master
            .as_ref()
            .map(|m| m.iter().cloned().collect::<Vec<_>>())
    }

    /// EMA-merge one session's adapted W̃ states into the master.
    ///
    /// ```text
    ///   W_master = decay * W_master + (1 - decay) * W_session
    /// ```
    ///
    /// The very first commit seeds the master directly from the session (no
    /// averaging against an empty master). Per-layer shapes must match
    /// `[d_model, d_model]`; a mismatch is rejected before any mutation so a bad
    /// session can never corrupt an existing master.
    pub fn commit_session(&mut self, session_states: &[Tensor]) -> Result<()> {
        if session_states.len() != self.n_layers {
            return Err(candle_core::Error::Msg(format!(
                "session has {} layers, expected {}",
                session_states.len(),
                self.n_layers
            )));
        }
        for (i, s) in session_states.iter().enumerate() {
            if s.dims() != [self.d_model, self.d_model] {
                return Err(candle_core::Error::Msg(format!(
                    "session layer {i} shape {:?} != [{}, {}]",
                    s.dims(),
                    self.d_model,
                    self.d_model
                )));
            }
            if !tensor_is_finite(s)? {
                return Err(candle_core::Error::Msg(format!(
                    "session layer {i} contains non-finite values; refusing to commit"
                )));
            }
        }

        let merged = match &self.master {
            // First commit: seed directly from the session.
            None => session_states
                .iter()
                .map(|s| s.to_dtype(DType::F32))
                .collect::<Result<Vec<_>>>()?,
            // Subsequent commits: exponential moving average per layer.
            Some(master) => {
                let keep = self.decay; // weight on the existing master
                let take = 1.0 - self.decay; // weight on the new session
                let mut out = Vec::with_capacity(self.n_layers);
                for (m, s) in master.iter().zip(session_states.iter()) {
                    let m_part = m.affine(keep, 0.0)?; // decay * W_master
                    let s_part = s.to_dtype(DType::F32)?.affine(take, 0.0)?; // (1-decay) * W_session
                    out.push(m_part.add(&s_part)?);
                }
                out
            }
        };
        self.master = Some(merged);
        Ok(())
    }

    /// Serialise the master to disk. Writes to a temp file then renames so a
    /// crash mid-write can never leave a half-written `axiom_master_vibe.bin`.
    /// No-op (returns `Ok`) when nothing has been committed yet.
    pub fn save(&self) -> Result<()> {
        let Some(master) = &self.master else {
            return Ok(());
        };
        let mut map: HashMap<String, Tensor> = HashMap::with_capacity(master.len());
        for (i, t) in master.iter().enumerate() {
            map.insert(Self::layer_key(i), t.clone());
        }
        let tmp = self.path.with_extension("bin.tmp");
        candle_core::safetensors::save(&map, &tmp)?;
        std::fs::rename(&tmp, &self.path)
            .map_err(|e| candle_core::Error::Msg(format!("vibe rename failed: {e}")))?;
        eprintln!("[vibe] saved master vibe -> {}", self.path.display());
        Ok(())
    }

    /// Commit then immediately persist. Convenience for the common
    /// "session concluded" path.
    pub fn commit_and_save(&mut self, session_states: &[Tensor]) -> Result<()> {
        self.commit_session(session_states)?;
        self.save()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn tensor_is_finite(tensor: &Tensor) -> Result<bool> {
    let values = tensor
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(values.into_iter().all(f32::is_finite))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn eye_states(n: usize, d: usize, dev: &Device) -> Vec<Tensor> {
        (0..n).map(|_| Tensor::eye(d, DType::F32, dev).unwrap()).collect()
    }

    #[test]
    fn first_commit_seeds_master() {
        let dev = Device::Cpu;
        let mut vibe = MasterVibe::load_or_init(
            std::env::temp_dir().join("axiom_vibe_test_seed.bin"),
            2,
            4,
            &dev,
            0.9,
        );
        assert!(!vibe.is_initialized());
        let s = eye_states(2, 4, &dev);
        vibe.commit_session(&s).unwrap();
        assert!(vibe.is_initialized());
        // Master should equal the session after the first (seeding) commit.
        let primed = vibe.prime_states().unwrap();
        for (a, b) in primed.iter().zip(s.iter()) {
            let da = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let db = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(da, db);
        }
    }

    #[test]
    fn ema_blends_toward_session() {
        let dev = Device::Cpu;
        let mut vibe = MasterVibe::load_or_init(
            std::env::temp_dir().join("axiom_vibe_test_ema.bin"),
            1,
            2,
            &dev,
            0.9,
        );
        // Seed master = identity.
        vibe.commit_session(&eye_states(1, 2, &dev)).unwrap();
        // New session = 2*identity. Expect 0.9*I + 0.1*(2I) = 1.1*I on the diagonal.
        let twice = vec![Tensor::eye(2, DType::F32, &dev).unwrap().affine(2.0, 0.0).unwrap()];
        vibe.commit_session(&twice).unwrap();
        let m = vibe.prime_states().unwrap();
        let diag = m[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // [0,0] entry: 0.9*1 + 0.1*2 = 1.1
        assert!((diag[0] - 1.1).abs() < 1e-5, "got {}", diag[0]);
    }

    #[test]
    fn rejects_layer_count_mismatch() {
        let dev = Device::Cpu;
        let mut vibe = MasterVibe::load_or_init(
            std::env::temp_dir().join("axiom_vibe_test_mismatch.bin"),
            2,
            4,
            &dev,
            0.9,
        );
        let wrong = eye_states(1, 4, &dev); // only 1 layer, expected 2
        assert!(vibe.commit_session(&wrong).is_err());
    }

    #[test]
    fn save_then_load_roundtrip() {
        let dev = Device::Cpu;
        let path = std::env::temp_dir().join("axiom_vibe_test_roundtrip.bin");
        let _ = std::fs::remove_file(&path);
        let mut vibe = MasterVibe::load_or_init(&path, 2, 3, &dev, 0.9);
        vibe.commit_and_save(&eye_states(2, 3, &dev)).unwrap();

        let reloaded = MasterVibe::load_or_init(&path, 2, 3, &dev, 0.9);
        assert!(reloaded.is_initialized());
        let _ = std::fs::remove_file(&path);
    }
}
