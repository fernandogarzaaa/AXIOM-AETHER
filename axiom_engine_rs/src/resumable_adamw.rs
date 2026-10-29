//! Resumable AdamW optimizer with serializable optimizer state.
//!
//! Candle's built-in `AdamW` keeps `first_moment` and `second_moment` in a
//! private `VarAdamW` with no accessor, and `step_t` is also private. When
//! training resumes from a checkpoint after a VM reboot, weights load fine
//! but Adam restarts cold (moments at zero, step_t at zero), causing loss
//! spikes and invalid comparisons against uninterrupted baselines.
//!
//! `ResumableAdamW` mirrors candle 0.10.2's AdamW update rule exactly, but
//! exposes the moment buffers and step counter so they can be saved to and
//! restored from a safetensors sidecar file alongside the model checkpoint.

use candle_core::{Device, Result, Tensor, Var};
use candle_nn::optim::ParamsAdamW;
use candle_nn::Optimizer;
use std::collections::HashMap;
use std::path::Path;

/// One parameter plus its Adam moment buffers.
#[derive(Debug)]
struct ResumableVarAdamW {
    name: String,
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

/// AdamW with serializable optimizer state.
///
/// Implements the exact same update rule as `candle_nn::optim::AdamW`
/// (candle 0.10.2), including bias correction via `step_t`.
#[derive(Debug)]
pub struct ResumableAdamW {
    vars: Vec<ResumableVarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

impl ResumableAdamW {
    /// Create from named variables. Names are used as keys when saving
    /// optimizer state, so save/load is order-independent.
    pub fn new_named(named_vars: Vec<(String, Var)>, params: ParamsAdamW) -> Result<Self> {
        let vars = named_vars
            .into_iter()
            .filter(|(_, var)| var.dtype().is_float())
            .map(|(name, var)| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(ResumableVarAdamW {
                    name,
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            step_t: 0,
            params,
        })
    }

    /// Current step counter (used for bias correction).
    pub fn step_count(&self) -> usize {
        self.step_t
    }

    /// Save optimizer state (moments + step counter) to a safetensors file.
    ///
    /// Format: for each parameter `name`, stores `{name}.m` (first moment)
    /// and `{name}.v` (second moment). The step counter is stored as a
    /// scalar u64 tensor under the key `__step_t__`.
    pub fn save_state<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for v in &self.vars {
            map.insert(format!("{}.m", v.name), v.first_moment.as_tensor().clone());
            map.insert(format!("{}.v", v.name), v.second_moment.as_tensor().clone());
        }
        // step_t as a scalar u32 tensor.
        let device = Device::Cpu;
        let step_tensor = Tensor::new(self.step_t as u32, &device)?;
        map.insert("__step_t__".to_string(), step_tensor);
        candle_core::safetensors::save(&map, path)?;
        Ok(())
    }

    /// Load optimizer state from a safetensors file written by `save_state`.
    ///
    /// Parameters whose moments are missing from the file keep their current
    /// (zero) moments; this makes loading tolerant of architecture changes.
    /// Returns the number of parameters whose state was restored.
    pub fn load_state<P: AsRef<Path>>(&mut self, path: P, device: &Device) -> Result<usize> {
        let map = candle_core::safetensors::load(path, device)?;
        let mut restored = 0usize;
        // Restore step counter first.
        if let Some(t) = map.get("__step_t__") {
            if let Ok(s) = t.to_scalar::<u32>() {
                self.step_t = s as usize;
            }
        }
        for v in &mut self.vars {
            let m_key = format!("{}.m", v.name);
            let v_key = format!("{}.v", v.name);
            let mut ok = true;
            if let Some(m) = map.get(&m_key) {
                if v.first_moment.set(m).is_err() {
                    ok = false;
                }
            } else {
                ok = false;
            }
            if let Some(vt) = map.get(&v_key) {
                if v.second_moment.set(vt).is_err() {
                    ok = false;
                }
            } else {
                ok = false;
            }
            if ok {
                restored += 1;
            }
        }
        Ok(restored)
    }
}

impl Optimizer for ResumableAdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        // Fallback: auto-generate names from indices. Prefer `new_named`.
        let named: Vec<(String, Var)> = vars
            .into_iter()
            .enumerate()
            .map(|(i, v)| (format!("param_{i}"), v))
            .collect();
        Self::new_named(named, params)
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        // Exact mirror of candle-nn 0.10.2 AdamW::step.
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

impl ResumableAdamW {
    /// Convenience: build from a `VarMap`'s named variables.
    pub fn from_varmap(varmap: &candle_nn::VarMap, params: ParamsAdamW) -> Result<Self> {
        let data = varmap.data();
        let guard = data.lock().unwrap();
        let named: Vec<(String, Var)> = guard.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        drop(guard);
        Self::new_named(named, params)
    }
}

/// Path for the optimizer-state sidecar given a checkpoint path.
pub fn optim_state_path<P: AsRef<Path>>(ckpt: P) -> std::path::PathBuf {
    let mut s = ckpt.as_ref().as_os_str().to_owned();
    s.push(".optim");
    std::path::PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Verify that save → load round-trips the exact optimizer state,
    /// including moments and step counter.
    #[test]
    fn test_optim_state_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        // Create Vars directly.
        let v1 = Var::zeros((4, 4), DType::F32, &device)?;
        let v2 = Var::zeros((8,), DType::F32, &device)?;
        let named = vec![("w1".to_string(), v1), ("w2".to_string(), v2)];

        let params = ParamsAdamW {
            lr: 0.01,
            ..Default::default()
        };
        let mut opt = ResumableAdamW::new_named(named, params)?;

        // Manually set non-zero moments to simulate trained state.
        for v in &opt.vars {
            let shape = v.var.shape();
            let m = Tensor::ones(shape, DType::F32, &device)?;
            let vv = (Tensor::ones(shape, DType::F32, &device)? * 0.5)?;
            v.first_moment.set(&m)?;
            v.second_moment.set(&vv)?;
        }
        opt.step_t = 42;

        // Snapshot.
        let before: Vec<(String, Tensor, Tensor)> = opt
            .vars
            .iter()
            .map(|v| {
                (
                    v.name.clone(),
                    v.first_moment.as_tensor().clone(),
                    v.second_moment.as_tensor().clone(),
                )
            })
            .collect();

        let path = std::env::temp_dir().join("test_resumable_adamw.optim");
        opt.save_state(&path)?;

        // Fresh optimizer then load.
        let v1b = Var::zeros((4, 4), DType::F32, &device)?;
        let v2b = Var::zeros((8,), DType::F32, &device)?;
        let named2 = vec![("w1".to_string(), v1b), ("w2".to_string(), v2b)];
        let params2 = ParamsAdamW {
            lr: 0.01,
            ..Default::default()
        };
        let mut opt2 = ResumableAdamW::new_named(named2, params2)?;
        assert_eq!(opt2.step_count(), 0);
        let restored = opt2.load_state(&path, &device)?;
        assert_eq!(restored, 2, "both params should restore");
        assert_eq!(opt2.step_count(), 42, "step counter must round-trip");

        for (name, m_before, v_before) in &before {
            let found = opt2.vars.iter().find(|x| &x.name == name).unwrap();
            let dm = ((found.first_moment.as_tensor() - m_before)?
                .abs()?
                .sum_all()?
                .to_scalar::<f32>()? as f64)
                .abs();
            let dv = ((found.second_moment.as_tensor() - v_before)?
                .abs()?
                .sum_all()?
                .to_scalar::<f32>()? as f64)
                .abs();
            assert!(dm < 1e-6, "first moment mismatch for {name}: {dm}");
            assert!(dv < 1e-6, "second moment mismatch for {name}: {dv}");
        }

        std::fs::remove_file(&path).ok();
        Ok(())
    }

    /// Loading from a missing file must fail cleanly (not panic).
    #[test]
    fn test_load_missing_file_errors() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let mut opt = ResumableAdamW::from_varmap(&varmap, ParamsAdamW::default()).unwrap();
        let r = opt.load_state("/tmp/definitely_not_here_12345.optim", &device);
        assert!(r.is_err(), "missing file should error");
    }

    /// Optimizer state path appends .optim to the checkpoint path.
    #[test]
    fn test_optim_state_path() {
        let p = optim_state_path("checkpoints/model.bin");
        assert_eq!(p.to_str().unwrap(), "checkpoints/model.bin.optim");
    }
}
