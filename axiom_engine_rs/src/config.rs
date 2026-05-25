pub const DEFAULT_CHECKPOINT_PATH: &str = "axiom_kernel_v1.safetensors";
pub const DEFAULT_EOS_TOKEN: u32 = 2;

/// Static hyper-parameters for the Axiom-TTT inference engine.
#[derive(Debug, Clone)]
pub struct AxiomConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    /// Inner-loop learning rate for the TTT weight update.
    pub lr_inner: f32,
    pub norm_eps: f32,
}

impl Default for AxiomConfig {
    fn default() -> Self {
        Self {
            d_model: 4096,
            n_layers: 32,
            vocab_size: 32000,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        }
    }
}
