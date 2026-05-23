pub const DEFAULT_CHECKPOINT_PATH: &str = "axiom_kernel_v1.safetensors";
pub const DEFAULT_EOS_TOKEN: u32 = 2;
pub const DEFAULT_LOG_SCAN_AUTO_THRESHOLD: usize = 100_000;

/// Static hyper-parameters for the Axiom-TTT inference engine.
#[derive(Debug, Clone)]
pub struct AxiomConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub num_heads: usize,
    /// Dimension per attention head (`d_model / num_heads`).
    pub head_dim: usize,
    pub vocab_size: usize,
    /// Inner-loop learning rate for the TTT weight update.
    pub lr_inner: f32,
    pub rms_norm_eps: f32,
    /// Force logarithmic associative scan during prefill.
    pub use_log_scan: bool,
    /// Auto-enable logarithmic prefill when sequence length exceeds this threshold.
    pub log_scan_auto_threshold: usize,
}

impl Default for AxiomConfig {
    fn default() -> Self {
        Self {
            d_model: 4096,
            n_layers: 32,
            num_heads: 32,
            head_dim: 128, // 4096 / 32
            vocab_size: 32000,
            lr_inner: 1e-3,
            rms_norm_eps: 1e-6,
            use_log_scan: false,
            log_scan_auto_threshold: DEFAULT_LOG_SCAN_AUTO_THRESHOLD,
        }
    }
}
