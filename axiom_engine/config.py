from dataclasses import dataclass


@dataclass(frozen=True)
class AxiomConfig:
    d_model: int = 4096
    n_layers: int = 32
    num_heads: int = 32
    vocab_size: int = 32000
    lr_inner: float = 1e-3
    rms_norm_eps: float = 1e-6
    max_context_tokens: int = 1024
