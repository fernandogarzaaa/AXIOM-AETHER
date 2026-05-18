from dataclasses import dataclass


@dataclass(frozen=True)
class AxiomConfig:
    d_model: int = 4096
    n_layers: int = 32
    lr_inner: float = 1e-3
    vocab_size: int = 32000
    ffn_multiplier: int = 4
    eps: float = 1e-6
    max_context_tokens: int = 1024
