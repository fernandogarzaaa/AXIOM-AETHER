import os
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class AxiomConfig:
    d_model: int = 4096
    n_layers: int = 32
    num_heads: int = 32
    vocab_size: int = 32000
    lr_inner: float = 1e-3
    rms_norm_eps: float = 1e-6
    max_context_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "AxiomConfig":
        """Build a config, honouring ``AXIOM_PY_PRESET`` and per-field overrides.

        The defaults above describe a ~7B-parameter model that needs tens of GB
        of RAM in fp32, which most laptops (and CI boxes) cannot allocate.
        ``AXIOM_PY_PRESET=tiny`` boots a toy model in well under a second for
        trying the API end to end; any field can also be set individually via
        ``AXIOM_PY_<FIELD>`` (for example ``AXIOM_PY_D_MODEL=256``).
        """
        base = {}
        preset = os.environ.get("AXIOM_PY_PRESET", "").strip().lower()
        if preset == "tiny":
            base = dict(d_model=64, n_layers=2, num_heads=4, vocab_size=512,
                        max_context_tokens=256)
        elif preset == "small":
            base = dict(d_model=512, n_layers=4, num_heads=8, vocab_size=8192,
                        max_context_tokens=512)
        elif preset not in ("", "default", "full"):
            raise ValueError(
                f"AXIOM_PY_PRESET={preset!r} is not one of: tiny, small, default"
            )
        for f in fields(cls):
            raw = os.environ.get(f"AXIOM_PY_{f.name.upper()}")
            if raw is not None and raw.strip():
                base[f.name] = type(f.default)(raw)
        return cls(**base)
