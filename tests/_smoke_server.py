"""Boot the Axiom server with a tiny config — for live smoke testing."""

from axiom_engine import server as srv
from axiom_engine.config import AxiomConfig


def _tiny_cfg() -> AxiomConfig:
    return AxiomConfig(
        d_model=16,
        n_layers=2,
        num_heads=2,
        vocab_size=64,
        lr_inner=1e-3,
        max_context_tokens=8,
    )


srv.AxiomConfig = _tiny_cfg


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(srv.app, host="127.0.0.1", port=8080, log_level="warning")
