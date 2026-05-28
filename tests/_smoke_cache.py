"""Boot the Axiom server with the cache enabled and prove repeated
identical requests skip generation.

Runs against the local untrained backend so we don't burn Anthropic
tokens — the *backend identity* is irrelevant for this demo; what
matters is that the second request returns instantly from cache.
"""

import os

os.environ["AXIOM_CACHE"] = "1"

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

    uvicorn.run(srv.app, host="127.0.0.1", port=8081, log_level="warning")
