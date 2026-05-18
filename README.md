# AXIOM-AETHER

A research project for new AI tech.

## Axiom-TTT engine scaffold

This repository now includes a self-contained `axiom_engine/` package implementing:

- `TTTLinearLayer` with online per-token dynamic weight updates
- async JIT context collection and context tensor packing
- stacked `AxiomTTTEngine` blocks (RMSNorm + TTT + SwiGLU FFN)
- streaming inference runner that updates dynamic state from retrieved context and flushes state after generation

### Quick run (Python)

```bash
python -m axiom_engine.inference "debug this failing framework upgrade" --d-model 256 --n-layers 4 --max-new-tokens 16
```

Use smaller dimensions for local experiments to avoid high memory use.

## Rust implementation (`axiom_engine_rs/`)

A production-ready Rust port of the same engine built with
[candle](https://github.com/huggingface/candle).  It mirrors the Python
architecture exactly:

| Module | Rust file | Python file |
|---|---|---|
| Config | `src/config.rs` | `axiom_engine/config.py` |
| TTT layer | `src/ttt_layer.rs` | `axiom_engine/ttt_layer.py` |
| Kernel (RMSNorm, SwiGLU, blocks, engine) | `src/kernel.rs` | `axiom_engine/kernel.py` |
| Demo binary | `src/main.rs` | – |

### Quick build and run

```bash
cd axiom_engine_rs
cargo run                   # demo with small config (d_model=64, n_layers=2)
cargo build --release       # optimised build
```

Requires a stable Rust toolchain (edition 2021).  No GPU needed for the
default CPU backend.
