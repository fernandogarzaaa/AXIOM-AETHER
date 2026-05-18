# AXIOM-AETHER

A research project for new AI tech.

## Axiom-TTT engine scaffold

This repository now includes a self-contained `axiom_engine/` package implementing:

- `TTTLinearLayer` with online per-token dynamic weight updates
- async JIT context collection and context tensor packing
- stacked `AxiomTTTEngine` blocks (RMSNorm + TTT + SwiGLU FFN)
- streaming inference runner that updates dynamic state from retrieved context and flushes state after generation

### Quick run

```bash
python -m axiom_engine.inference "debug this failing framework upgrade" --d-model 256 --n-layers 4 --max-new-tokens 16
```

Use smaller dimensions for local experiments to avoid high memory use.
