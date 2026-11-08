# Proof loop — 2026-11-08

| Metric | Value | Config |
|---|---|---|
| Autonomous repair | score: 32/32 = 100% | built-in fixtures, CPU, no LLM |
| Compression (legacy tokenizer) | 84.5% token savings (260383 -> 40366) with 3085/3085 = 100.0% round-trip | `AXIOM_PRODUCTION_BPE=0`, `axiom_engine_rs/src` |
| Compression (production BPE) | skipped (no checkpoint at checkpoints/axiom_production_bpe.bin) | `AXIOM_PRODUCTION_BPE=1`, `axiom_engine_rs/src` |
| Trust gate | [trust] 100% supported coverage, neural contradiction catch-rate 10% -> 100% (threshold 0.750) | `bench/trust/claims.jsonl` |

_Cost on real traffic is measured separately — see `bench/cvm/*RESULTS*.md`._
