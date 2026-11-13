# Proof loop — 2026-11-13

| Metric | Value | Config |
|---|---|---|
| Autonomous repair | score: 32/32 = 100% | built-in fixtures, CPU, no LLM |
| Compression (legacy tokenizer) | 85.7% token savings (317014 -> 45297) with 3608/3608 = 100.0% round-trip | `AXIOM_PRODUCTION_BPE=0`, `axiom_engine_rs/src` |
| Compression (production BPE) | 87.6% token savings (942678 -> 117041) with 3608/3608 = 100.0% round-trip | `AXIOM_PRODUCTION_BPE=1`, `axiom_engine_rs/src` |
| Trust gate | [trust] 100% supported coverage, neural contradiction catch-rate 10% -> 100% (threshold 0.750) | `bench/trust/claims.jsonl` |

_Cost on real traffic is measured separately — see `bench/cvm/*RESULTS*.md`._
