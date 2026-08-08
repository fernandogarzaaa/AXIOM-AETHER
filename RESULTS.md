# AxiomBench Results

Generated (unix): 1783089448

| Pillar | Headline | n | Read as |
|---|---|---|---|
| cognition | 3/3 symbol exact-recovery | **3** | smoke check, not a rate |
| trust | 100% supported coverage; neural contradiction catch-rate 10% -> 100% (threshold 0.750) | 240 calibration claims | calibrated result |
| fleet | node B pre-immunized in 0.17 ms; signed DWE applied in 0.10 ms (pass) | 2 nodes | pass/fail, not a benchmark |
| cost | 75.1% byte reduction | **3 replayed records from 3 sessions** | indicative only |

## How to read these

Two of the four pillars run on a **very small sample**, and the `n` column exists
so a percentage cannot be mistaken for a rate. `3/3` is a smoke check that the
round-trip works at all; it is not evidence of a 100% recovery rate, and neither
is a 75.1% reduction measured over three records evidence of a 75% reduction in
general. Both are deliberately reported as counts.

For the compression figure that **is** measured at scale, run
`axiom bench axiom_engine_rs/src`: **82.4% token savings (173,446 -> 30,474) with
2181/2181 = 100.0% signature round-trip** over 120 files. That number carries a
real sample size and is the one to cite. It also runs with no checkpoint, since
skeleton compression is structural rather than model-dependent.

Answer-quality effects are not measured anywhere here and are not claimed. Token
savings and structural fidelity are measured offline; whether a smaller context
changes what a model produces requires a separate upstream evaluation.

## Reproduce

Deterministic pillars:

```bash
cargo run --release --features tools --bin axiombench
```

Compression at scale:

```bash
cargo run --release --bin axiom -- bench axiom_engine_rs/src
```

Live cost pillar without upstream credentials:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_axiombench_cost_mock.ps1
```
