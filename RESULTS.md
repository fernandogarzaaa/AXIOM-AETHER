# AxiomBench Results

Generated (unix): 1783078456

| Pillar | Headline |
|---|---|
| cognition | 100% symbol exact-recovery (3/3) |
| trust | 91% supported-claim coverage, 36% false-positive @ delta=0.10 (threshold 0.667) |
| fleet | node B pre-immunized in 0.32 ms; fragment auth enforced (pass) |
| cost | 75.1% byte reduction over 3 replayed record(s) from 3 session(s) |

Reproduce deterministic pillars: `cargo run --release --features tools --bin axiombench`.
Reproduce the live cost pillar without upstream credentials: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_axiombench_cost_mock.ps1`.
