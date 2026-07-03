# AxiomBench Results

Generated (unix): 1783042338

| Pillar | Headline |
|---|---|
| cognition | 100% symbol exact-recovery (3/3) |
| trust | 91% supported-claim coverage, 36% false-positive @ delta=0.10 (threshold 0.667) |
| fleet | node B pre-immunized in 0.12 ms; fragment auth enforced (pass) |

Reproduce: `cargo run --release --features tools --bin axiombench` (add `--live` for the cost pillar).
