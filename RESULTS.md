# AxiomBench Results

Generated (unix): 1783060971

| Pillar | Headline |
|---|---|
| cognition | 100% symbol exact-recovery (3/3) |
| trust | 91% supported-claim coverage, 36% false-positive @ delta=0.10 (threshold 0.667) |
| fleet | node B pre-immunized in 0.18 ms; fragment auth enforced (pass) |
| cost | no successful live cost replays (3 errored, 0 skipped) on 3 corpus session(s) |

Reproduce: `cargo run --release --features tools --bin axiombench` (add `--live` for the cost pillar).
