# Trust calibration dataset

`claims.jsonl` contains labeled `{claim, evidence, supported, family}` rows used
to calibrate the conformal grounding gate (`hallucination::ConformalGate`) and
to measure the neural-tier contradiction catch-rate in AxiomBench.

## Format

One JSON object per line:

```json
{"claim":"...","evidence":"...","supported":true,"family":"supported"}
```

- `claim` - a single declarative sentence to verify.
- `evidence` - the context the claim is checked against.
- `supported` - ground-truth label: `true` if the evidence genuinely supports
  the claim, `false` otherwise.
- `family` - one of `supported`, `unsupported`, or `contradiction`.

## Composition

The corpus has 240 deterministic, synthetic, secret-free rows:

1. `supported` - the claim's facts appear in the evidence.
2. `unsupported` - the claim's facts are absent from the evidence.
3. `contradiction` - the claim reuses the evidence vocabulary but asserts the
   opposite; these are the hardest negatives for a lexical gate.

Contradictions are at least one-third of the dataset, and rows are interleaved
so the deterministic even/odd split includes every family on both sides.

## Use

`axiom_engine_rs/tests/trust_calibration.rs` splits the set deterministically:
even indices calibrate, odd indices test. The test calibrates a threshold at
delta=0.10 via `calibrate_conformal_threshold`, asserts per-family coverage, and
checks that the deterministic neural-tier stand-in improves contradiction
catch-rate without weakening supported-claim coverage.

The resulting threshold is shipped in `hallucination.rs` as
`SHIPPED_CONFORMAL_THRESHOLD` and documented in `axiom.env` as
`AXIOM_CONFORMAL_THRESHOLD`.
