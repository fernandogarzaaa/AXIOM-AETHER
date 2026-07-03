# Trust calibration dataset

`claims.jsonl` — labeled `{claim, evidence, supported}` rows used to calibrate the
conformal grounding gate (`hallucination::ConformalGate`).

## Format

One JSON object per line:

```json
{"claim": "…", "evidence": "…", "supported": true}
```

- `claim` — a single declarative sentence to verify.
- `evidence` — the context the claim is checked against.
- `supported` — ground-truth label: `true` if the evidence genuinely supports the
  claim, `false` otherwise.

## Composition

Rows span three families so the calibration reflects real gate behavior:

1. **Supported** — the claim's facts appear in the evidence.
2. **Unsupported** — the claim's facts are absent from the evidence.
3. **Vocabulary-sharing contradictions** — the claim reuses the evidence's terms but
   asserts the opposite; these are the hardest negatives for a lexical gate.

## Use

`axiom_engine_rs/tests/trust_calibration.rs` splits the set deterministically
(even indices calibrate, odd indices test), calibrates a threshold at δ=0.10 via
`calibrate_conformal_threshold`, and asserts the calibrated gate covers ≥(1−δ) of
genuinely supported held-out claims. The resulting threshold is shipped in
`axiom.env` as `AXIOM_CONFORMAL_THRESHOLD`.

The data is synthetic, deterministic, and secret-free.
