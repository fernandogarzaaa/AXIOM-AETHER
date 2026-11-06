# Repair corpus — 2026-11-06

Corpus: `/tmp/seeded-violations` · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`.

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| gumbel-L18-minmax-26 | rust | FAIL | no repair found | 46.7s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| gumbel-L26-range-21 | rust | FAIL | no repair found | 46.0s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| mesh-L112-bool-17 | rust | FAIL | no repair found | 46.0s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L257-arith-35 | rust | FAIL | no repair found | 44.0s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L269-logic-25 | rust | FAIL | no repair found | 44.4s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L269-neg-33 | rust | FAIL | no repair found | 44.6s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L277-neg-13 | rust | FAIL | no repair found | 45.6s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L324-minmax-12 | rust | FAIL | no repair found | 45.9s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| mesh-L324-minmax-32 | rust | FAIL | no repair found | 42.5s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| mesh-L414-neg-40 | rust | FAIL | no repair found | 42.1s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L464-offbyone-28 | rust | FAIL | no repair found | 45.2s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| mesh-L489-flow-04 | rust | FAIL | no repair found | 43.8s, tree intact; verifier artifacts: proptest-regressions |

**pass: 0 · fail: 12 · skipped: 0 · not broken: 0 · rollback violations: 0**

_12 failed fixture(s) left test-framework artifacts that the verifier wrote while a candidate was failing (e.g. `proptest-regressions/`). These are not solver edits and are not counted as rollback violations; they are what running the same tests by hand on a broken candidate would leave._
