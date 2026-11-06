# Repair corpus — 2026-11-06

Corpus: `/tmp/claude-0/-home-user-AXIOM-AETHER/3625c33e-e5a7-566a-8b5d-bee72903a15e/scratchpad/seeded_dbg` · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`.

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| gumbel-L18-minmax-26 | rust | FAIL | no repair found | 45.8s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| gumbel-L21-arith-23 | rust | PASS | mutation repair verified green (arithmetic:L21 * -> /) | 19.9s, authored fix, pinned 0 file(s) intact |
| gumbel-L26-range-21 | rust | FAIL | no repair found | 46.6s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| idc-L162-logic-20 | rust | PASS | mutation repair verified green (comparison:L162 <= -> >=) | 23.9s, different fix, pinned 0 file(s) intact |
| idc-L94-range-07 | rust | PASS | mutation repair verified green (off-by-one:L94 ..= -> ..) | 13.3s, authored fix, pinned 0 file(s) intact |
| mesh-L112-bool-17 | rust | FAIL | no repair found | 46.9s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L148-idswap-19 | rust | FAIL | no repair found | 51.2s, tree intact |
| mesh-L190-cmp-30 | rust | PASS | mutation repair verified green (comparison:L190 != -> ==) | 14.4s, authored fix, pinned 0 file(s) intact |
| mesh-L230-offbyone-06 | rust | PASS | mutation repair verified green (off-by-one:L230 drop `- 1`) | 34.7s, authored fix, pinned 0 file(s) intact |
| mesh-L257-arith-35 | rust | FAIL | no repair found | 47.6s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L269-logic-25 | rust | FAIL | no repair found | 46.1s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L269-neg-33 | rust | FAIL | no repair found | 48.4s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L276-range-15 | rust | PASS | mutation repair verified green (off-by-one:L276 insert `- 1` after length) | 25.8s, different fix, pinned 0 file(s) intact |
| mesh-L277-neg-13 | rust | FAIL | no repair found | 50.5s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L295-cmp-36 | rust | PASS | mutation repair verified green (comparison:L295 >= -> >) | 13.4s, authored fix, pinned 0 file(s) intact |
| mesh-L296-argswap-08 | rust | FAIL | no repair found | 47.7s, tree intact |
| mesh-L316-neg-38 | rust | PASS | mutation repair verified green (comparison:L316 != -> ==) | 26.6s, different fix, pinned 0 file(s) intact |
| mesh-L318-idswap-39 | rust | FAIL | no repair found | 50.2s, tree intact |
| mesh-L323-argswap-16 | rust | PASS | mutation repair verified green (swap-args:L323 swap args `residual` <-> `payload`) | 14.7s, authored fix, pinned 0 file(s) intact |
| mesh-L324-minmax-12 | rust | FAIL | no repair found | 49.7s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| mesh-L324-minmax-32 | rust | FAIL | no repair found | 47.9s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| mesh-L324-offbyone-14 | rust | PASS | mutation repair verified green (off-by-one:L324 drop `- 1`) | 16.7s, authored fix, pinned 0 file(s) intact |
| mesh-L325-idswap-05 | rust | FAIL | no repair found | 49.6s, tree intact |
| mesh-L350-range-34 | rust | PASS | mutation repair verified green (off-by-one:L350 ..= -> ..) | 23.1s, authored fix, pinned 0 file(s) intact |
| mesh-L397-cmp-03 | rust | PASS | mutation repair verified green (off-by-one:L397 insert `- 1` after length) | 13.9s, different fix, pinned 0 file(s) intact |
| mesh-L397-idswap-31 | rust | FAIL | no repair found | 47.4s, tree intact |
| mesh-L408-idswap-37 | rust | FAIL | no repair found | 49.0s, tree intact |
| mesh-L412-argswap-01 | rust | PASS | mutation repair verified green (swap-args:L412 swap args `residual` <-> `payload`) | 14.1s, authored fix, pinned 0 file(s) intact |
| mesh-L414-neg-40 | rust | FAIL | no repair found | 46.6s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L420-idswap-24 | rust | FAIL | no repair found | 49.1s, tree intact |
| mesh-L464-offbyone-28 | rust | FAIL | no repair found | 46.7s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| mesh-L474-idswap-10 | rust | FAIL | no repair found | 49.1s, tree intact |
| mesh-L476-argswap-22 | rust | PASS | mutation repair verified green (swap-args:L476 swap args `residual` <-> `payload`) | 13.7s, authored fix, pinned 0 file(s) intact |
| mesh-L484-bool-02 | rust | PASS | mutation repair verified green (boolean:L484 true -> false) | 29.6s, authored fix, pinned 0 file(s) intact |
| mesh-L488-cmp-18 | rust | PASS | mutation repair verified green (off-by-one:L492 1 -> 2) | 23.1s, different fix, pinned 0 file(s) intact |
| mesh-L488-logic-11 | rust | PASS | mutation repair verified green (logic:L488 && -> ||) | 31.2s, authored fix, pinned 0 file(s) intact |
| mesh-L489-flow-04 | rust | FAIL | no repair found | 46.8s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L491-bool-09 | rust | PASS | mutation repair verified green (boolean:L491 false -> true) | 29.5s, authored fix, pinned 0 file(s) intact |
| mesh-L502-range-29 | rust | PASS | mutation repair verified green (off-by-one:L502 ..= -> ..) | 41.3s, authored fix, pinned 0 file(s) intact |
| mesh-L503-neg-27 | rust | FAIL | no repair found | 46.4s, tree intact |

**pass: 19 · fail: 21 · skipped: 0 · not broken: 0 · rollback violations: 0**

_12 failed fixture(s) left test-framework artifacts that the verifier wrote while a candidate was failing (e.g. `proptest-regressions/`). These are not solver edits and are not counted as rollback violations; they are what running the same tests by hand on a broken candidate would leave._
