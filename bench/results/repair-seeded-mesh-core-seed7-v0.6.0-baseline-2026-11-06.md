# Repair corpus — 2026-11-06

Corpus: `/tmp/seeded7` (`seed_faults.py --seed 7 --balance`) · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`.

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| gumbel-L26-range-16 | rust | FAIL | no repair found | 36.8s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| gumbel-L77-arith-10 | rust | FAIL | no repair found | 35.8s, tree intact; verifier artifacts: proptest-regressions |
| gumbel-L80-idswap-27 | rust | FAIL | no repair found | 13.7s, tree intact |
| idc-L126-idswap-14 | rust | FAIL | no repair found | 38.4s, tree intact |
| idc-L162-logic-21 | rust | PASS | mutation repair verified green (comparison:L162 <= -> >=) | 3.3s, different fix, pinned 0 file(s) intact |
| idc-L70-idswap-40 | rust | FAIL | no repair found | 39.4s, tree intact |
| idc-L78-offbyone-08 | rust | PASS | mutation repair verified green (off-by-one:L78 drop `- 1`) | 4.2s, authored fix, pinned 0 file(s) intact |
| idc-L80-const-13 | rust | PASS | mutation repair verified green (off-by-one:L80 64 -> 63) | 4.1s, different fix, pinned 0 file(s) intact |
| idc-L80-idswap-35 | rust | FAIL | no repair found | 38.7s, tree intact |
| idc-L94-range-37 | rust | PASS | mutation repair verified green (off-by-one:L94 ..= -> ..) | 4.1s, authored fix, pinned 0 file(s) intact |
| mesh-L112-bool-11 | rust | FAIL | no repair found | 36.3s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L230-idswap-04 | rust | FAIL | no repair found | 38.7s, tree intact |
| mesh-L248-const-39 | rust | PASS | mutation repair verified green (off-by-one:L248 1 -> 0) | 13.9s, authored fix, pinned 0 file(s) intact |
| mesh-L269-logic-05 | rust | FAIL | no repair found | 35.1s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L276-range-24 | rust | PASS | mutation repair verified green (off-by-one:L276 insert `- 1` after length) | 15.4s, different fix, pinned 0 file(s) intact |
| mesh-L277-cmp-12 | rust | FAIL | no repair found | 35.2s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L278-arith-18 | rust | FAIL | no repair found | 34.7s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L281-minmax-36 | rust | FAIL | no repair found | 33.8s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| mesh-L296-argswap-30 | rust | FAIL | no repair found | 36.2s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| mesh-L323-argswap-17 | rust | PASS | mutation repair verified green (swap-args:L323 swap args `residual` <-> `payload`) | 4.1s, authored fix, pinned 0 file(s) intact |
| mesh-L324-minmax-06 | rust | FAIL | no repair found | 36.4s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| mesh-L324-minmax-28 | rust | FAIL | no repair found | 35.2s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| mesh-L350-range-34 | rust | PASS | mutation repair verified green (off-by-one:L350 ..= -> ..) | 17.7s, authored fix, pinned 0 file(s) intact |
| mesh-L356-idswap-32 | rust | FAIL | no repair found | 38.7s, tree intact |
| mesh-L397-cmp-31 | rust | PASS | mutation repair verified green (off-by-one:L397 insert `- 1` after length) | 3.2s, different fix, pinned 0 file(s) intact |
| mesh-L397-offbyone-23 | rust | PASS | mutation repair verified green (off-by-one:L397 drop `- 1`) | 3.2s, authored fix, pinned 0 file(s) intact |
| mesh-L408-neg-22 | rust | PASS | mutation repair verified green (boolean:L408 drop `!`) | 12.0s, different fix, pinned 0 file(s) intact |
| mesh-L412-argswap-01 | rust | PASS | mutation repair verified green (swap-args:L412 swap args `residual` <-> `payload`) | 4.2s, authored fix, pinned 0 file(s) intact |
| mesh-L414-neg-07 | rust | PASS | mutation repair verified green (boolean:L414 drop `!`) | 11.7s, different fix, pinned 0 file(s) intact |
| mesh-L414-neg-29 | rust | FAIL | no repair found | 34.5s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L471-const-26 | rust | PASS | mutation repair verified green (off-by-one:L471 2 -> 1) | 7.6s, authored fix, pinned 0 file(s) intact |
| mesh-L476-argswap-25 | rust | PASS | mutation repair verified green (swap-args:L476 swap args `residual` <-> `payload`) | 4.3s, authored fix, pinned 0 file(s) intact |
| mesh-L476-idswap-20 | rust | FAIL | no repair found | 40.9s, tree intact |
| mesh-L484-bool-02 | rust | PASS | mutation repair verified green (boolean:L484 true -> false) | 18.7s, authored fix, pinned 0 file(s) intact |
| mesh-L488-cmp-38 | rust | PASS | mutation repair verified green (off-by-one:L492 1 -> 2) | 12.2s, different fix, pinned 0 file(s) intact |
| mesh-L488-logic-15 | rust | PASS | mutation repair verified green (logic:L488 && -> ||) | 20.2s, authored fix, pinned 0 file(s) intact |
| mesh-L489-flow-03 | rust | FAIL | no repair found | 36.0s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L491-bool-19 | rust | PASS | mutation repair verified green (boolean:L491 false -> true) | 19.3s, authored fix, pinned 0 file(s) intact |
| mesh-L502-range-09 | rust | PASS | mutation repair verified green (off-by-one:L502 ..= -> ..) | 32.7s, authored fix, pinned 0 file(s) intact |
| mesh-L503-neg-33 | rust | FAIL | no repair found | 38.2s, tree intact |

**pass: 20 · fail: 20 · skipped: 0 · not broken: 0 · rollback violations: 0**

_12 failed fixture(s) left test-framework artifacts that the verifier wrote while a candidate was failing (e.g. `proptest-regressions/`). These are not solver edits and are not counted as rollback violations; they are what running the same tests by hand on a broken candidate would leave._
