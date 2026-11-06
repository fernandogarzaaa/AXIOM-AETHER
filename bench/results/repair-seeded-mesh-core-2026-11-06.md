# Repair corpus — 2026-11-06

Corpus: `/tmp/claude-0/-home-user-AXIOM-AETHER/3625c33e-e5a7-566a-8b5d-bee72903a15e/scratchpad/seeded_dbg` · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`.

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| gumbel-L18-minmax-26 | rust | FAIL | no repair found | 47.3s, TREE MODIFIED: Files /tmp/axiom_bench_repair_gumbel-L18-minmax-26_snap_kjQvSR/proptest-regressions/mesh.txt and /tmp/axiom_bench_repair_gumbel-L18-minmax-26_kwvROZ/proptest-regressions/mesh.txt differ  |
| gumbel-L21-arith-23 | rust | PASS | mutation repair verified green (arithmetic:L21 * -> /) | 20.1s, authored fix, pinned 0 file(s) intact |
| gumbel-L26-range-21 | rust | FAIL | no repair found | 47.6s, TREE MODIFIED: Files /tmp/axiom_bench_repair_gumbel-L26-range-21_snap_yoksgp/proptest-regressions/mesh.txt and /tmp/axiom_bench_repair_gumbel-L26-range-21_3vMHOJ/proptest-regressions/mesh.txt differ  |
| idc-L162-logic-20 | rust | PASS | mutation repair verified green (comparison:L162 <= -> >=) | 13.9s, different fix, pinned 0 file(s) intact |
| idc-L94-range-07 | rust | PASS | mutation repair verified green (off-by-one:L94 ..= -> ..) | 13.8s, authored fix, pinned 0 file(s) intact |
| mesh-L112-bool-17 | rust | FAIL | no repair found | 45.3s, TREE MODIFIED: Only in /tmp/axiom_bench_repair_mesh-L112-bool-17_OiL6T2: proptest-regressions  |
| mesh-L148-idswap-19 | rust | FAIL | no repair found | 48.7s, tree intact |
| mesh-L190-cmp-30 | rust | PASS | mutation repair verified green (comparison:L190 != -> ==) | 14.0s, authored fix, pinned 0 file(s) intact |
| mesh-L230-offbyone-06 | rust | PASS | mutation repair verified green (off-by-one:L230 drop `- 1`) | 34.0s, authored fix, pinned 0 file(s) intact |
| mesh-L257-arith-35 | rust | FAIL | no repair found | 46.6s, TREE MODIFIED: Only in /tmp/axiom_bench_repair_mesh-L257-arith-35_fozyRQ: proptest-regressions  |
| mesh-L269-logic-25 | rust | FAIL | no repair found | 46.1s, TREE MODIFIED: Only in /tmp/axiom_bench_repair_mesh-L269-logic-25_pd1VAD: proptest-regressions  |
| mesh-L269-neg-33 | rust | FAIL | no repair found | 47.2s, TREE MODIFIED: Only in /tmp/axiom_bench_repair_mesh-L269-neg-33_whbFTm: proptest-regressions  |
| mesh-L276-range-15 | rust | PASS | mutation repair verified green (off-by-one:L276 insert `- 1` after length) | 25.4s, different fix, pinned 0 file(s) intact |
| mesh-L277-neg-13 | rust | FAIL | no repair found | 46.0s, TREE MODIFIED: Only in /tmp/axiom_bench_repair_mesh-L277-neg-13_UsVigd: proptest-regressions  |
| mesh-L295-cmp-36 | rust | PASS | mutation repair verified green (comparison:L295 >= -> >) | 12.9s, authored fix, pinned 0 file(s) intact |
| mesh-L296-argswap-08 | rust | FAIL | no repair found | 52.5s, tree intact |
| mesh-L316-neg-38 | rust | PASS | mutation repair verified green (comparison:L316 != -> ==) | 26.6s, different fix, pinned 0 file(s) intact |
| mesh-L318-idswap-39 | rust | FAIL | no repair found | 48.8s, tree intact |
| mesh-L323-argswap-16 | rust | PASS | mutation repair verified green (swap-args:L323 swap args `residual` <-> `payload`) | 14.3s, authored fix, pinned 0 file(s) intact |
| mesh-L324-minmax-12 | rust | FAIL | no repair found | 45.4s, TREE MODIFIED: Files /tmp/axiom_bench_repair_mesh-L324-minmax-12_snap_SnoHhT/proptest-regressions/mesh.txt and /tmp/axiom_bench_repair_mesh-L324-minmax-12_waF0WZ/proptest-regressions/mesh.txt differ  |
| mesh-L324-minmax-32 | rust | FAIL | no repair found | 44.6s, TREE MODIFIED: Files /tmp/axiom_bench_repair_mesh-L324-minmax-32_snap_EQRWsF/proptest-regressions/mesh.txt and /tmp/axiom_bench_repair_mesh-L324-minmax-32_cv2nnE/proptest-regressions/mesh.txt differ  |
| mesh-L324-offbyone-14 | rust | PASS | mutation repair verified green (off-by-one:L324 drop `- 1`) | 15.5s, authored fix, pinned 0 file(s) intact |
| mesh-L325-idswap-05 | rust | FAIL | no repair found | 48.8s, tree intact |
| mesh-L350-range-34 | rust | PASS | mutation repair verified green (off-by-one:L350 ..= -> ..) | 28.4s, different fix, pinned 0 file(s) intact |
| mesh-L397-cmp-03 | rust | PASS | mutation repair verified green (off-by-one:L397 insert `- 1` after length) | 15.2s, different fix, pinned 0 file(s) intact |
| mesh-L397-idswap-31 | rust | FAIL | no repair found | 49.1s, tree intact |
| mesh-L408-idswap-37 | rust | FAIL | no repair found | 49.0s, tree intact |
| mesh-L412-argswap-01 | rust | PASS | mutation repair verified green (swap-args:L412 swap args `residual` <-> `payload`) | 14.0s, authored fix, pinned 0 file(s) intact |
| mesh-L414-neg-40 | rust | FAIL | no repair found | 45.7s, TREE MODIFIED: Only in /tmp/axiom_bench_repair_mesh-L414-neg-40_Kf2ZKc: proptest-regressions  |
| mesh-L420-idswap-24 | rust | FAIL | no repair found | 48.8s, tree intact |
| mesh-L464-offbyone-28 | rust | FAIL | no repair found | 47.6s, TREE MODIFIED: Files /tmp/axiom_bench_repair_mesh-L464-offbyone-28_snap_MBwOBe/proptest-regressions/mesh.txt and /tmp/axiom_bench_repair_mesh-L464-offbyone-28_D2j2sf/proptest-regressions/mesh.txt differ  |
| mesh-L474-idswap-10 | rust | FAIL | no repair found | 49.4s, tree intact |
| mesh-L476-argswap-22 | rust | PASS | mutation repair verified green (swap-args:L476 swap args `residual` <-> `payload`) | 13.7s, authored fix, pinned 0 file(s) intact |
| mesh-L484-bool-02 | rust | PASS | mutation repair verified green (boolean:L484 true -> false) | 28.9s, authored fix, pinned 0 file(s) intact |
| mesh-L488-cmp-18 | rust | PASS | mutation repair verified green (off-by-one:L492 1 -> 2) | 22.8s, different fix, pinned 0 file(s) intact |
| mesh-L488-logic-11 | rust | PASS | mutation repair verified green (logic:L488 && -> ||) | 31.9s, different fix, pinned 0 file(s) intact |
| mesh-L489-flow-04 | rust | FAIL | no repair found | 46.2s, TREE MODIFIED: Only in /tmp/axiom_bench_repair_mesh-L489-flow-04_Nc5Sdv: proptest-regressions  |
| mesh-L491-bool-09 | rust | PASS | mutation repair verified green (boolean:L491 false -> true) | 29.2s, authored fix, pinned 0 file(s) intact |
| mesh-L502-range-29 | rust | PASS | mutation repair verified green (off-by-one:L502 ..= -> ..) | 41.7s, different fix, pinned 0 file(s) intact |
| mesh-L503-neg-27 | rust | FAIL | no repair found | 46.3s, tree intact |

**pass: 19 · fail: 21 · skipped: 0 · not broken: 0 · rollback violations: 12**
