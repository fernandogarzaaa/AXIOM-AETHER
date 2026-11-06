# Repair corpus — 2026-11-06

Corpus: `/tmp/claude-0/-home-user-AXIOM-AETHER/3625c33e-e5a7-566a-8b5d-bee72903a15e/scratchpad/seeded` · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`.

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| gumbel-L18-minmax-26 | rust | FAIL | no repair found | 25.8s, tree intact |
| gumbel-L21-arith-23 | rust | FAIL | no repair found | 20.7s, TREE MODIFIED: Only in /tmp/axiom_bench_repair_gumbel-L21-arith-23_mQcBXQ: proptest-regressions  |
| gumbel-L26-range-21 | rust | FAIL | no repair found | 28.8s, tree intact |
| idc-L162-logic-20 | rust | FAIL | no repair found | 23.4s, tree intact |
| idc-L94-range-07 | rust | PASS | mutation repair verified green (off-by-one:L94 ..= -> ..) | 16.6s, authored fix, pinned 0 file(s) intact |
| mesh-L112-bool-17 | rust | FAIL | no repair found | 29.3s, tree intact |
| mesh-L148-idswap-19 | rust | FAIL | no repair found | 44.1s, tree intact |
| mesh-L190-cmp-30 | rust | FAIL | no repair found | 29.2s, tree intact |
| mesh-L230-offbyone-06 | rust | FAIL | no repair found | 29.2s, tree intact |
| mesh-L257-arith-35 | rust | FAIL | no repair found | 28.2s, tree intact |
| mesh-L269-logic-25 | rust | FAIL | no repair found | 28.3s, tree intact |
| mesh-L269-neg-33 | rust | FAIL | no repair found | 28.2s, tree intact |
| mesh-L276-range-15 | rust | FAIL | no repair found | 28.2s, tree intact |
| mesh-L277-neg-13 | rust | FAIL | no repair found | 26.0s, tree intact |
| mesh-L295-cmp-36 | rust | FAIL | no repair found | 26.2s, tree intact |
| mesh-L296-argswap-08 | rust | FAIL | no repair found | 26.8s, tree intact |
| mesh-L316-neg-38 | rust | FAIL | no repair found | 28.9s, tree intact |
| mesh-L318-idswap-39 | rust | FAIL | no repair found | 42.5s, tree intact |
| mesh-L323-argswap-16 | rust | PASS | mutation repair verified green (swap-args:L323 swap args `residual` <-> `payload`) | 16.0s, authored fix, pinned 0 file(s) intact |
| mesh-L324-minmax-12 | rust | FAIL | no repair found | 26.0s, tree intact |
| mesh-L324-minmax-32 | rust | FAIL | no repair found | 27.3s, tree intact |
| mesh-L324-offbyone-14 | rust | FAIL | no repair found | 27.0s, tree intact |
| mesh-L325-idswap-05 | rust | FAIL | no repair found | 41.5s, tree intact |
| mesh-L350-range-34 | rust | FAIL | no repair found | 27.0s, tree intact |
| mesh-L397-cmp-03 | rust | FAIL | no repair found | 28.9s, tree intact |
| mesh-L397-idswap-31 | rust | FAIL | no repair found | 42.3s, tree intact |
| mesh-L408-idswap-37 | rust | FAIL | no repair found | 42.2s, tree intact |
| mesh-L412-argswap-01 | rust | PASS | mutation repair verified green (swap-args:L412 swap args `residual` <-> `payload`) | 16.4s, authored fix, pinned 0 file(s) intact |
| mesh-L414-neg-40 | rust | FAIL | no repair found | 29.4s, tree intact |
| mesh-L420-idswap-24 | rust | FAIL | no repair found | 41.8s, tree intact |
| mesh-L464-offbyone-28 | rust | FAIL | no repair found | 25.3s, tree intact |
| mesh-L474-idswap-10 | rust | FAIL | no repair found | 43.3s, tree intact |
| mesh-L476-argswap-22 | rust | PASS | mutation repair verified green (swap-args:L476 swap args `residual` <-> `payload`) | 16.8s, authored fix, pinned 0 file(s) intact |
| mesh-L484-bool-02 | rust | FAIL | no repair found | 28.3s, tree intact |
| mesh-L488-cmp-18 | rust | FAIL | no repair found | 29.1s, tree intact |
| mesh-L488-logic-11 | rust | FAIL | no repair found | 26.5s, tree intact |
| mesh-L489-flow-04 | rust | FAIL | no repair found | 28.3s, tree intact |
| mesh-L491-bool-09 | rust | FAIL | no repair found | 29.0s, tree intact |
| mesh-L502-range-29 | rust | FAIL | no repair found | 26.4s, tree intact |
| mesh-L503-neg-27 | rust | FAIL | no repair found | 27.7s, tree intact |

**pass: 4 · fail: 36 · skipped: 0 · not broken: 0 · rollback violations: 1**
