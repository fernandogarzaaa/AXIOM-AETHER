# Repair corpus — 2026-11-13

Corpus: `/tmp/claude-0/-home-user-AXIOM-AETHER/3625c33e-e5a7-566a-8b5d-bee72903a15e/scratchpad/release/seeded7` · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`.

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| gumbel-L18-minmax-39 | rust | PASS | mutation repair verified green (api-swap:L18 min -> max) | 12.3s, authored fix, pinned 0 file(s) intact |
| gumbel-L26-range-30 | rust | PASS | mutation repair verified green (off-by-one:L76 insert `- 1` after length) | 5.2s, different fix, pinned 0 file(s) intact |
| gumbel-L77-arith-01 | rust | PASS | mutation repair verified green (arithmetic:L77 * -> /) | 36.4s, authored fix, pinned 0 file(s) intact |
| gumbel-L78-deref-13 | rust | FAIL | no repair found | 31.9s, tree intact |
| gumbel-L88-offbyone-29 | rust | PASS | mutation repair verified green (off-by-one:L88 drop `- 1`) | 2.9s, authored fix, pinned 0 file(s) intact |
| idc-L123-cast-34 | rust | FAIL | no repair found | 30.7s, tree intact |
| idc-L162-logic-15 | rust | FAIL | no repair found | 22.9s, tree intact |
| idc-L69-cast-03 | rust | FAIL | no repair found | 30.5s, tree intact |
| idc-L78-cast-12 | rust | FAIL | no repair found | 30.7s, tree intact |
| idc-L80-shift-09 | rust | FAIL | no repair found | 25.9s, tree intact |
| idc-L96-deref-25 | rust | FAIL | no repair found | 29.5s, tree intact |
| mesh-L112-bool-11 | rust | PASS | mutation repair verified green (boolean:L112 false -> true) | 6.3s, authored fix, pinned 0 file(s) intact |
| mesh-L191-arith-33 | rust | PASS | mutation repair verified green (arithmetic:L191 -= -> +=) | 2.4s, authored fix, pinned 0 file(s) intact |
| mesh-L192-cast-23 | rust | FAIL | no repair found | 30.2s, tree intact |
| mesh-L214-cmp-35 | rust | PASS | mutation repair verified green (comparison:L214 == -> !=) | 2.0s, authored fix, pinned 0 file(s) intact |
| mesh-L233-idswap-14 | rust | FAIL | no repair found | 30.9s, tree intact |
| mesh-L269-logic-38 | rust | PASS | mutation repair verified green (logic:L269 && -> \|\|) | 17.3s, authored fix, pinned 0 file(s) intact |
| mesh-L275-minmax-16 | rust | PASS | mutation repair verified green (off-by-one:L275 1 -> 2) | 1.8s, different fix, pinned 0 file(s) intact |
| mesh-L283-idswap-37 | rust | PASS | mutation repair verified green (ident-swap:L283 swap `total_visits` <-> `config`) | 6.0s, authored fix, pinned 0 file(s) intact |
| mesh-L314-ret-08 | rust | FAIL | no repair found | 30.9s, tree intact |
| mesh-L316-neg-28 | rust | PASS | mutation repair verified green (comparison:L316 != -> ==) | 2.7s, different fix, pinned 0 file(s) intact |
| mesh-L316-offbyone-18 | rust | PASS | mutation repair verified green (off-by-one:L316 drop `+ 1`) | 4.1s, authored fix, pinned 0 file(s) intact |
| mesh-L323-argswap-32 | rust | PASS | mutation repair verified green (swap-args:L323 swap args `residual` <-> `payload`) | 2.9s, authored fix, pinned 0 file(s) intact |
| mesh-L324-minmax-06 | rust | PASS | mutation repair verified green (api-swap:L324 max -> min) | 18.8s, authored fix, pinned 0 file(s) intact |
| mesh-L324-offbyone-40 | rust | PASS | mutation repair verified green (off-by-one:L324 drop `+ 1`) | 4.3s, authored fix, pinned 0 file(s) intact |
| mesh-L350-idswap-26 | rust | PASS | mutation repair verified green (ident-swap:L350 swap `fan_out` <-> `_`) | 3.2s, authored fix, pinned 0 file(s) intact |
| mesh-L350-range-07 | rust | PASS | mutation repair verified green (off-by-one:L350 ..= -> ..) | 23.2s, authored fix, pinned 0 file(s) intact |
| mesh-L395-ret-20 | rust | FAIL | no repair found | 31.9s, tree intact |
| mesh-L406-deref-36 | rust | FAIL | no repair found | 29.8s, tree intact |
| mesh-L412-argswap-10 | rust | PASS | mutation repair verified green (swap-args:L412 swap args `residual` <-> `payload`) | 2.8s, authored fix, pinned 0 file(s) intact |
| mesh-L461-ret-31 | rust | FAIL | no repair found | 29.2s, tree intact |
| mesh-L464-neg-17 | rust | PASS | mutation repair verified green (comparison:L464 != -> ==) | 18.2s, different fix, pinned 0 file(s) intact |
| mesh-L476-argswap-21 | rust | PASS | mutation repair verified green (swap-args:L476 swap args `residual` <-> `payload`) | 2.9s, authored fix, pinned 0 file(s) intact |
| mesh-L484-bool-02 | rust | PASS | mutation repair verified green (boolean:L484 true -> false) | 15.5s, authored fix, pinned 0 file(s) intact |
| mesh-L488-cmp-24 | rust | PASS | mutation repair verified green (comparison:L488 > -> >=) | 12.7s, authored fix, pinned 0 file(s) intact |
| mesh-L488-logic-27 | rust | PASS | mutation repair verified green (logic:L488 && -> \|\|) | 20.1s, authored fix, pinned 0 file(s) intact |
| mesh-L489-flow-04 | rust | PASS | mutation repair verified green (api-swap:L489 break -> continue) | 23.5s, authored fix, pinned 0 file(s) intact |
| mesh-L491-bool-22 | rust | PASS | mutation repair verified green (boolean:L491 false -> true) | 16.6s, authored fix, pinned 0 file(s) intact |
| mesh-L502-range-19 | rust | PASS | mutation repair verified green (off-by-one:L502 ..= -> ..) | 4.5s, authored fix, pinned 0 file(s) intact |
| residual-L44-idswap-05 | rust | FAIL | no repair found | 25.0s, tree intact |

**pass: 26 · fail: 14 · skipped: 0 · not broken: 0 · rollback violations: 0**
