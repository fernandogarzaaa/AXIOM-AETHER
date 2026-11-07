# Repair corpus — 2026-11-07

Corpus: `/tmp/seeded` (`seed_faults.py --seed 2026 --balance`) · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`.

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| gumbel-L18-minmax-26 | rust | PASS | mutation repair verified green (api-swap:L18 min -> max) | 15.7s, authored fix, pinned 0 file(s) intact |
| gumbel-L21-arith-23 | rust | PASS | mutation repair verified green (arithmetic:L21 * -> /) | 11.7s, authored fix, pinned 0 file(s) intact |
| gumbel-L26-range-21 | rust | PASS | mutation repair verified green (off-by-one:L76 insert `- 1` after length) | 7.5s, different fix, pinned 0 file(s) intact |
| idc-L162-logic-20 | rust | PASS | mutation repair verified green (comparison:L162 <= -> >=) | 3.2s, different fix, pinned 0 file(s) intact |
| idc-L94-range-07 | rust | PASS | mutation repair verified green (off-by-one:L94 ..= -> ..) | 4.3s, authored fix, pinned 0 file(s) intact |
| mesh-L112-bool-17 | rust | PASS | mutation repair verified green (boolean:L353 negate `if` condition) | 20.4s, different fix, pinned 0 file(s) intact |
| mesh-L148-idswap-19 | rust | PASS | mutation repair verified green (ident-swap:L148 swap `config` <-> `affinity`) | 5.1s, authored fix, pinned 0 file(s) intact |
| mesh-L190-cmp-30 | rust | PASS | mutation repair verified green (comparison:L190 != -> ==) | 7.8s, authored fix, pinned 0 file(s) intact |
| mesh-L230-offbyone-06 | rust | PASS | mutation repair verified green (off-by-one:L230 drop `- 1`) | 7.8s, authored fix, pinned 0 file(s) intact |
| mesh-L257-arith-35 | rust | FAIL | no repair found | 35.6s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L269-logic-25 | rust | PASS | mutation repair verified green (comparison:L269 != -> ==) | 20.4s, different fix, pinned 0 file(s) intact |
| mesh-L269-neg-33 | rust | PASS | mutation repair verified green (comparison:L269 != -> ==) | 19.4s, different fix, pinned 0 file(s) intact |
| mesh-L276-range-15 | rust | PASS | mutation repair verified green (off-by-one:L276 insert `- 1` after length) | 9.0s, different fix, pinned 0 file(s) intact |
| mesh-L277-neg-13 | rust | PASS | mutation repair verified green (comparison:L277 != -> ==) | 20.2s, different fix, pinned 0 file(s) intact |
| mesh-L295-cmp-36 | rust | PASS | mutation repair verified green (comparison:L295 >= -> >) | 2.3s, authored fix, pinned 0 file(s) intact |
| mesh-L296-argswap-08 | rust | PASS | mutation repair verified green (swap-args:L296 swap args `self.config.tau_max` <-> `self.config.tau_min`) | 4.6s, authored fix, pinned 0 file(s) intact |
| mesh-L316-neg-38 | rust | PASS | mutation repair verified green (comparison:L316 != -> ==) | 4.2s, different fix, pinned 0 file(s) intact |
| mesh-L318-idswap-39 | rust | PASS | mutation repair verified green (ident-swap:L318 swap `len` <-> `payload`) | 4.3s, authored fix, pinned 0 file(s) intact |
| mesh-L323-argswap-16 | rust | PASS | mutation repair verified green (swap-args:L323 swap args `residual` <-> `payload`) | 4.8s, authored fix, pinned 0 file(s) intact |
| mesh-L324-minmax-12 | rust | PASS | mutation repair verified green (api-swap:L324 max -> min) | 22.3s, authored fix, pinned 0 file(s) intact |
| mesh-L324-minmax-32 | rust | PASS | mutation repair verified green (api-swap:L324 min -> max) | 11.9s, authored fix, pinned 0 file(s) intact |
| mesh-L324-offbyone-14 | rust | PASS | mutation repair verified green (off-by-one:L324 drop `- 1`) | 5.8s, authored fix, pinned 0 file(s) intact |
| mesh-L325-idswap-05 | rust | PASS | mutation repair verified green (ident-swap:L325 swap `residual` <-> `tau`) | 5.8s, authored fix, pinned 0 file(s) intact |
| mesh-L350-range-34 | rust | PASS | mutation repair verified green (off-by-one:L350 ..= -> ..) | 27.4s, authored fix, pinned 0 file(s) intact |
| mesh-L397-cmp-03 | rust | PASS | mutation repair verified green (off-by-one:L397 insert `- 1` after length) | 3.3s, different fix, pinned 0 file(s) intact |
| mesh-L397-idswap-31 | rust | PASS | mutation repair verified green (ident-swap:L397 swap `len` <-> `payload`) | 5.1s, authored fix, pinned 0 file(s) intact |
| mesh-L408-idswap-37 | rust | PASS | mutation repair verified green (ident-swap:L408 swap `is_empty` <-> `eligible_idx`) | 4.6s, authored fix, pinned 0 file(s) intact |
| mesh-L412-argswap-01 | rust | PASS | mutation repair verified green (swap-args:L412 swap args `residual` <-> `payload`) | 4.4s, authored fix, pinned 0 file(s) intact |
| mesh-L414-neg-40 | rust | PASS | mutation repair verified green (boolean:L414 negate `if` condition) | 14.2s, authored fix, pinned 0 file(s) intact |
| mesh-L420-idswap-24 | rust | PASS | mutation repair verified green (ident-swap:L420 swap `residual` <-> `tau`) | 5.6s, authored fix, pinned 0 file(s) intact |
| mesh-L464-offbyone-28 | rust | PASS | mutation repair verified green (off-by-one:L464 drop `- 1`) | 5.2s, authored fix, pinned 0 file(s) intact |
| mesh-L474-idswap-10 | rust | PASS | mutation repair verified green (ident-swap:L474 swap `len` <-> `payloads`) | 5.4s, authored fix, pinned 0 file(s) intact |
| mesh-L476-argswap-22 | rust | PASS | mutation repair verified green (swap-args:L476 swap args `residual` <-> `payload`) | 4.4s, authored fix, pinned 0 file(s) intact |
| mesh-L484-bool-02 | rust | PASS | mutation repair verified green (boolean:L484 true -> false) | 17.9s, authored fix, pinned 0 file(s) intact |
| mesh-L488-cmp-18 | rust | PASS | mutation repair verified green (off-by-one:L492 1 -> 2) | 11.9s, different fix, pinned 0 file(s) intact |
| mesh-L488-logic-11 | rust | PASS | mutation repair verified green (logic:L488 && -> ||) | 22.8s, authored fix, pinned 0 file(s) intact |
| mesh-L489-flow-04 | rust | PASS | mutation repair verified green (api-swap:L489 break -> continue) | 25.4s, authored fix, pinned 0 file(s) intact |
| mesh-L491-bool-09 | rust | PASS | mutation repair verified green (boolean:L491 false -> true) | 19.6s, authored fix, pinned 0 file(s) intact |
| mesh-L502-range-29 | rust | PASS | mutation repair verified green (off-by-one:L502 ..= -> ..) | 5.8s, authored fix, pinned 0 file(s) intact |
| mesh-L503-neg-27 | rust | PASS | mutation repair verified green (boolean:L503 negate closure body) | 20.6s, authored fix, pinned 0 file(s) intact |

**pass: 39 · fail: 1 · skipped: 0 · not broken: 0 · rollback violations: 0**

_1 failed fixture(s) left test-framework artifacts that the verifier wrote while a candidate was failing (e.g. `proptest-regressions/`). These are not solver edits and are not counted as rollback violations; they are what running the same tests by hand on a broken candidate would leave._
