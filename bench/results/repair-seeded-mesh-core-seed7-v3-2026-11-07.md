# Repair corpus — 2026-11-07

Corpus: `/tmp/seeded7` (`seed_faults.py --seed 7 --balance`) · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`.

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| gumbel-L26-range-16 | rust | PASS | mutation repair verified green (off-by-one:L76 insert `- 1` after length) | 7.1s, different fix, pinned 0 file(s) intact |
| gumbel-L77-arith-10 | rust | PASS | mutation repair verified green (arithmetic:L77 * -> /) | 45.5s, authored fix, pinned 0 file(s) intact |
| gumbel-L80-idswap-27 | rust | PASS | mutation repair verified green (ident-swap:L80 swap `soft` <-> `winner`) | 4.3s, authored fix, pinned 0 file(s) intact |
| idc-L126-idswap-14 | rust | PASS | mutation repair verified green (ident-swap:L126 swap `failures` <-> `f`) | 4.0s, authored fix, pinned 0 file(s) intact |
| idc-L162-logic-21 | rust | PASS | mutation repair verified green (comparison:L162 <= -> >=) | 3.1s, different fix, pinned 0 file(s) intact |
| idc-L70-idswap-40 | rust | PASS | mutation repair verified green (ident-swap:L70 swap `wrapping_mul` <-> `h`) | 4.6s, authored fix, pinned 0 file(s) intact |
| idc-L78-offbyone-08 | rust | PASS | mutation repair verified green (off-by-one:L78 drop `- 1`) | 4.5s, authored fix, pinned 0 file(s) intact |
| idc-L80-const-13 | rust | PASS | mutation repair verified green (off-by-one:L80 64 -> 63) | 5.0s, different fix, pinned 0 file(s) intact |
| idc-L80-idswap-35 | rust | PASS | mutation repair verified green (ident-swap:L80 swap `h` <-> `sign`) | 8.4s, authored fix, pinned 0 file(s) intact |
| idc-L94-range-37 | rust | PASS | mutation repair verified green (off-by-one:L94 ..= -> ..) | 4.1s, authored fix, pinned 0 file(s) intact |
| mesh-L112-bool-11 | rust | PASS | mutation repair verified green (boolean:L353 negate `if` condition) | 20.8s, different fix, pinned 0 file(s) intact |
| mesh-L230-idswap-04 | rust | PASS | mutation repair verified green (ident-swap:L230 swap `nodes` <-> `n`) | 14.4s, authored fix, pinned 0 file(s) intact |
| mesh-L248-const-39 | rust | PASS | mutation repair verified green (off-by-one:L248 1 -> 0) | 14.5s, authored fix, pinned 0 file(s) intact |
| mesh-L269-logic-05 | rust | PASS | mutation repair verified green (comparison:L269 != -> ==) | 20.0s, different fix, pinned 0 file(s) intact |
| mesh-L276-range-24 | rust | PASS | mutation repair verified green (off-by-one:L276 insert `- 1` after length) | 9.2s, different fix, pinned 0 file(s) intact |
| mesh-L277-cmp-12 | rust | PASS | mutation repair verified green (comparison:L277 == -> !=) | 20.7s, authored fix, pinned 0 file(s) intact |
| mesh-L278-arith-18 | rust | FAIL | no repair found | 32.2s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L281-minmax-36 | rust | FAIL | no repair found | 33.6s, tree intact; verifier artifacts: proptest-regressions/mesh.txt |
| mesh-L296-argswap-30 | rust | PASS | mutation repair verified green (swap-args:L296 swap args `self.config.tau_max` <-> `self.config.tau_min`) | 4.7s, authored fix, pinned 0 file(s) intact |
| mesh-L323-argswap-17 | rust | PASS | mutation repair verified green (swap-args:L323 swap args `residual` <-> `payload`) | 4.5s, authored fix, pinned 0 file(s) intact |
| mesh-L324-minmax-06 | rust | PASS | mutation repair verified green (api-swap:L324 max -> min) | 22.2s, authored fix, pinned 0 file(s) intact |
| mesh-L324-minmax-28 | rust | PASS | mutation repair verified green (api-swap:L324 min -> max) | 11.8s, authored fix, pinned 0 file(s) intact |
| mesh-L350-range-34 | rust | PASS | mutation repair verified green (off-by-one:L350 ..= -> ..) | 23.5s, authored fix, pinned 0 file(s) intact |
| mesh-L356-idswap-32 | rust | PASS | mutation repair verified green (ident-swap:L356 swap `adhesion` <-> `sample`) | 5.2s, authored fix, pinned 0 file(s) intact |
| mesh-L397-cmp-31 | rust | PASS | mutation repair verified green (off-by-one:L397 insert `- 1` after length) | 3.2s, different fix, pinned 0 file(s) intact |
| mesh-L397-offbyone-23 | rust | PASS | mutation repair verified green (off-by-one:L397 drop `- 1`) | 3.1s, authored fix, pinned 0 file(s) intact |
| mesh-L408-neg-22 | rust | PASS | mutation repair verified green (boolean:L408 drop `!`) | 13.7s, different fix, pinned 0 file(s) intact |
| mesh-L412-argswap-01 | rust | PASS | mutation repair verified green (swap-args:L412 swap args `residual` <-> `payload`) | 4.4s, authored fix, pinned 0 file(s) intact |
| mesh-L414-neg-07 | rust | PASS | mutation repair verified green (boolean:L414 drop `!`) | 14.1s, different fix, pinned 0 file(s) intact |
| mesh-L414-neg-29 | rust | PASS | mutation repair verified green (boolean:L414 negate `if` condition) | 14.0s, authored fix, pinned 0 file(s) intact |
| mesh-L471-const-26 | rust | PASS | mutation repair verified green (off-by-one:L471 2 -> 1) | 8.3s, authored fix, pinned 0 file(s) intact |
| mesh-L476-argswap-25 | rust | PASS | mutation repair verified green (swap-args:L476 swap args `residual` <-> `payload`) | 4.4s, authored fix, pinned 0 file(s) intact |
| mesh-L476-idswap-20 | rust | PASS | mutation repair verified green (ident-swap:L476 swap `payload` <-> `field`) | 7.4s, authored fix, pinned 0 file(s) intact |
| mesh-L484-bool-02 | rust | PASS | mutation repair verified green (boolean:L484 true -> false) | 18.7s, authored fix, pinned 0 file(s) intact |
| mesh-L488-cmp-38 | rust | PASS | mutation repair verified green (off-by-one:L492 1 -> 2) | 12.5s, different fix, pinned 0 file(s) intact |
| mesh-L488-logic-15 | rust | PASS | mutation repair verified green (logic:L488 && -> ||) | 22.0s, authored fix, pinned 0 file(s) intact |
| mesh-L489-flow-03 | rust | PASS | mutation repair verified green (api-swap:L489 break -> continue) | 26.7s, authored fix, pinned 0 file(s) intact |
| mesh-L491-bool-19 | rust | PASS | mutation repair verified green (boolean:L491 false -> true) | 19.1s, authored fix, pinned 0 file(s) intact |
| mesh-L502-range-09 | rust | PASS | mutation repair verified green (off-by-one:L502 ..= -> ..) | 6.0s, authored fix, pinned 0 file(s) intact |
| mesh-L503-neg-33 | rust | PASS | mutation repair verified green (boolean:L503 negate closure body) | 21.7s, authored fix, pinned 0 file(s) intact |

**pass: 38 · fail: 2 · skipped: 0 · not broken: 0 · rollback violations: 0**

_2 failed fixture(s) left test-framework artifacts that the verifier wrote while a candidate was failing (e.g. `proptest-regressions/`). These are not solver edits and are not counted as rollback violations; they are what running the same tests by hand on a broken candidate would leave._
