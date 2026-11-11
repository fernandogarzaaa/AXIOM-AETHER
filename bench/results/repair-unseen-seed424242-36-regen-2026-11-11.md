# Repair corpus — 2026-11-11

Corpus: `bench/corpus/unseen-seed424242` · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`.

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| gumbel-L19-arith-16 | rust | PASS | mutation repair verified green (arithmetic:L19 + -> -) | 58.4s, authored fix, pinned 0 file(s) intact |
| gumbel-L78-idswap-15 | rust | PASS | mutation repair verified green (ident-swap:L78 swap `softmax` <-> `soft`) | 10.9s, authored fix, pinned 0 file(s) intact |
| idc-L111-deref-02 | rust | FAIL | no repair found | 50.9s, tree intact |
| idc-L124-cast-29 | rust | FAIL | no repair found | 51.2s, tree intact |
| idc-L127-deref-34 | rust | FAIL | no repair found | 48.6s, tree intact |
| idc-L162-cmp-09 | rust | FAIL | no repair found | 40.6s, tree intact |
| idc-L184-arith-05 | rust | PASS | mutation repair verified green (arithmetic:L184 + -> *) | 4.2s, authored fix, pinned 0 file(s) intact |
| idc-L78-cast-25 | rust | FAIL | no repair found | 48.0s, tree intact |
| idc-L80-shift-10 | rust | FAIL | no repair found | 43.9s, tree intact |
| idc-L81-arith-01 | rust | PASS | mutation repair verified green (arithmetic:L81 / -> *) | 30.2s, authored fix, pinned 0 file(s) intact |
| mesh-L113-const-10 | rust | PASS | mutation repair verified green (off-by-one:L113 2 -> 1) | 13.7s, authored fix, pinned 0 file(s) intact |
| mesh-L192-arith-02 | rust | PASS | mutation repair verified green (arithmetic:L192 + -> -) | 9.1s, authored fix, pinned 0 file(s) intact |
| mesh-L199-offbyone-07 | rust | PASS | mutation repair verified green (comparison:L199 != -> ==) | 5.3s, different fix, pinned 0 file(s) intact |
| mesh-L207-idswap-03 | rust | PASS | mutation repair verified green (ident-swap:L207 swap `last` <-> `nodes`) | 6.3s, authored fix, pinned 0 file(s) intact |
| mesh-L214-cmp-21 | rust | PASS | mutation repair verified green (comparison:L214 == -> !=) | 4.5s, authored fix, pinned 0 file(s) intact |
| mesh-L214-offbyone-14 | rust | PASS | mutation repair verified green (comparison:L214 != -> ==) | 3.1s, different fix, pinned 0 file(s) intact |
| mesh-L246-idswap-22 | rust | FAIL | no repair found | 49.6s, tree intact |
| mesh-L257-deref-06 | rust | FAIL | no repair found | 49.1s, tree intact |
| mesh-L266-cast-08 | rust | FAIL | no repair found | 49.0s, tree intact |
| mesh-L275-cast-33 | rust | FAIL | no repair found | 49.6s, tree intact |
| mesh-L276-idswap-12 | rust | PASS | mutation repair verified green (ident-swap:L276 swap `logits` <-> `i`) | 23.7s, authored fix, pinned 0 file(s) intact |
| mesh-L276-offbyone-18 | rust | PASS | mutation repair verified green (off-by-one:L276 drop `- 1`) | 21.3s, authored fix, pinned 0 file(s) intact |
| mesh-L283-arith-12 | rust | FAIL | no repair found | 51.6s, tree intact; verifier artifacts: proptest-regressions |
| mesh-L347-offbyone-04 | rust | PASS | mutation repair verified green (off-by-one:L347 drop `+ 1`) | 22.3s, authored fix, pinned 0 file(s) intact |
| mesh-L348-idswap-05 | rust | PASS | mutation repair verified green (ident-swap:L348 swap `fan_out` <-> `active`) | 7.9s, authored fix, pinned 0 file(s) intact |
| mesh-L351-deref-30 | rust | FAIL | no repair found | 98.7s, tree intact |
| mesh-L353-idswap-17 | rust | PASS | mutation repair verified green (ident-swap:L353 swap `w` <-> `weights`) | 9.3s, authored fix, pinned 0 file(s) intact |
| mesh-L394-neg-07 | rust | PASS | mutation repair verified green (boolean:L394 drop `!`) | 19.6s, different fix, pinned 0 file(s) intact |
| mesh-L395-ret-28 | rust | FAIL | no repair found | 48.4s, tree intact |
| mesh-L409-ret-15 | rust | FAIL | no repair found | 48.1s, tree intact |
| mesh-L461-ret-36 | rust | FAIL | no repair found | 48.8s, tree intact |
| mesh-L471-minmax-24 | rust | PASS | mutation repair verified green (api-swap:L471 min -> max) | 29.5s, authored fix, pinned 0 file(s) intact |
| mesh-L481-deref-11 | rust | FAIL | no repair found | 48.9s, tree intact |
| mesh-L492-const-14 | rust | PASS | mutation repair verified green (off-by-one:L492 0 -> 1) | 9.9s, authored fix, pinned 0 file(s) intact |
| mesh-L499-neg-13 | rust | PASS | mutation repair verified green (boolean:L499 negate closure body) | 24.4s, authored fix, pinned 0 file(s) intact |
| mesh-L516-ret-32 | rust | FAIL | no repair found | 46.9s, tree intact |

**pass: 19 · fail: 17 · skipped: 0 · not broken: 0 · rollback violations: 0**

_1 failed fixture(s) left test-framework artifacts that the verifier wrote while a candidate was failing (e.g. `proptest-regressions/`). These are not solver edits and are not counted as rollback violations; they are what running the same tests by hand on a broken candidate would leave._
