# Repair corpus — 2026-11-09

Corpus: `bench/corpus/unseen-codeflaws` · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`.

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| 131-A-bug-18069181-18073244 | c | PASS | mutation repair verified green (off-by-one:L21 1 -> 0) | 3.8s, —, pinned 129 file(s) intact |
| 133-A-bug-18048780-18048793 | c | PASS | mutation repair verified green (off-by-one:L4 4 -> 3) | 1.6s, —, pinned 181 file(s) intact |
| 136-A-bug-17499279-17499383 | c | PASS | mutation repair verified green (off-by-one:L4 2 -> 3) | 5.1s, —, pinned 233 file(s) intact |
| 231-A-bug-18055428-18055539 | c | PASS | mutation repair verified green (ident-swap:L9 swap `p` <-> `v`) | 1.1s, —, pinned 33 file(s) intact |
| 271-A-bug-18358749-18358771 | c | FAIL | no repair found | 66.5s, tree intact |
| 4-A-bug-18258406-18258457 | c | FAIL | no repair found | 1.8s, tree intact |
| 467-A-bug-18348808-18348843 | c | PASS | mutation repair verified green (comparison:L11 >= -> <=) | 1.3s, —, pinned 61 file(s) intact |
| 58-A-bug-17762757-17762804 | c | PASS | mutation repair verified green (off-by-one:L4 0 -> 1) | 3.1s, —, pinned 97 file(s) intact |
| 58-A-bug-18163826-18163845 | c | FAIL | no repair found | 3.2s, tree intact |
| 82-A-bug-18221798-18221807 | c | FAIL | no repair found | 3.2s, tree intact |
| 96-A-bug-18173471-18173484 | c | FAIL | no repair found | 3.7s, tree intact |
| 96-A-bug-18284844-18284870 | c | PASS | mutation repair verified green (off-by-one:L35 drop `+ 1`) | 2.8s, —, pinned 99 file(s) intact |

**pass: 7 · fail: 5 · skipped: 0 · not broken: 0 · rollback violations: 0**
