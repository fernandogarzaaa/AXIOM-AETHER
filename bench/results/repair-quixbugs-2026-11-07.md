# Repair corpus: QuixBugs (Python) — 2026-11-06/07

Corpus: `bench/repair-quixbugs` (`scripts/import_quixbugs.py`, QuixBugs @ 4257f44; run in four chunks of ten) · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`.

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| quixbugs-bitcount | python | FAIL | no repair found | 324.7s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-breadth_first_search | python | FAIL | no repair found | 81.9s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-bucketsort | python | FAIL | no repair found | 122.5s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-depth_first_search | python | FAIL | no repair found | 59.9s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-detect_cycle | python | FAIL | no repair found | 81.1s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-find_first_in_sorted | python | PASS | mutation repair verified green (comparison:L5 <= -> <) | 36.9s, —, pinned 5 file(s) intact |
| quixbugs-find_in_sorted | python | FAIL | no repair found | 159.9s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-flatten | python | FAIL | no repair found | 161.8s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-gcd | python | PASS | mutation repair verified green (swap-args:L5 swap args `a % b` <-> `b`) | 24.3s, —, pinned 5 file(s) intact |
| quixbugs-get_factors | python | FAIL | no repair found | 175.5s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-hanoi | python | FAIL | no repair found | 164.5s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-is_valid_parenthesization | python | FAIL | no repair found | 122.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-kheapsort | python | FAIL | no repair found | 143.4s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-knapsack | python | PASS | mutation repair verified green (comparison:L12 < -> <=) | 98.0s, —, pinned 5 file(s) intact |
| quixbugs-kth | python | FAIL | no repair found | 159.9s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-lcs_length | python | FAIL | no repair found | 157.4s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-levenshtein | python | PASS | mutation repair verified green (arithmetic:L6 + -> *) | 177.2s, —, pinned 5 file(s) intact |
| quixbugs-lis | python | FAIL | no repair found | 150.2s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-longest_common_subsequence | python | FAIL | no repair found | 186.1s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-max_sublist_sum | python | FAIL | no repair found | 142.7s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-mergesort | python | FAIL | no repair found | 276.2s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-minimum_spanning_tree | python | FAIL | no repair found | 109.3s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-next_palindrome | python | PASS | mutation repair verified green (off-by-one:L15 insert `- 1` after length) | 40.3s, —, pinned 5 file(s) intact |
| quixbugs-next_permutation | python | PASS | mutation repair verified green (comparison:L6 < -> >) | 73.4s, —, pinned 5 file(s) intact |
| quixbugs-pascal | python | FAIL | no repair found | 154.4s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-possible_change | python | FAIL | no repair found | 181.2s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-powerset | python | FAIL | no repair found | 139.3s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-quicksort | python | PASS | mutation repair verified green (comparison:L6 < -> <=) | 20.6s, —, pinned 5 file(s) intact |
| quixbugs-reverse_linked_list | python | FAIL | no repair found | 116.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-rpn_eval | python | PASS | mutation repair verified green (swap-args:L3 swap args `a` <-> `b`) | 39.0s, —, pinned 5 file(s) intact |
| quixbugs-shortest_path_length | python | FAIL | no repair found | 120.4s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-shortest_path_lengths | python | FAIL | no repair found | 118.1s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-shortest_paths | python | FAIL | no repair found | 108.9s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-shunting_yard | python | FAIL | no repair found | 148.5s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-sieve | python | PASS | mutation repair verified green (api-swap:L4 any -> all) | 43.9s, —, pinned 5 file(s) intact |
| quixbugs-sqrt | python | FAIL | no repair found | 399.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-subsequences | python | FAIL | no repair found | 261.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-to_base | python | FAIL | no repair found | 242.7s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-topological_ordering | python | FAIL | no repair found | 201.8s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-wrap | python | FAIL | no repair found | 271.0s, tree intact; verifier artifacts: `__pycache__/` |

**pass: 9 · fail: 31 · skipped: 0 · not broken: 0 · rollback violations: 0**
