# Repair corpus: QuixBugs (Python) — 2026-11-06/07

Corpus: `bench/repair-quixbugs` (`scripts/import_quixbugs.py`, QuixBugs @ 4257f44; run in four chunks of ten, on the build with process-group kill; 0 leftover verifier processes after each chunk) · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`.

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| quixbugs-bitcount | python | FAIL | no repair found | 309.4s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-breadth_first_search | python | FAIL | no repair found | 18.3s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-bucketsort | python | FAIL | no repair found | 29.9s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-depth_first_search | python | FAIL | no repair found | 14.0s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-detect_cycle | python | FAIL | no repair found | 19.7s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-find_first_in_sorted | python | PASS | mutation repair verified green (comparison:L5 <= -> <) | 35.4s, —, pinned 5 file(s) intact |
| quixbugs-find_in_sorted | python | FAIL | no repair found | 28.7s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-flatten | python | FAIL | no repair found | 29.3s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-gcd | python | PASS | mutation repair verified green (swap-args:L5 swap args `a % b` <-> `b`) | 4.5s, —, pinned 5 file(s) intact |
| quixbugs-get_factors | python | FAIL | no repair found | 30.2s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-hanoi | python | FAIL | no repair found | 30.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-is_valid_parenthesization | python | FAIL | no repair found | 23.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-kheapsort | python | FAIL | no repair found | 28.3s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-knapsack | python | PASS | mutation repair verified green (comparison:L12 < -> <=) | 19.5s, —, pinned 5 file(s) intact |
| quixbugs-kth | python | FAIL | no repair found | 30.0s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-lcs_length | python | FAIL | no repair found | 28.5s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-levenshtein | python | PASS | mutation repair verified green (off-by-one:L6 1 -> 0) | 11.1s, —, pinned 5 file(s) intact |
| quixbugs-lis | python | FAIL | no repair found | 28.2s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-longest_common_subsequence | python | FAIL | no repair found | 36.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-max_sublist_sum | python | FAIL | no repair found | 27.3s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-mergesort | python | FAIL | no repair found | 53.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-minimum_spanning_tree | python | FAIL | no repair found | 20.8s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-next_palindrome | python | PASS | mutation repair verified green (off-by-one:L15 insert `- 1` after length) | 8.2s, —, pinned 5 file(s) intact |
| quixbugs-next_permutation | python | PASS | mutation repair verified green (comparison:L6 < -> >) | 15.5s, —, pinned 5 file(s) intact |
| quixbugs-pascal | python | FAIL | no repair found | 29.9s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-possible_change | python | FAIL | no repair found | 37.8s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-powerset | python | FAIL | no repair found | 24.2s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-quicksort | python | PASS | mutation repair verified green (comparison:L6 < -> <=) | 3.6s, —, pinned 5 file(s) intact |
| quixbugs-reverse_linked_list | python | FAIL | no repair found | 25.5s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-rpn_eval | python | PASS | mutation repair verified green (swap-args:L3 swap args `a` <-> `b`) | 6.6s, —, pinned 5 file(s) intact |
| quixbugs-shortest_path_length | python | FAIL | no repair found | 20.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-shortest_path_lengths | python | FAIL | no repair found | 22.7s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-shortest_paths | python | FAIL | no repair found | 20.4s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-shunting_yard | python | FAIL | no repair found | 28.0s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-sieve | python | PASS | mutation repair verified green (api-swap:L4 any -> all) | 8.3s, —, pinned 5 file(s) intact |
| quixbugs-sqrt | python | FAIL | no repair found | 321.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-subsequences | python | FAIL | no repair found | 30.2s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-to_base | python | FAIL | no repair found | 34.0s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-topological_ordering | python | FAIL | no repair found | 21.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-wrap | python | FAIL | no repair found | 29.8s, tree intact; verifier artifacts: `__pycache__/` |

**pass: 9 · fail: 31 · skipped: 0 · not broken: 0 · rollback violations: 0**
