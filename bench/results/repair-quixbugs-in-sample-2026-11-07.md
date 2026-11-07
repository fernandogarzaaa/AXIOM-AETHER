# Repair corpus: QuixBugs (Python), in-sample — 2026-11-07

Corpus: `bench/repair-quixbugs` (`scripts/import_quixbugs.py`, QuixBugs @ 4257f44; four chunks of ten; 0 leftover verifier processes) · runner: `scripts/bench_repair.sh` · deterministic, no LLM · candidate cap: `AXIOM_REPAIR_MAX_CANDIDATES=64`. **In-sample:** the `arg-shift` class and bitwise swaps were chosen from this corpus's failures; the held-out result is [`repair-quixbugs-2026-11-07.md`](repair-quixbugs-2026-11-07.md).

| Fixture | Lang | Result | How | Notes |
|---|---|---|---|---|
| quixbugs-bitcount | python | PASS | mutation repair verified green (arithmetic:L5 ^= -> &=) | 136.4s, —, pinned 5 file(s) intact |
| quixbugs-breadth_first_search | python | FAIL | no repair found | 20.5s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-bucketsort | python | FAIL | no repair found | 31.4s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-depth_first_search | python | FAIL | no repair found | 16.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-detect_cycle | python | FAIL | no repair found | 19.9s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-find_first_in_sorted | python | PASS | mutation repair verified green (comparison:L5 <= -> <) | 35.5s, —, pinned 5 file(s) intact |
| quixbugs-find_in_sorted | python | PASS | mutation repair verified green (arg-shift:L9 `mid` -> `mid + 1`) | 3.8s, —, pinned 5 file(s) intact |
| quixbugs-flatten | python | FAIL | no repair found | 32.5s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-gcd | python | PASS | mutation repair verified green (swap-args:L5 swap args `a % b` <-> `b`) | 4.6s, —, pinned 5 file(s) intact |
| quixbugs-get_factors | python | FAIL | no repair found | 32.1s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-hanoi | python | FAIL | no repair found | 33.1s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-is_valid_parenthesization | python | FAIL | no repair found | 25.1s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-kheapsort | python | FAIL | no repair found | 30.4s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-knapsack | python | PASS | mutation repair verified green (comparison:L12 < -> <=) | 19.1s, —, pinned 5 file(s) intact |
| quixbugs-kth | python | FAIL | no repair found | 31.7s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-lcs_length | python | PASS | mutation repair verified green (arg-shift:L9 `j` -> `j + 1`) | 16.6s, —, pinned 5 file(s) intact |
| quixbugs-levenshtein | python | PASS | mutation repair verified green (off-by-one:L6 1 -> 0) | 11.0s, —, pinned 5 file(s) intact |
| quixbugs-lis | python | FAIL | no repair found | 30.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-longest_common_subsequence | python | FAIL | no repair found | 36.7s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-max_sublist_sum | python | FAIL | no repair found | 29.3s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-mergesort | python | FAIL | no repair found | 55.1s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-minimum_spanning_tree | python | FAIL | no repair found | 20.4s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-next_palindrome | python | PASS | mutation repair verified green (off-by-one:L15 insert `- 1` after length) | 8.0s, —, pinned 5 file(s) intact |
| quixbugs-next_permutation | python | PASS | mutation repair verified green (comparison:L6 < -> >) | 14.6s, —, pinned 5 file(s) intact |
| quixbugs-pascal | python | FAIL | no repair found | 31.7s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-possible_change | python | FAIL | no repair found | 39.5s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-powerset | python | FAIL | no repair found | 28.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-quicksort | python | PASS | mutation repair verified green (comparison:L6 < -> <=) | 3.9s, —, pinned 5 file(s) intact |
| quixbugs-reverse_linked_list | python | FAIL | no repair found | 25.3s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-rpn_eval | python | PASS | mutation repair verified green (swap-args:L3 swap args `a` <-> `b`) | 6.6s, —, pinned 5 file(s) intact |
| quixbugs-shortest_path_length | python | FAIL | no repair found | 21.8s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-shortest_path_lengths | python | PASS | mutation repair verified green (arg-shift:L13 swap subscript `j` <-> `k`) | 21.3s, —, pinned 4 file(s) intact |
| quixbugs-shortest_paths | python | FAIL | no repair found | 20.7s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-shunting_yard | python | FAIL | no repair found | 29.6s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-sieve | python | PASS | mutation repair verified green (api-swap:L4 any -> all) | 8.3s, —, pinned 5 file(s) intact |
| quixbugs-sqrt | python | FAIL | no repair found | 329.8s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-subsequences | python | FAIL | no repair found | 31.5s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-to_base | python | PASS | mutation repair verified green (arg-shift:L9 swap operands `result` + `alphabet[i]`) | 19.1s, —, pinned 5 file(s) intact |
| quixbugs-topological_ordering | python | FAIL | no repair found | 21.7s, tree intact; verifier artifacts: `__pycache__/` |
| quixbugs-wrap | python | FAIL | no repair found | 33.4s, tree intact; verifier artifacts: `__pycache__/` |

**pass: 14 · fail: 26 · skipped: 0 · not broken: 0 · rollback violations: 0**
