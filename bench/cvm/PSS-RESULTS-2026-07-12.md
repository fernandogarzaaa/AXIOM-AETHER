# PSS v2 Live-Eval Results — 2026-07-12

Live run of the dependent chain in `bench/cvm/pss-eval-tasks.tsv` through a
local Axiom proxy, flags OFF vs all PSS levers ON, via
`claude -p --continue --model claude-haiku-4-5` (one growing session).

> **Credit note:** non-interactive `claude -p` spend since 2026-06-15 draws
> from the separate Agent-SDK credit pool, not the main subscription window.
> These are real token/quota savings; the *spend* hit that pool.

| Metric | Flags off | PSS on | Gate |
|---|---|---|---|
| Correctness | 10/13 | 2/13 | on >= off - 1 |
| Quota units | 265815.125000 | 50790.700000 | on strictly lower |
| Quota savings | — | 80.9% | target >= 50% |
| Cost (USD) | 0.531627 | 0.101580 | (secondary) |
| Fault rate | — | 0.00% (0/16) | <= 5% |

PSS-on lever activity: L-B local-answered=0 · R1 routed=0 (saved 0.000000 units, 0 fallbacks).

**Result: FAIL.** Defaults stay OFF. See the failed gate(s) above; per plan
step P5.3-FAIL, return to brainstorming.
