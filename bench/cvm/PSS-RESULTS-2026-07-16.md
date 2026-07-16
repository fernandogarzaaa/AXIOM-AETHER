# PSS v2 Live-Eval Results — 2026-07-16

Live run of the dependent chain in `bench/cvm/pss-eval-tasks.tsv` through a
local Axiom proxy, flags OFF vs all PSS levers ON, via
`claude -p --continue --model claude-haiku-4-5` (one growing session).

> **Credit note:** non-interactive `claude -p` spend since 2026-06-15 draws
> from the separate Agent-SDK credit pool, not the main subscription window.
> These are real token/quota savings; the *spend* hit that pool.

| Metric | Flags off | PSS on | Gate |
|---|---|---|---|
| Correctness | 12/13 | 12/13 | on >= off - 1 |
| Quota units | 201780.750000 | 179668.300000 | on strictly lower |
| Quota savings | — | 11.0% | target >= 50% |
| Cost (USD) | 0.403563 | 0.359335 | (secondary) |
| Fault rate | — | 0.00% (0/16) | <= 5% |

PSS-on lever activity: L-B local-answered=0 · R1 routed=0 (saved 0.000000 units, 0 fallbacks).

**Result: PASS on gates, BELOW 50% target (11.0%).** Levers are
correctness-preserving and cheaper, but under goal. Human decides whether to flip
defaults now or return to brainstorming for more headroom.
