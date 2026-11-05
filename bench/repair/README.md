# `bench/repair/` — on-disk autonomous-repair corpus

The deterministic fixtures that gate CI live in Rust
(`axiom_engine_rs/src/agentic_eval.rs`, `axiom eval-agentic`). This directory is
the **on-disk format** from [`docs/BENCH-REPAIR.md`](../../docs/BENCH-REPAIR.md)
for fixtures that outgrow that file — a corpus an external bug set (SWE-bench
instances, Defects4J, QuixBugs, your own bugs) can be dropped into without
touching the engine, and a runner that scores it with a baseline column.

```
bench/repair/<name>/
  meta.json      # { "lang", "verify": ["cmd", "args"...], "kind": "single|agentic", "source": "..." }
  broken/        # the project tree that fails `verify`
  expected.diff  # optional: a known-good patch, so the report can say "found the authored fix" vs "found a fix"
```

Run it:

```bash
scripts/bench_repair.sh            # table to stdout + bench/results/repair-<date>.md
scripts/bench_repair.sh --check    # CI: exit non-zero if any fixture is not actually broken,
                                   #     is not repaired, or a failed attempt left the tree modified
```

For each fixture the runner (1) confirms **No-op** — `verify` fails on the
untouched tree, so a "pass" means something; (2) runs `axiom solve` in a scratch
copy with no file named, so the loop must localize and repair on its own;
(3) re-runs `verify`; (4) on failure checks **rollback correctness** — the tree
is byte-identical to `broken/`. The three seed fixtures here are deliberately
small and distinct from the built-in suite; they exist to prove the runner, not
to inflate the number.
