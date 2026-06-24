# axiom_engine

The Rust inference engine behind **[AXIOM-AETHER](https://github.com/fernandogarzaaa/AXIOM-AETHER)** —
a local-first runtime for online Test-Time Training (TTT), context compression,
grounding checks, self-healing execution, and the in-tree ChimeraLang DSL.

Installs the `axiom` command.

## Install

```bash
cargo install axiom_engine          # builds from source (needs a Rust toolchain)
pip install axiom-aether            # prebuilt binary, no toolchain (recommended)
curl -fsSL https://raw.githubusercontent.com/fernandogarzaaa/AXIOM-AETHER/main/scripts/install.sh | bash
```

## Quick start

```bash
axiom init                  # scaffold ~/.axiom (offline checkpoint bootstrap)
axiom --mode doctor         # hardware-aware device pick
axiom run -- <cmd>          # self-healing supervised run
axiom solve -- <verify>     # drive a failing verify command to green
axiom chimera run prog.chimera   # run a ChimeraLang program
```

See the [main repository](https://github.com/fernandogarzaaa/AXIOM-AETHER) for
full documentation, the capability table, and the architecture overview.

## License

MIT
