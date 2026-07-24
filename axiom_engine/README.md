# `axiom_engine/` — Python reference implementation

**This is a reference implementation, not the shipped product.**

The engine that ships and is documented in the top-level [`README`](../README.md)
— the one behind `pip install axiom-aether`, the Docker image, the release
binaries, the MCP server, and the whole CVM/PSS cost stack — is the **Rust**
crate in [`axiom_engine_rs/`](../axiom_engine_rs/). That is the authoritative
implementation.

This `axiom_engine/` package is a smaller, pure-Python (PyTorch) implementation
of the core Test-Time Training ideas — the TTT kernel, an inference pipeline, an
OpenAI-compatible FastAPI server, and a response cache. It exists for reference,
experimentation, and readability, and is published to PyPI as **`axiom-engine`**
(distinct from the Rust binary wheel **`axiom-aether`**).

## Which one do I want?

| You want… | Use |
|---|---|
| The real runtime, all documented features, best performance | `axiom_engine_rs/` (Rust) — `pip install axiom-aether` |
| A readable PyTorch sketch of the TTT core, or to hack on the ideas in Python | this package — `pip install axiom-engine` |

## Status

- Covered by CI: the test suite (`tests/test_*.py`) runs on every PR via the
  `Python reference engine` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
- It does **not** track the Rust engine feature-for-feature. New capabilities
  (compression proxy, self-heal, swarm/fleet, grounding gate, mesh routing, …)
  land in Rust; this package is not expected to mirror them.

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[server,dev]"
pytest tests/ -q
axiom-server            # OpenAI-compatible server (see pyproject [project.scripts])
```
