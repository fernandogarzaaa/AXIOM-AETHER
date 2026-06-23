# ChimeraLang ↔ AXIOM-AETHER — Audit & Integration

Integrates the ChimeraLang AI-cognition language
(https://github.com/fernandogarzaaa/ChimeraLang) into AXIOM-AETHER as a
**first-class, in-tree Rust implementation** (`axiom_engine_rs/src/chimera.rs`).

## Audit summary

**ChimeraLang** (`chimeralang` 0.2.0, Python ≥3.11): a language whose primitives
model reasoning under uncertainty. Core is **zero-dependency** (stdlib only);
optional extras `ml`(torch)/`vector`(numpy)/`sign`(cryptography). Pipeline
`Lexer → Parser → TypeChecker → {VM | CIR}`; CLI `chimera`
(`run/check/prove/verify/compile/rag/repl`) over a clean programmatic API. Key
constructs: belief path (`belief/inquire/resolve/guard/evolve`) over a
`BetaDist(α,β)` with Dempster-Shafer fusion + temporal decay; VM path
(`val/emit/for/match/fn/gate`) with confidence propagation; certificates
(SHA-256 + HMAC + Ed25519); pluggable `inquiry_adapter`.

**The decisive finding — these projects already converge.** AXIOM's
`belief.rs::BetaBelief` is explicitly *"ported and adapted from ChimeraLang's
`cir/nodes.py::BetaDist`"*; `provenance.rs` mirrors ChimeraLang's certificates;
`heal_memory` already does Dempster-Shafer with conflict; `hallucination.rs`
parallels `detect.py`. Integration is **reuniting convergent designs**, not
bolting on a foreign runtime.

| ChimeraLang | AXIOM | Resolution |
|---|---|---|
| `BetaDist(α,β)` | `belief::BetaBelief` | beliefs run directly on `BetaBelief` |
| `combine_ds` | `BetaBelief::combine_ds` | shared fusion |
| cert SHA-256/HMAC | `provenance::{sign,verify}_export` | shared cert format |
| `inquiry_adapter` | `chimera::InquiryAdapter` trait | AXIOM model grounds inquiries |

## Decision

**Port to Rust** (not subprocess/PyO3), extending the existing `BetaDist` port,
and deliver **Phases 0–4 in one PR**. No Python runtime, no IPC; ChimeraLang's
belief/cert primitives become the engine's own.

## What shipped (this PR)

A faithful **core** in `src/chimera.rs` — lexer → parser → AST → {VM, CIR} →
certificate — reusing `belief.rs` and `provenance.rs`:

- **VM path:** `val`, `emit`, `for…in…end`, arithmetic/string-concat/comparison
  expressions, list literals, member access (`.confidence`/`.raw`), confidence
  propagated as the min over operands.
- **CIR/belief path:** `belief NAME := inquire { prompt, agents, ttl }`,
  `resolve … with consensus { threshold }`, `guard … against hallucination
  { max_risk }`, `evolve … until stable { max_iter }`, `emit NAME`. Beliefs are
  `BetaBelief`; multiple agents fuse via `combine_ds`; guards are variance-aware
  (a high mean with high variance still fails); `ttl: 0` ages via `decayed`.
- **InquiryAdapter trait** + `MockAdapter` (the Phase-2 grounding seam).
- **Certificates** via `provenance::sign_export` (SHA-256 + optional HMAC from
  `AXIOM_FLEET_KEY`) — one offline-verifiable format shared with AXIOM patches.

### Phase status
| Phase | Goal | Status |
|---|---|---|
| 0 | In-tree module + contract | ✅ `src/chimera.rs`, registered in lib + bin |
| 1 | ChimeraLang as a repair/verify target | ✅ `Language::Chimera` (marker `chimera.toml` / `*.chimera`), `default_verify → axiom chimera check`, `.chimera` in source-ext set |
| 2 | AXIOM grounds `inquire` | ◑ `InquiryAdapter` seam shipped; backend-backed adapter is the follow-up |
| 3 | Cert + service unification | ✅ certs via `provenance`; CLI `axiom chimera {check,run,prove,verify}`; HTTP `POST /v1/chimera/run` |
| 4 | Rust port | ✅ the core is a Rust port (this module) |

### Verification
- `cargo test --lib`: **297 passing** (+10: chimera core ×9, fault_locate ×1, server ×1).
- `cargo clippy --lib` clean; `cargo build --release --locked` OK; bin builds.

## Deferred (tracked follow-ups)
- **Phase 2 deep wiring:** an `InquiryAdapter` backed by AXIOM's model/backend
  (`claude_backend` / `/v1/messages`) so `inquire` returns grounded `{answer,
  confidence}` instead of the mock.
- **Language surface:** `gate` quantum-consensus branches, `fn`/`match`/`goal`/
  `reason`, the type checker's capability enforcement (`allow/forbidden`), and
  the PyTorch/LLVM compiler + RAG + symbol-emergence subsystems. The module is
  layered (lexer → parser → {vm, cir}) so these slot in incrementally.
- **Ed25519:** ChimeraLang's asymmetric certs; AXIOM `provenance` is SHA-256+HMAC
  today — adding Ed25519 there unifies third-party-verifiable certs.

## Invariant preserved
ChimeraLang execution is pure/offline on the VM path and gated by AXIOM's own
verifier when used as a repair target — re-verify-before-trust is unchanged.
