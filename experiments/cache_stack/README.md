# cache_stack (L0+L1)

First mergeable frontier-economics unit, built onto upstream modules rather than the proxy fork.

- **L0** complements `cache_safety.rs` (which *honors* breakpoints) with a constructor that *places* one at the static/dynamic boundary on unmarked bodies — and refuses when the client already marks its own.
- **L1** ports `axiom_engine/response_cache.py` fingerprint semantics to Rust for the proxy path (SHA-256 over canonical `{model, system, messages, max_tokens}`; temperature excluded by design).
- **L2** companions `cost_ledger.rs` (Anthropic USD) with OpenAI `cached_tokens` counts + hit ratio. No OpenAI price table is invented.

Manifest: `manifest.json` (frozen). Repro: `run.ps1` (unit tests only, no network).

No savings claimed. The pilot that measures hit rates on real traffic is a separate, future manifest.
