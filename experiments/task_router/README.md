# task_router (L3)

Deterministic tier-routing bridge: caller-declared `TaskKind` → `Capability`, so orchestrator callers get tier floors without naming models and without any inference from prompt text.

- `backend_router::TaskKind` picks *provider*; `model_router::Capability` picks *tier*. This experiment is the missing bridge between the two vocabularies, implemented in `model_router.rs` (which owns `Capability` + `select_model`).
- Mapping: `CodeRepair → General`, `Reasoning → Reasoning`, `General → None` (router does nothing for unclassified traffic — safe default, never a guessed downgrade).
- Hard ban (frozen): no capability is ever inferred from prompt text, length, or keywords. That family is killed; `General` maps to `None` rather than to a guess.

Manifest: `manifest.json` (frozen). Repro: `run.ps1` (unit tests only).
