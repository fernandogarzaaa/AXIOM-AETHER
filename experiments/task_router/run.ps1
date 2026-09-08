$ErrorActionPreference = "Stop"
# task_router repro: deterministic unit tests only (no network, no models).
Set-Location (Join-Path $PSScriptRoot ".." ".." "axiom_engine_rs")
cargo test --lib model_router
