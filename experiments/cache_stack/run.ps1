$ErrorActionPreference = "Stop"
# cache_stack repro: deterministic unit tests only (no network, no models).
Set-Location (Join-Path $PSScriptRoot ".." ".." "axiom_engine_rs")
cargo test --lib cache_stack
