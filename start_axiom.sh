#!/usr/bin/env bash
#
# start_axiom.sh — boot the Axiom-TTT context-compression proxy.
#
# This launches the Rust `axiom_engine` server in `--mode server`. The server
# exposes an Anthropic-compatible POST /v1/messages endpoint. When compression
# is enabled it absorbs "heavy" context locally (TTT) and forwards a lean,
# fingerprinted payload to the REAL Anthropic API.
#
# IMPORTANT — upstream vs. client routing
# ----------------------------------------
# The forwarder reads ANTHROPIC_BASE_URL to choose ITS OWN upstream
# (see src/anthropic_forwarder.rs). If you point ANTHROPIC_BASE_URL at this
# proxy (127.0.0.1:3000) in the SAME shell that runs the server, the proxy
# forwards to itself -> infinite loop. This script therefore pins the server's
# upstream to the real Anthropic API and ignores any inherited client value.
# Client redirection belongs in a SEPARATE shell — see axiom.env.
#
set -euo pipefail

# --- Resolve paths ---------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$REPO_ROOT/axiom_engine_rs"
BIN="$CRATE_DIR/target/release/axiom_engine"

# --- Network boundary ------------------------------------------------------
HOST="${AXIOM_HOST:-127.0.0.1}"
PORT="${AXIOM_PORT:-3000}"

# --- Upstream (real Anthropic) — never the proxy itself --------------------
# Allow an explicit override via AXIOM_UPSTREAM_URL, but default to the real API
# and refuse to forward to ourselves.
UPSTREAM="${AXIOM_UPSTREAM_URL:-https://api.anthropic.com}"
case "$UPSTREAM" in
    *"$HOST:$PORT"*|*127.0.0.1:"$PORT"*|*localhost:"$PORT"*)
        echo "[start_axiom] FATAL: upstream ($UPSTREAM) points back at this proxy ($HOST:$PORT)."
        echo "[start_axiom] That would create an infinite forward loop. Set AXIOM_UPSTREAM_URL"
        echo "[start_axiom] to the real Anthropic API (https://api.anthropic.com)."
        exit 1
        ;;
esac

# --- Compression config ----------------------------------------------------
# Compression is the whole point of the proxy, so default it ON. Override with
# AXIOM_TTT_COMPRESS=0 to run a pure passthrough.
export AXIOM_TTT_COMPRESS="${AXIOM_TTT_COMPRESS:-1}"
export AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS="${AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS:-512}"
export AXIOM_TTT_COMPRESS_TOP_K="${AXIOM_TTT_COMPRESS_TOP_K:-32}"

# --- Compute device: GPU-first, CPU fallback -------------------------------
# Default to "auto": the engine selects CUDA when the `cuda` feature is compiled
# in AND a GPU + runtime libraries are available, else it falls back to CPU
# gracefully (cuda_if_available in main.rs never errors). Force with
# AXIOM_DEVICE=cpu (e.g. to keep the display GPU idle) or AXIOM_DEVICE=cuda.
export AXIOM_DEVICE="${AXIOM_DEVICE:-auto}"
# If a local CUDA 12.6 toolkit is present (the GPU build links cudarc against it),
# put its runtime libraries on PATH so they load at device-init time. Skipped
# harmlessly on machines without it — AXIOM_DEVICE=auto then just uses CPU.
AXIOM_CUDA_HOME="${AXIOM_CUDA_HOME:-$HOME/cuda-12.6}"
if [ -d "$AXIOM_CUDA_HOME/bin" ]; then
    export PATH="$AXIOM_CUDA_HOME/bin:$AXIOM_CUDA_HOME/nvvm/bin:$PATH"
    export CUDA_PATH="$AXIOM_CUDA_HOME"
fi

# --- Production BPE model (Objective 1.3 swap) -----------------------------
# Officially deprecate the legacy 256-character-hash model: when the scaled BPE
# artifacts are present, the proxy runs the d_model=256 / n_layers=4 / BPE-vocab
# semantic model with the calibrated deterministic drift gate. Falls back to the
# legacy model automatically if the artifacts are missing.
PROD_BPE_CKPT="$REPO_ROOT/checkpoints/axiom_production_bpe.bin"
PROD_BPE_TOK="$REPO_ROOT/checkpoints/axiom_bpe.json"
if [ -f "$PROD_BPE_CKPT" ] && [ -f "$PROD_BPE_TOK" ]; then
    export AXIOM_PRODUCTION_BPE=1
    export AXIOM_TOKENIZER="$PROD_BPE_TOK"
    export AXIOM_BPE_CKPT="$PROD_BPE_CKPT"
    # Prefer the eval-recalibrated drift gate when eval_model wrote one.
    GATE_FILE="$REPO_ROOT/checkpoints/axiom_drift_gate.txt"
    if [ -f "$GATE_FILE" ]; then
        export AXIOM_DRIFT_THRESHOLD="$(cat "$GATE_FILE")"
    else
        export AXIOM_DRIFT_THRESHOLD="${AXIOM_DRIFT_THRESHOLD:-7.03}"
    fi
    echo "[start_axiom] Production model: BPE semantic (dims from sidecar, drift_gate=$AXIOM_DRIFT_THRESHOLD)"
else
    echo "[start_axiom] Production model: legacy 256-hash (BPE artifacts not found)"
fi

# The server's outbound bridge uses this to reach the REAL API.
export ANTHROPIC_BASE_URL="$UPSTREAM"

# --- Checkpoint resolution -------------------------------------------------
# The mission references ./checkpoints/axiom_production.bin. That artifact is
# gitignored and not present in a fresh clone. If you have it, drop it in;
# otherwise we fall back to the crate default (fresh in-memory init).
PROD_CKPT="$REPO_ROOT/checkpoints/axiom_production.bin"
CKPT_ARGS=()
if [ -f "$PROD_CKPT" ]; then
    echo "[start_axiom] Using production checkpoint: $PROD_CKPT"
    CKPT_ARGS=(--checkpoint "$PROD_CKPT")
else
    echo "[start_axiom] WARNING: $PROD_CKPT not found — booting with the crate's"
    echo "[start_axiom]          default fresh init (a small CPU model, d_model=64,"
    echo "[start_axiom]          n_layers=2, vocab=256). The compression fingerprint"
    echo "[start_axiom]          from this model is LOW FIDELITY. Do not route real"
    echo "[start_axiom]          coding traffic through it expecting lossless context."
fi

# --- Preflight -------------------------------------------------------------
if [ ! -x "$BIN" ]; then
    echo "[start_axiom] Release binary missing: $BIN"
    echo "[start_axiom] Build it first:  cargo build --release --manifest-path \"$CRATE_DIR/Cargo.toml\""
    exit 1
fi

if [ "${AXIOM_TTT_COMPRESS}" = "1" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "[start_axiom] No ANTHROPIC_API_KEY set -> AUTH-PASSTHROUGH mode."
    echo "[start_axiom]   The proxy holds no key of its own and relays each client's"
    echo "[start_axiom]   own Authorization / x-api-key headers upstream. This is the"
    echo "[start_axiom]   correct mode for a Claude SUBSCRIPTION (Claude Code OAuth):"
    echo "[start_axiom]   point a client shell's ANTHROPIC_BASE_URL at this proxy and"
    echo "[start_axiom]   its OAuth bearer token is forwarded to Anthropic for you."
fi

LOG_FILE="${AXIOM_LOG_FILE:-$REPO_ROOT/axiom_server.log}"

echo "[start_axiom] Launching Axiom-TTT proxy"
echo "[start_axiom]   bind        : http://$HOST:$PORT"
echo "[start_axiom]   upstream    : $ANTHROPIC_BASE_URL"
echo "[start_axiom]   compression : $AXIOM_TTT_COMPRESS (threshold=$AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS tokens, top_k=$AXIOM_TTT_COMPRESS_TOP_K)"
if [ -d "$AXIOM_CUDA_HOME/bin" ]; then
    echo "[start_axiom]   device      : $AXIOM_DEVICE (GPU-first; CUDA 12.6 toolkit on PATH, CPU fallback)"
else
    echo "[start_axiom]   device      : $AXIOM_DEVICE (no local CUDA toolkit — CPU unless system CUDA present)"
fi
echo "[start_axiom]   log         : $LOG_FILE"
echo

# Tee so the compression metric lines ([axiom-ttt] ... recall_norm=...) are
# visible live AND captured for the smoke test to inspect.
exec "$BIN" --mode server --host "$HOST" --port "$PORT" "${CKPT_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
