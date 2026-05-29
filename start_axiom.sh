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
echo "[start_axiom]   log         : $LOG_FILE"
echo

# Tee so the compression metric lines ([axiom-ttt] ... recall_norm=...) are
# visible live AND captured for the smoke test to inspect.
exec "$BIN" --mode server --host "$HOST" --port "$PORT" "${CKPT_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
