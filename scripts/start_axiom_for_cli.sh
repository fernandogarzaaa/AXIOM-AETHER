#!/usr/bin/env bash
#
# start_axiom_for_cli.sh — run the Axiom proxy on loopback for a vendor CLI.
#
# The hybrid integration (docs/CLI-INTEGRATION.md): your coding CLI keeps its
# own login (Claude Code, Codex, Aider/Cline/Continue API keys — read the
# vendor-terms section of docs/CLI-INTEGRATION.md before routing a
# subscription login) and only its *base URL* is pointed at this proxy. Axiom
# compresses the traffic and relays the CLI's own Authorization / x-api-key /
# anthropic-version / anthropic-beta / OpenAI-Organization headers upstream
# verbatim. The proxy holds NO key of its own: this script never reads
# ANTHROPIC_API_KEY / OPENAI_API_KEY into the server and never sets
# AXIOM_API_KEY (loopback binds are open by design; a non-loopback bind
# requires AXIOM_API_KEY, see docs/ENVIRONMENT.md).
#
# What it does:
#   1. Starts `axiom_engine --mode server` on AXIOM_HOST:AXIOM_PORT
#      (default 127.0.0.1:8080) in the background, logging to AXIOM_LOG_FILE.
#      Idempotent: if a healthy proxy already answers there, it is reused.
#   2. Waits for GET /readyz.
#   3. Prints the exact `export` lines for each supported CLI, plus a one-line
#      verification command.
#
# Usage:
#   ./scripts/start_axiom_for_cli.sh            # start (or reuse) and print exports
#   ./scripts/start_axiom_for_cli.sh --stop     # stop the proxy this script started
#   AXIOM_PORT=9090 ./scripts/start_axiom_for_cli.sh
#   eval "$(./scripts/start_axiom_for_cli.sh --exports-only)"   # just the exports
#
# Upstream vs client routing (same rule as start_axiom.sh): the server's own
# outbound bridge reads ANTHROPIC_BASE_URL / OPENAI_BASE_URL to pick ITS
# upstream. This script pins both to the real vendor APIs for the server
# process and ignores any inherited client value, so the proxy can never
# forward to itself. Put the printed exports in the CLI's shell, not here.
#
set -uo pipefail

# --- Resolve paths ---------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_DIR="$REPO_ROOT/axiom_engine_rs"
BIN="${AXIOM_BIN:-$CRATE_DIR/target/release/axiom_engine}"
[ -x "$BIN" ] || { [ -x "$BIN.exe" ] && BIN="$BIN.exe"; }   # Windows / Git Bash

# --- Network boundary ------------------------------------------------------
HOST="${AXIOM_HOST:-127.0.0.1}"
PORT="${AXIOM_PORT:-8080}"
BASE="http://$HOST:$PORT"
STATE_DIR="${AXIOM_CLI_PROXY_STATE_DIR:-$REPO_ROOT/.axiom-cli-proxy}"
PID_FILE="$STATE_DIR/axiom-$PORT.pid"
LOG_FILE="${AXIOM_LOG_FILE:-$STATE_DIR/axiom-$PORT.log}"
READY_TIMEOUT_S="${AXIOM_READY_TIMEOUT_S:-120}"

# --- Upstreams (real vendor APIs) — never the proxy itself -----------------
ANTHROPIC_UPSTREAM="${AXIOM_UPSTREAM_URL:-https://api.anthropic.com}"
OPENAI_UPSTREAM="${AXIOM_OPENAI_UPSTREAM_URL:-https://api.openai.com}"
for u in "$ANTHROPIC_UPSTREAM" "$OPENAI_UPSTREAM"; do
    case "$u" in
        *"$HOST:$PORT"*|*127.0.0.1:"$PORT"*|*localhost:"$PORT"*)
            echo "[start_axiom_for_cli] FATAL: upstream ($u) points back at this proxy ($HOST:$PORT)."
            echo "[start_axiom_for_cli] That would create an infinite forward loop. Set AXIOM_UPSTREAM_URL /"
            echo "[start_axiom_for_cli] AXIOM_OPENAI_UPSTREAM_URL to the real vendor APIs."
            exit 1
            ;;
    esac
done

log() { echo "[start_axiom_for_cli] $*" >&2; }

healthy() { curl -sf --max-time 2 "$BASE/healthz" > /dev/null 2>&1; }
ready()   { curl -sf --max-time 2 "$BASE/readyz"  > /dev/null 2>&1; }

print_exports() {
    cat <<EOF
# --- Axiom proxy for vendor CLIs: $BASE ---------------------------------
# Paste into the shell that runs the CLI (NOT the proxy's shell). Each CLI
# keeps its own login; only the base URL changes. Routes hit:
#   Anthropic shape -> POST $BASE/v1/messages
#   OpenAI shapes   -> POST $BASE/v1/chat/completions  and  $BASE/v1/responses

# Claude Code (OAuth subscription login or ANTHROPIC_API_KEY, unchanged):
export ANTHROPIC_BASE_URL="$BASE"

# OpenAI Codex CLI: the current config reference documents the base URL as a
# config.toml key, not an env var. Put this in ~/.codex/config.toml (Codex
# speaks the Responses API, so traffic lands on $BASE/v1/responses):
#   openai_base_url = "$BASE/v1"
# Older Codex releases also read OPENAI_BASE_URL; harmless to export both:
export OPENAI_BASE_URL="$BASE/v1"

# Aider (OpenAI-compatible models, \`aider --model openai/<name>\`; aider
# documents no base-URL override for its Anthropic path):
export OPENAI_API_BASE="$BASE/v1"

# Cline / Continue / OpenCode: set the provider's Base URL field in their
# settings to "$BASE" (Anthropic provider) or "$BASE/v1" (OpenAI-compatible
# provider) -- docs/CLI-INTEGRATION.md has the exact keys.

# Verify (no credentials needed for these two):
#   curl -fsS $BASE/healthz && curl -s $BASE/metrics | grep -E '^axiom_(savings|cost|quota)_'
# Savings show up as axiom_savings_* / axiom_cost_* deltas while the CLI runs.
EOF
}

# --- --stop / --exports-only -----------------------------------------------
case "${1:-}" in
    --stop)
        if [ -f "$PID_FILE" ]; then
            pid="$(cat "$PID_FILE" 2>/dev/null || true)"
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                kill "$pid" && log "stopped axiom_engine pid $pid"
            else
                log "no live process for pid file $PID_FILE"
            fi
            rm -f "$PID_FILE"
        else
            log "nothing to stop (no pid file at $PID_FILE)"
        fi
        exit 0
        ;;
    --exports-only)
        print_exports
        exit 0
        ;;
    --help|-h)
        sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
esac

# --- Idempotent: reuse a healthy proxy already on this address -------------
if healthy; then
    log "a proxy already answers on $BASE — reusing it (nothing started)."
    print_exports
    exit 0
fi

# --- Preflight -------------------------------------------------------------
if [ ! -x "$BIN" ]; then
    log "Release binary missing: $BIN"
    log "Build it first:  cargo build --release --locked --manifest-path \"$CRATE_DIR/Cargo.toml\""
    exit 1
fi
command -v curl > /dev/null 2>&1 || { log "curl is required"; exit 1; }
mkdir -p "$STATE_DIR"

# A stale pid file from a dead process must not block a fresh start.
if [ -f "$PID_FILE" ]; then
    old="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
        log "pid $old from $PID_FILE is alive but $BASE/healthz does not answer yet; waiting for it."
    else
        rm -f "$PID_FILE"
    fi
fi

# --- Compression config (same defaults as start_axiom.sh) ------------------
export AXIOM_TTT_COMPRESS="${AXIOM_TTT_COMPRESS:-1}"
export AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS="${AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS:-400}"
export AXIOM_TTT_COMPRESS_TOP_K="${AXIOM_TTT_COMPRESS_TOP_K:-32}"
export AXIOM_DEVICE="${AXIOM_DEVICE:-auto}"

# --- Production BPE model when present (same resolution as start_axiom.sh) --
PROD_BPE_CKPT="$REPO_ROOT/checkpoints/axiom_production_bpe.bin"
PROD_BPE_TOK="$REPO_ROOT/checkpoints/axiom_bpe.json"
if [ -f "$PROD_BPE_CKPT" ] && [ -f "$PROD_BPE_TOK" ]; then
    export AXIOM_PRODUCTION_BPE=1
    export AXIOM_TOKENIZER="$PROD_BPE_TOK"
    export AXIOM_BPE_CKPT="$PROD_BPE_CKPT"
    log "model: BPE semantic ($PROD_BPE_CKPT)"
else
    log "model: default (no checkpoints/axiom_production_bpe.bin; compression fingerprint is low fidelity)"
fi

# --- Start (if not already running from a live pid) ------------------------
if [ ! -f "$PID_FILE" ]; then
    log "launching $BIN --mode server --host $HOST --port $PORT"
    log "  upstreams   : anthropic=$ANTHROPIC_UPSTREAM  openai=$OPENAI_UPSTREAM"
    log "  compression : $AXIOM_TTT_COMPRESS (threshold=$AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS, top_k=$AXIOM_TTT_COMPRESS_TOP_K)"
    log "  auth        : pass-through (the proxy holds no vendor key; AXIOM_API_KEY unset on loopback)"
    log "  log         : $LOG_FILE"
    # `env -u`: the server process must never inherit a client-side base URL
    # (loop) nor a vendor key (it would inject it next to the CLI's own auth);
    # `env -i` is avoided so PATH/HOME/proxy settings still apply.
    env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u AXIOM_API_KEY \
        ANTHROPIC_BASE_URL="$ANTHROPIC_UPSTREAM" OPENAI_BASE_URL="$OPENAI_UPSTREAM" \
        nohup "$BIN" --mode server --host "$HOST" --port "$PORT" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
fi

# --- Wait for /readyz ------------------------------------------------------
deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
until ready; do
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
        log "FATAL: axiom_engine (pid $pid) exited before becoming ready. Last log lines:"
        tail -n 20 "$LOG_FILE" >&2 2>/dev/null || true
        rm -f "$PID_FILE"
        exit 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        log "FATAL: $BASE/readyz not ready after ${READY_TIMEOUT_S}s. Last log lines:"
        tail -n 20 "$LOG_FILE" >&2 2>/dev/null || true
        exit 1
    fi
    sleep 0.5
done
log "ready: $BASE/readyz OK (pid $(cat "$PID_FILE"))"
log "stop with: $0 --stop"
echo >&2
print_exports
