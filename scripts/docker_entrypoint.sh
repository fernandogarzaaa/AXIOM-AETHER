#!/usr/bin/env sh
# Container entrypoint for the Axiom-TTT engine.
#
# The image ships WITHOUT trained weights (see .dockerignore) so it stays small
# and reproducible. At boot we optionally seed the read-only production
# checkpoint + tokenizer from object storage / a release URL, then hand off to
# the server. Mutable learned state (heal memory, immunity) converges across the
# fleet at runtime via /v1/immunity{,/merge} — it is NOT fetched here.
#
# Env:
#   AXIOM_CHECKPOINT_URL   optional; if set and the local checkpoint is absent,
#                          download it to $AXIOM_BPE_CKPT (default checkpoints/axiom_production_bpe.bin)
#   AXIOM_TOKENIZER_URL    optional; downloaded to $AXIOM_TOKENIZER (default checkpoints/axiom_bpe.json)
#   AXIOM_BPE_CKPT         target checkpoint path (default: /app/checkpoints/axiom_production_bpe.bin)
#   AXIOM_TOKENIZER        target tokenizer path  (default: /app/checkpoints/axiom_bpe.json)
#
# If neither URL is set and no checkpoint is mounted, the server boots with a
# random-initialized model (already supported) — fine for a smoke deploy, but
# the drift gate / surprisal signals are only meaningful with a trained model.
set -eu

CKPT_PATH="${AXIOM_BPE_CKPT:-/app/checkpoints/axiom_production_bpe.bin}"
TOK_PATH="${AXIOM_TOKENIZER:-/app/checkpoints/axiom_bpe.json}"

fetch() {
    url="$1"
    dest="$2"
    if [ -f "$dest" ]; then
        echo "[entrypoint] $dest already present — skipping download"
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    echo "[entrypoint] fetching $url -> $dest"
    # -f: fail on HTTP errors; -L: follow redirects (release assets / signed URLs)
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$dest"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$dest" "$url"
    else
        echo "[entrypoint] ERROR: neither curl nor wget available to fetch $url" >&2
        return 1
    fi
}

if [ -n "${AXIOM_CHECKPOINT_URL:-}" ]; then
    fetch "$AXIOM_CHECKPOINT_URL" "$CKPT_PATH"
fi
if [ -n "${AXIOM_TOKENIZER_URL:-}" ]; then
    fetch "$AXIOM_TOKENIZER_URL" "$TOK_PATH"
fi

if [ -f "$CKPT_PATH" ]; then
    echo "[entrypoint] checkpoint present at $CKPT_PATH"
else
    echo "[entrypoint] WARNING: no checkpoint at $CKPT_PATH — server will random-init (drift signals weak)"
fi

exec /app/axiom_engine "$@"
