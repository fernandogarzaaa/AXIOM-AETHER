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
#   AXIOM_CHECKPOINT_SHA256 / AXIOM_TOKENIZER_SHA256
#                          optional pinned SHA-256 for the corresponding download. Every
#                          `release.yml` run publishes a SHA256SUMS.txt release asset
#                          alongside the checkpoint/tokenizer -- copy the matching line's
#                          hash here to pin it. A mismatch deletes the partial download and
#                          exits non-zero rather than booting on an unverified checkpoint;
#                          this mirrors the same-named env vars in axiom_engine_rs/src/config.rs
#                          (the `axiom init` path), which this container image doesn't use.
#                          Unset (the default) skips verification -- same as before.
#
# If neither URL is set and no checkpoint is mounted, the server boots with a
# random-initialized model (already supported) — fine for a smoke deploy, but
# the drift gate / surprisal signals are only meaningful with a trained model.
set -eu

CKPT_PATH="${AXIOM_BPE_CKPT:-/app/checkpoints/axiom_production_bpe.bin}"
TOK_PATH="${AXIOM_TOKENIZER:-/app/checkpoints/axiom_bpe.json}"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        echo "[entrypoint] ERROR: neither sha256sum nor shasum available to verify $1" >&2
        return 1
    fi
}

# Verify $1 against pinned hash $2 (case-insensitive), when $2 is non-empty.
# On mismatch, deletes $1 and returns non-zero -- refusing to leave an
# unverified checkpoint on disk under its expected name.
verify_checksum() {
    dest="$1"
    expected="$2"
    if [ -z "$expected" ]; then
        actual=$(sha256_of "$dest") || return 1
        echo "[entrypoint] $dest sha256=$actual (unverified -- no expected hash configured; pin it to verify future fetches)"
        return 0
    fi
    actual=$(sha256_of "$dest") || return 1
    expected_lc=$(echo "$expected" | tr 'A-F' 'a-f')
    actual_lc=$(echo "$actual" | tr 'A-F' 'a-f')
    if [ "$expected_lc" != "$actual_lc" ]; then
        echo "[entrypoint] ERROR: checksum mismatch for $dest: expected sha256=$expected, got sha256=$actual. Deleting unverified download." >&2
        rm -f "$dest"
        return 1
    fi
    echo "[entrypoint] $dest sha256 verified"
}

fetch() {
    url="$1"
    dest="$2"
    expected_sha256="${3:-}"
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
    verify_checksum "$dest" "$expected_sha256"
}

if [ -n "${AXIOM_CHECKPOINT_URL:-}" ]; then
    fetch "$AXIOM_CHECKPOINT_URL" "$CKPT_PATH" "${AXIOM_CHECKPOINT_SHA256:-}"
fi
if [ -n "${AXIOM_TOKENIZER_URL:-}" ]; then
    fetch "$AXIOM_TOKENIZER_URL" "$TOK_PATH" "${AXIOM_TOKENIZER_SHA256:-}"
fi

if [ -f "$CKPT_PATH" ]; then
    echo "[entrypoint] checkpoint present at $CKPT_PATH"
else
    echo "[entrypoint] WARNING: no checkpoint at $CKPT_PATH — server will random-init (drift signals weak)"
fi

exec /app/axiom_engine "$@"
