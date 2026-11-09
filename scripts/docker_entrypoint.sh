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
#                          optional pinned SHA-256 for the matching download.
#                          When set, the download is verified before it is
#                          promoted, a mismatch refuses install (non-zero exit),
#                          and a pre-existing file with a wrong hash is
#                          re-downloaded instead of trusted. When unset, the
#                          computed hash is printed so it can be pinned.
#   AXIOM_ENTRYPOINT_NOEXEC=1
#                          test hook: run the fetch/verify steps, then exit 0
#                          instead of execing the server (see
#                          tests/test_docker_entrypoint_atomic.sh).
#
# Install is atomic: every download goes to a temp file in the same directory,
# is verified there, and is renamed onto the final path only on success. A
# failed or interrupted download never leaves a partial file under the final
# name, so a later boot never mistakes a truncated download for a checkpoint.
#
# If neither URL is set and no checkpoint is mounted, the server boots with a
# random-initialized model (already supported) — fine for a smoke deploy, but
# the drift gate / surprisal signals are only meaningful with a trained model.
set -eu

CKPT_PATH="${AXIOM_BPE_CKPT:-/app/checkpoints/axiom_production_bpe.bin}"
TOK_PATH="${AXIOM_TOKENIZER:-/app/checkpoints/axiom_bpe.json}"

# Print the SHA-256 of $1 on stdout. Prefers sha256sum, falls back to shasum.
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

fetch() {
    url="$1"
    dest="$2"
    expected="${3:-}"
    if [ -f "$dest" ]; then
        if [ -n "$expected" ]; then
            actual="$(sha256_of "$dest")"
            if [ "$actual" = "$expected" ]; then
                echo "[entrypoint] $dest already present (sha256 verified) — skipping download"
                return 0
            fi
            echo "[entrypoint] $dest present but sha256 mismatch (expected $expected, got $actual) — re-downloading"
            rm -f "$dest"
        else
            echo "[entrypoint] $dest already present — skipping download (no pinned hash configured; hash unverified)"
            return 0
        fi
    fi
    mkdir -p "$(dirname "$dest")"
    tmp="$dest.tmp.$$"
    rm -f "$tmp"
    echo "[entrypoint] fetching $url -> $dest (via temp file)"
    # -f: fail on HTTP errors; -L: follow redirects (release assets / signed URLs).
    # A failure here must delete the temp file: curl exits non-zero on a
    # truncated transfer (Content-Length unfulfilled), and the partial bytes
    # must never be promoted or left under the final name.
    if command -v curl >/dev/null 2>&1; then
        if ! curl -fsSL "$url" -o "$tmp"; then
            echo "[entrypoint] ERROR: download failed for $url (partial download discarded)" >&2
            rm -f "$tmp"
            return 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        if ! wget -qO "$tmp" "$url"; then
            echo "[entrypoint] ERROR: download failed for $url (partial download discarded)" >&2
            rm -f "$tmp"
            return 1
        fi
    else
        echo "[entrypoint] ERROR: neither curl nor wget available to fetch $url" >&2
        return 1
    fi
    actual="$(sha256_of "$tmp")"
    if [ -n "$expected" ]; then
        if [ "$actual" != "$expected" ]; then
            echo "[entrypoint] ERROR: sha256 mismatch for $url (expected $expected, got $actual) — refusing to install" >&2
            rm -f "$tmp"
            return 1
        fi
        echo "[entrypoint] sha256 verified: $actual"
    else
        echo "[entrypoint] fetched $url (sha256=$actual, unverified — pin it with AXIOM_CHECKPOINT_SHA256 / AXIOM_TOKENIZER_SHA256)"
    fi
    # Same-directory rename: atomic promotion, no window where $dest is partial.
    mv "$tmp" "$dest"
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

if [ "${AXIOM_ENTRYPOINT_NOEXEC:-}" = "1" ]; then
    exit 0
fi

exec /app/axiom_engine "$@"
