#!/usr/bin/env bash
set -euo pipefail

REPO="${AXIOM_REPO:-fernandogarzaaa/AXIOM-AETHER}"
INSTALL_DIR="${AXIOM_INSTALL_DIR:-$HOME/.local/bin}"
API_URL="https://api.github.com/repos/${REPO}/releases/latest"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "axiom installer: missing required command '$1'" >&2
    exit 1
  }
}

# sha256_of <file>: print the lowercase hex SHA-256 of a file, or return 1 if
# no hashing tool is available.
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print tolower($1)}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print tolower($1)}'
  else
    return 1
  fi
}

# verify_archive_checksum <archive> <sidecar-or-empty>
# Returns 0 when the archive may be installed, non-zero when it must not be.
#   sidecar present, hash matches    -> 0 (prints "checksum verified")
#   sidecar present, hash differs    -> 1
#   sidecar absent                   -> 0 with a warning (1 if AXIOM_REQUIRE_CHECKSUM=1)
#   no sha256sum/shasum on the host  -> 0 with a warning (1 if AXIOM_REQUIRE_CHECKSUM=1)
verify_archive_checksum() {
  local archive="$1" sidecar="${2:-}" expected actual

  if [[ -z "$sidecar" || ! -s "$sidecar" ]]; then
    if [[ "${AXIOM_REQUIRE_CHECKSUM:-}" == "1" ]]; then
      echo "axiom installer: no .sha256 sidecar published for this release and AXIOM_REQUIRE_CHECKSUM=1; refusing to install" >&2
      return 1
    fi
    echo "axiom installer: warning: no .sha256 sidecar published for this release; skipping verification" >&2
    return 0
  fi

  if ! actual="$(sha256_of "$archive")"; then
    if [[ "${AXIOM_REQUIRE_CHECKSUM:-}" == "1" ]]; then
      echo "axiom installer: neither sha256sum nor shasum found and AXIOM_REQUIRE_CHECKSUM=1; refusing to install" >&2
      return 1
    fi
    echo "axiom installer: warning: neither sha256sum nor shasum found; skipping verification" >&2
    return 0
  fi

  expected="$(awk 'NR==1 {print tolower($1)}' "$sidecar")"
  if [[ "$expected" != "$actual" ]]; then
    echo "axiom installer: checksum mismatch" >&2
    echo "  expected: ${expected}" >&2
    echo "  actual:   ${actual}" >&2
    echo "axiom installer: refusing to install a corrupted or substituted archive" >&2
    return 1
  fi

  echo "axiom installer: checksum verified (${actual})"
}

main() {
  need curl
  need tar

  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$os:$arch" in
    linux:x86_64|linux:amd64)
      suffix="linux-x86_64"
      ext="tar.gz"
      ;;
    darwin:arm64|darwin:aarch64)
      suffix="macos-arm64"
      ext="tar.gz"
      ;;
    *)
      echo "axiom installer: no prebuilt binary for ${os}/${arch}" >&2
      echo "  Prebuilt releases cover linux/x86_64, macOS/arm64 and Windows/x86_64." >&2
      echo "  Build from source instead (Rust >= 1.90 plus a C toolchain):" >&2
      echo "    Debian/Ubuntu: sudo apt-get install -y build-essential pkg-config libssl-dev" >&2
      echo "    curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal" >&2
      echo "    cargo install axiom_engine        # or: cargo build --release in axiom_engine_rs/" >&2
      exit 1
      ;;
  esac

  echo "axiom installer: resolving latest release from ${REPO}"
  release_json="$(curl -fsSL "$API_URL")"
  asset_url="$(printf '%s' "$release_json" |
    sed -n 's/.*"browser_download_url": "\(.*axiom-ttt-.*-'"$suffix"'\.'"$ext"'\)".*/\1/p' |
    head -n 1)"

  if [[ -z "$asset_url" ]]; then
    echo "axiom installer: no release asset found for ${suffix}" >&2
    exit 1
  fi

  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  archive="$tmp/axiom.${ext}"

  echo "axiom installer: downloading ${asset_url}"
  curl -fL "$asset_url" -o "$archive"

  # Releases built after the checksum-manifest change publish <asset>.sha256
  # (sha256sum -c format). Older releases have none.
  sidecar="$tmp/axiom.${ext}.sha256"
  if ! curl -fsSL "${asset_url}.sha256" -o "$sidecar" 2>/dev/null; then
    rm -f "$sidecar"
    sidecar=""
  fi
  verify_archive_checksum "$archive" "$sidecar" || exit 1

  tar -xzf "$archive" -C "$tmp"

  binary="$(find "$tmp" -type f -name axiom_engine -perm -111 | head -n 1)"
  if [[ -z "$binary" ]]; then
    echo "axiom installer: release archive did not contain axiom_engine" >&2
    exit 1
  fi

  mkdir -p "$INSTALL_DIR"
  install -m 0755 "$binary" "$INSTALL_DIR/axiom"

  echo "axiom installer: installed $INSTALL_DIR/axiom"
  if ! printf '%s' "$PATH" | tr ':' '\n' | grep -Fxq "$INSTALL_DIR"; then
    echo "axiom installer: add this to PATH: export PATH=\"$INSTALL_DIR:\$PATH\""
  fi

  "$INSTALL_DIR/axiom" init
}

if [[ "${BASH_SOURCE[0]:-$0}" == "$0" ]]; then
  main "$@"
fi
