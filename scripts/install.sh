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
    echo "axiom installer: unsupported platform ${os}/${arch}" >&2
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
