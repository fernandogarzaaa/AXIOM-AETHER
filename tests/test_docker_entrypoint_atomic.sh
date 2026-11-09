#!/usr/bin/env sh
# Tests for scripts/docker_entrypoint.sh atomic, verified checkpoint install.
#
# Exercises the real entrypoint against a local HTTP server:
#   1. a good download lands atomically with the pinned hash,
#   2. a digest mismatch refuses install and leaves nothing behind,
#   3. a truncated download is rejected and a later retry succeeds,
#   4. a pre-existing partial file is re-downloaded when a hash is pinned,
#   5. a pre-existing file whose hash matches is skipped without a request.
#
# Run: sh tests/test_docker_entrypoint_atomic.sh (exit 0 = all pass)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENTRY="$ROOT/scripts/docker_entrypoint.sh"
TMP="$(mktemp -d)"
PASS=0
FAIL=0

note() { # note <ok|bad> <label>
    if [ "$1" = ok ]; then PASS=$((PASS + 1)); echo "ok: $2";
    else FAIL=$((FAIL + 1)); echo "FAIL: $2"; fi
}

cleanup() {
    [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

GOOD="checkpoint-bytes-0123456789"
printf '%s' "$GOOD" > "$TMP/good.bin"
GOOD_SHA="$(sha256sum "$TMP/good.bin" | cut -d' ' -f1)"

cat > "$TMP/server.py" <<'PY'
import http.server, os, sys

base = sys.argv[1]
port_file = os.path.join(base, "port")
log_file = os.path.join(base, "requests.log")
good = open(os.path.join(base, "good.bin"), "rb").read()

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass
    def do_GET(self):
        with open(log_file, "a") as f:
            f.write(self.path + "\n")
        if self.path == "/good.bin":
            body = good
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/trunc.bin":
            # Lie about the length, then close mid-body: a truncated download.
            self.send_response(200)
            self.send_header("Content-Length", str(len(good) * 10))
            self.end_headers()
            self.wfile.write(good[:10])
            self.wfile.flush()
            self.close_connection = True
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self.close_connection = True

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
with open(port_file, "w") as f:
    f.write(str(srv.server_address[1]))
srv.serve_forever()
PY

python3 "$TMP/server.py" "$TMP" &
SERVER_PID=$!
for _ in $(seq 1 50); do [ -f "$TMP/port" ] && break; sleep 0.1; done
PORT="$(cat "$TMP/port")"
BASE="http://127.0.0.1:$PORT"

# run_entry <url> <dest> <sha> ; exit code in $?
run_entry() {
    AXIOM_CHECKPOINT_URL="$1" AXIOM_BPE_CKPT="$2" AXIOM_CHECKPOINT_SHA256="$3" \
        AXIOM_TOKENIZER_URL="" AXIOM_ENTRYPOINT_NOEXEC=1 sh "$ENTRY" \
        >/dev/null 2>"$TMP/last.err"
}

leftovers() { # leftovers <dir> : temp files the install should never leave
    find "$1" -name '*.tmp.*' -o -name '*.download' | grep -q .
}

D="$TMP/work"
mkdir -p "$D"

# 1. Good download lands atomically.
run_entry "$BASE/good.bin" "$D/ckpt.bin" "$GOOD_SHA"
rc=$?
if [ "$rc" -eq 0 ] && [ "$(cat "$D/ckpt.bin")" = "$GOOD" ] && ! leftovers "$D"; then
    note ok "good download installs with matching hash"
else
    note bad "good download installs with matching hash (rc=$rc)"; cat "$TMP/last.err"
fi

# 2. Digest mismatch refuses install.
run_entry "$BASE/good.bin" "$D/mismatch.bin" "0000000000000000000000000000000000000000000000000000000000000000"
rc=$?
if [ "$rc" -ne 0 ] && [ ! -e "$D/mismatch.bin" ] && ! leftovers "$D"; then
    note ok "digest mismatch refuses install, leaves nothing"
else
    note bad "digest mismatch refuses install, leaves nothing (rc=$rc)"; cat "$TMP/last.err"
fi

# 3. Truncated download is rejected, then a retry succeeds.
run_entry "$BASE/trunc.bin" "$D/trunc.bin" "$GOOD_SHA"
rc=$?
if [ "$rc" -ne 0 ] && [ ! -e "$D/trunc.bin" ] && ! leftovers "$D"; then
    note ok "truncated download rejected, leaves nothing"
else
    note bad "truncated download rejected, leaves nothing (rc=$rc)"; cat "$TMP/last.err"
fi
run_entry "$BASE/good.bin" "$D/trunc.bin" "$GOOD_SHA"
rc=$?
if [ "$rc" -eq 0 ] && [ "$(cat "$D/trunc.bin")" = "$GOOD" ]; then
    note ok "retry after truncation installs"
else
    note bad "retry after truncation installs (rc=$rc)"; cat "$TMP/last.err"
fi

# 4. Pre-existing partial file is re-downloaded when a hash is pinned.
printf 'partial-garbage' > "$D/partial.bin"
run_entry "$BASE/good.bin" "$D/partial.bin" "$GOOD_SHA"
rc=$?
if [ "$rc" -eq 0 ] && [ "$(cat "$D/partial.bin")" = "$GOOD" ]; then
    note ok "pre-existing partial file is replaced under a pinned hash"
else
    note bad "pre-existing partial file is replaced under a pinned hash (rc=$rc)"; cat "$TMP/last.err"
fi

# 5. Pre-existing file with a matching hash is skipped without a request.
: > "$TMP/requests.log"
run_entry "$BASE/gone.bin" "$D/ckpt.bin" "$GOOD_SHA"
rc=$?
if [ "$rc" -eq 0 ] && [ "$(cat "$D/ckpt.bin")" = "$GOOD" ] && [ ! -s "$TMP/requests.log" ]; then
    note ok "matching existing file is skipped without a request"
else
    note bad "matching existing file is skipped without a request (rc=$rc)"; cat "$TMP/last.err"
fi

echo "pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
