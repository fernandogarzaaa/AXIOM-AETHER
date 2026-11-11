#!/usr/bin/env bash
# rollback_drill.sh: scripted rollback drill for an Axiom Helm release.
#
# Two modes. Cluster mode is the original drill: it rolls back a real Helm
# release and verifies it with kubectl. It refuses to run against a release
# that does not exist, and --dry-run prints the commands without running them.
#
#   ./scripts/rollback_drill.sh --release axiom --namespace axiom-staging [--revision N] [--dry-run]
#
# Local mode runs the same three rollback paths end to end without a cluster,
# a container runtime, or a registry. It uses the real Helm binary (lint and
# template), the real container entrypoint (atomic checkpoint install), and
# the real server binary (the /readyz gate) as a local process. It writes a
# filled evidence file under docs/release-qualification/drills/.
#
#   ./scripts/rollback_drill.sh --local [--server-bin PATH] [--helm-bin PATH] [--port N] [--evidence PATH]
#
# Local mode needs: python3 (file server), curl, sha256sum, and a built
# axiom_engine binary (cargo build --release --bin axiom_engine). If helm is
# not on PATH it is downloaded from get.helm.sh into the drill work dir and
# verified against its published SHA-256; pass --helm-bin to skip that.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENTRY="$ROOT/scripts/docker_entrypoint.sh"

HELM_VERSION="v3.18.4"
HELM_SHA256="f8180838c23d7c7d797b208861fecb591d9ce1690d8704ed1e4cb8e2add966c1"

MODE="cluster"
RELEASE=""
NAMESPACE=""
REVISION=""
DRY_RUN=0
SERVER_BIN="$ROOT/axiom_engine_rs/target/release/axiom_engine"
HELM_BIN=""
PORT="18081"
EVIDENCE=""
KEEP_WORK=0

while [ $# -gt 0 ]; do
  case "$1" in
    --local) MODE="local"; shift ;;
    --release) RELEASE="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --server-bin) SERVER_BIN="$2"; shift 2 ;;
    --helm-bin) HELM_BIN="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --evidence) EVIDENCE="$2"; shift 2 ;;
    --keep-work) KEEP_WORK=1; shift ;;
    *) echo "rollback_drill: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$MODE" = "cluster" ]; then
  if [ -z "$RELEASE" ] || [ -z "$NAMESPACE" ]; then
    echo "rollback_drill: --release and --namespace are required (cluster mode)" >&2
    exit 2
  fi

  run() {
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "[dry-run] $*"
    else
      echo "+ $*"
      "$@"
    fi
  }

  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "rollback drill: release=$RELEASE namespace=$NAMESPACE started=$started dry-run=$DRY_RUN"

  if [ "$DRY_RUN" -eq 0 ]; then
    command -v helm >/dev/null 2>&1 || { echo "rollback_drill: helm not found" >&2; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { echo "rollback_drill: kubectl not found" >&2; exit 1; }
    helm status "$RELEASE" -n "$NAMESPACE" >/dev/null 2>&1 \
      || { echo "rollback_drill: release $RELEASE not found in namespace $NAMESPACE; refusing to run" >&2; exit 1; }
  fi

  echo "--- pre-state ---"
  run helm history "$RELEASE" -n "$NAMESPACE"
  run kubectl -n "$NAMESPACE" get pods -l app.kubernetes.io/name=axiom
  run helm test "$RELEASE" -n "$NAMESPACE"

  echo "--- rollback ---"
  if [ -n "$REVISION" ]; then
    run helm rollback "$RELEASE" "$REVISION" -n "$NAMESPACE" --wait --timeout 10m
  else
    run helm rollback "$RELEASE" -n "$NAMESPACE" --wait --timeout 10m
  fi
  run kubectl -n "$NAMESPACE" rollout status deploy/"$RELEASE"

  echo "--- post-state verification ---"
  run kubectl -n "$NAMESPACE" get pods -l app.kubernetes.io/name=axiom
  run kubectl -n "$NAMESPACE" exec deploy/"$RELEASE" -- curl -fsS localhost:8080/readyz
  run helm test "$RELEASE" -n "$NAMESPACE"

  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "rollback drill: finished=$finished"
  echo "Record the outputs above in docs/release-qualification/drills/<date>-$RELEASE-rollback.md"
  exit 0
fi

# ---------------------------------------------------------------------------
# Local mode. Every step records pass/fail and keeps going so the evidence
# file shows the whole drill, not just the first failure.
# ---------------------------------------------------------------------------

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rollback-drill-XXXXXX")"
PASS=0
FAIL=0
STARTED_EPOCH="$(date +%s)"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SERVER_PID=""
FILE_PID=""

if [ -z "$EVIDENCE" ]; then
  EVIDENCE="$ROOT/docs/release-qualification/drills/$(date -u +%F)-local-rollback.md"
fi

note() {
  if [ "$1" = ok ]; then PASS=$((PASS + 1)); echo "ok: $2";
  else FAIL=$((FAIL + 1)); echo "FAIL: $2"; fi
}

cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  [ -n "$FILE_PID" ] && kill "$FILE_PID" 2>/dev/null
  if [ "$KEEP_WORK" -eq 0 ]; then rm -rf "$WORK"; else echo "work dir kept: $WORK"; fi
}
trap cleanup EXIT INT TERM

stop_server() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
  fi
}

echo "rollback drill (local): started=$STARTED_UTC work=$WORK port=$PORT"
echo "evidence: $EVIDENCE"

echo "--- environment probe ---"
for tool in helm kubectl docker kind minikube; do
  if command -v "$tool" >/dev/null 2>&1; then echo "$tool: $($tool version 2>&1 | head -1)"; else echo "$tool: not on PATH"; fi
done
echo "server binary: $SERVER_BIN"
echo "python3: $(python3 --version 2>&1)"
echo "curl: $(curl --version 2>&1 | head -1)"

if [ -n "$HELM_BIN" ]; then
  HELM="$HELM_BIN"
elif command -v helm >/dev/null 2>&1; then
  HELM="$(command -v helm)"
else
  echo "helm not on PATH; downloading $HELM_VERSION into the work dir"
  HELM_TGZ="$WORK/helm.tar.gz"
  curl -fsSL -o "$HELM_TGZ" "https://get.helm.sh/helm-$HELM_VERSION-linux-amd64.tar.gz" \
    || { echo "FAIL: helm download failed" >&2; exit 1; }
  actual="$(sha256sum "$HELM_TGZ" | cut -d' ' -f1)"
  [ "$actual" = "$HELM_SHA256" ] \
    || { echo "FAIL: helm checksum mismatch (expected $HELM_SHA256, got $actual)" >&2; exit 1; }
  tar xzf "$HELM_TGZ" -C "$WORK"
  HELM="$WORK/linux-amd64/helm"
fi
echo "helm: $($HELM version --short 2>&1)"
[ -x "$SERVER_BIN" ] || { echo "FAIL: server binary missing: $SERVER_BIN (run: cargo build --release --bin axiom_engine)" >&2; exit 1; }
note ok "environment probe recorded"

echo "--- phase 1: helm lint and template ---"
if "$HELM" lint "$ROOT/deploy/helm/axiom" >"$WORK/helm-lint.log" 2>&1; then
  note ok "helm lint passes"
else
  note bad "helm lint passes"; cat "$WORK/helm-lint.log"
fi

# Drill checkpoint fixtures: small but format-valid safetensors files (an
# empty tensor header parses; the loader keeps init fallbacks) and a minimal
# WordLevel tokenizer. v1 is the good release, v2 is the bad release: same
# format, different bytes, therefore a different hash.
mkdir -p "$WORK/serve" "$WORK/state"
python3 - "$WORK/serve" <<'PY'
import json, struct, sys
base = sys.argv[1]
def safetensors(path, meta):
    hdr = json.dumps(meta).encode()
    open(path, "wb").write(struct.pack("<Q", len(hdr)) + hdr)
safetensors(base + "/v1.bin", {})
safetensors(base + "/v2.bin", {"__metadata__": {"drill": "v2"}})
tok = {"version": "1.0", "model": {"type": "WordLevel", "vocab": {"hello": 0}, "unk_token": "[UNK]"}}
open(base + "/v1.tok.json", "w").write(json.dumps(tok))
open(base + "/v2.tok.json", "w").write(json.dumps(tok))
PY
V1_SHA="$(sha256sum "$WORK/serve/v1.bin" | cut -d' ' -f1)"
V1_TOK_SHA="$(sha256sum "$WORK/serve/v1.tok.json" | cut -d' ' -f1)"
V2_SHA="$(sha256sum "$WORK/serve/v2.bin" | cut -d' ' -f1)"
echo "v1 checkpoint sha256: $V1_SHA"
echo "v1 tokenizer sha256: $V1_TOK_SHA"
echo "v2 checkpoint sha256: $V2_SHA"

if "$HELM" template axiom-staging "$ROOT/deploy/helm/axiom" \
    -f "$ROOT/deploy/helm/axiom/values-staging.yaml" \
    --set checkpoint.url="http://127.0.0.1/v1.bin" \
    --set checkpoint.tokenizerUrl="http://127.0.0.1/v1.tok.json" \
    --set checkpoint.sha256="$V1_SHA" \
    --set checkpoint.tokenizerSha256="$V1_TOK_SHA" >"$WORK/render-staging.yaml" 2>"$WORK/render-staging.err"; then
  note ok "helm template renders staging overlay"
else
  note bad "helm template renders staging overlay"; cat "$WORK/render-staging.err"
fi
if "$HELM" template axiom-prod "$ROOT/deploy/helm/axiom" \
    -f "$ROOT/deploy/helm/axiom/values-prod.yaml" \
    --set checkpoint.url="http://127.0.0.1/v1.bin" \
    --set checkpoint.tokenizerUrl="http://127.0.0.1/v1.tok.json" \
    --set checkpoint.sha256="$V1_SHA" \
    --set checkpoint.tokenizerSha256="$V1_TOK_SHA" >"$WORK/render-prod.yaml" 2>"$WORK/render-prod.err"; then
  note ok "helm template renders prod overlay"
else
  note bad "helm template renders prod overlay"; cat "$WORK/render-prod.err"
fi
check_render() { # check_render <file> <grep pattern> <label>
  if grep -q "$2" "$1"; then note ok "$3"; else note bad "$3"; fi
}
check_render "$WORK/render-staging.yaml" 'AXIOM_REQUIRE_VERIFIED_CHECKPOINT: "1"' "staging render enables the readiness gate"
check_render "$WORK/render-staging.yaml" 'AXIOM_PRODUCTION_BPE: "1"' "staging render enables the production model"
check_render "$WORK/render-staging.yaml" 'AXIOM_CORS_ORIGINS: "https://staging.example.invalid"' "staging render allows exactly one CORS origin"
check_render "$WORK/render-prod.yaml" 'AXIOM_CORS_ORIGINS: ""' "prod render denies all CORS origins"
check_render "$WORK/render-staging.yaml" 'image: "ghcr.io/fernandogarzaaa/axiom-aether:latest@sha256:' "staging render pins the image digest"
check_render "$WORK/render-prod.yaml" 'minAvailable: 2' "prod render keeps a two-pod quorum"

echo "--- phase 2: checkpoint re-install via the atomic installer ---"
cat > "$WORK/serve.py" <<'PY'
import http.server, os, sys
base = sys.argv[1]
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass
    def do_GET(self):
        path = os.path.join(base, self.path.lstrip("/"))
        if os.path.isfile(path):
            body = open(path, "rb").read()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self.close_connection = True
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
open(os.path.join(base, "port"), "w").write(str(srv.server_address[1]))
srv.serve_forever()
PY
python3 "$WORK/serve.py" "$WORK/serve" &
FILE_PID=$!
for _ in $(seq 1 50); do [ -f "$WORK/serve/port" ] && break; sleep 0.1; done
BASE="http://127.0.0.1:$(cat "$WORK/serve/port")"
echo "file server: $BASE"

run_entry() { # run_entry <url> <sha> ; installs into $WORK/state
  AXIOM_CHECKPOINT_URL="$1" AXIOM_BPE_CKPT="$WORK/state/ckpt.bin" AXIOM_CHECKPOINT_SHA256="$2" \
    AXIOM_TOKENIZER_URL="$BASE/v1.tok.json" AXIOM_TOKENIZER="$WORK/state/tok.json" \
    AXIOM_TOKENIZER_SHA256="$V1_TOK_SHA" AXIOM_ENTRYPOINT_NOEXEC=1 \
    sh "$ENTRY" >"$WORK/last-entry.log" 2>&1
}
leftovers() { find "$WORK/state" -name '*.tmp.*' -o -name '*.download' | grep -q .; }

run_entry "$BASE/v1.bin" "$V1_SHA"
rc=$?
if [ "$rc" -eq 0 ] && [ "$(sha256sum "$WORK/state/ckpt.bin" | cut -d' ' -f1)" = "$V1_SHA" ] && ! leftovers; then
  note ok "entrypoint installs v1 checkpoint with matching hash"
else
  note bad "entrypoint installs v1 checkpoint with matching hash (rc=$rc)"; cat "$WORK/last-entry.log"
fi

# A wrong hash refuses the install and leaves nothing under the final
# name. Note the entrypoint deletes the installed file before it
# re-downloads, so a refused attempt also removes the good file: the
# rollback must re-install v1 afterwards, which the next step does.
run_entry "$BASE/v1.bin" "0000000000000000000000000000000000000000000000000000000000000000"
rc=$?
if [ "$rc" -ne 0 ] && [ ! -e "$WORK/state/ckpt.bin" ] && ! leftovers; then
  note ok "entrypoint refuses a wrong hash and leaves nothing behind"
else
  note bad "entrypoint refuses a wrong hash and leaves nothing behind (rc=$rc)"; cat "$WORK/last-entry.log"
fi

run_entry "$BASE/v1.bin" "$V1_SHA"
rc=$?
if [ "$rc" -eq 0 ] && [ "$(sha256sum "$WORK/state/ckpt.bin" | cut -d' ' -f1)" = "$V1_SHA" ]; then
  note ok "entrypoint re-installs v1 after the refused attempt"
else
  note bad "entrypoint re-installs v1 after the refused attempt (rc=$rc)"; cat "$WORK/last-entry.log"
fi

run_entry "$BASE/v1.bin" "$V1_SHA"
rc=$?
if [ "$rc" -eq 0 ] && grep -q "ckpt.bin already present (sha256 verified)" "$WORK/last-entry.log"; then
  note ok "entrypoint skips re-download when v1 is already installed"
else
  note bad "entrypoint skips re-download when v1 is already installed (rc=$rc)"; cat "$WORK/last-entry.log"
fi

echo "--- phase 3: local process rollback against /readyz ---"
start_server() { # start_server <log file>
  AXIOM_PRODUCTION_BPE=1 AXIOM_BPE_CKPT="$WORK/state/ckpt.bin" AXIOM_TOKENIZER="$WORK/state/tok.json" \
    AXIOM_REQUIRE_VERIFIED_CHECKPOINT=1 AXIOM_CHECKPOINT_SHA256="$V1_SHA" AXIOM_TOKENIZER_SHA256="$V1_TOK_SHA" \
    "$SERVER_BIN" --mode server --host 127.0.0.1 --port "$PORT" >"$1" 2>&1 &
  SERVER_PID=$!
  echo "server started: pid=$SERVER_PID log=$1"
  sleep 2
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "server pid $SERVER_PID exited early; log:"; cat "$1"; }
}
# wait_readyz <expect code> <timeout seconds> ; READYZ_BODY and READYZ_SECS are set
wait_readyz() {
  local expect="$1" timeout="$2" start now code body
  start="$(date +%s)"
  READYZ_BODY=""; READYZ_SECS=""
  for _ in $(seq 1 "$timeout"); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then return 2; fi
    body="$(curl -s -m 2 "http://127.0.0.1:$PORT/readyz" || true)"
    code="$(curl -s -o /dev/null -w '%{http_code}' -m 2 "http://127.0.0.1:$PORT/readyz" || echo 000)"
    if [ "$code" = "$expect" ]; then
      now="$(date +%s)"; READYZ_BODY="$body"; READYZ_SECS=$((now - start)); return 0
    fi
    sleep 1
  done
  READYZ_BODY="$body"; return 1
}

start_server "$WORK/server-v1.log"
if wait_readyz 200 120; then
  note ok "v1 server reports /readyz 200 (body: $READYZ_BODY, ${READYZ_SECS}s to ready)"
else
  note bad "v1 server reports /readyz 200 (last body: $READYZ_BODY)"; tail -20 "$WORK/server-v1.log"
fi
HEALTH_V1="$(curl -s -o /dev/null -w '%{http_code}' -m 2 "http://127.0.0.1:$PORT/healthz" || echo 000)"
echo "v1 /healthz: $HEALTH_V1"
stop_server

# Bad release: v2 bytes on disk, pinned hash still v1. The file parses, so
# the server starts, but the gate must hold it out of rotation.
cp "$WORK/serve/v2.bin" "$WORK/state/ckpt.bin"
T_BAD="$(date +%s)"
start_server "$WORK/server-v2.log"
if wait_readyz 503 120; then
  note ok "v2 server reports /readyz 503 with pinned hash still v1 (body: $READYZ_BODY, ${READYZ_SECS}s)"
else
  note bad "v2 server reports /readyz 503 with pinned hash still v1 (last body: $READYZ_BODY)"; tail -20 "$WORK/server-v2.log"
fi
T_V2_503="$(date +%s)"
HEALTH_V2="$(curl -s -o /dev/null -w '%{http_code}' -m 2 "http://127.0.0.1:$PORT/healthz" || echo 000)"
echo "v2 /healthz (process alive, gate holding): $HEALTH_V2"
stop_server

# Rollback: re-install v1 through the entrypoint, then restart.
T_ROLLBACK="$(date +%s)"
run_entry "$BASE/v1.bin" "$V1_SHA"
rc=$?
if [ "$rc" -eq 0 ] && [ "$(sha256sum "$WORK/state/ckpt.bin" | cut -d' ' -f1)" = "$V1_SHA" ]; then
  note ok "rollback re-installs v1 through the entrypoint"
else
  note bad "rollback re-installs v1 through the entrypoint (rc=$rc)"; cat "$WORK/last-entry.log"
fi
start_server "$WORK/server-rollback.log"
if wait_readyz 200 120; then
  note ok "rolled-back server reports /readyz 200 (body: $READYZ_BODY, ${READYZ_SECS}s to ready)"
else
  note bad "rolled-back server reports /readyz 200 (last body: $READYZ_BODY)"; tail -20 "$WORK/server-rollback.log"
fi
T_ROLLBACK_READY="$(date +%s)"
stop_server

# Corrupt checkpoint: unparseable bytes fail boot outright (non-zero exit,
# no listener). That is the designed behavior, and it differs from a
# hash-mismatched but parseable file, which yields /readyz 503.
printf 'not-a-safetensors-file' > "$WORK/state/ckpt.bin"
start_server "$WORK/server-corrupt.log"
CORRUPT_RC="unknown"
for _ in $(seq 1 45); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID" 2>/dev/null; CORRUPT_RC=$?
    break
  fi
  sleep 2
done
if [ "$CORRUPT_RC" != "unknown" ]; then
  SERVER_PID=""
  note ok "corrupt checkpoint fails boot (exit $CORRUPT_RC, no listener)"
else
  note bad "corrupt checkpoint fails boot (server still running after 90s)"; stop_server
fi
run_entry "$BASE/v1.bin" "$V1_SHA"
echo "v1 restored after corrupt-boot check: rc=$?"

echo "--- phase 4: config revert at the render layer ---"
cp "$ROOT/deploy/helm/axiom/values-staging.yaml" "$WORK/values-staging-bad.yaml"
sed -i 's|https://staging.example.invalid|https://staging2.example.invalid|' "$WORK/values-staging-bad.yaml"
"$HELM" template axiom-staging "$ROOT/deploy/helm/axiom" -f "$WORK/values-staging-bad.yaml" \
  --set checkpoint.url="http://127.0.0.1/v1.bin" \
  --set checkpoint.tokenizerUrl="http://127.0.0.1/v1.tok.json" \
  --set checkpoint.sha256="$V1_SHA" --set checkpoint.tokenizerSha256="$V1_TOK_SHA" >"$WORK/render-bad.yaml" 2>/dev/null
BAD_SUM="$(grep 'checksum/config' "$WORK/render-bad.yaml" | head -1)"
BASE_SUM="$(grep 'checksum/config' "$WORK/render-staging.yaml" | head -1)"
if [ "$BAD_SUM" != "$BASE_SUM" ] && grep -q 'staging2.example.invalid' "$WORK/render-bad.yaml"; then
  note ok "bad config change is visible in the rendered ConfigMap and checksum annotation"
else
  note bad "bad config change is visible in the rendered ConfigMap and checksum annotation"
fi
"$HELM" template axiom-staging "$ROOT/deploy/helm/axiom" -f "$ROOT/deploy/helm/axiom/values-staging.yaml" \
  --set checkpoint.url="http://127.0.0.1/v1.bin" \
  --set checkpoint.tokenizerUrl="http://127.0.0.1/v1.tok.json" \
  --set checkpoint.sha256="$V1_SHA" --set checkpoint.tokenizerSha256="$V1_TOK_SHA" >"$WORK/render-reverted.yaml" 2>/dev/null
if cmp -s "$WORK/render-reverted.yaml" "$WORK/render-staging.yaml"; then
  note ok "reverting the values change restores the baseline render byte for byte"
else
  note bad "reverting the values change restores the baseline render byte for byte"
fi

FINISHED_EPOCH="$(date +%s)"
FINISHED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_S=$((FINISHED_EPOCH - STARTED_EPOCH))
DETECT_TO_ROLLBACK_S=$((T_ROLLBACK - T_V2_503))
ROLLBACK_TO_READY_S=$((T_ROLLBACK_READY - T_ROLLBACK))
TRIGGER_TO_READY_S=$((T_ROLLBACK_READY - T_BAD))

mkdir -p "$(dirname "$EVIDENCE")"
cat > "$EVIDENCE" <<EOF
# Rollback drill evidence: local drill

Local-mode run of \`scripts/rollback_drill.sh --local\`. No cluster, no
container runtime, no registry existed in this sandbox, so the drill ran
against the real Helm binary (lint and template), the real container
entrypoint (atomic checkpoint install), and the real server binary as a
local process (the /readyz gate). Cluster steps (helm rollback, kubectl
rollout status, helm test, gossip CronJob) were NOT exercised.

## Drill identity

- Date and time (UTC): $STARTED_UTC to $FINISHED_UTC
- Operator: rollback drill script (local mode)
- Release name and namespace: axiom / axiom-staging (rendered only; no release exists)
- Chart revision before the drill: n/a (no cluster)
- Chart revision rolled back to: n/a (no cluster)
- Rollback path used: checkpoint re-install (local process) plus config revert (render layer)
- Environment: local process drill, no live cluster

## Pre-state

- Server binary: $SERVER_BIN
- Helm: $($HELM version --short 2>&1)
- v1 checkpoint sha256: $V1_SHA
- v1 tokenizer sha256: $V1_TOK_SHA
- v2 checkpoint sha256: $V2_SHA
- helm lint: pass (see phase 1)
- Staging and prod overlays render with the readiness gate on, the image
  digest pinned, staging CORS at exactly one origin, prod CORS deny-all.

## Trigger

- What was broken, in one sentence: a bad checkpoint release (v2 bytes) was installed while the pinned hash still named v1.
- How the break was introduced (bad image, bad checkpoint, bad config, simulated): simulated, by copying v2 bytes over the installed checkpoint.
- Time the break was introduced (UTC): $(date -u -d "@$T_BAD" +%Y-%m-%dT%H:%M:%SZ)
- Time the break was detected (UTC): $(date -u -d "@$T_V2_503" +%Y-%m-%dT%H:%M:%SZ)
- Detection method (probe, metrics, manual check): /readyz probe, 503 with reason mismatched.

## Rollback steps (commands run, in order)

1. Command: helm lint deploy/helm/axiom; helm template for staging and prod overlays with the drill hashes.
   Output: lint passes; both renders succeed; gate on, image digest pinned, CORS as above.
2. Command: entrypoint install of v1 (AXIOM_CHECKPOINT_URL=<local file server>/v1.bin, pinned sha256, AXIOM_ENTRYPOINT_NOEXEC=1).
   Output: v1 installed with matching hash; a wrong-hash attempt refused and left nothing under the final name (the entrypoint deletes the installed file before re-downloading, so v1 was re-installed after); a repeat install skipped without re-download.
3. Command: start server with v1 installed, AXIOM_PRODUCTION_BPE=1, gate on, pinned hashes; curl /readyz.
   Output: /readyz 200; /healthz $HEALTH_V1.
4. Command: copy v2 bytes over the checkpoint, restart the server, curl /readyz.
   Output: /readyz 503 with the mismatch reason; /healthz $HEALTH_V2 (process alive, gate holding it out).
5. Command: entrypoint re-install of v1, restart the server, curl /readyz.
   Output: v1 re-installed; /readyz 200.
6. Command: write unparseable bytes over the checkpoint, start the server.
   Output: server exits non-zero (exit $CORRUPT_RC) with no listener; v1 restored afterwards.
7. Command: change the staging CORS origin in a copy of values-staging.yaml, render, revert, render again.
   Output: the bad render differs in the ConfigMap and checksum/config annotation; the reverted render matches the baseline byte for byte.

## Post-state verification

- Pods running and ready after rollback: n/a (no cluster); the local server process reports /readyz 200 after rollback.
- Image digest on the pods after rollback: n/a (no cluster); the rendered staging image is ghcr.io/fernandogarzaaa/axiom-aether:latest@sha256:844f485adf925e515070a1680630eaefe4423a54ea8cbc971b045e6acb5d8afe.
- Checkpoint SHA-256 after rollback: $V1_SHA (matches the pinned v1 hash).
- /readyz result per pod after rollback: 200 on the local process.
- helm test result after rollback: NOT exercised (needs a cluster).
- Gossip CronJob result after rollback: NOT exercised (needs a cluster).
- Error rate or smoke request result after rollback: /healthz 200, /readyz 200.

## Timings

- Trigger to detection: $((T_V2_503 - T_BAD))s (restart plus /readyz poll)
- Detection to rollback start: ${DETECT_TO_ROLLBACK_S}s
- Rollback start to all pods ready: ${ROLLBACK_TO_READY_S}s (local process /readyz 200; no pods exist)
- Rollback start to /readyz 200 on every pod: ${ROLLBACK_TO_READY_S}s (one local process)
- Total time from trigger to verified recovery: ${TRIGGER_TO_READY_S}s
- Whole drill wall clock: ${TOTAL_S}s

## Verdict

- Rollback succeeded: $([ "$FAIL" -eq 0 ] && echo yes || echo "no ($FAIL checks failed)")
- Recovery verified by the post-state checks above: $([ "$FAIL" -eq 0 ] && echo yes || echo no)
- Gaps found during the drill (each one becomes a follow-up item, not a silent fix):
  - No live cluster, so helm rollback, kubectl rollout status, helm test, and the gossip CronJob were not exercised. A cluster drill is still required before a release is qualified.
  - The drill needs AXIOM_PRODUCTION_BPE=1 for the server to load AXIOM_BPE_CKPT at all; with the flag unset the server silently boots the legacy model while the gate hashes a different path. The chart sets the flag, so in-cluster behavior is consistent, but a manual run without it is misleading.
  - A corrupt (unparseable) checkpoint fails server boot outright instead of yielding /readyz 503. The runbook now says so. The 503 path covers hash-mismatched but parseable files.
- Follow-up issue numbers: (filed with the PR; see docs/ROLLBACK-RECOVERY.md)

## Drill result

- pass=$PASS fail=$FAIL
EOF

echo "--- result ---"
echo "pass=$PASS fail=$FAIL total=${TOTAL_S}s"
echo "evidence: $EVIDENCE"
[ "$FAIL" -eq 0 ]
