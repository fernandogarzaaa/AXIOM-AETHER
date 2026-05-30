#!/usr/bin/env bash
#
# axiom-wipe — instant reset button for Axiom's persistent "vibe memory".
#
# Axiom accumulates structural codebase patterns into a master fast-weight
# tensor (axiom_master_vibe.bin) via an EMA merge on every session drop. If a
# messy debugging session poisons that tensor with a toxic gradient, this script
# resets it.
#
# IMPORTANT — the running proxy holds the master vibe IN MEMORY and re-saves it
# on the next session drop. Deleting the file alone is therefore NOT a true
# reset: the in-memory master would just re-create it. So by default this script
# backs up the file, deletes it, AND restarts the proxy so it boots with a fresh
# (empty) master. Use --no-restart to skip the bounce (file-only).
#
# Usage:
#   axiom-wipe            backup + delete the vibe file + restart the proxy
#   axiom-wipe --hard     skip the backup (irreversible delete), then restart
#   axiom-wipe --no-restart   backup + delete file only (in-memory state persists)
#   axiom-wipe --restore [FILE]   restore newest (or named) backup, then restart
#   axiom-wipe --list     list available backups
#   axiom-wipe -h|--help  show this help
#
# Environment:
#   AXIOM_VIBE_PATH   override the vibe file location (default: <repo>/axiom_master_vibe.bin)
#   AXIOM_HOST        proxy host for restart (default: 127.0.0.1)
#   AXIOM_PORT        proxy port for restart (default: 3000)
set -euo pipefail

# --- Resolve paths ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Vibe file: honour AXIOM_VIBE_PATH (absolute or repo-relative), else default.
if [ -n "${AXIOM_VIBE_PATH:-}" ]; then
    case "$AXIOM_VIBE_PATH" in
        /*|[A-Za-z]:*) VIBE_PATH="$AXIOM_VIBE_PATH" ;;   # absolute (unix or windows)
        *)             VIBE_PATH="$REPO_ROOT/$AXIOM_VIBE_PATH" ;;
    esac
else
    VIBE_PATH="$REPO_ROOT/axiom_master_vibe.bin"
fi
BACKUP_DIR="$REPO_ROOT/.vibe_backups"
HOST="${AXIOM_HOST:-127.0.0.1}"
PORT="${AXIOM_PORT:-3000}"

log() { echo "[axiom-wipe] $*"; }
die() { echo "[axiom-wipe] ERROR: $*" >&2; exit 1; }

# --- Proxy control ---------------------------------------------------------
proxy_pid() {
    netstat -ano 2>/dev/null \
        | grep -E ":${PORT}[[:space:]].*LISTENING" \
        | awk '{print $5}' | head -1
}

stop_proxy() {
    local pid
    pid="$(proxy_pid)"
    if [ -z "$pid" ]; then
        log "no proxy listening on :${PORT} (nothing to stop)"
        return 0
    fi
    log "stopping proxy PID=$pid"
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command "Stop-Process -Id $pid -Force" >/dev/null 2>&1 || true
    else
        kill -9 "$pid" 2>/dev/null || true
    fi
    local i
    for i in $(seq 1 20); do
        [ -z "$(proxy_pid)" ] && { log "port :${PORT} freed"; return 0; }
        sleep 0.5
    done
    die "proxy on :${PORT} did not stop"
}

start_proxy() {
    log "starting proxy via start_axiom.sh"
    # Match the production launch: repo-root CWD, no proxy-owned key
    # (subscription passthrough). start_axiom.sh tees its own log.
    ( cd "$REPO_ROOT" && unset ANTHROPIC_API_KEY \
        && nohup bash ./start_axiom.sh >> "$REPO_ROOT/axiom_boot.log" 2>&1 & ) >/dev/null 2>&1
    local i code
    for i in $(seq 1 40); do
        code="$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/v1/models" 2>/dev/null || true)"
        [ "$code" = "200" ] && { log "proxy back up (http 200) after ${i}s"; return 0; }
        sleep 1
    done
    die "proxy did not come back up on :${PORT}; check $REPO_ROOT/axiom_server.log"
}

restart_proxy_if_running() {
    if [ -n "$(proxy_pid)" ]; then
        stop_proxy
        start_proxy
    else
        log "proxy not running; next start will boot with a fresh master vibe"
    fi
}

# --- Backup helpers --------------------------------------------------------
make_backup() {
    [ -f "$VIBE_PATH" ] || { log "no vibe file at $VIBE_PATH (nothing to back up)"; return 1; }
    mkdir -p "$BACKUP_DIR"
    local stamp dest
    stamp="$(date +%Y%m%d-%H%M%S)"
    dest="$BACKUP_DIR/axiom_master_vibe.$stamp.bin"
    cp "$VIBE_PATH" "$dest"
    log "backed up -> $dest ($(wc -c < "$dest") bytes)"
    return 0
}

newest_backup() {
    ls -1t "$BACKUP_DIR"/axiom_master_vibe.*.bin 2>/dev/null | head -1
}

# --- Subcommands -----------------------------------------------------------
cmd_list() {
    if [ -d "$BACKUP_DIR" ] && ls "$BACKUP_DIR"/axiom_master_vibe.*.bin >/dev/null 2>&1; then
        log "backups in $BACKUP_DIR:"
        ls -lt "$BACKUP_DIR"/axiom_master_vibe.*.bin | awk '{print "  "$0}'
    else
        log "no backups in $BACKUP_DIR"
    fi
    if [ -f "$VIBE_PATH" ]; then
        log "live vibe: $VIBE_PATH"
        log "  size: $(wc -c < "$VIBE_PATH") bytes | modified: $(date -r "$VIBE_PATH" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo '?')"
    else
        log "live vibe: none at $VIBE_PATH"
    fi
}

cmd_restore() {
    local src="${1:-}"
    [ -z "$src" ] && src="$(newest_backup)"
    [ -z "$src" ] && die "no backup to restore (looked in $BACKUP_DIR)"
    [ -f "$src" ] || die "backup not found: $src"
    cp "$src" "$VIBE_PATH"
    log "restored $src -> $VIBE_PATH"
    restart_proxy_if_running
    log "restore complete."
}

cmd_wipe() {
    local hard="$1" do_restart="$2"
    if [ "$hard" = "1" ]; then
        if [ -f "$VIBE_PATH" ]; then
            rm -f "$VIBE_PATH"
            log "HARD wipe — deleted $VIBE_PATH (no backup)"
        else
            log "no live vibe file at $VIBE_PATH"
        fi
    else
        make_backup || true
        if [ -f "$VIBE_PATH" ]; then
            rm -f "$VIBE_PATH"
            log "deleted $VIBE_PATH"
        fi
    fi

    if [ "$do_restart" = "1" ]; then
        restart_proxy_if_running
    else
        log "--no-restart: file cleared, but the running proxy still holds the"
        log "old master in memory and will re-save it on the next session drop."
        log "Restart the proxy for a full reset."
    fi
    log "done."
}

usage() {
    sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# --- Arg parsing -----------------------------------------------------------
# Default (no flags): backup + delete + restart. Flags may be combined,
# e.g. `axiom-wipe --hard --no-restart`. Subcommands run standalone.
HARD=0
RESTART=1
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)     usage; exit 0 ;;
        --list)        cmd_list; exit 0 ;;
        --restore)     shift; cmd_restore "${1:-}"; exit 0 ;;
        --hard)        HARD=1 ;;
        --no-restart)  RESTART=0 ;;
        *)             die "unknown argument: $1 (try --help)" ;;
    esac
    shift
done

cmd_wipe "$HARD" "$RESTART"
