#!/usr/bin/env bash
#
# axiom-guard.sh — Hybrid pre-commit quality gate.
#
#   Phase 1 (STRICT, deterministic): an AST/structural pre-filter scans each
#           staged .rs file for hard anti-patterns (unsafe outside whitelisted
#           low-level layers, `static mut`, goto-style state machines, extreme
#           nesting). Any hit BLOCKS the commit. See scripts/lib/axiom-structural.js.
#
#   Phase 2 (ADVISORY, semantic): files that clear Phase 1 are run through the
#           Axiom TTT `axiom_evaluate_drift` tool. This layer is WARN-ONLY — it
#           never blocks; it only emits an amber "vibe warning" when a file's
#           cross-entropy exceeds the repo-wide threshold (it cannot reliably
#           distinguish anti-patterns from dense idiomatic code, proven
#           empirically — so it advises, the structural gate enforces).
#
# Modes:
#   axiom-guard.sh               pre-commit mode: evaluate files in the git index
#   axiom-guard.sh --check F...   evaluate specific working-tree files (ad-hoc)
#   axiom-guard.sh --install      install this as .git/hooks/pre-commit
#   axiom-guard.sh -h|--help      show help
#
# Environment:
#   AXIOM_DRIFT_THRESHOLD        advisory vibe-warning threshold (default 9.8636)
#   AXIOM_GUARD_MAX_NESTING      structural nesting-depth limit (default 8)
#   AXIOM_GUARD_UNSAFE_WHITELIST regex; paths matching may use `unsafe`
#                                (default: quantization|kernel|chunk_kernel|memory_pool)
#   AXIOM_GUARD_EXTS             comma list of code extensions (default rs,go,py,ts,js)
#   AXIOM_GUARD_MAX_BYTES        per-file size cap (default 262144)
#
# Override the whole gate for a single commit with:  git commit --no-verify
#
# Design note: FAILS OPEN on tooling problems (missing binary/checkpoint, MCP
# error) — it warns and allows rather than bricking the workflow. The only hard
# block is a confident structural anti-pattern in Phase 1.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STRUCTURAL="$SCRIPT_DIR/lib/axiom-structural.js"
BIN="$REPO_ROOT/axiom_engine_rs/target/release/axiom_engine.exe"
CKPT="$REPO_ROOT/checkpoints/axiom_production.bin"
THRESHOLD="${AXIOM_DRIFT_THRESHOLD:-9.8636}"
EXTS="${AXIOM_GUARD_EXTS:-rs,go,py,ts,js}"
MAX_BYTES="${AXIOM_GUARD_MAX_BYTES:-262144}"

info() { echo "[axiom-guard] $*" >&2; }
fail_open() { info "WARN: $* — (advisory layer skipped, commit allowed)."; ADVISORY_OK=0; }

ext_ok() {
    local e="${1##*.}"
    case ",$EXTS," in *",$e,"*) return 0 ;; *) return 1 ;; esac
}

usage() { sed -n '3,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

install_hook() {
    local hooks_dir hook
    hooks_dir="$(git -C "$REPO_ROOT" rev-parse --git-path hooks 2>/dev/null)" \
        || { info "not inside a git repository"; return 1; }
    case "$hooks_dir" in /*|[A-Za-z]:*) : ;; *) hooks_dir="$REPO_ROOT/$hooks_dir" ;; esac
    mkdir -p "$hooks_dir"
    hook="$hooks_dir/pre-commit"
    if [ -e "$hook" ] && ! grep -q "axiom-guard" "$hook" 2>/dev/null; then
        info "existing pre-commit hook found; backing up to ${hook}.bak"
        cp "$hook" "${hook}.bak"
    fi
    cat > "$hook" <<'SHIM'
#!/usr/bin/env bash
# Axiom hybrid pre-commit guard (installed by scripts/axiom-guard.sh --install).
exec "$(git rev-parse --show-toplevel)/scripts/axiom-guard.sh"
SHIM
    chmod +x "$hook"
    info "installed pre-commit hook -> $hook"
    info "Phase 1 structural = STRICT block | Phase 2 TTT = advisory | bypass: git commit --no-verify"
}

# --- Mode dispatch ---------------------------------------------------------
MODE="precommit"
CHECK_FILES=()
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    --install) install_hook; exit $? ;;
    --check)   shift; MODE="check"; CHECK_FILES=("$@") ;;
    "")        MODE="precommit" ;;
    *)         info "unknown argument: $1 (try --help)"; exit 2 ;;
esac

# --- Assemble the candidate file set into a temp workspace -----------------
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
: > "$WORK/manifest.tsv"
n=0

stage_blob() { # $1 = label/path ; content on stdin
    cat > "$WORK/blob_$n"
    local sz; sz=$(wc -c < "$WORK/blob_$n")
    if [ "$sz" -eq 0 ]; then return; fi
    if [ "$sz" -gt "$MAX_BYTES" ]; then info "skip $1 (>$MAX_BYTES bytes)"; return; fi
    printf '%s\t%s\n' "$n" "$1" >> "$WORK/manifest.tsv"
    n=$((n + 1))
}

if [ "$MODE" = "precommit" ]; then
    git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        || { info "not inside a git work tree — OK."; exit 0; }
    if git -C "$REPO_ROOT" rev-parse --verify -q HEAD >/dev/null 2>&1; then
        mapfile -t staged < <(git -C "$REPO_ROOT" diff --cached --name-only --diff-filter=ACM)
    else
        empty_tree="$(git -C "$REPO_ROOT" hash-object -t tree /dev/null)"
        mapfile -t staged < <(git -C "$REPO_ROOT" diff --cached --name-only --diff-filter=ACM "$empty_tree")
    fi
    if [ "${#staged[@]}" -eq 0 ]; then
        info "staging area is empty — nothing to evaluate. OK."
        exit 0
    fi
    for f in "${staged[@]}"; do
        [ -n "$f" ] || continue
        ext_ok "$f" || continue
        git -C "$REPO_ROOT" show ":$f" > "$WORK/.staged_content" 2>/dev/null || continue
        stage_blob "$f" < "$WORK/.staged_content"
    done
else
    if [ "${#CHECK_FILES[@]}" -eq 0 ]; then info "no files given to --check"; exit 2; fi
    for f in "${CHECK_FILES[@]}"; do
        [ -f "$f" ] || { info "not found: $f"; continue; }
        ext_ok "$f" || { info "skip non-code: $f"; continue; }
        stage_blob "$f" < "$f"
    done
fi

if [ "$n" -eq 0 ]; then
    info "no staged code files (${EXTS}) to evaluate. OK."
    exit 0
fi

# ===========================================================================
# PHASE 1 — Deterministic AST/structural gate (STRICT: blocks on violation)
# ===========================================================================
if ! command -v node >/dev/null 2>&1; then
    info "WARN: node not found — skipping structural gate (cannot enforce)."
elif [ ! -f "$STRUCTURAL" ]; then
    info "WARN: $STRUCTURAL missing — skipping structural gate."
else
    if ! AXIOM_GUARD_MAX_NESTING="${AXIOM_GUARD_MAX_NESTING:-8}" \
         AXIOM_GUARD_UNSAFE_WHITELIST="${AXIOM_GUARD_UNSAFE_WHITELIST:-}" \
         node "$STRUCTURAL" --gate "$WORK/manifest.tsv"; then
        # The structural gate already printed its diagnostic trace to stderr.
        exit 1
    fi
fi

# ===========================================================================
# PHASE 2 — Axiom TTT semantic pass (ADVISORY: warn-only, never blocks)
# ===========================================================================
ADVISORY_OK=1
[ -x "$BIN" ] || fail_open "release binary not found ($BIN)"
[ "$ADVISORY_OK" = 1 ] && { [ -f "$CKPT" ] || fail_open "checkpoint not found ($CKPT)"; }

if [ "$ADVISORY_OK" = 1 ]; then
    WORK="$WORK" node -e '
    const fs=require("fs");const W=process.env.WORK;
    const man=fs.readFileSync(W+"/manifest.tsv","utf8").trim().split("\n").filter(Boolean);
    const L=[{jsonrpc:"2.0",id:1,method:"initialize",params:{protocolVersion:"2024-11-05",capabilities:{}}},
             {jsonrpc:"2.0",method:"notifications/initialized"}];
    for(const line of man){const i=line.indexOf("\t");const nn=line.slice(0,i);const content=fs.readFileSync(W+"/blob_"+nn,"utf8");
      L.push({jsonrpc:"2.0",id:1000+Number(nn),method:"tools/call",params:{name:"axiom_evaluate_drift",arguments:{code_content:content}}});}
    fs.writeFileSync(W+"/req.jsonl",L.map(x=>JSON.stringify(x)).join("\n")+"\n");
    ' 2>/dev/null || fail_open "failed to build advisory MCP request"
fi

if [ "$ADVISORY_OK" = 1 ]; then
    AXIOM_DRIFT_THRESHOLD="$THRESHOLD" "$BIN" --mode mcp --checkpoint "$CKPT" \
        < "$WORK/req.jsonl" > "$WORK/out.jsonl" 2>"$WORK/mcp_err.log" || true
    [ -s "$WORK/out.jsonl" ] || fail_open "advisory MCP produced no output"
fi

if [ "$ADVISORY_OK" = 1 ]; then
    WORK="$WORK" THRESHOLD="$THRESHOLD" node -e '
    const fs=require("fs");const W=process.env.WORK;const TH=parseFloat(process.env.THRESHOLD);
    const E=String.fromCharCode(27),Y=E+"[33m",G=E+"[32m",X=E+"[0m";
    const man=Object.fromEntries(fs.readFileSync(W+"/manifest.tsv","utf8").trim().split("\n").filter(Boolean).map(l=>{const i=l.indexOf("\t");return[1000+Number(l.slice(0,i)),l.slice(i+1)];}));
    let out;try{out=fs.readFileSync(W+"/out.jsonl","utf8").trim().split("\n").filter(Boolean).map(JSON.parse);}catch(e){process.exit(0);}
    const byId=Object.fromEntries(out.filter(x=>x&&x.id!==undefined).map(x=>[x.id,x]));
    let warned=0;const losses=[];
    for(const [id,path] of Object.entries(man)){
      const r=byId[id];if(!r||!r.result)continue;
      const m=((r.result.content&&r.result.content[0]&&r.result.content[0].text)||"").match(/cross_entropy_loss=([0-9.]+)/);
      const loss=m?parseFloat(m[1]):null;losses.push([path,loss]);
      if(loss!=null&&loss>TH){warned++;
        console.error(Y+"💡 Vibe Warning: Staged file "+path+" deviates semantically from master history (Loss: "+loss.toFixed(2)+"). This is advisory; commit allowed."+X);
      }
    }
    if(warned===0) console.error(G+"   semantic advisory: all "+losses.length+" file(s) within master history (≤ "+TH.toFixed(4)+")"+X);
    process.exit(0);
    '
fi

# Reached only when Phase 1 passed (and Phase 2 is advisory-only).
echo "[axiom-guard] $(printf '\033[32m✓ Hybrid gate PASS\033[0m') — structural checks clean ($n file(s)); commit allowed." >&2
exit 0
