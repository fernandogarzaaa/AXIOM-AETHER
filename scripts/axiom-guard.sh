#!/usr/bin/env bash
#
# axiom-guard.sh — Axiom TTT pre-commit quality gate.
#
# Evaluates staged code against the engine's persistent "codebase DNA" using the
# native MCP tool `axiom_evaluate_drift`. Files whose cross-entropy loss exceeds
# the drift threshold are flagged. WARN-ONLY by default (does not block);
# set AXIOM_GUARD_STRICT=1 to turn it into a hard commit gate.
#
# Modes:
#   axiom-guard.sh               pre-commit mode: evaluate files in the git index
#   axiom-guard.sh --check F...   evaluate specific working-tree files (ad-hoc)
#   axiom-guard.sh --install      install this as .git/hooks/pre-commit
#   axiom-guard.sh -h|--help      show help
#
# Environment:
#   AXIOM_GUARD_STRICT      1 = BLOCK on drift; 0 = warn-only advisory (default 0)
#   AXIOM_DRIFT_THRESHOLD   flag above this cross-entropy loss (default 9.8636 =
#                           repo-calibrated mu+3sigma over src/; the earlier
#                           single-file 7.4736 was unrepresentative — see below)
#   AXIOM_GUARD_EXTS        comma list of code extensions (default rs,go,py,ts,js)
#   AXIOM_GUARD_MAX_BYTES   per-file size cap (default 262144)
#
# Override the gate for a single commit with:  git commit --no-verify
#
# Design notes:
# * FAILS OPEN. If the engine binary/checkpoint is missing or the MCP call
#   errors, it warns and ALLOWS the commit rather than bricking the workflow.
# * WARN-ONLY BY DEFAULT. Empirically, per-file absolute cross-entropy from the
#   current small model (d_model=64, vocab=256, hash tokenizer) is NOT a reliable
#   discriminator of architectural quality: legitimate idiomatic files span
#   ~3.1-8.0 and a hand-built anti-pattern landed mid-distribution. So a hard
#   block on absolute CE produces false positives. The gate therefore only warns
#   unless AXIOM_GUARD_STRICT=1, and the default threshold is the repo's own
#   mu+3sigma so even strict mode passes current clean code.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$REPO_ROOT/axiom_engine_rs/target/release/axiom_engine.exe"
CKPT="$REPO_ROOT/checkpoints/axiom_production.bin"
THRESHOLD="${AXIOM_DRIFT_THRESHOLD:-9.8636}"
STRICT="${AXIOM_GUARD_STRICT:-0}"
EXTS="${AXIOM_GUARD_EXTS:-rs,go,py,ts,js}"
MAX_BYTES="${AXIOM_GUARD_MAX_BYTES:-262144}"

info() { echo "[axiom-guard] $*" >&2; }
fail_open() { info "WARN: $* — allowing commit (fail-open)."; exit 0; }

ext_ok() {
    local e="${1##*.}"
    case ",$EXTS," in *",$e,"*) return 0 ;; *) return 1 ;; esac
}

usage() { sed -n '3,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

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
# Axiom pre-commit guard (installed by scripts/axiom-guard.sh --install).
exec "$(git rev-parse --show-toplevel)/scripts/axiom-guard.sh"
SHIM
    chmod +x "$hook"
    info "installed pre-commit hook -> $hook"
    info "mode: $([ "$STRICT" = 1 ] && echo STRICT-block || echo warn-only)  | bypass any commit with: git commit --no-verify"
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
        || fail_open "not inside a git work tree"
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
        # Redirection (not a pipe) so stage_blob runs in THIS shell and its
        # counter increments propagate; a pipe would subshell it and lose `n`.
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

# --- Preflight the engine (fail-open if unavailable) -----------------------
[ -x "$BIN" ] || fail_open "release binary not found ($BIN); build with: cargo build --release"
[ -f "$CKPT" ] || fail_open "checkpoint not found ($CKPT)"

# --- Build the JSON-RPC batch (one evaluate_drift per file) ----------------
WORK="$WORK" node -e '
const fs=require("fs");const W=process.env.WORK;
const man=fs.readFileSync(W+"/manifest.tsv","utf8").trim().split("\n").filter(Boolean);
const L=[{jsonrpc:"2.0",id:1,method:"initialize",params:{protocolVersion:"2024-11-05",capabilities:{}}},
         {jsonrpc:"2.0",method:"notifications/initialized"}];
for(const line of man){const i=line.indexOf("\t");const nn=line.slice(0,i);const content=fs.readFileSync(W+"/blob_"+nn,"utf8");
  L.push({jsonrpc:"2.0",id:1000+Number(nn),method:"tools/call",params:{name:"axiom_evaluate_drift",arguments:{code_content:content}}});}
fs.writeFileSync(W+"/req.jsonl",L.map(x=>JSON.stringify(x)).join("\n")+"\n");
' 2>/dev/null || fail_open "failed to build MCP request payload"

# --- Drive the MCP server (threshold drives the isError flag) --------------
AXIOM_DRIFT_THRESHOLD="$THRESHOLD" "$BIN" --mode mcp --checkpoint "$CKPT" \
    < "$WORK/req.jsonl" > "$WORK/out.jsonl" 2>"$WORK/mcp_err.log" || true
if [ ! -s "$WORK/out.jsonl" ]; then
    sed 's/^/[axiom-guard:mcp] /' "$WORK/mcp_err.log" >&2 2>/dev/null || true
    fail_open "MCP evaluation produced no output"
fi

# --- Parse responses, render verdict, set exit code ------------------------
WORK="$WORK" THRESHOLD="$THRESHOLD" STRICT="$STRICT" node -e '
const fs=require("fs");const W=process.env.WORK;const TH=parseFloat(process.env.THRESHOLD);
const STRICT=process.env.STRICT==="1";
const E=String.fromCharCode(27);const G=E+"[32m",R=E+"[31m",Y=E+"[33m",X=E+"[0m";
const man=Object.fromEntries(fs.readFileSync(W+"/manifest.tsv","utf8").trim().split("\n").filter(Boolean).map(l=>{const i=l.indexOf("\t");return[1000+Number(l.slice(0,i)),l.slice(i+1)];}));
let out;
try{out=fs.readFileSync(W+"/out.jsonl","utf8").trim().split("\n").filter(Boolean).map(JSON.parse);}
catch(e){console.error("[axiom-guard] WARN: unparseable MCP output — allowing commit.");process.exit(0);}
const byId=Object.fromEntries(out.filter(x=>x&&x.id!==undefined).map(x=>[x.id,x]));
const rows=[];let viol=0,evaluated=0;
for(const [id,path] of Object.entries(man)){
  const r=byId[id];
  if(!r||!r.result){rows.push({path,flagged:false,loss:null,note:"no-response(skipped)"});continue;}
  evaluated++;
  const txt=(r.result.content&&r.result.content[0]&&r.result.content[0].text)||"";
  const m=txt.match(/cross_entropy_loss=([0-9.]+)/);
  const loss=m?parseFloat(m[1]):null;
  const flagged=(r.result.isError===true)||(loss!==null&&loss>TH);
  if(flagged)viol++;
  rows.push({path,flagged,loss});
}
if(viol>0){
  const head=STRICT? R+"✖ Axiom Quality Gate: COMMIT BLOCKED"+X
                    : Y+"⚠ Axiom Quality Gate: DRIFT ADVISORY (not blocking)"+X;
  console.error("");
  console.error(head+"  ("+viol+" file(s) exceed the drift threshold)");
  console.error("  Drift threshold (cross-entropy): "+TH.toFixed(4)+(STRICT?"":"   [warn-only; set AXIOM_GUARD_STRICT=1 to block]"));
  console.error("");
  for(const r of rows){if(r.flagged){
    const d=r.loss!=null?("   (Δ +"+(r.loss-TH).toFixed(4)+")"):"";
    console.error("  "+(STRICT?R+"✖"+X:Y+"⚠"+X)+" "+r.path);
    console.error("      Current Loss: "+(r.loss!=null?r.loss.toFixed(4):"n/a")+"   vs   Threshold: "+TH.toFixed(4)+d);
  }}
  console.error("");
  console.error(Y+"  These files drift from the absorbed codebase DNA. Options:"+X);
  console.error("    1. Refactor toward the repo idioms, then re-stage.");
  console.error("    2. If this is an INTENTIONAL new architectural pattern, teach Axiom it is");
  console.error("       canonical by merging it into the master vibe, then re-commit:");
  console.error("         (MCP) axiom_compress_path  on the new path/dir");
  if(STRICT) console.error("    3. Override the gate for this one commit:  git commit --no-verify");
  console.error("");
  process.exit(STRICT?1:0);
}
console.error(G+"✓ Axiom Quality Gate: PASS"+X+"  ("+evaluated+" file(s) within codebase DNA <= "+TH.toFixed(4)+(STRICT?", strict":", warn-only")+")");
for(const r of rows){console.error("    "+G+"ok"+X+"  "+r.path+(r.loss!=null?"   (L="+r.loss.toFixed(4)+")":"   ("+(r.note||"skipped")+")"));}
process.exit(0);
'
exit $?
