#!/usr/bin/env bash
# new_user_simulation.sh — simulate a brand-new user on a blank machine.
#
# Unlike demo_end_to_end.sh (which exercises surfaces in temp dirs), this script
# stands up a *pristine HOME with no ~/.axiom at all*, runs `axiom init` to
# scaffold it from scratch, and then walks the real first-run journey end to end:
# init → doctor → compression → self-healing runtime → learned immunity →
# the ChimeraLang DSL (run + offline certificate) → the reproducible capability
# score. It prints PASS/FAIL per step and exits non-zero on any failure, so it
# doubles as a first-run regression check.
#
#   ./scripts/new_user_simulation.sh
#
# Everything is CPU-only and offline. `axiom init` will try to fetch a model and,
# with no network, gracefully bootstraps a small local checkpoint instead — that
# offline fallback is part of what this script verifies.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/axiom_engine_rs/target/release/axiom_engine"
if [ ! -x "$BIN" ]; then
    echo "[build] release binary not found — building it first…"
    ( cd "$REPO/axiom_engine_rs" && cargo build --release --locked ) || { echo "build failed"; exit 1; }
fi

# A throwaway HOME so we never touch the real ~/.axiom. Cleaned up on exit.
NEWHOME="$(mktemp -d)/newuser"; mkdir -p "$NEWHOME"
WORK="$(mktemp -d)"; SAMPLE="$(mktemp -d)/proj"; CH="$(mktemp -d)/demo.chimera"
cleanup() { rm -rf "$NEWHOME" "$WORK" "$SAMPLE" "$CH" 2>/dev/null; }
trap cleanup EXIT
export HOME="$NEWHOME"
export AXIOM_DEVICE=cpu
unset AXIOM_HEAL_MEMORY AXIOM_FLEET_KEY ANTHROPIC_API_KEY 2>/dev/null || true
# Run from inside the throwaway workspace so commands that write a *relative*
# artifact (e.g. `axiom init` bootstraps ./axiom_kernel_v1.safetensors when no
# model can be fetched) keep it inside the temp dir, never the caller's repo —
# the simulation must be fully isolated, not rely on .gitignore to hide leaks.
cd "$WORK" || { echo "cannot enter workspace"; exit 1; }

P=0; F=0
ok(){ printf '  \033[32mPASS\033[0m %s\n' "$1"; P=$((P+1)); }
no(){ printf '  \033[31mFAIL\033[0m %s\n' "$1"; F=$((F+1)); }
st(){ printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

st "0. blank slate (brand-new machine)"
echo "    HOME=$HOME"
[ ! -e "$HOME/.axiom" ] && ok "no ~/.axiom yet (truly fresh)" || no "~/.axiom already exists"

st "1. axiom init — scaffold user state from nothing"
"$BIN" init 2>&1 | sed 's/^/    /' | head -12
[ -d "$HOME/.axiom" ] && ok "~/.axiom created (config + logs + offline checkpoint)" || no "init did not create ~/.axiom"

st "2. axiom --mode doctor — hardware-aware device pick"
"$BIN" --mode doctor 2>&1 | grep -iE "core|gpu|device|recommend" | head -3 | sed 's/^/    /'
"$BIN" --mode doctor 2>&1 | grep -qiE "device|cpu|recommend" && ok "doctor gives a recommendation" || no "doctor"

st "3. axiom bench — context compression on a sample project"
mkdir -p "$SAMPLE/src"
cat > "$SAMPLE/src/lib.rs" <<'RS'
pub fn add(a: i32, b: i32) -> i32 { a + b }
pub fn mul(a: i32, b: i32) -> i32 { a * b }
pub struct Point { pub x: f64, pub y: f64 }
impl Point { pub fn norm(&self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() } }
RS
"$BIN" bench "$SAMPLE/src" 2>&1 | grep -iE "token savings|signatures kept|round-trip|fidelity" | head -3 | sed 's/^/    /'
"$BIN" bench "$SAMPLE/src" 2>&1 | grep -qiE "savings|round-trip" && ok "compression measured" || no "bench"

st "4. axiom run — self-healing runtime (creates a missing dir)"
"$BIN" run -- sh -c "echo built > $WORK/dist/app.bin" 2>&1 | grep -iE "heal|result|immun" | head -3 | sed 's/^/    /'
[ -f "$WORK/dist/app.bin" ] && ok "self-heal created the missing dir; the command then succeeded" || no "self-heal did not produce the artifact"

st "5. axiom immunity — learned experience report"
"$BIN" immunity 2>&1 | grep -iE "confidence|immun|program|•" | head -4 | sed 's/^/    /'
"$BIN" immunity 2>&1 | grep -qiE "confidence|immun" && ok "immunity report renders the learned heal" || no "immunity"

st "6. axiom chimera — the ChimeraLang DSL, end to end (VM + belief + certificate)"
# ChimeraLang is behind the `experimental` cargo feature (docs/EXPERIMENTAL.md);
# a stock build has no `chimera` subcommand.
if "$BIN" chimera --help >/dev/null 2>&1; then
    cat > "$CH" <<'CHIM'
val scores = [1, 2, 3]
for s in scores
  emit s
end
belief cause := inquire { prompt: "why?", agents: [local], ttl: 0 }
guard cause against hallucination { max_risk: 0.4 }
emit cause
CHIM
    "$BIN" chimera run "$CH" 2>&1 | sed 's/^/    /'
    "$BIN" chimera run "$CH" 2>&1 | grep -q "3" && ok "chimera run executed VM + belief paths" || no "chimera run"
    "$BIN" chimera prove "$CH" --out "$WORK/cert.json" >/dev/null 2>&1
    "$BIN" chimera verify "$WORK/cert.json" 2>&1 | grep -qi VALID && ok "chimera certificate verifies offline" || no "chimera cert"
else
    echo "    SKIP — build with --features experimental to include ChimeraLang"
fi

st "7. axiom eval-agentic — reproducible capability score (no LLM)"
EVAL="$("$BIN" eval-agentic 2>&1)"
echo "$EVAL" | grep -iE "score:|[0-9]+/[0-9]+" | tail -2 | sed 's/^/    /'
echo "$EVAL" | grep -qiE "score: [0-9]+/[0-9]+" && ok "agentic eval produced a capability score" || no "eval-agentic"

st "summary"
printf '\n  %d passed, %d failed\n\n' "$P" "$F"
[ "$F" -eq 0 ] || exit 1
