#!/usr/bin/env bash
# run_benchmark.sh — measure how online Test-Time Training (/v1/adapt) changes
# answer quality on a held-out Rust-knowledge eval set, scored locally with
# BLEU-4 and Rouge-L (no external API).
#
# Pipeline:
#   1. Start the Axiom server (trained BPE checkpoint) if not already running.
#   2. Create a fresh TTT session.
#   3. baseline       — generate an answer per question with NO adaptation.
#   4. after_adapt_10 — adapt the session on 10 domain docs, re-measure.
#   5. after_adapt_50 — adapt cumulatively on 50 domain docs, re-measure.
#   6. Write benchmarks/results.json.
#
# To keep the comparison clean, the adapted W̃ is snapshotted after adaptation
# and restored before each question (generation itself mutates session state),
# so every question sees the same adapted weights rather than accumulating drift.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PORT="${AXIOM_BENCH_PORT:-3000}"
BASE="http://127.0.0.1:${PORT}"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
BIN="$REPO/axiom_engine_rs/target/release/axiom_engine"
CKPT="$REPO/checkpoints/axiom_production_bpe.bin"
TOK="$REPO/checkpoints/axiom_bpe.json"

STARTED_SERVER=0
if ! curl -s -m2 "$BASE/v1/models" >/dev/null 2>&1; then
    echo "[bench] starting Axiom server on :$PORT (trained BPE checkpoint)"
    AXIOM_PRODUCTION_BPE=1 AXIOM_TOKENIZER="$TOK" AXIOM_BPE_CKPT="$CKPT" \
        AXIOM_DRIFT_THRESHOLD="$(cat "$REPO/checkpoints/axiom_drift_gate.txt" 2>/dev/null || echo 5.0)" \
        AXIOM_DEVICE=cpu AXIOM_TTT_COMPRESS=0 \
        "$BIN" --mode server --host 127.0.0.1 --port "$PORT" \
        > /tmp/axiom_bench_server.log 2>&1 &
    SRV_PID=$!
    STARTED_SERVER=1
    for _ in $(seq 1 90); do
        curl -s -m2 "$BASE/v1/models" >/dev/null 2>&1 && break
        sleep 1
    done
fi

if ! curl -s -m2 "$BASE/v1/models" >/dev/null 2>&1; then
    echo "[bench] FAIL: server did not come up on :$PORT" >&2
    tail -20 /tmp/axiom_bench_server.log >&2 || true
    exit 1
fi

AXIOM_BASE="$BASE" AXIOM_REPO="$REPO" AXIOM_SCORE="$HERE/score.py" \
AXIOM_EVAL="$HERE/eval_set.jsonl" AXIOM_OUT="$HERE/results.json" \
AXIOM_MAXTOK="${AXIOM_BENCH_MAXTOK:-48}" \
"$PY" - <<'PYEOF'
import json, os, sys, time, urllib.request, urllib.error, importlib.util

BASE  = os.environ["AXIOM_BASE"]
REPO  = os.environ["AXIOM_REPO"]
EVAL  = os.environ["AXIOM_EVAL"]
OUT   = os.environ["AXIOM_OUT"]
MAXT  = int(os.environ["AXIOM_MAXTOK"])

# Load the scorer from score.py (the dedicated local BLEU-4 / Rouge-L module).
spec = importlib.util.spec_from_file_location("score", os.environ["AXIOM_SCORE"])
score = importlib.util.module_from_spec(spec); spec.loader.exec_module(score)

def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"content-type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return json.load(r)

def put(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"content-type": "application/json"},
                                 method="PUT")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

# Eval set.
pairs = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
docs = []
for p in pairs:
    docs.extend(p.get("domain_docs", []))
print(f"[bench] {len(pairs)} eval pairs, {len(docs)} domain docs available", flush=True)

def prompt_for(q):
    # A QA lead-in so the model continues with an answer rather than the question.
    return f"Question: {q}\nAnswer:"

def gen(question, session_id=None):
    body = {"prompt": prompt_for(question), "max_tokens": MAXT}
    if session_id:
        body["session_id"] = session_id
    out = post("/v1/completions", body)
    return out["choices"][0]["text"]

def measure(session_id, snapshot):
    sp = []
    for p in pairs:
        if session_id and snapshot is not None:
            # Restore the adapted W̃ so each question sees the same weights.
            put(f"/v1/sessions/{session_id}/checkpoint", snapshot)
        hyp = gen(p["question"], session_id)
        sp.append({"hypothesis": hyp, "reference": p["answer"]})
    return score.score_pairs(sp)

# Model dims from the checkpoint sidecar.
meta = json.load(open(os.path.join(REPO, "checkpoints/axiom_production_bpe.meta.json")))
model = {"d_model": meta["d_model"], "n_layers": meta["n_layers"],
         "val_ce": round(float(meta["val_ce"]), 4), "vocab": meta["vocab_size"]}

# --- baseline: stateless generation, no adaptation -------------------------
print("[bench] baseline (no adaptation)...", flush=True)
baseline = measure(None, None)
print("        ", baseline, flush=True)

# --- session for the adapted conditions ------------------------------------
sid = post("/v1/sessions", {})["session_id"]

print("[bench] adapt on 10 docs...", flush=True)
post("/v1/adapt", {"session_id": sid, "corpus": docs[:10], "steps": 4})
snap10 = get(f"/v1/sessions/{sid}/checkpoint")
adapt10 = measure(sid, snap10)
print("        ", adapt10, flush=True)

print("[bench] adapt cumulatively to 50 docs...", flush=True)
post("/v1/adapt", {"session_id": sid, "corpus": docs[:50], "steps": 4})
snap50 = get(f"/v1/sessions/{sid}/checkpoint")
adapt50 = measure(sid, snap50)
print("        ", adapt50, flush=True)

results = {
    "model": model,
    "conditions": [
        {"label": "baseline",       "bleu4": baseline["bleu4"], "rougeL": baseline["rougeL"]},
        {"label": "after_adapt_10", "bleu4": adapt10["bleu4"],  "rougeL": adapt10["rougeL"]},
        {"label": "after_adapt_50", "bleu4": adapt50["bleu4"],  "rougeL": adapt50["rougeL"]},
    ],
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(results, open(OUT, "w"), indent=2)
print("[bench] wrote", OUT, flush=True)
print(json.dumps(results, indent=2))
PYEOF
RC=$?

if [ "$STARTED_SERVER" = "1" ] && [ -n "${SRV_PID:-}" ]; then
    kill "$SRV_PID" 2>/dev/null
fi
exit "$RC"
