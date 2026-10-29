# Model operations: training more, the recall embedder, and the Hub artifacts

A runbook for the three model-side gaps the 2026-10-29 live test left open
(`LIVE-TEST-2026-10-29.md`). Everything here was written against the trainer
binaries and scripts in this tree; commands are copy-pasteable from the
repository root on a Linux/macOS shell (PowerShell equivalents are noted
where they differ).

## 0. Should the main model be trained more?

Short answer: **not before two cheaper things**, and only with a held-out
eval in place. The evidence:

- Token savings do not depend on the weights. `axiom bench` measured 86.3% on
  the optimfix, baseline and v0.4.1 checkpoints alike, because signature
  extraction is structural and only the tokenizer changes the percentage.
  More training will not move the headline compression number.
- What the weights *do* drive is drift scoring, absorption (the `tension`
  CE delta in `axiom run`) and the fingerprint's recall layout. The optimfix
  checkpoint already separates clean code (CE 5.4–5.9) from garbage (8.9) by
  a margin of 3 nats on the published 7.03 gate.
- The only held-out number on record is bad in the other direction: the Hub
  card's own caveat says optimfix scored 8.04 on one 431-byte `server.rs`
  sample, worse than the baseline. One sample is not a rate, but it is the
  reason the baseline file is preserved. Training more without a real
  held-out set would make that number unknowable, not better.

So the order is:

1. **Train and publish the recall embedder** (§2). It is the one model the
   product is missing outright; recall is exact-match only without it.
2. **Build a held-out evaluation** the next training run has to beat: 30–50
   files the model has never seen, across the six languages, scored with
   `eval_model` (CE) and the drift margin. Keep it out of the corpus crawl.
3. **Then** train more, on *more data* rather than more epochs: the Hub card
   says 2,039 files / 1M tokens. The engine's own `corpus_crawl` + `harvest`
   binaries exist to grow that. A d128/2-layer model is capacity-limited;
   if the held-out CE plateaus, the next lever is `scripts/train_d384.sh`
   (d384 / 6 layers, needs a ~6 GB GPU), which was built for exactly this
   and writes to a separate checkpoint so production is never clobbered.

Promotion rule, unchanged from the repo's own: a new checkpoint replaces
`axiom_production_bpe.bin` only after `eval_model` passes on it, and the
previous file stays on the Hub under a `_baseline_*` name.

## 1. Training the main model (when you get to it)

CPU, from scratch, ~20 minutes on 4 cores (the validated recipe):

```bash
./scripts/train_cpu_quickstart.sh
```

It builds the `tools` binaries, stages a corpus from this repo, trains the
tokenizer and the d128/2-layer model with early stopping on held-out CE, and
runs the acceptance eval. On PASS it prints the three exports; with the
`init`-installed model path you can instead copy the artifacts into
`~/.axiom/models/` and leave `AXIOM_PRODUCTION_BPE` unset.

To train on *more* data (the recommended lever), point the corpus at more
trees before the quickstart's train step:

```bash
cargo build --release --locked --features tools --bins --manifest-path axiom_engine_rs/Cargo.toml
# Extra source roots are positional arguments; output goes to AXIOM_CORPUS_OUT
# (default checkpoints/corpus), capped by AXIOM_CORPUS_MAX_MB (default 200).
AXIOM_CORPUS_OUT=checkpoints/corpus AXIOM_CORPUS_MAX_MB=400 \
  ./axiom_engine_rs/target/release/corpus_crawl /path/to/repoA /path/to/repoB
```

GPU, bigger model: `./scripts/train_d384.sh` (reads its knobs from the
environment, trains to `axiom_d384.bin`, never touches the production file).

## 2. Training and publishing the recall embedder

The MCP memory tools (`axiom_remember` / `axiom_recall` / `search`) embed text
with `axiom_embedder.bin` when it exists next to the checkpoint, and fall
back to pooled TTT hidden states when it does not. The fallback finds a
memory by its exact text and almost nothing else. The embedder is a separate
contrastive bi-encoder trained on (query, content) pairs.

### 2.1 Build the pair set

```bash
cargo build --release --locked --features tools --bins --manifest-path axiom_engine_rs/Cargo.toml
# Mines doc<->body and markdown pairs from source trees. Add any repos whose
# conventions you want recall to understand.
AXIOM_PAIR_DIRS="$PWD;/path/to/another/repo" \
AXIOM_PAIRS_OUT=checkpoints/pairs.jsonl \
  ./axiom_engine_rs/target/release/build_pairs
```

Two ways to add natural-language questions (the kind of query recall gets):

- Hand-authored, no LLM: `python3 scripts/author_synth_pairs.py` appends its
  curated (question, content) set (66 pairs) to `checkpoints/pairs.jsonl`;
  extend the list inside the script with pairs about your own conventions.
- Synthetic, via the running proxy: set `AXIOM_SYNTH=200` on `build_pairs`
  with the proxy up on `AXIOM_PROXY_URL` (default `http://127.0.0.1:3000`)
  and a backend configured; each chunk gets generated questions. This spends
  upstream tokens.

Aim for a few thousand pairs. Check the file: `wc -l checkpoints/pairs.jsonl`.

### 2.2 Train

```bash
AXIOM_BPE=checkpoints/axiom_bpe.json \
AXIOM_PAIRS_OUT=checkpoints/pairs.jsonl \
AXIOM_EMB_CKPT=checkpoints/axiom_embedder.bin \
AXIOM_EMB_EPOCHS=10 AXIOM_EMB_BS=32 \
  ./axiom_engine_rs/target/release/train_embedder
```

Defaults: d 384, 6 layers, 6 heads, seq 256, lr 3e-4, InfoNCE with in-batch
negatives, best checkpoint by validation recall@1 saved with a sidecar. It
uses the GPU when one is available (move the proxy to CPU first); on CPU
expect tens of minutes per epoch at batch 32. The tokenizer must be the
same `axiom_bpe.json` the production checkpoint uses.

### 2.3 Verify before publishing

```bash
AXIOM_EMB_CKPT=checkpoints/axiom_embedder.bin ./axiom_engine_rs/target/release/eval_recall
```

Then the live check that failed in the live test: start `axiom --mode mcp`
with `AXIOM_EMB_CKPT` pointing at the file, `axiom_remember` "Use cargo fmt
before every commit", and `axiom_recall` with the query "cargo fmt". It must
return the memory; the stderr banner must say the contrastive embedder is
present, not "absent (pipeline fallback)".

### 2.4 Publish next to the checkpoint

```bash
pip install -U huggingface_hub
huggingface-cli login                       # once; needs a write token for fggg23
huggingface-cli upload fggg23/axiom-multilang-v1 \
  checkpoints/axiom_embedder.bin axiom_embedder.bin \
  --commit-message "Add contrastive recall embedder (recall@1 = <value from eval_recall>)"
huggingface-cli upload fggg23/axiom-multilang-v1 \
  checkpoints/axiom_embedder.meta.json axiom_embedder.meta.json \
  --commit-message "Embedder sidecar"
```

Add the file to the model card's Files list with its SHA-256
(`sha256sum checkpoints/axiom_embedder.bin`). Then make the release ship it
too: add `checkpoints/axiom_embedder.bin` (and its sidecar) to the
`checkpoint` job's staging and `files:` lists in `.github/workflows/release.yml`,
so `SHA256SUMS.txt` covers it, and have `axiom init` fetch it: add an
`embedder_file` to `ModelFetchConfig` in `axiom_engine_rs/src/config.rs`
alongside `tokenizer_file` and set `AXIOM_EMB_CKPT` from the installed path
in `resolve_production_model` (the same pattern this change set added for the
tokenizer).

## 3. Fixing the v0.4.1 sidecar on the Hub

`fggg23/axiom-aether-v0.4.1/axiom_production_bpe.meta.json` has
`"tokenizer": "/home/runner/work/AXIOM-AETHER/AXIOM-AETHER/checkpoints/axiom_bpe.json"`,
the CI runner's absolute path. The engine ignores the field at runtime (it
takes the tokenizer from `AXIOM_TOKENIZER` or the installed path), but
`eval_model` reads it and anyone reading the card is misled.

Edit it to the relative name the multilang repo already uses:

```bash
huggingface-cli download fggg23/axiom-aether-v0.4.1 axiom_production_bpe.meta.json --local-dir /tmp/v041
python3 - <<'PY'
import json
p = "/tmp/v041/axiom_production_bpe.meta.json"
m = json.load(open(p))
m["tokenizer"] = "axiom_bpe.json"
json.dump(m, open(p, "w"), indent=2)
PY
huggingface-cli upload fggg23/axiom-aether-v0.4.1 \
  /tmp/v041/axiom_production_bpe.meta.json axiom_production_bpe.meta.json \
  --commit-message "meta: tokenizer path relative to the repo, not the CI runner"
```

Or edit the file in the Hub web UI (Files → the sidecar → Edit) and change
the one value. While there, the same repo's card still quotes "85.7% token
savings"; the 2026-10-29 live test measured 86.3% on this checkpoint with the
current `bench`, so the number is fine to leave but is tokenizer-dependent.

Going forward, the release workflow's `train_semantic` step writes that path
from `AXIOM_BPE`; passing a relative path there (`checkpoints/axiom_bpe.json`
rather than the runner's absolute one) keeps future sidecars portable.
