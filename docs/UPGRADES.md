# Axiom upgrade roadmap (2024–2026 research synthesis)

Distilled from a deep-research sweep across Axiom's four pillars. Each item lists
the method, what it improves, a practicality verdict for a Rust/candle CPU/modest-GPU
engine, and a citation. Status: ✅ implemented · 🚧 scaffolded · ⬜ planned.

## P0 — cheap, high-confidence, mostly reuse existing infra

- ✅ **Self-correction grounding guardrail.** Intrinsic self-correction can *degrade*
  output unless gated on an external signal (Huang et al., ICLR 2024,
  arXiv:2310.01798; TACL 2024 survey, arXiv:2406.01297). `ground_correct_round`
  now re-verifies the revision against the evidence and keeps it **only if it is
  strictly more grounded** than the original; otherwise the original stands.
- ✅ **DARE-TIES fleet merging.** Replaces the uniform alpha-blend (which suffers
  sign cancellation + redundancy) with DARE drop-and-rescale (arXiv:2311.03099)
  + TIES trim/elect-sign/disjoint-merge (arXiv:2306.01708). Data-free, single-pass,
  reproducible (seeded). `cluster_merge` now uses it; alpha-blend remains as a
  legacy fallback. +1–5pp over averaging at our scale; DARE sparsity also shrinks
  per-node gossip payloads.
- ✅ **Dempster-Shafer conflict fix + reliability weighting.** Raw Dempster's rule
  misfires under high conflict (Zadeh's paradox). `BetaBelief` gains
  `discounted()` (down-weight a peer by trust before combining),
  `combine_ds_reliable()`, `murphy_average()`, and `combine_ds_conflict_aware()`
  (Dempster when reconcilable, Murphy averaging + conflict flag otherwise).
  Refs: PMC10047774; OptiGradTrust arXiv:2507.23638.
- 🚧 **Semantic Entropy Probe (SEP).** Single forward-pass linear probe on hidden
  states recovers most of Nature-2024 semantic-entropy signal at ~zero cost
  (arXiv:2406.15927; SE: Nature s41586-024-07421-0). *Needs an offline-trained
  probe weight file*, so it can't be fully completed in-repo without a training
  run — to be added behind a loadable probe tensor.

## P1 — medium effort, strong payoff

- ⬜ **Conformal factuality gate** — calibrate detector scores into a keep/drop
  threshold with a distribution-free guarantee that ≥(1−δ) of retained claims are
  correct (Mohri & Hashimoto, arXiv:2402.10978). Runtime = one quantile.
- ⬜ **LLMLingua-2 second-stage squeeze** (encoder-only, candle-portable,
  arXiv:2403.12968) + **query-aware expansion ranking** (LongLLMLingua,
  arXiv:2310.06839; "Beyond RAG", arXiv:2503.04973) on the compression path.
- ⬜ **FLTrust root-of-trust + centered-clipping w/ momentum** Byzantine gate —
  survives 40–60% malicious peers and provably resists ALIE/IPM
  (arXiv:2012.13995; PMLR v139 karimireddy21a).

## P2 — bigger / offline / research

- 🚧 **Gated DeltaNet TTT core** (arXiv:2412.06464). The existing TTT update is
  already the delta rule (`W̃ ← W̃(I − ηkkᵀ) + ηvkᵀ`). **Implemented:**
  - the **scalar forget gate** — parameter-free α ∈ (0,1] on the retained-memory
    term (`W̃ ← α·W̃(I − ηkkᵀ) + ηvkᵀ`), opt-in via `AXIOM_FORGET_GATE`,
    byte-identical at α=1; with normalized keys spectral radius ≤ α.
  - the **learned, data-dependent gate** — a per-layer `w_α: Linear(d→1)` so the
    network decides what to forget per token: α_t = α_min + (1−α_min)·σ(w_α·x),
    warm-started near α≈1 (`GATE_INIT_LOGIT`) so training departs from the proven
    dynamics. Opt-in via `AXIOM_LEARNED_GATE` at training; recorded in the
    `ModelMeta` sidecar (`learned_gate`) so inference builds the matching
    architecture. Adds `w_alpha` weights → a checkpoint trained with it is needed;
    default-off is parameter-identical, so existing d128/d256 checkpoints load
    unchanged.
  **Follow-ups:** train + eval a learned-gate checkpoint to quantify the gain vs
  the ungated baseline; the chunkwise (C=64) parallel form for training throughput.
- ⬜ **RWKV-7-style vector gating** (arXiv:2503.14456) — per-channel gate +
  in-context LR (more expressive than the scalar gate; also a checkpoint bump).
- ⬜ **KV-cache eviction** (PyramidKV, arXiv:2406.02069) in the decode loop.
- ⬜ **EigenScore** second sampling-based detector (arXiv:2402.03744).
- ⬜ **Sakana CMA-ES** offline merge-recipe tuning (arXiv:2403.13187).
- ⬜ **zkML on the dispute path** / TEE attestation for provenance beyond HMAC.

## Honest negatives (verified across sources — do *not* adopt naively)

- Intrinsic self-correction without external feedback can regress (→ guardrail above).
- Uniform model soup ≈ no better than the current alpha-blend (greedy/val-gated is).
- Proof-of-Learning is spoofable; treat as soft evidence only.
- Titans' headline long-context numbers are not yet independently reproduced
  ("Titans Revisited", arXiv:2510.09551).
- Full TTT-MLP / Titans-MLP / ATLAS-Muon need inner-loop backprop / second-order
  ops — heavy and unstable on CPU/modest GPU; research-only for now.
