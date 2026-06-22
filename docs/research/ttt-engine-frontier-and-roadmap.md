# Test-Time Training (TTT) for AXIOM-AETHER — Frontier Survey + Roadmap

> Deep-research report. Two halves: **(A)** a cited survey of the TTT frontier,
> **(B)** a ranked, effort/impact-mapped set of next steps grounded in the
> existing engine (`axiom_engine_rs/src/ttt_block.rs`, `axiom_engine/ttt_layer.py`).
>
> Claims carry confidence + primary sources. Where sources disagree, the
> contradiction is flagged rather than smoothed over.

---

## 0. What AXIOM already has (baseline)

`NativeTTTBlock` / `TTTLinearLayer` is a **TTT-as-architecture** primitive — a
sequence mixer that *replaces* attention, in the same family as Sun et al. 2024
(§A.2). Already implemented:

- Single `[d,d]` fast-weight `W̃`, **one self-supervised gradient step per token**, O(1) state.
- **Dual-form parallel prefill** via causal Gram matrix — the ">5× vs naive" trick from the TTT-layers paper.
- **Gated-DeltaNet forget gate** — scalar `α` *and* an optional learned data-dependent `w_α` (α init ≈ 0.98), giving spectral radius ≤ α.
- **Stabilization**: key L2-normalization + element clamp (`STAB_CLAMP=10`) to stop d384/d512 NaN divergence.
- Multi-head; shared `inner_lr` atomic with cosine decay during meta-training.

So the architecture half is genuinely at the frontier. The **largest untapped
gains are in the *other* meaning of TTT** — per-task test-time *adaptation of the
repair agent* (§A.3, §B.1), and in **safe online-update gating** (§A.5, §B.2).

---

## A. Frontier survey

### A.1 Foundational TTT / test-time adaptation (the "fix distribution shift" lineage)

- **Original TTT** (Sun, Wang, Liu, Miller, Efros, Hardt — ICML 2020, arXiv:1909.13231): turn each unlabeled test input into a **self-supervised task (rotation prediction, 0/90/180/270°)** on a Y-shaped net (shared trunk + main head + aux head); update before predicting. Standard = 10 steps/sample (discarded); online = 1 step/sample (persisted). *High confidence.* https://arxiv.org/abs/1909.13231
  - Benefit is **theoretically conditioned on positive gradient correlation** between main and aux losses (Thm 1; measured ≈0.93 standard / ≈0.89 online). *Med-high.* https://ar5iv.labs.arxiv.org/html/1909.13231
- **TENT** (Wang, Shelhamer, Liu, Olshausen, Darrell — ICLR 2021 spotlight, arXiv:2006.10726): **fully test-time** (source-free, online) — minimize prediction **entropy**, updating only BatchNorm affine (γ,β) + stats. On ImageNet-C/ResNet-50: **44.0% error vs 50.2%** robust-training SOTA, one epoch. *High confidence.* https://arxiv.org/abs/2006.10726
- **TTT++** (Liu et al. — NeurIPS 2021): naive TTT can **degrade** under severe shift; swap rotation → **contrastive (SimCLR)** aux task and add **offline feature-moment alignment** (store source mean/cov, match online). *High confidence.* https://openreview.net/forum?id=86NHK__yFDl
- **Survey**: Liang, He, Tan — *A Comprehensive Survey on TTA under Distribution Shifts* (IJCV 2024, arXiv:2303.15361). Taxonomy: source-free DA / Test-Time-Batch / Online-TTA. *High confidence.* https://arxiv.org/abs/2303.15361

**Takeaway for AXIOM:** the aux-loss choice matters and the *sign of benefit is
not guaranteed* — adaptation must be gated, not assumed-good.

### A.2 TTT as a sequence-modeling architecture (AXIOM's lineage)

- **"Learning to (Learn at Test Time): RNNs with Expressive Hidden States"** (Sun, Li, Dalal et al. — ICML 2025, arXiv:2407.04620): RNN hidden state **is a model** (weights W), update rule = **one SSL gradient step per token** (reconstruction loss `‖f(θ_K x;W) − θ_V x‖²`). **TTT-Linear** (W linear) and **TTT-MLP** (W = 2-layer MLP). *High confidence.* https://arxiv.org/abs/2407.04620
  - **O(1)/token** vs attention's growing KV-cache; **perplexity keeps dropping to 32K context** where **Mamba plateaus ≈16K**. *High confidence (Fig. 2).*
  - **Dual form** → >5× faster training than naive; mini-batch TTT → up to ~4× inference throughput. *High (mechanism); 4× via secondary.* https://www.emergentmind.com/topics/test-time-training-ttt-layers
  - **TTT-MLP** is more expressive but **memory-I/O-bound** — long-context win is potential, not fully realized at scale in the paper.
- **Follow-up — "End-to-End TTT for Long Context"** (arXiv:2512.23675): a Transformer w/ sliding-window attention that adapts at test time via next-token prediction; 3B/164B-tokens scales with context like full attention (Mamba2/Gated-DeltaNet don't), **2.7× faster than full attention at 128K**, constant latency. *High confidence.* https://arxiv.org/abs/2512.23675
- **Critique** — NVIDIA "TTT with KV Binding Is Secretly Linear Attention" (2026): argues TTT-style layers ≈ linear attention. *Med confidence (thesis not fully rendered).* https://research.nvidia.com/labs/sil/projects/tttla/

**Takeaway:** AXIOM's Gated-DeltaNet path already counters the precise weakness
(memory decay/plateau) that the linear-attention critique targets. The open lever
is **TTT-MLP-style expressivity** (§B.5).

### A.3 TTT for reasoning / abstraction — the result most relevant to code repair

- **"The Surprising Effectiveness of TTT for Abstract Reasoning"** (Akyürek, Damani, Qiu, Guo, Kim, Andreas — MIT, Nov 2024, arXiv:2411.07279; v2 retitled "...for Few-Shot Learning"). Recipe: **per-task LoRA adapters** trained on **leave-one-out in-context examples** + **invertible augmentations** (rot/flip/color-perm), then **hierarchical self-consistency voting**. *High confidence.* https://arxiv.org/abs/2411.07279
  - **8B + TTT → 53% on ARC public val (~6× over base FT)**; **+ program-synthesis (BARC) ensemble → 61.9%**, vs avg human **60.2%**. Static prompting: Claude ≈21%, GPT-4o ≈9%. *High confidence.*
  - Three crucial components: (1) FT on similar tasks, (2) aux format + augmentations, (3) **per-instance test-time training**. *High confidence.*
- **ARC Prize 2024 report** (arXiv:2412.04604): **all** top transduction entries use TTT; **no static-inference transduction >~11%**. MindsAI 55.5% (closed), ARChitects **53.5%** (open, won). *High confidence.*
- **TTT ≠ test-time compute**: TTT does **real gradient updates** at inference; o1/o3 scale *compute* without weight changes (o3: 75.7%→87.5% on ARC-AGI-1). *High confidence.* https://arcprize.org/blog/oai-o3-pub-breakthrough
- **Contradictions flagged:** version drift (v1 53% @8B vs v2 tables 47.1% FT / 53.0% BARC+TTT); **LoRA rank 128** (paper) vs **32** (ARC-AGI-2 reimpl.); public-val 53% vs semi-private 47.5% vs private 53.5% are *different splits* — don't conflate.

**Takeaway — the headline for AXIOM:** the single biggest demonstrated TTT win
is **per-task adaptation at inference** (LoRA + augmentation + voting). AXIOM has
TTT in the *backbone* but does **not** yet wrap its repair agent in this loop.

### A.4 TTT/adaptation in code & program repair

- **Self-Debugging** (Chen et al., Google 2023, arXiv:2304.05128): "rubber-duck" — explain + inspect execution, no human feedback. **+2–3% Spider (+9% hardest), up to +12% TransCoder/MBPP**, matches baselines sampling **>10×** more. *High confidence.*
- **Reflexion** (Shinn et al. 2023, arXiv:2303.11366): **linguistic** feedback in episodic memory, **no weight updates** → **91% HumanEval pass@1** (vs GPT-4 80%). *High confidence.*
- **Self-repair across 7 models** (2026, arXiv:2604.10508): every model improves (**+4.9–17.1pp HumanEval, +16–30pp MBPP**); **first round dominates, 2 rounds capture 76–95%** of gains. *Med-high (preprint).*
- **SOAR** (Pourcel et al., ICML 2025, arXiv:2507.14172): evolutionary LLM search + **hindsight fine-tuning on its own search traces** = explicit **test-time training without ground-truth**; **52% of ARC-AGI public test**. *High confidence (ARC, not SWE-bench).*
- **Retrieval adaptation**: RelRepair (retrieve signatures/snippets) **101/255 Defects4J v1.2**; ReAPR/SelRepair dual-RAG. *High/med.* https://arxiv.org/pdf/2509.16701
- **SWE-bench Verified** (~mid-2025 snapshot, secondary): ByteDance 75.2%, Refact.ai 74.4%, Claude-based 73.2%; SWE-Agent 66%, Agentless-1.5 50.8%. *Med confidence.* https://arxiv.org/html/2506.17208v2
- **Caveat — test overfitting**: execution-gated patches can pass provided tests but fail held-out ones ("is the cure worse than the disease," 2025). *Med confidence.* https://arxiv.org/html/2511.16858v1

**Takeaway:** code repair already lives in a **generate-and-validate** loop (AXIOM's
strength). The under-used adaptation routes are (a) **hindsight fine-tuning on the
node's own verified/failed traces** (SOAR) and (b) **per-bug test-time adapters**
(Akyürek) — both gated by AXIOM's existing re-verification.

### A.5 Practical engineering — cost, collapse, forgetting, safe gating

- **Cost**: gradient TTT adds backward passes; ~**150ms/step on Llama-3.2-1B** (≈15s for 100 steps), **1.7–2.5× serving overhead**. *Med (aggregated, treat as order-of-magnitude).* https://arxiv.org/pdf/2505.20633
- **Sample selection cuts cost**: **EATA** (ICML 2022) skips unreliable/redundant samples — **29.7k vs 50k backward passes (~41%↓)** on ImageNet-C. *High.* https://arxiv.org/html/2403.11491v2
- **Collapse**: entropy-min TTA **collapses to one class** under long non-stationary streams. **SAR** (ICLR 2023) blames BatchNorm; fixes via large-gradient sample filtering + **sharpness-aware (flat-minimum)** updates; flags **mixed shift / small batch / imbalanced labels** as failure regimes. *High.* https://arxiv.org/abs/2302.12400
- **Forgetting**: **CoTTA** (CVPR 2022) — EMA mean-teacher + augmentation-averaged pseudo-labels + **stochastic restoration** (randomly reset a few neurons to source each step). **EATA** — **Fisher-importance regularizer** anchors ID-important params. *High.* https://arxiv.org/abs/2203.13591
- **Reset > clever adaptation on long horizons**: **RDumb** (NeurIPS 2023) — 7 SOTA TTA methods collapse below a non-adapting baseline on CCC; **periodic reset to pretrained weights (~every 1000 steps)** matches/beats them (CCC-Medium: Tent →1.4%, CoTTA →7.7%, **EATA 35.4%, RDumb 38.9%**, no-adapt 17.3%). *High/med-high.* https://arxiv.org/abs/2306.05401
- **Episodic vs online matters hugely**: MEMO **31.5% episodic → 66.6% online error** — episodic-tuned methods fail when updates accumulate. *Med-high.* https://proceedings.mlr.press/v202/zhao23d/zhao23d.pdf

**Takeaway:** AXIOM's `forget_gate` (geometric decay) + `STAB_CLAMP` are exactly the
right instincts, but for **persistent online** updates the literature is blunt:
add **drift-aware reset**, **sample selection**, and an **anti-forgetting anchor**.

---

## B. Ranked roadmap for AXIOM (effort × impact)

> Ordered by impact/effort. Each maps to a survey section and the existing code.
>
> **Status: all six items are now implemented** (one PR). Each entry links the
> module + tests that land it. Effort labels were revised during review to match
> the realized scope (B.1 → HIGH; B.2 orchestration spelled out).
>
> **Preconditions actually leveraged** (these existed before this work and are
> reused, which is what keeps the items tractable): the verifier-gated repair
> loop (`agentic.rs`), the verified-patch store with provenance + `verified_count`
> (`patch_memory.rs`, #49/#58), the meta-training pipeline (`meta_train.rs`), the
> shared per-layer atomic control pattern (`ttt_block.rs`/`model.rs`), and
> surprisal estimation (`surprisal.rs`). No new training infrastructure was
> required — B.1/B.3 adapt the existing pieces.

### B.1 ⭐ Per-bug test-time adapter for the repair loop — **HIGH impact / HIGH effort** — ✅ implemented
The biggest demonstrated TTT win (§A.3, §A.4) is *not* in the backbone — it's
wrapping inference in per-task adaptation. Implemented in **`test_time_adapter.rs`**
as the Akyürek recipe adapted to AXIOM's candidate-based repair: `canonical_form`
gives an **augmentation-invariant** form (alpha-rename identifiers → positional
placeholders, preserve keywords, drop insignificant whitespace) so fixes that
differ only by variable names/formatting collapse together; `rerank_by_self_consistency`
clusters candidates by that form and ranks by **cluster vote mass** (summed
`verified_count`), then prior, then stable index — the hierarchical
self-consistency vote. Wired into **`PatchMemory::try_candidates`** so the verifier
tries fleet candidates consensus-first (plus `candidates_reranked`). The verify
gate is unchanged, so **re-verify-before-trust is preserved** — the adapter only
reorders. *Effort revised MED→HIGH per review*: a full LoRA gradient-trained
adapter is the heavier future form; this ships the augmentation + voting core
(the part that drives the gain) on the existing patch store + verify gate.

### B.2 ⭐ Safe-online-update gate for `NativeTTTBlock` — **HIGH impact / MED effort** — ✅ implemented
The persistent 1-step/token path risks collapse/forgetting on long sessions (§A.5).
Implemented as a shared `OnlineGuards` cell in **`ttt_block.rs`** (threaded like
`inner_lr`, exposed via `AxiomTTTLM::set_online_guards`), all default-disabled so
the default path stays **byte-identical and device-sync-free** (`all_disabled()`
short-circuits the whole block). **Orchestration is fixed and layered so the three
guards cannot conflict** (clarified per review):
1. **Token selection** (EATA): skip the update when the reconstruction error
   ‖pred−v‖ is below threshold (low-information token) — runs *first*, so a
   skipped token is never anchored or reset.
2. **Anti-forgetting anchor** (EATA Fisher / CoTTA restore): pull the *kept*
   update toward init, `W̃ ← (1−λ)W̃ + λI` — runs *second*, on whatever survived (1).
3. **Drift-aware reset** (RDumb): if the resulting `‖W̃‖_F` exceeds threshold, snap
   `W̃` back to init — runs *last*, the final backstop on the post-anchor state.

The init snapshot is simply the identity (`init_states` uses `Tensor::eye`), so
reset/anchor are parameter-free. *Effort revised LOW-MED→MED per review* (three
orthogonal controls + tests).

### B.3 Hindsight fine-tuning on the node's own traces (SOAR-style) — **HIGH impact / MED-HIGH effort** — ✅ implemented
§A.4. Implemented in **`hindsight.rs`**. The training signal is sourced where it
actually lives (**corrected per Codex review**): verified-fix **contents** — the
positive targets — come from **`patch_memory.rs`** (`PatchCandidate.content` +
`verified_count`), *not* from `heal_memory.rs`, which holds only `ce_mean`/dirs and
is used here purely as auxiliary failure-tension context. `collect_from_patch_memory`
+ `write_corpus` materialize verified fixes (deduped) into a corpus; `fine_tune`
runs it end-to-end through the existing **`meta_train.rs`** pipeline and returns a
`FineTuneReport` (final loss) so the new checkpoint can be **gated behind the
agentic-eval benchmark before promotion** (re-verify-before-trust preserved).

### B.4 Held-out verification split to kill test-overfitting — **MED impact / LOW effort** — ✅ implemented
§A.4 caveat: patches passing provided tests can fail held-out ones. Implemented as
**`agentic::agentic_loop_with_holdout`**: a candidate is committed only when it
passes the train subset **and** a held-out subset; a train-only (overfit) patch is
rolled back and counted as a rejection so the loop keeps searching. Held-out runs
cheap-first (only after train passes). Directly hardens the honesty of the
capability number.

### B.5 TTT-MLP expressivity option for the backbone — **MED impact / HIGH effort** — ✅ implemented (opt-in primitive)
§A.2. Implemented in **`ttt_mlp.rs`** as `NativeTTTMlpBlock`: a 2-layer-MLP hidden
state `pred(k)=W₂·tanh(W₁·k)` with its own `MlpState`, one closed-form gradient
step through both layers per token (tanh for an exact cheap derivative). Provided
as a **standalone, tested primitive** rather than forced into `AxiomTTTLM`'s
single-`[d,d]`-tensor state plumbing (which the linear path keeps byte-identical) —
model integration is the measured follow-up, per "measure first": only worth
turning on if long-context code understanding is a measured bottleneck
(perplexity-vs-context like Fig. 2).

### B.6 Contrastive / multi-view aux loss ablation — **LOW-MED impact / MED effort** — ✅ implemented
§A.1/§A.2: TTT++ found contrastive > rotation for vision. Implemented as an opt-in
flag on `OnlineGuards` (`aux_loss_normalized`, via `AxiomTTTLM::set_aux_loss_normalized`):
when on, the inner loss L2-normalizes both the predicted and value views, turning
MSE reconstruction into a directional/cosine (contrastive multi-view) objective.
Default off ⇒ exact reconstruction (byte-identical). An ablation knob to keep only
if it beats reconstruction on the benchmark.

---

## C. Confidence & caveats
- Strongest evidence: §A.1–A.3 (peer-reviewed, reproduced). §A.4 SWE-bench ranks and several 2026-dated preprints are **snapshots** — re-verify before quoting.
- Latency multipliers (§A.5) are aggregated, order-of-magnitude.
- ARC numbers: mind split (public/semi-private/private) and paper-version drift.
- The roadmap deliberately keeps AXIOM's invariant: **adaptation only proposes; the verifier still gates trust.** Every item above is revertible and benchmark-gated.

### Primary sources
Sun 2020 (1909.13231) · TENT 2021 (2006.10726) · TTT++ 2021 (NeurIPS) · TTA survey 2024 (2303.15361) · TTT-layers 2024/25 (2407.04620) · E2E-TTT long-context (2512.23675) · Akyürek ARC 2024 (2411.07279) · ARC Prize 2024 (2412.04604) · SOAR 2025 (2507.14172) · Self-Debugging (2304.05128) · Reflexion (2303.11366) · RelRepair (2509.16701) · CoTTA (2203.13591) · EATA (2403.11491) · SAR (2302.12400) · RDumb (2306.05401) · Pitfalls-of-TTA (Zhao 2023).
