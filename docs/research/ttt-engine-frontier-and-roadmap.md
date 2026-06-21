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

### B.1 ⭐ Per-bug test-time adapter for the repair loop  — **HIGH impact / MED effort**
The biggest demonstrated TTT win (§A.3, §A.4) is *not* in the backbone — it's
wrapping inference in per-task adaptation. **For each fault**: build leave-one-out
examples from the repo's own surrounding code + prior verified patches
(`patch_memory.rs`), train a small **LoRA-style adapter**, generate candidates
under **augmentations** (variable renames, equivalent refactors), and **vote**
(`fault_locate.rs` → candidate ranking). Gate every candidate through the existing
re-verification before trust. *Why now:* reuses the verify-gate and patch store
you already merged (#49/#58); adapter is discarded per-bug (episodic = safe, §A.5).

### B.2 ⭐ Safe-online-update gate for `NativeTTTBlock` — **HIGH impact / LOW-MED effort**
The persistent 1-step/token path risks collapse/forgetting on long sessions (§A.5).
Add three small, revertible guards alongside `forget_gate`/`stabilize`:
1. **Drift-aware reset** (RDumb): snapshot `W̃₀`; if a cheap drift signal (output-norm or surprisal spike via `surprisal.rs`) trips, reset `W̃ ← W̃₀`. Cheaper than the element clamp catching divergence after the fact.
2. **Token/sample selection** (EATA): skip the inner update on low-information tokens (low surprisal) — cuts compute ~40% (§A.5) and reduces noise.
3. **Anti-forgetting anchor** (EATA Fisher / CoTTA stochastic restore): bias `W̃` weakly toward the meta-trained init.

### B.3 Hindsight fine-tuning on the node's own traces (SOAR-style) — **HIGH impact / MED-HIGH effort**
§A.4: periodically fine-tune (slow weights, not `W̃`) on the node's **own verified
successes *and* failed-but-informative traces** from `heal_memory.rs`. Pairs with
fleet patch-sharing: a node learns from peers' *verified* patches without trusting
them blindly (re-verify-before-trust preserved). Schedule offline; gate the new
checkpoint behind the agentic-eval benchmark (8/8) before promotion.

### B.4 Held-out verification split to kill test-overfitting — **MED impact / LOW effort**
§A.4 caveat: patches passing provided tests can fail held-out ones. Split the
verification suite — adapt/select on one subset, **confirm on a held-out subset**
before a patch is recorded as verified. Cheap insurance for the metric's honesty;
directly hardens the capability number you've been careful to keep real.

### B.5 TTT-MLP expressivity option for the backbone — **MED impact / HIGH effort**
§A.2: a 2-layer-MLP hidden state is more expressive than linear at long context,
but memory-I/O-bound. Add as an **opt-in** `NativeTTTBlock` variant (keep linear
default byte-identical). Only worth it if long-context code understanding is a
measured bottleneck — measure first (perplexity-vs-context like Fig. 2).

### B.6 Contrastive / multi-view aux loss ablation — **LOW-MED impact / MED effort**
§A.1/§A.2: TTT++ found contrastive > rotation for vision; AXIOM uses
reconstruction (correct default for sequences). Low-priority ablation — try a
multi-view code-specific corruption (e.g., mask-and-reconstruct identifiers) for
the inner SSL loss; keep only if it beats reconstruction on the benchmark.

---

## C. Confidence & caveats
- Strongest evidence: §A.1–A.3 (peer-reviewed, reproduced). §A.4 SWE-bench ranks and several 2026-dated preprints are **snapshots** — re-verify before quoting.
- Latency multipliers (§A.5) are aggregated, order-of-magnitude.
- ARC numbers: mind split (public/semi-private/private) and paper-version drift.
- The roadmap deliberately keeps AXIOM's invariant: **adaptation only proposes; the verifier still gates trust.** Every item above is revertible and benchmark-gated.

### Primary sources
Sun 2020 (1909.13231) · TENT 2021 (2006.10726) · TTT++ 2021 (NeurIPS) · TTA survey 2024 (2303.15361) · TTT-layers 2024/25 (2407.04620) · E2E-TTT long-context (2512.23675) · Akyürek ARC 2024 (2411.07279) · ARC Prize 2024 (2412.04604) · SOAR 2025 (2507.14172) · Self-Debugging (2304.05128) · Reflexion (2303.11366) · RelRepair (2509.16701) · CoTTA (2203.13591) · EATA (2403.11491) · SAR (2302.12400) · RDumb (2306.05401) · Pitfalls-of-TTA (Zhao 2023).
