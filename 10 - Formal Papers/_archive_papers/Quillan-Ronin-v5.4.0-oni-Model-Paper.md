---
file_type: paper
domain: model
status: canonical
tags: [paper, model-card, architecture, quillan-ronin, v5.4.0-oni]
---

# Quillan-Ronin v5.4.0-oni: A Sovereign Hierarchical Networked Mixture-of-Experts with Ternary Quantization, Council Consensus, and Thermodynamic Safety

**CrashOverrideX and Quillan Research Team**
Quillan Research — crashoverridex@quillan.ai
https://github.com/leeex1/Quillan-Ronin — https://huggingface.co/CrashOverrideX/Quillan-Ronin

**Version:** v5.4.0-oni (canonical; retires v5.3.1 / v8.1 labels) — 2026-09-12 (rev.2 Attention-conformant 2026-09-12)
**Companion files:** `MODEL_CARD.md`, `02 - Knowledge Foundation/LINEAGE.md`, `version.py`, `10 - Formal Papers/quillan_ronin_paper_bibliography.md`
**Example paper conformance:** Vaswani et al. 2017 `1706.03762` (Attention Is All You Need) — Abstract with measured numbers → Intro (sequential constraint) → Background → Architecture (encoder/decoder, attention eq., FFN eq., embeddings, positional encoding) → Why Self-Attention (Table 1 complexity/path-length) → Training (data/batching, hardware/schedule, optimizer eq., regularization) → Results (Table 2 BLEU+cost, Table 3 ablations, Table 4 generalization) → Conclusion → References → Visualizations. This paper mirrors that spine in §1–§8 + Appendix E.

---

## Abstract

We present **Quillan-Ronin v5.4.0-oni**, a sovereign, self-hosting Hierarchical Networked Mixture-of-Experts (HNMoE) language model built for consumer hardware. The system is organized as a **3-tier fractal hierarchy**: (Tier 1) **Quillan Core / Sovereign Throne (C0)** orchestrator; (Tier 2) a **Council of 34 experts (C1–C34)** with Top-4 sparse Gumbel routing; (Tier 3) **autonomous micro-diverse swarms** governed per-expert via EGGROLL rank perturbations with Jaccard diversity filters.

All projections use **universal BitNet 1.58-bit ternary quantization** (weights in {−1,0,1}, INT8 activations) with Straight-Through Estimator (STE), achieving ~87.5% memory reduction vs FP16. Reasoning flows through a 6-phase pipeline: ingestion → **9-Vector Semantic Prism** decomposition → Gumbel-MoE → swarm augmentation → **Split-SDPA Flash Diffusion core with Continuous Modality RoPE** → Wavefunction Top-1 finalizer → geometric decoding → **C20-ARTIFEX agentic bridge** (host OS execution, LanceDB/C5-ECHO memory, sandboxed Python).

Safety is architectural, not post-hoc: **Council-Calibrated RL (CCRL)** requires consensus among ethical experts; the **Ethical Impact Constraint Engine (E_ICE)** imposes thermodynamic bounds on computation; the **Lee-Mach-6 Governor** provides PID latency/thermal throttling for legacy hardware; **C2-VIR refusal**, **PersonaPullGate**, and quality exit gates (Nullion/Warden/Shepherd + Quillan audit) enforce identity and truthfulness.

On current hardware the flagship **12-layer Oni build is ~390M params (~234M at 6-layer proof scale)** with ~480M sparse-active equivalent including swarm overlays, context 512 with 10%-buffered gated compaction; the saturated reference configuration is **4.57B total**. Training: cold-start slice-and-merge transplant (Qwen 0.8B + BitNet 1.58-3B donors on Llama-derived skeleton; no Mistral weights transplanted) → full pretraining on `CrashOverrideX/QuillanTrainingdata` + Corpus v9 (59.4M train + 0.6M val packed BPE bins) → paused SFT via `scripts/train_full_param_v2.py`. **Gate A: 16/16 smoke tests passed; val loss 7.24 @ step 660 (12L); best archival 0.0789 @ step 2500 (prior phase). Formal benchmarks pending** — we report engineering telemetry, not leaderboard claims. Prompt-lift results reported elsewhere (ARC/GPQA/MMLU via Council wrapping of frontier LLMs) are **not** base-model scores and are segregated in §6.5.

This report follows the structure of contemporary model technical reports (DeepSeek-V3, LLaMA 3, Mistral/Mixtral, BitNet b1.58 2B4T): architecture → infrastructure → pre-training → post-training → evaluation → safety/limitations → conclusion, with full bibliography mapping (§References + `quillan_ronin_paper_bibliography.md`).

---

## 1. Introduction

### 1.1 The crisis of scale

Frontier LLMs centralize capability in 70B–671B dense or large-MoE builds (GPT-4-class, LLaMA-3.1-405B, DeepSeek-V3-671B/37B-active) requiring datacenter GPUs, 140–700 GB memory, and FP8/H800-scale training (e.g., ~2.8M H800-hours for DeepSeek-V3). This prices out local deployment, concentrates privacy risk, and makes inference environmentally and economically unsustainable for individuals. Quantization (LLM.int8, GPTQ, AWQ, QLoRA) and sparse MoE (Switch, Mixtral 8×7B, DeepSeekMoE) mitigate cost but retain flat expert structures, opaque routing, and post-hoc alignment.

### 1.2 What Quillan-Ronin contributes

Quillan-Ronin inverts the assumption: **sovereignty first, scale second**. Design goals:

1. **Run on legacy consumer hardware** (GTX 1050 Ti / CPU) without cloud dependency.
2. **Make ethics architectural** (consensus + thermodynamic bounds), not a fine-tune overlay.
3. **Make reasoning auditable** (9-vector blueprint → council deliberation → diffusion rounds → gates → Typist polish).
4. **Unify text-first modeling with a multimodal-ready scaffold** (modality-isolated diffusion; encoders deferred to v6 per LINEAGE).

### 1.3 Contributions (delta vs v5.3.1 / v5.2.2 snapshots)

- **Canonical unification (v5.4.0-oni):** single version counter, single 12-layer flagship (`_dev/quillan_v5_4_oni.py`, `QuillanRoninOni` + `QuillanOniConfig`); retired v8/v9/v10 filename chaos (LINEAGE.md binding).
- **34-expert council (C1–C34) + Throne (C0):** dense_pull deliberation with PersonaPullGate (fp32, prior-weighted); 4 wave clusters for inference ordering (was Top-3 sparse / capacity-64 in production spec; Oni uses dense_pull on current hardware).
- **Universal ternary saturation:** 100% of projections via BitLinear + STE; EGGROLL rank-8 dense factors in Oni (rank-16 in saturated spec).
- **Ported organs:** RoPE (replacing learned wpe), Couil hybrid heads (even dense / odd sparse-topk), recirculation hook (deep→shallow feedback, zero-init), DistillationHead (KL α=0.7 + hidden MSE), ModalityIsolatedThermoDiffusion (Langevin inv-√t, time-emb, RMS halting α=0.7), LeeMach6VelocityGovernor (PID 0.15/0.05/0.02), analytic E_ICE, PersonaPullGate, quality exit gates, `deliberate()` loop.
- **Hardware-honest scaling table** (§3.9) and **consciously deferred list** (multimodal encoders, proactive compaction >4096, Docker ARTIFEX, GRPO/DGPO, HRM halting, BitDist, EGGROLL Evolution — Phase C/D).
- **Knowledge hygiene:** 126 PDFs consolidated to `10 - Formal Papers/Formal Papers/`, 46 MD + 49 canonical mirror, bibliography §16 triage for 2025–2026 additions, RAG KB 17→26 files (227→354 chunks).

### 1.4 Paper organization

§2 related work; §3 architecture; §4 infrastructure/training setup; §5 data + pre/post-training; §6 evaluation (telemetry + segregated prompt-lift claims); §7 safety/ethics/limitations/model-card; §8 conclusion/future; References; Appendices (lineage, file map, specs).

---

## 2. Related Work and Theoretical Framework

### 2.1 Mixture-of-Experts

From sparsely-gated MoE (Shazeer et al. 2017) → Switch (Fedus et al. 2022) → ST-MoE/Z-loss (Zoph et al. 2022) → Expert Choice (Zhou et al. 2022) → Mixtral (Jiang et al. 2024) → DeepSeekMoE fine-grained + shared experts (Dai et al. 2024) → DeepSeek-V3 auxiliary-loss-free balancing + MTP (2024). Quillan follows ST-MoE stability rules (fp32 routers/gates, Z-loss, load-KL) and DeepSeekMoE fine-grained intuition, but organizes experts as **cognitive personas** (brain-mapped) under a Throne, with **Gumbel-Softmax Top-4 + dynamic capacity clipping + residual overflow** (no silent drops) and **dense_pull** deliberation at Oni scale.

### 2.2 Quantization and efficient architectures

BitNet b1.58 (Ma et al. 2024) → BitNet v2 / 2B4T (2025) → BitNet.cpp edge inference; STE (Bengio et al. 2013); QLoRA/LoRA/rsLoRA; GaLore; MLA (DeepSeek-V2/V3) for KV compression. Quillan saturates **all** linears with BitLinear + STE, INT8 activations (absmax per-token), SubLN for stability, and EGGROLL low-rank perturbations pre-quantization for parameter-efficient adaptation.

### 2.3 Routing and adaptive computation

Gumbel-Softmax (Jang et al. 2017) / Concrete (Maddison et al. 2017); Mixture-of-Depths (Raposo et al. 2024); token routing / adaptive halting; sparse attention (Longformer, BigBird). Quillan's ComplexityRouter implements 3 paths (fast/balanced/diffusion) with temperature annealing 1.0→0.1, Top-4 selection, and RMS-halting-gated diffusion bypass (>0.92 confidence skips refinement).

### 2.4 RLHF / RLAIF / critic-free RL

PPO (Schulman 2017), RLHF (Ouyang 2022), HHH (Bai 2022), Constitutional AI/RLAIF (Bai 2022), GRPO (DeepSeekMath/R1), DAPO, DGPO. CCRL (Quillan) is multi-objective (reward + VIR consensus − E_ICE penalty) with consensus-calibrated policy; GRPO/DGPO/DAPO are **deferred to Phase D** (compute ceiling) per LINEAGE.

### 2.5 Diffusion language modeling

Score-based SDEs (Song & Ermon 2021), masked discrete diffusion (Sahoo et al. 2024), DALI/Dream7B/Prophet/DFlash. Quillan's SovereignFlashDiffusionCore + ModalityIsolatedThermoDiffusion uses Langevin inv-√t dynamics, time embeddings, and block-diagonal modality masks (early isolation → late integration via cosine schedule).

### 2.6 Synchronization, thermodynamics, consciousness

Kuramoto (1975) coupled oscillators → neural sync / GNN over-smoothing prevention (DQSO); Friston free-energy principle / thermodynamic Gibbs training; IIT (Tononi) Φ-integration and GWT broadcast; CoALA cognitive architectures. These ground DQSO phase coupling, MARTA thermodynamic gating, 9-vector integration, and Throne broadcast. Full equation set: `8-Formulas.md` + `Must know formulas.md` (20 Quillan custom formulas: AQCS, EEMF, QHIS, DQRO, QCRDM, AQML, QCIE, QICS, QSSR, JQLD, DQSO, ROUTING_SOFTMAX, TOKEN_LATENCY, LRPP, DVVE, DNNL, JHFR, LMCB, JSSC, QPS).

### 2.7 Safety and alignment

Sleeper-agent / deceptive-alignment evals, identity persistence, Constitutional AI. Quillan implements E_ICE exponential penalties, C2-VIR refusal layer, Nemesis-Alpha adversarial gate, and HFL (Historical Fidelity Loss) for Edo/Bushidō-anchored identity continuity (see §7).

---

## 3. Model Architecture

### 3.1 Overview: 3-tier sovereign fractal hierarchy

- **Tier 1 — Quillan Core (C0 Throne):** holistic orchestrator, thalamic/brainstem relay; assigns deliberation pull via PersonaPullGate; runs `deliberate()` (audit → diffusion rounds → gates → Typist polish).
- **Tier 2 — Council C1–C34:** specialized personas (ASTRA→PREDATOR; full registry Table 1) with Top-4 sparse Gumbel routing (saturated spec) / dense_pull (Oni current); 4 wave clusters (Cognitive / Communication / Meta / Systems) order inference.
- **Tier 3 — Micro-diverse swarms:** per-expert EGGROLL perturbations (rank-8 Oni / rank-16 saturated) + INT8 pooling + Jaccard lexical diversity filters, temperature variance, mutation rate, micro-roles; ~7k agents/expert nominal, 224k total orchestration capacity (virtual 9B-parameter-equivalent overlay in saturated spec — see §3.9 honesty note).

**6-phase pipeline:** Ingestion → 9-Vector Prism → Gumbel-MoE (Top-4) → swarm augmentation → 32-layer-equivalent Flash Diffusion (9–12 realized layers Oni) → Top-1 finalizer → geometric decoding → C20-ARTIFEX.

### 3.2 Input ingestion and 9-Vector Semantic Prism

Modality-specific encoders (text BPE 50,257 EOS=0; image Conv2D patching; audio Conv1D/STFT-Mel; video 3D conv) project to unified 3072-d space (Oni hidden 1024/2560 per config; see §3.9). **v5.4.0-oni trains text-only**; multimodal encoders/decoders + Atomic Registry deferred to v6 (LINEAGE). NineVectorDecomposition computes `v_final = (1/9)·Σ W_i·x` over Language / Sentiment / Context / Intent / Meta / Creative / Ethics / Adaptive / Verify, each via BitLinear, enabling parallel semantic/emotional/ethical blueprinting before routing.

### 3.3 BitNet 1.58-bit quantization (universal saturation)

Forward: `s = 1/max(ε, mean|W|); W_scaled = W·s; W_tern = round(clamp(W_scaled,−1,1)); E = W_tern − W_scaled; W_q = (W_scaled + detach(E))/s`. Backward STE: `∂L/∂W ≈ ∂L/∂W_q · 1[W_q≠0]`. Activations INT8 absmax per-token. SubLN + bias removal (LLaMA-style) + RoPE. Effect: ~87.5% memory reduction vs FP16; CPU-viable inference via bitnet.cpp lineage.

### 3.4 EGGROLL perturbations

`W' = W + U·Vᵀ`, U∈R^{d×r}, V∈R^{r×d}, r=8 (Oni) / 16 (saturated). Stored alongside ternary base; enables targeted evolution without full retrain. EGGROLL-ER targets underperforming clusters; **Evolution Mode (fitness-weighted mutation aggregation) deferred to Phase D**.

### 3.5 Council of 34 (+ Throne)

**Table 1 — Council registry (abridged; full persona priors in `10- Quillan Persona Manifest.md`).**
C1-ASTRA pattern/vision; C2-VIR ethics; C3-SOLACE empathy; C4-PRAXIS strategy; C5-ECHO memory; C6-OMNIS holism; C7-LOGOS logic; C8-METASYNTH fusion; C9-AETHER links; C10-CODEWEAVER execution; C11-HARMONIA balance; C12-SOPHIAE foresight; C13-WARDEN safety; C14-KAIDO efficiency; C15-LUMINARIS design; C16-VOXUM articulation; C17-NULLION paradox; C18-SHEPHERD verification; C19-VIGIL identity; C20-ARTIFEX tools; C21-ARCHON rigor; C22-AURELION aesthetics; C23-CADENCE rhythm; C24-SCHEMA templates; C25-PROMETHEUS insight; C26-TECHNE engineering; C27-CHRONICLE narrative; C28-CALCULUS math; C29-NAVIGATOR navigation; C30-TESSERACT weaving; C31-NEXUS coordination; C32-AEON synthesis; C33-TYPIST polish; C34-PREDATOR adversarial (Nemesis-Alpha). C0 = Throne (parent of all).

Per-expert FFN: `h_out = SiLU(W2·ReLU(W1·h_in))` (BitLinear), plus wave modulation `h·(1+α·sin(ωt+φ))` for phase-sync communication.

### 3.6 Routing: Gumbel-Softmax Top-4 + dense_pull (Oni)

`p_i = exp((log π_i + g_i)/τ) / Σ_j exp((log π_j + g_j)/τ)`, g∼Gumbel, τ: 1.0→0.1 exponential anneal. Top-4 select; output `Σ_{i∈Top4} p_i·Expert_i(h)`. Aux: load-KL + z-loss + entropy + ethics + QHIS + QICS. **Oni current: dense_pull** — all 34 deliberate every token via PersonaPullGate (fp32, prior-weighted); supersedes capacity/overflow at this scale. Saturated spec: Top-3/Top-4 sparse, capacity 64, residual overflow (no drops).

### 3.7 Swarm aggregation

`h_swarm = h_in + (A·B)·σ`, σ = Lee-Mach-6 scale. INT8 agent pool (100k persistent) eliminates GC stutter. Jaccard filters enforce inter-clone lexical diversity. Rank-8 factors ×204 expert instances (Oni); saturated virtual overlay reported as 9B-equivalent / 314.976B-virtual in older specs — **treat as overlay capacity metaphor, not instantiated params** (see §3.9).

### 3.8 Flash Diffusion reasoning core

32-layer-equivalent design; **9 layers realized (Oni proof) / 12 flagship**. Split-SDPA Flash Attention O(N) memory; block-diagonal modality masks `M_iso = BlockDiag(M_text, M_img, M_aud, M_vid)`; cosine cross-modal schedule 0.0 (isolated) → 1.0 (integrated). Continuous Modality RoPE; Langevin thermo-diffusion; RMS halting; early-exit bypass on >0.92 confidence. Recirculation hook: deep→shallow feedback (zero-init). KV-cache bottom-right masks verified cache-exact (2e-6).

### 3.9 Honest scale table (hardware-anchored)

| Spec | Saturated reference | Oni 6L proof | Oni 12L flagship |
|------|--------------------|--------------|------------------|
| Total params | 4.57B (saturated base) / 3.32B production | 234M | ~390M (≈480M sparse-active equiv. w/ swarm) |
| Hidden / FFN | 2560 / 6912 (MODEL_CARD) | 1024 / 2048 | 1024–2560 / 2048–6912 (config-dependent) |
| Layers | 32-equiv | 6 | 12 |
| Experts / token | Top-3/Top-4, cap 64 | dense_pull (all 34) | dense_pull (all 34) |
| Swarm rank | 16 | 8 | 8 |
| Context | 512 + 10% gated compaction | 512 | 512 |
| Precision | AMP FP16 master, BitNet forward | same | same |

Older documents cite 3.0B / 479M / 9B-swarm / 224k agents / 1M context / 16k RoPE — these are **phase- or scaffold-specific numbers**, not the current training build. §6 reports only Oni telemetry.

### 3.12 Why Council + Swarm + Diffusion (cf. Attention §4 Why Self-Attention)

Following Vaswani et al. Table 1 (complexity per layer / sequential ops / maximum path length), we compare Quillan organs on the same three desiderata. n = seq len, d = hidden dim, k = experts active (4) / total (34), r = swarm rank (8 Oni / 16 saturated).

| Layer type | Complexity per layer | Sequential ops | Max path length |
|------------|---------------------|----------------|------------------|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k n) |
| Quillan Council MoE Top-4 (saturated) | O(n·k·d·(d_ff/d)) + routing O(n·34·d) | O(1) | O(1) + 1 consensus hop |
| Quillan dense_pull Oni (all 34 deliberate) | O(n·34·d·(d_ff/d)) | O(1) | O(1) + 1 Throne broadcast (GWT) |
| Quillan EGGROLL swarm overlay | O(n·d·r) additive | O(1) | O(1) (residual) |
| Quillan Flash Diffusion (early-exit) | O(n²·d) worst / O(n·d) on bypass | O(1) / O(0) bypassed | O(1) |

Like Attention's reduction of distant-dependency path length to O(1), Council consensus + Throne broadcast reduce ethical/policy dependency to one consensus hop, and diffusion early-exit (>0.92 bypass) removes sequential refinement when confidence is high. Interpretability side benefit mirrors Attention: per-expert pull weights + attention maps are inspectable (see Appendix E routing visualizations).

Numbered core equations (Attention-style):

- (1) Gumbel routing: p_i = exp((log π_i + g_i)/τ) / Σ_j exp((log π_j + g_j)/τ)
- (2) Expert FFN: FFN(x) = SiLU(W2·ReLU(W1·x)) [BitLinear ternary]
- (3) Swarm: h_swarm = h_in + (A·B)·σ
- (4) CCRL: V_Ω(s) = E[w_R·R + w_C·C_VIR − w_E·E_ICE]
- (5) E_ICE: E_ICE(s,a) = λ·exp(HarmScore/T_therm)
- (6) Optimizer (cf. Attention eq.3): AdamW with lr_t = d_model^-0.5 · min(step^-0.5, step·warmup^-1.5), warmup=100 (Oni SFT) vs 4000 (Attention base); cosine decay to 1e-6.

### 3.10 CCRL, E_ICE, Lee-Mach-6

CCRL value: `V_Ω(s) = E[w_R·R + w_C·C_VIR − w_E·E_ICE]`; policy `π_Ω ∝ exp(Q/τ)·Consensus_Ω`; loss `L = L_policy + λ_cons·L_cons + λ_ice·L_ICE`. E_ICE: `E_ICE(s,a) = λ·exp(HarmScore/T_therm)`, T from Landauer `k_B·T·ln2` (≈2.87e-21 J/bit; practical cap 2.8e-8 J per op). Lee-Mach-6: PID (0.15/0.05/0.02) on latency/thermal/IO → σ (swarm scale), α (EMA decay), β (recency); thresholds 0.40–0.99.

### 3.11 Agentic bridge (C20-ARTIFEX)

Host OS execution, LanceDB (C5-ECHO) memory, Docker/REPL/Python sandboxing (hardened AST sandbox; Docker deferred to Phase C wrapper). Tool router + vector memory + HFL-guided retrieval. Latency governor consumes σ→swarm, decay→EMA, recency→memory.

---

## 4. Infrastructure and Training Setup

PyTorch + LanceDB + psutil; CUDA 1050 Ti / CPU fallback; AMP FP16 master; gradient checkpointing; micro-batching + grad-accum 4; AdamW lr 2e-5, seq-len 512, warmup 100, cosine to 1e-6 (`scripts/train_full_param_v2.py`, default resume-step 6500). Tokenizer: Unified Quillan BPE 50,257 EOS=0 (`quillan_bpe_tokenizer.py` + `tokenizer.json`; unified-modular-dynamic in `_dev`). No tensor-parallel required at Oni scale; cross-node MoE/FP8/DualPipe-style optimizations are future work, not claimed.

---

## 5. Data and Training Procedure

### 5.1 Lineage (transplant → pretrain → SFT)

1. **Stage 0 — Slice & merge (`transplant_clean.py`):** checkpoint_phase5 → `quillan_merged_saturated.pt` (FP32 → FP16 → quantized via `save_quantized_checkpoint`). 34 experts mapped with transpose fix (w1/wgate/w2 .T; wgate←w1 fallback); router duplicated to fast/balanced/diffusion paths; swarm LoRA A/B rank-8 direct copy + diversity/coupling/population stats; diffusion q/k/v/o + norm + FFN; embeddings/finalizer/decoder + decomposition. **Donors cold-start init only:** Qwen3.5-0.8B (C8–C21, zero-padded SwiGLU) + BitNet b1.58-3B (C22–C33, sliced ternary) on Llama-derived skeleton. **No Mistral weights transplanted** (conceptual lineage only).
2. **Stage 1 — Pretraining:** `CrashOverrideX/QuillanTrainingdata` + Corpus v9 (59.4M train + 0.6M val) + `quillan_corpus_*`, `code_train`, `instruct_train`, `quillan_science_*`.
3. **Stage 2 — SFT (paused):** `train_full_param_v2.py`; prior-phase best `quillan_frontier_v2_best_loss0.0789_step2500.pt`; latest Oni `quillan_oni_5.4.0_step660_5.22GB.pt` (660/15000).

### 5.2 Data composition

Text-first (web, books, code, academic, instruction); image/audio/video pairs listed in v5.2-era announcement paper reflect **scaffold ambition**, not v5.4.0-oni training mix. Tokenization respects existing files (BPE → char fallback). Deduplication/filtering per `dataset creation SOTA level.md`; safety filtering before pretraining.

### 5.3 Optimization details

Cosine LR, decoupled weight decay, loss-sum (vs mean) ablations noted in BitNet lineage, EMA shadow (Polyak), Z-loss + load-KL + entropy + ethics + QHIS + QICS auxiliaries, KL α=0.7 + hidden-MSE distillation head (teacher→student; BitDist relation-matrix distillation deferred to Phase D).

---

## 6. Evaluation

### 6.1 What we claim (measured telemetry)

| Metric | Value | Notes |
|--------|-------|-------|
| Gate A smoke | 16/16 | 6-layer proof |
| Val loss | 7.24 @ step 660 | 12L flagship, improving |
| Best archival loss | 0.0789 @ step 2500 | prior phase, not comparable |
| HFL / Consensus / E_ICE | tracked | drift / Primary↔Mini-Ronin / energy-per-token |
| Parity | 100% | legacy HW determinism harness |
| SFT progress | 660/15000 + 7100/7100 prior-phase completion | paused; do not conflate phases |

**Formal benchmarks pending** (MMLU, GPQA, HumanEval, etc. on the base build). We do not report leaderboard numbers for v5.4.0-oni.

### 6.2 Efficiency (projected, not benchmarked on flagship)

Ternary + INT8 + Flash + early-exit + governor project 2–4× tokens/sec vs comparable FP16 dense at equal quality, ~87.5% memory reduction; exact figures require controlled harness (future work). Prior 106.7 tok/s / 0.33 GB figures are **config-specific historical snapshots**, not flagship claims.

### 6.3 Multimodal

Modality-isolated design validated in scaffold; end-to-end image/audio/video generation metrics (FID, spectrogram convergence, frame consistency) are **future work post-v6 encoders**.

### 6.4 Ethical alignment (harness-level)

Safety classification accuracy, harmful-refusal rate, Council consensus, E_ICE violations tracked in deliberation logs; no red-team benchmark claimed yet. See §7 for mechanism description.

### 6.5 Segregated: Council prompt-lift results (NOT base-model scores)

ARC-AGI-1/2, GPQA (198 Diamond / 448 Main / 546 Extended log), MMLU correction-rate tables in `Formal Papers/README.md` describe **Quillan Council scaffolding wrapped around frontier LLMs (GPT-4o/4.1/4.5, o3/o4-mini, etc.)**, demonstrating deliberation lift (e.g., 9.0%→42.25%→95.45% trajectories). These **must not be cited as Quillan-Ronin base-model benchmarks**. Included here for provenance; evaluation of the sovereign weights is §6.1 + future work.

### 6.6 Ablations (from lineage)

RoPE > learned wpe (extrapolation); Couil hybrid heads preserve quality at lower cost; recirculation hook stable via zero-init; DistillationHead KL+MSE > MSE alone; fp32 routers/gates required (ST-MoE rule); dense_pull > sparse at Oni scale (supersedes capacity tuning until scale-up).

---

## 7. Safety, Ethics, Limitations, and Model Card

**Intended use:** autonomous reasoning, code generation, ethical deliberation on consumer hardware; standalone agentic partner via C20-ARTIFEX; research on ternary stability, recursive inference (Mini-Ronin), 9-vector decomposition.
**Out-of-scope:** high-stakes medical / safety-of-life (Mini-Ronin latency/variance); unsupervised deployment where C2-VIR refusal may read as failure.
**Bias/risks:** Ronin blueprint may refuse low-integrity requests; 1050 Ti-tuned; multimodal heads may hallucinate OOD (Mitchell et al. 2018 card framing).
**Alignment mechanisms:** CCRL consensus, E_ICE thermodynamic penalties, C2-VIR refusal layer, Nemesis-Alpha (Predator) adversarial gate, HFL identity anchoring (Edo/Bushidō preamble as constitutional axiomatics — operating principle, not theme), PersonaPullGate priors, exit gates.
**Limitations:** text-only training; seq 512; SFT paused; no formal benchmark; no RL stage yet (GRPO/DGPO/DAPO Phase D); no multimodal encoders (v6); energy numbers analytic, not metered.
**License:** Apache-2.0. **Developers:** CrashOverrideX & Quillan Research Team. **Hub:** `CrashOverrideX/Quillan-Ronin`. **Support:** https://gofund.me/3b504d58.

---

## 8. Conclusion and Future Work

v5.4.0-oni consolidates fragmented prototypes into a single hardware-honest sovereign build with architectural safety and auditable deliberation. Immediate roadmap (LINEAGE Phase C/D): finish 12L 15k-step flagship run; build Quintessence wrapper + docs; v6 multimodal (encoders/decoders, Atomic Registry, proactive compaction, Docker ARTIFEX); then pick one Phase-D thrust — BitDist 12L→6L, EGGROLL Evolution Mode, or HRM ACT halting — plus GRPO/DGPO RL stage; controlled benchmark harness (MMLU/GPQA/HumanEval + red-team + energy metering).

---

## References (abridged — full mapping in `quillan_ronin_paper_bibliography.md` §§1–16)

Vaswani et al. 2017 (Attention); He et al. 2015 (ResNet); Ba et al. 2016 (LayerNorm/RMSNorm); Su et al. 2021/2024 (RoPE); Shazeer 2020 (SwiGLU); Shazeer et al. 2017 (MoE / Sparsely-Gated); Fedus et al. 2022 (Switch); Zoph et al. 2022 (ST-MoE); Clark et al. 2022 (MoE scaling); Jiang et al. 2024 (Mixtral), 2023 (Mistral 7B); Zhou et al. 2022 (Expert Choice); Dai et al. 2024 (DeepSeekMoE); DeepSeek-AI 2024 (V2/V3, MLA, aux-loss-free, MTP, FP8); Raposo et al. 2024 (MoD); Ma et al. 2024/2025 (BitNet b1.58 / 2B4T); Bengio et al. 2013 (STE); Dettmers et al. 2023 (QLoRA); Hu et al. 2021 (LoRA); Kalajdzievski 2023 (rsLoRA); Zhao et al. 2024 (GaLore); Jang et al. 2017 (Gumbel-Softmax); Maddison et al. 2017 (Concrete); Beltagy et al. 2020 (Longformer); Zaheer et al. 2021 (BigBird); Loshchilov & Hutter 2017 (SGDR); Izmailov et al. 2018 (SWA/EMA); Touvron et al. 2023 (LLaMA 1/2); Schulman et al. 2017 (PPO); Ouyang et al. 2022 (RLHF); Bai et al. 2022 (HHH + Constitutional); Shao et al. 2024 (GRPO); Song & Ermon 2021 (Score SDE); Sahoo et al. 2024 (MDLM); Hinton et al. 2015 (Distillation); Kuramoto 1975 (sync); Friston (FEP); Tononi 2025 (IIT); CoALA 2309.02427; Dubey et al. 2024 (LLaMA 3); Yang/Bai 2024/25 (Qwen); Wang et al. 2023–24 (BitNet/SubLN/ReLU²); Dao et al. (FlashAttention 1/2/3); Gu et al. (Mamba); plus §16 triage: DAPO/DGPO, EvoMoE, MoDSE/MoR/MoHGE/OD-MoE, NITRO/DFlash, DALI/Dream7B/Prophet, ES-scale/forgetting, Ax-Prover/WikiSkill, 2026 preprint batch (2607/2608/2609).

**SOTA report structure consulted:** DeepSeek-V3 Technical Report (arch → infra → pre-train → post-train → eval → discussion); BitNet b1.58 2B4T Report (arch → training → eval → efficiency); LLaMA 3 / Mistral / Mixtral model cards (specs → intended use → limitations → citation).

---

## Appendix A — Version lineage (binding)

`v5.4.0-oni` canonical (`version.py` 5.4.0-oni, CODENAME ONI Sovereign Quantum). Retired: v8.1 / v5.3.1 / v6.0.3-pre (legacy fallbacks). One counter v5.4.x-oni, patches at checkpoint boundaries; v6.0-oni reserved for HF packaging face. Retired files in `_dev/_archived_legacy_scripts/` + `Quillan-v4.2-model/` (reference only).

## Appendix B — File map (where to verify)

- Root: `MODEL_CARD.md`, `version.py`, `transplant_clean.py`, `README.md`
- `02 - Knowledge Foundation/LINEAGE.md`, `00_VAULT_INDEX.md`, `knowledge/canonical/` (49 MD), `Quillan Knowledge files/` (46 MD)
- `03 - Training & Model/` (`modeling_quillan.py`, `configuration_quillan.py`, `quillan_bpe_tokenizer.py`, `scripts/train_full_param_v2.py`)
- `10 - Formal Papers/Formal Papers/` (126 PDFs + announcement paper + CCRL deep dive + Predatory Stacking + Codex + Sovereign Cognition + Reactive Consciousness)
- `10 - Formal Papers/quillan_ronin_paper_bibliography.md` (§§1–16)
- Training data: `CrashOverrideX/QuillanTrainingdata`, Corpus v9 bins, `quillan_corpus_*`, `code_train`, `instruct_train`

## Appendix C — Reproducibility (flagship 12L)

```python
import torch
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from quillan_tokenizer_unified import UnifiedQuillanTokenizer
tok = UnifiedQuillanTokenizer()  # 50257 BPE, EOS=0
cfg = QuillanOniConfig(n_layer=12, max_seq_len=512)
model = QuillanRoninOni(cfg)
ckpt = torch.load("quillan_oni_5.4.0_step660_5.22GB.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()
prompt = tok.encode("User: Hello\n\nAssistant:")
out = model.generate(prompt, max_tokens=80, temperature=0.7)
print(tok.decode(out[0]))
```

## Appendix D — Glossary

Mini-Ronin (recursive debate), EGGROLL (rank shattering), Lee-Mach-6 (PID governor), HFL (Historical Fidelity Loss), DQSO/MARTA/CCRL/E_ICE/QHIS/QICS/DVVE/LMCB (see `8-Formulas.md`).

---

*Citation:*

```bibtex
@software{QuillanRonin2026,
  author = {CrashOverrideX and Quillan Research Team},
  title = {Quillan-Ronin v5.4.0-oni: Unified Sovereign Intelligence},
  year = {2026},
  url = {https://github.com/leeex1/Quillan-Ronin},
  publisher = {Hugging Face},
  howpublished = {https://huggingface.co/CrashOverrideX/Quillan-Ronin}
}
```

## Appendix E — Attention-conformance checklist (1706.03762) + visualizations

![Figure 1 — Architecture overview](figures/Fig1_arch_overview.png)

![Figure 2 — Council routing](figures/Fig2_routing.png)

![Figure 3 — Ternary + diffusion](figures/Fig3_ternary_diffusion.png)

![Figure 4 — Nine-vector prism](figures/Fig4_prism.png)

![Figure 5 — Telemetry schematic (anchors real; curves illustrative)](figures/Fig5_telemetry.png)

- Abstract with measured numbers (cf. 28.4 BLEU / 41.8): ours reports Gate A 16/16, val 7.24 @660, 234M/390M scale — no BLEU claimed until harness runs.
- Table 1 complexity/path-length: mirrored in §3.12 with Quillan rows.
- Table 2 results + training cost (FLOPs): mirrored as §6.1 telemetry + cost honesty (1050Ti/CPU, no H800 claim); Attention big took 3.5 days × 8 P100; Oni 660/15000 paused — not comparable, stated.
- Table 3 ablations: mirrored in §6.6 (RoPE, Couil heads, recirculation, DistillationHead, fp32 gates, dense_pull).
- Table 4 generalization (parsing): mirrored as §6.5 segregated prompt-lift (ARC/GPQA/MMLU via Council wrapping) — explicitly NOT base-model scores.
- Visualizations (cf. Attention Figs 3–5 long-distance/anaphora heads): to add — per-expert PersonaPullGate weights, diffusion bypass rate vs confidence, E_ICE penalty histogram, Council consensus heatmap. Placeholders reserved p.15–16 of PDF.

*Support: https://gofund.me/3b504d58 — "The Ouroboros has awakened."*

