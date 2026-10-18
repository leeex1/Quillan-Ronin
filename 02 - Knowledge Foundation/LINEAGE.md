# QUILLAN-RONIN — CANONICAL LINEAGE & VERSION REGISTRY
**Single source of truth for versioning. Established 2026-08-26.**

---

## Current Canonical Build

| | |
|---|---|
| **Version** | **v5.4.0-oni** |
| **File** | `_dev\quillan_v5_4_oni.py` |
| **Class** | `QuillanRoninOni` (config: `QuillanOniConfig`) |
| **Status** | Gate A verified — 16/16 smoke tests passed |
| **Tokenizer** | Unified Modular Dynamic (`_dev\quillan_tokenizer_unified.py`) — custom Quillan BPE 50,257, EOS=0 |
| **Roster** | Throne (Quillan Core, separate) + Council C1–C34 (ASTRA→PREDATOR), 4 wave clusters |
| **Council mode** | dense_pull — all 34 deliberate every token, PersonaPullGate (fp32, prior-weighted) |
| **Data** | `training_data\v9\` — 59.4M train + 0.6M val tokens, unified-BPE packed bins |

---

## Version Counter Rules (binding)

1. **One counter**: v5.4.x-oni. Patches increment x at checkpoint boundaries only.
2. **v6.0-oni** = reserved for the HF packaging face (`modeling_quillan.py` wraps current).
3. **No new version names without a LINEAGE.md entry.** The v8/v9/v10 filename chaos (file names ≠ internal branding ≠ chronology) is retired permanently.
4. Retired files live in `_dev\_archived_legacy_scripts\` and `Quillan-v4.2-model\` — reference only, never train from them.

---

## Organ → Parent Map (what Oni unified, and from where)

| Organ | Source | Disposition |
|---|---|---|
| Unrolled blocks (attn + council-MoE) | Samurai.md embedded ref / v9 | core |
| 34 CouncilExperts (LoRA + EGGROLL swarm) | Samurai.md / v9 | core, rank-8 dense |
| 9-Vector Prism per attention | Samurai.md / v9 | core |
| Dual Q1/Q2 ingestion + dual finalizers + comm gate | Samurai.md / v9 | core |
| Tied embeddings, custom BPE, EOS=0 | v9 + tokenizer unification | core |
| E_ICE / MARTA / DQSO / Covenant / CCRL / 10 Quantum Formulas | Samurai.md / v9 / 8-Formulas.md | core |
| Latency governor (σ→swarm, decay→EMA, recency→memory) | AGI paper Alg.2 / v9 | core, consumed |
| AST sandbox, tool router, vector memory | v9 (hardened, no exec) | core |
| Aux losses: load-KL, z-loss, entropy, ethics, QHIS, QICS | v9 (papers: Mixtral/ST-MoE/CCRL) | core |
| KV-cache bottom-right masks (cache-exact) | v9, verified 2e-6 | core |
| **RoPE** (replaces learned wpe) | v10 branch / Samurai :3545 | **ported v5.4** |
| **Couil hybrid heads** (even dense / odd sparse-topk, absolute window) | 117KB v8 + Quintessence | **ported v5.4** |
| **Recirculation hook** (deep→shallow feedback, zero-init) | v10 branch / Recirculation paper | **ported v5.4** |
| **DistillationHead** (KL α=0.7 + hidden MSE) | 117KB v8 / AGI eq.35 | **ported v5.4** |
| **ModalityIsolatedThermoDiffusion** (Langevin inv-√t, time-emb, RMS halting, α=0.7) | Samurai :4151 | **ported v5.4** |
| **LeeMach6VelocityGovernor** (PID 0.15/0.05/0.02, threshold 0.40–0.99) | Samurai :8358 | **ported v5.4** |
| **Analytic E_ICE** (E_ω = I_s·γ²·k_B·T·ln2) | Samurai :3279 | **ported v5.4** |
| **PersonaPullGate** (Throne assigns deliberation pull; persona priors from Knowledge file 10) | user canon | **new v5.4** |
| **Quality exit gates** (Nullion/Warden/Shepherd + Quillan audit) | user canon + telemetry spec | **new v5.4** |
| **deliberate() loop** (audit → diffusion rounds → gates → Typist polish) | user canon | **new v5.4** |
| fp32 routers/pull-gates | ST-MoE | core rule |
| Wave clusters (Cognitive/Communication/Meta/Systems) | Hierarchy Chain v5.3.3 | inference ordering |

## Consciously Deferred (with reasons)

| Item | Reason | Slot |
|---|---|---|
| Multimodal encoders/decoders + Atomic Registry | text-only corpus | v6 multimodal |
| Proactive compaction (>4096 tok, Conv1d s2) | seq 512 training | wrapper, Phase C |
| Docker ARTIFEX bridge | AST sandbox covers safe-exec | wrapper, Phase C |
| World Modeling Engine (planetary sim) | wrapper-layer system | Phase C build |
| GRPO/DGPO RL stage | compute ceiling | stretch after Phase C |
| HRM ACT learned halting (Q-head halt/continue, deep supervision) | ArXiv Ultima file — upgrade over RMS halting | Phase D |
| BitDist relation-matrix distillation (Q/K/V KL, ternary student vs FP16 teacher) | ArXiv Ultima file — upgrade over hidden-MSE | Phase D |
| EGGROLL Evolution Mode (fitness-weighted mutation aggregation) | gradient-free refinement pass | Phase D |

## Scaling Ratios (documented, hardware-honest)

| Spec (production) | Oni (this hardware) |
|---|---|
| 3.32B params, ~300M active | 234M (6L) / ~390M (12L flagship) |
| FFN 12288 | FFN 2048 |
| Top-3 sparse, capacity 64 | dense_pull (all 34; supersedes capacity/overflow) |
| Swarm 314.976B virtual, rank-64 | rank-8 factors × 204 expert instances |
| hidden 1024–8192 adaptive | hidden 1024 |

## Training Phases

- **Phase B (proof)**: 6-layer Oni, 1,000 steps, existing bins → val-curve + English-sample gate
- **Phase C (flagship)**: 12-layer, 15k steps (~1 epoch) — build Quintessence wrapper + docs during downtime
- **Phase D (pick one)**: BitDist distillation 12L→6L · EGGROLL Evolution Mode · HRM halting upgrade
