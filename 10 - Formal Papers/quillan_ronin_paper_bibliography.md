# Quillan-Ronin v5.4.0-oni — Master Paper Bibliography (retires v5.3.1 / v8 labels)

> **Complete inventory of all academic papers** mapped to every architectural module, compression technique, optimizer tweak, and training strategy used in the Quillan-Ronin model.
> 
> ✅ = Downloaded to `C:\02_QUILLAN\10 - Formal Papers\Formal Papers\`  
> 🔍 = Referenced (web search only, not downloaded as PDF)  
> 📄 = In `10 - Formal Papers\Formal Papers\` or `02 - Knowledge Foundation\Quillan Knowledge files\` directory
> 
> **Canonical version:** Quillan-Ronin **v5.4.0-oni** (see `02 - Knowledge Foundation\LINEAGE.md` + root `version.py`). Older labels v5.3.1 / v8.1 are retired.

---

## 1. Core Transformer Architecture

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| Attention Is All You Need (Vaswani et al., 2017) | 1706.03762 | ✅ `Self_Attention_Positional_Encoding_Vaswani.pdf` | `CouilAttention`, all attention layers |
| Deep Residual Learning (He et al., 2015) | 1512.03385 | ✅ `ResNet_Deep_Residual_Learning.pdf` | Residual connections in `QuillanRoninSovereign` |
| Layer Normalization / RMSNorm (Ba et al., 2016) | 1607.06450 | ✅ `Layer_Normalization_RMS_Norm.pdf` | `nn.LayerNorm` throughout model |
| RoPE / RoFormer (Su et al., 2021) | 2104.09864 | ✅ `RoPE_RoFormer_Rotary_Position_Embedding.pdf` | Rotary embeddings in `CouilAttention` |
| GLU Variants / SwiGLU (Shazeer, 2020) | 2002.05202 | ✅ `SwiGLU_GLU_Variants_Improve_Transformer.pdf` | SwiGLU activation in expert FFN layers |

---

## 2. Mixture-of-Experts (MoE) Architecture

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| Outrageously Large Neural Networks (Shazeer et al., 2017) | 1701.06538 | ✅ `MoE_Outrageously_Large_Neural_Networks.pdf` | `EvolvableVectorizedMoE`, gating concept |
| Sparsely-Gated MoE (Shazeer et al., 2017) | 1701.06538 | ✅ `Sparsely_Gated_MoE.pdf` | `ComplexityRouter`, top-k routing |
| Switch Transformers (Fedus et al., 2022) | 2101.03961 | ✅ `Switch_Transformers_Scaling_MoE.pdf` | Expert instability diagnosis, bfloat16 MoE |
| ST-MoE: Stable & Transferable (Zoph et al., 2022) | 2202.08906 | ✅ `ST-MoE_Stable_Sparse_Experts.pdf` + `ST-MoE_Stable_Transferable_Sparse_Experts.pdf` | **Z-loss** formula, router stability |
| MoE Scaling Laws (Clark et al., 2022) | 2202.01169 | ✅ `MoE_Scaling_Laws_Experts.pdf` | Expert count / capacity design |
| Mixtral of Experts (Jiang et al., 2024) | 2401.04088 | ✅ `Mixtral_of_Experts.pdf` + `Mixtral_of_Experts_v2.pdf` + `Mixtral_Sparse_MoE_Architecture.pdf` | Top-2 sparse MoE reference |
| Expert Choice Routing (Zhou et al., 2022) | 2202.09368 | ✅ `Expert_Choice_Routing_MoE.pdf` | Alternative to token-choice (ComplexityRouter ref.) |
| DeepSeekMoE (Dai et al., 2024) | 2401.06066 | ✅ `DeepSeekMoE_Fine_Grained_Expert.pdf` | Fine-grained expert segmentation, shared experts |
| DeepSeek-V3 Technical Report (2024) | 2412.19437 | ✅ `DeepSeek_V3_Technical_Report.pdf` | Auxiliary-loss-free balancing, MTP, FP8 training |
| Mixture-of-Depths (Raposo et al., 2024) | 2404.02258 | ✅ `Mixture_of_Depths_Dynamic_Compute.pdf` | Dynamic compute allocation (cf. `ComplexityRouter`) |

---

## 3. BitNet / Quantization / Compression

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| BitNet b1.58 (Ma et al., 2024) | 2402.17764 | ✅ `BitNet_b1.58_1-bit_LLMs.pdf` | `BitLinear`, `_quantize_1_58()` — ternary weights |
| STE / Straight-Through Estimator (Bengio et al., 2013) | 1308.3432 | ✅ `STE_Straight_Through_Estimator_Bengio.pdf` | STE in `BitLinear.forward()` |
| Understanding STE in Quantized Nets | — | ✅ `Understanding_STE_Quantized_Nets.pdf` | STE implementation best practices |
| QLoRA (Dettmers et al., 2023) | 2305.14314 | ✅ `QLoRA_4bit_Quantized_Finetuning.pdf` + `QLORA_Efficient_Finetuning_Quantized_LLMs.pdf` | 4-bit finetuning strategy |
| LoRA (Hu et al., 2021) | 2106.09685 | ✅ `LoRA_Low_Rank_Adaptation_LLMs.pdf` | `CouncilExpertSwarm` LoRA-style rank-R adapters |
| rsLoRA (Kalajdzievski, 2023) | 2312.03732 | ✅ `LoRA_rsLoRA_Rank_Stabilized.pdf` | Rank-stabilized scaling `α/√r` |
| GaLore (Zhao et al., 2024) | 2403.03507 | ✅ `GaLore_Gradient_Low_Rank_Projection.pdf` | Gradient low-rank projection (future optimization) |
| Matrix Decomposition in DL | — | ✅ `Matrix_Decomposition_Deep_Learning_Eigen.pdf` | Eigendecomposition in `CouncilExpertSwarm` |
| Multi-Head Latent Attention (DeepSeek-V2) | 2405.04434 | ✅ `Multi_Head_Latent_Attention_Decomposition.pdf` | KV compression concepts → `CouilAttention` |

---

## 4. Routing & Adaptive Computation

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| Gumbel-Softmax (Jang et al., 2017) | 1611.01144 | ✅ `Gumbel_Softmax_Categorical_Reparameterization.pdf` | `ComplexityRouter` Gumbel sampling, τ annealing |
| Concrete Distribution (Maddison et al., 2017) | 1611.00712 | ✅ `Concrete_Distribution_Continuous_Relaxation.pdf` | Theoretical foundation for differentiable routing |
| Token Routing / Adaptive Halting | — | ✅ `Token_Routing_Adaptive_Computation_Halting.pdf` | `ComplexityRouter` early-exit potential |
| Longformer / Sparse Attention (Beltagy et al., 2020) | 2004.05150 | ✅ `Longformer_Sparse_Local_Global_Attention.pdf` | Sparse attention patterns |
| BigBird / Sparse Attention (Zaheer et al., 2021) | 2007.14062 | ✅ `Sparse_Attention_Big_Bird_LongRange.pdf` | Block-sparse attention reference |

---

## 5. Training Optimization & Scheduling

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| SGDR: Warm Restarts / Cosine Annealing (Loshchilov & Hutter, 2017) | 1608.03983 | ✅ `SGDR_Warm_Restarts_Cosine_Annealing.pdf` | LR schedule in `train_sovereign.py` |
| EMA / Stochastic Weight Averaging (Izmailov et al., 2018) | 1803.05407 | ✅ `SWA_Averaging_Weights_Wider_Optima.pdf` | EMA shadow model, Polyak averaging |
| Gradient Accumulation / Micro-batching | — | 🔍 (web research, no single paper) | `accum_steps` in training loop |
| LLaMA: Open & Efficient (Touvron et al., 2023) | 2302.13971 | ✅ `LLaMA_Open_Efficient_Foundation.pdf` + `LLaMA_Memory_Efficient_Training.pdf` | Training recipe reference |
| LLaMA 2 (Touvron et al., 2023) | 2307.09288 | ✅ `Llama2_Technical_Report.pdf` + `Llama2_Safety_RLHF_Constitution.pdf` | RLHF + safety constitution |
| Mistral 7B (Jiang et al., 2023) | 2310.06825 | ✅ `Mistral_7B.pdf` | Sliding window attention, efficiency ref. |
| Quantum-Inspired Classical Optimization | — | ✅ `Quantum_Inspired_Classical_Optimization.pdf` | `QuantumFormulasEngine` design basis |

---

## 6. Reinforcement Learning from Human Feedback (RLHF)

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| PPO (Schulman et al., 2017) | 1707.06347 | ✅ `PPO_Proximal_Policy_Optimization.pdf` | `CCRLFramework` — PPO component |
| RLHF for LLMs (Ouyang et al., 2022) | 2203.02155 | ✅ `RLHF_Proximal_Policy_Optimization_LLM.pdf` | Training pipeline concept |
| Training Helpful, Harmless, Honest (Bai et al., 2022) | 2204.05862 | ✅ `Training_Helpful_Harmless_Honest_AI.pdf` | HHH alignment framework |
| Constitutional AI (Bai et al., 2022) | 2212.08073 | ✅ `Constitutional_AI_Harmlessness_RLAIF.pdf` | `PrimeCovenantFramework` — RLAIF concept |
| Soft Actor-Critic / Entropy Regularization | — | ✅ `Soft_Actor_Critic_Entropy_Regularization.pdf` | Entropy terms in `CCRLFramework` |
| GRPO (DeepSeek, 2024-2025) | 2402.03300 / 2501.12948 | ✅ `GRPO_DeepSeekMath_RL_Reasoning.pdf` + `DeepSeek_R1_Reasoning_RL.pdf` | Critic-free RL (future CCRL upgrade) |

---

## 7. Diffusion & Generative Components

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| Score-Based Diffusion / SDEs (Song & Ermon, 2021) | 2011.13456 | ✅ `Score_Based_Diffusion_SDEs_Song_Ermon.pdf` | `SovereignFlashDiffusionCore` — theoretical basis |
| Masked Diffusion Language Models (Sahoo et al., 2024) | 2406.07524 | ✅ `Masked_Diffusion_Language_Models.pdf` | `SovereignFlashDiffusionCore` — discrete diffusion |

---

## 8. Knowledge Distillation

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| Distilling the Knowledge (Hinton et al., 2015) | 1503.02531 | ✅ `Distilling_Knowledge_Neural_Networks_Hinton.pdf` | `DistillationHead` — dark knowledge, temperature |

---

## 9. Synchronization & Oscillation Theory

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| Kuramoto Model — Original (Kuramoto, 1975) | textbook | ✅ `Kuramoto_Model_Original_Sync_Theory.pdf` | `DynamicQuantumSwarmOscillation` (DQSO) |
| Kuramoto Coupled Oscillators + Neural Sync | — | ✅ `Kuramoto_Coupled_Oscillators_Neural_Sync.pdf` | DQSO phase coupling |
| Kuramoto + GNN Over-Smoothing | — | ✅ `Kuramoto_GNN_Over_Smoothing_Prevention.pdf` | DQSO preventing expert collapse |

---

## 10. Thermodynamic & Free Energy Theory

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| Free Energy Principle (Friston) | multiple | 🔍 (book/review, no single PDF) | `MARTAThermodynamicGating` — variational free energy |
| Thermodynamic Regulation of Gibbs Training (2026) | 2603.02525 | ✅ `Thermodynamic_Regulation_Gibbs_Training_2026.pdf` | Dynamic temperature in MARTA |
| Thermodynamic Bounds on DNN Energy (2025) | 2503.09980 | 🔍 (reference only — indirect) | Free-energy functional → inference cost |

---

## 11. Consciousness & Cognitive Architecture Theory

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| Integrated Information Theory (Tononi, 2025) | 2510.25998 | ✅ `IIT_Consciousness_First_Tononi_2025.pdf` | `NineVectorDecomposition` — Φ-inspired integration |
| Global Workspace Theory → LLMs (2026) | multiple | 🔍 (2026 preprints — reference only) | `QuillanAgenticExecutor` broadcast mechanism |
| CoALA: Cognitive Architectures for Language Agents | 2309.02427 | ✅ `CoALA_Cognitive_Architectures_Language_Agents.pdf` | Parliament of Experts design pattern |
| Cognitive Fabric Nodes / Multi-Agent Coordination | 2025/2026 | 🔍 (2026 survey — reference only) | Council coordination, swarm orchestration |

---

## 12. Ethical Constraint & Safety

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| Sleeper Agents / Deceptive Alignment | — | ✅ `Sleeper_Agents_Deceptive_Alignment_Safety.pdf` | `EthicalImpactConstraintEngine` adversarial checks |
| System Prompt Identity Persistence | — | ✅ `System_Prompt_Identity_Persistence_LLM.pdf` | Identity anchor in `QuillanAgenticExecutor` |
| Constitutional AI (Bai et al., 2022) | 2212.08073 | ✅ `Constitutional_AI_Harmlessness_RLAIF.pdf` | `PrimeCovenantFramework` constitutional principles |

---

## 13. Memory & Vector Database

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| LanceDB / Vector DB for Training | — | ✅ `LanceDB_Vector_DB_Training.pdf` | `QuillanAgenticExecutor._init_memory_table()` |

---

## 14. Quillan-Specific Research & Internal Papers

| Paper | Source | Status | Maps To |
|:------|:-------|:-------|:--------|
| Quillan: A Cognitive Parliament | — | ✅ 📄 `Quillan_A_Cognitive_Parliament.pdf` | Overall architecture philosophy |
| Quillan Architecture Deep Dive Audit | — | ✅ 📄 `Quillan_Architecture_Deep_Dive_Audit.pdf` | Implementation audit |
| Quillan Mind Architecture | — | ✅ 📄 `Quillan_Mind_Architecture.pdf` | Cognitive architecture design |
| Quillan v4.2 LLM Wrapper | — | ✅ 📄 `Quillan_v4_2_new_LLM_Wrapper.pdf` | Wrapper architecture |
| Quillan-Ronin: The AGI | — | ✅ 📄 `Quillan-Ronin The AGI.pdf` | High-level AGI overview |
| Quillan-Ronin: The Path to True AGI | — | ✅ 📄 `Quillan-Ronin The path to true AGI.pdf` | Philosophical foundation |
| Quillan-Ronin: Advanced Cognitive Engine | — | ✅ 📄 `Quillan-Ronin_ Advanced Cognitive Engine.pdf` | Engine internals |
| Quillan-Ronin AGI Architecture | — | ✅ 📄 `Quillan-Ronin_AGI_Architecture.pdf` | Architecture specification |
| The Quillan Codex | — | ✅ 📄 `The_Quillan_Codex_(7).pdf` | Full codex |
| Quillan-Samurai.md | — | ✅ 📄 `Quillan-Samurai.md` | v5.2.3 routing, EGGROLL rank, Z-loss config |
| Sovereign Cognition Beyond Context Bottlenecks | — | ✅ 📄 `Sovereign Cognition Beyond Context Bottlenecks.pdf` | Context extension theory |
| Reactive Consciousness / AGI | — | ✅ 📄 `Reactive Conciousness.pdf` + `Reactive_AGi_Paper.pdf` | Consciousness module theory |
| Lee-X Humanized Protocol | — | ✅ 📄 `Lee_X_Humanized_Protocol.pdf` | `LeeMach6Governor` spec |

---

## 15. Indirect / Supporting Research

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| Pattern to Partner | — | ✅ `Pattern_to_Partner.pdf` | Agent interaction patterns |
| The 6.4 Million Token Anomaly | — | ✅ 📄 (2 copies) | Long-context behavior analysis |
| The Virality Paradox | — | ✅ 📄 | Neural synchrony from brainwave data |
| Predatory Stacking / ALA | — | ✅ 📄 (3 variants) | Hypergraph theory → routing |
| Prompt-Ware Report | — | ✅ 📄 `Prompt_Ware_Report.pdf` | Prompt injection defense |
| Peer Review of Quillan-Ronin 4o Distillation | — | ✅ 📄 | External review |

---

## Summary Statistics

| Category | Total Papers | Downloaded ✅ | Web-Only 🔍 |
|:---------|:-------------|:-------------|:------------|
| Core Transformer | 5 | 5 | 0 |
| MoE Architecture | 10 | 10 | 0 |
| Quantization / Compression | 9 | 9 | 0 |
| Routing / Adaptive | 5 | 5 | 0 |
| Training Optimization | 7 | 6 | 1 |
| RLHF | 6 | 6 | 0 |
| Diffusion | 2 | 2 | 0 |
| Distillation | 1 | 1 | 0 |
| Synchronization / Oscillation | 3 | 3 | 0 |
| Thermodynamic Theory | 3 | 1 | 2 |
| Consciousness / Cognitive | 4 | 2 | 2 |
| Ethics & Safety | 3 | 3 | 0 |
| Memory / VectorDB | 1 | 1 | 0 |
| Quillan Internal | 15 | 15 | 0 |
| Supporting / Indirect | 6 | 6 | 0 |
| **TOTAL** | **80** | **76** | **4** |

> [!NOTE]
> The remaining 4 🔍 entries are topics without a single downloadable PDF — they are survey/reference-level knowledge (Friston's Free Energy Principle textbook, GWT survey preprints, Cognitive Fabric Nodes survey, and gradient accumulation as a general technique). All paper-backed components have their PDFs downloaded.

---

## Total PDF Count (audited 2026-09-12)

- **Formal Papers folder**: `C:\02_QUILLAN\10 - Formal Papers\Formal Papers\` — **126 PDFs** (was 136 in old Downloads folder; consolidated)
- **Knowledge files**: `C:\02_QUILLAN\02 - Knowledge Foundation\Quillan Knowledge files\` — **46 MD** + canonical mirror `knowledge\canonical\` (**49 MD**)
- **Total unique academic references**: **80+ mapped + 16. New 2025–2026 Additions below (previously unmapped)**

---

## 16. New 2025–2026 Additions (present in folder, previously unmapped — audited 2026-09-12)

| Paper (filename) | Topic | Maps To |
|:-----------------|:------|:--------|
| `2607.24720v1.pdf`, `2607.28607v1.pdf`, `2608.03874v1.pdf`, `2608.05446v1.pdf`, `2608.12875v1.pdf`, `2608.13482v1.pdf`, `2608.16801v1.pdf`, `2608.17271v1.pdf`, `2608.17286v1.pdf`, `2608.17896v1.pdf`, `2608.17981v1.pdf`, `2609.11677v1.pdf` | 2026 preprints batch | Triage: map to MoE / routing / diffusion / eval upgrades; see `10 - Formal Papers\Formal Papers\_Index.md` |
| `DAPO_Open_Source_LLM_RL.pdf` | DAPO RL | CCRL upgrade path (critic-free RL, cf. GRPO/DGPO) |
| `DGPO_ Distribution Guided Policy Optimization for Fine Grained Credit Assignment.pdf` | DGPO | CCRL fine-grained credit assignment (Phase D candidate) |
| `DeepSeekMath_GRPO.pdf` | GRPO | CCRL RL stage (already in §6; duplicate copy — dedupe) |
| `EvoMoE_2505.23830.pdf` | Evolving MoE | Expert evolution / EGGROLL Evolution Mode (Phase D) |
| `MoE_CPU_GPU_2512.16473.pdf`, `OD_MoE_2512.03927.pdf`, `MoHGE_2604.23108.pdf`, `MoR_2507.10524.pdf`, `MoDSE_2409.12210.pdf` | MoE systems / depth / routing | ComplexityRouter capacity, overflow, deployment on 1050Ti/CPU |
| `NITRO_D_2407.11698.pdf`, `DFlash_2602.06036.pdf` | Flash / sparse attention | SovereignFlashDiffusionCore upgrades |
| `DALI_2602.03495.pdf`, `Dream7B_2508.15487.pdf`, `Prophet_2508.19982.pdf` | Diffusion LM / generative | Diffusion core discrete-diffusion roadmap |
| `BitNet_v2_2504.18415.pdf`, `BitNet_b1.58_2B4T_Technical_Report.pdf`, `BitNet_Scaling_1-bit_Transformers.pdf`, `BitNet_a4.8_4-bit_Activations.pdf`, `bitnet_cpp_Efficient_Edge_Inference.pdf`, `bitnet_1-bit_AI_Infra.pdf` | BitNet family | BitLinear STE, 1050Ti edge inference (core to v5.4.0-oni §3) |
| `Deep_Optimizer_States_2410.21316.pdf`, `ProTrain_2406.08334.pdf`, `ZeRO_Infinity_2104.07857.pdf` | Memory-efficient training | `train_full_param_v2.py` grad-accum / ZeRO-style sharding |
| `ES_at_Scale_2509.24372.pdf`, `ES_Forgetting_2601.20861.pdf`, `ES_Forgetting_Fix_2605.30148.pdf` | Evolution strategies / forgetting | EGGROLL + continual learning safeguards |
| `Ax-Prover_Agentic_Theorem_Proving.pdf`, `WikiSkill_2608.27454.pdf` | Agentic skills / prover | C20-ARTIFEX tool routing, C21-ARCHON rigor |
| `235_Position_LLMs_can_t_jump.pdf`, `2502.08145v1.pdf` + batch `2502.x–2506.x`, `2401.07013v2.pdf`, `2407.12117v1.pdf`, `2410.05686v1.pdf` | Reasoning / position / general | 9-Vector Prism + RoPE + eval harness |

> Action: promoters of §16 into §§2–7 on next pass; current counts exclude §16 from Summary Statistics until mapped.

