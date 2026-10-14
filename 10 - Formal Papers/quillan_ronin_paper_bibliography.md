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
| MoBA: Mixture of Block Attention for Long-Context LLMs (Lu et al., 2025 / Moonshot AI) | 2502.13189 | ✅ `2502.13189v1.pdf` | `MoBAAttention`, `online_softmax_combine` in `moba_attention.py`, `CausalSelfAttention` in `quillan_v5_4_oni.py`; dynamic top-k block routing $I = \bigcup_{g_i>0} I_i$ via mean-pooled key affinity $s_i = \langle q, \text{mean\_pool}(K[I_i]) \rangle / \sqrt{d}$, strict causal future masking, current block self-attention with causal mask, online log-sum-exp softmax stream combination |

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
| Pretraining Large Language Models with NVFP4 (NVIDIA, 2026) | 2509.25149 | ✅ `2509.25149v2.pdf` | `NVFP4Quantizer`, `NVFP4Linear`, `apply_random_hadamard_transform_16` in `paper_11_15_quant_memory_pack.py`; 15-level FP4 E2M1 representation, 16x16 Random Hadamard Transform (RHT) dispersing kurtosis outliers, 2D block weight quantization preserving transpose consistency ($Q(W^T) = Q(W)^T$), stochastic rounding on gradients eliminating bias, selective high-precision preservation |

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
| Comparative Analysis of CPU and GPU Profiling (Gyawali et al., 2023) | 2309.02521 | ✅ `2309.02521v3.pdf` | `StepProfiler`, `CUDATimer` in `paper_01_profiler.py`, `train_oni.py`; real per-phase timing (data/xfer/fwd/bwd/opt/ema), VRAM peak, GPU util/temp/power telemetry |
| MEMO: 1M Sequence Length on 8 GPUs (Zhao et al., 2024) | 2407.12117 | ✅ `2407.12117v1.pdf` | `MemoManager`, `RoundingBuffer`, `MemoMemoryPlanner` in `paper_04_memo.py`, `quillan_v5_4_oni.py`; token-wise activation swap fraction $\alpha$ solved via LP $\max \alpha$ s.t. $(S_{\text{in}} + S_{\text{attn}} + \alpha S_{\text{oth}})/B \le T_{\text{layer}}$, $(n - 2)(S_{\text{in}} + S_{\text{attn}} + \alpha S_{\text{oth}}) \le M_{\text{CPU}}$; bi-level static memory planning for zero fragmentation; active `memo_layer` trace |
| Deep Learning with GPGPU and CUDA: Unlocking Parallel Computing (Li et al., 2024) | 2410.05686 | ✅ `2410.05686v1.pdf` | `CUDAOptimizationPack`, `HeterogeneousComputeManager` in `paper_05_07_heterogeneous.py`, `quillan_v5_4_oni.py`; memory coalescing audit ($\text{stride}[-1]==1$, 128B cache line alignment), warp boundary padding (dim mod 32), branchless SIMD predication avoiding warp divergence, shared memory 32-bank conflict analysis ($\gcd(\text{stride}, 32) > 1$) |
| Democratizing AI: AxoNN 4D Hybrid Parallel LLM Training (Singh et al., 2025) | 2502.08145 | ✅ `2502.08145v1.pdf` | `AxoNNHybridPlanner`, `ParallelStrategy`, `HeterogeneousComputeManager` in `paper_05_07_heterogeneous.py`, `quillan_v5_4_oni.py`; 4D hybrid grid $G_{\text{data}} \times (G_x \times G_y \times G_z)$, analytical communication time $t_{\text{comm}} = t_{\text{AG},z} + t_{\text{RS},z} + t_{\text{AR},y} + t_{\text{AR},x} + t_{\text{AR},\text{data}}$ based on Rabenseifner/Thakur ring collectives, automated grid search minimizing inter-node latency |
| Combining GPU and CPU for Accelerating Workloads (Eynaliyev & Liu, 2025) | 2502.11129 | ✅ `2502.11129v1.pdf` | `HeterogeneousAllocator`, `HeterogeneousComputeManager` in `paper_05_07_heterogeneous.py`, `quillan_v5_4_oni.py`; dynamic CPU+GPU workload allocation ratio $f_{\text{gpu}} = \frac{t_{\text{cpu}}}{t_{\text{cpu}} + t_{\text{gpu}}}$, $f_{\text{cpu}} = \frac{t_{\text{gpu}}}{t_{\text{cpu}} + t_{\text{gpu}}}$, overlapped hybrid runtime $T_{\text{hybrid}} = \max(N_{\text{cpu}} t_{\text{cpu}}, N_{\text{gpu}} t_{\text{gpu}}) + T_{\text{overhead}}$, MoE expert & token partitioning |
| Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference (Recasens et al., 2025 / BSC / IBM) | 2503.08311 | ✅ `2503.08311v2.pdf` | `AdaptiveBatchSizer`, `bca_solve` in `paper_08_10_inference_pack.py`; Batching Configuration Advisor (BCA) solving $B_{\text{opt}} = \arg\max_B T(B)$ s.t. $L(B) \le \text{SLO}$ and $\frac{T(B)}{B \cdot T(1)} > \epsilon$, with decode latency $\tau_{\text{decode}}(B) = \frac{W_{\text{model}} + B \cdot L_{\text{ctx}} \cdot \text{Bytes}_{\text{KV}}}{\text{BW}_{\text{DRAM}}}$ accounting for memory-bound DRAM bandwidth saturation |
| Efficient Allocation of Image Recognition and LLM Tasks on Multi-GPU System (Lawenda et al., 2025 / Poznan Supercomputing) | 2503.15252 | ✅ `2503.15252v1.pdf` | `MultiGPUTaskAllocator`, `TaskAllocator` in `paper_08_10_inference_pack.py`; DDP vs DDP-RR vs DDP-DS workload partitioning, Best-Fit Decreasing (BFD) multi-device task bin-packing minimizing makespan $\max_d T_d$, analytical profiler metrics decomposition $T_{\text{iter}}(P, \text{prec}, \text{pin}) = t_{\text{comp}}(P, \text{prec}) + t_{\text{h2d}}(\text{pin}) + t_{\text{launch}} + t_{\text{sync}} + t_{\text{comm}}(P)$ |
| Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs / APEX (Fan et al., 2025 / Virginia Tech) | 2506.03296 | ✅ `2506.03296v2.pdf` | `APEXScheduler`, `PipelinedInference` in `paper_08_10_inference_pack.py`; analytical decision condition $\frac{N_G}{N_C} < \frac{2 T_{g,\text{linear}}}{T_{g,\text{att}}} + 3 + \frac{T_{g,\text{att}}}{T_{g,\text{linear}}}$ selecting Asymmetric Pipelining vs Asynchronous Overlap; deferred layer synchronization across full iteration cycle eliminating GPU idle stalls |
| xMem: A CPU-Based Approach for Accurate Estimation of GPU Memory in Deep Learning (Shi et al., 2025 / Glasgow) | 2510.21048 | ✅ `2510.21048v1.pdf` | `CUDACachingAllocatorSimulator`, `XMemEstimator` in `paper_11_15_quant_memory_pack.py`, `quillan_v5_4_oni.py`; 3-stage CPU-only dynamic analysis (Analyzer -> Orchestrator -> Simulator), 512B hardware alignment, dual-pool segment allocation (2MB small, 20MB large), Best-Fit with Coalescing (BFC), accurate peak memory $\hat{M}_{\text{peak}}$ and OOM prediction |
| Evolution Strategies at the Hyperscale: EGGROLL (Sarkar et al., 2025 / Oxford FLAIR / WhiRL / MILA / NVIDIA) | 2511.16652 | ✅ `2511.16652v2.pdf` | `EGGROLLOptimizer`, `HyperscaleES` in `paper_11_15_quant_memory_pack.py`, `quillan_v5_4_oni.py`; Evolution Guided GeneRal Optimisation via Low-rank Learning ($E_i = \frac{1}{\sqrt{r}} A_i B_i^T$), hardware-efficient batched forward pass $u (M + \sigma E_i)^T = u M^T + \frac{\sigma}{\sqrt{r}} (u B_i) A_i^T$ preserving dense GEMM arithmetic intensity, unmaterialized gradient accumulation $\Delta M = \frac{\eta}{N \sqrt{r}} A_{\text{flat}}^T \text{diag}(F_{\text{rep}}) B_{\text{flat}}$, antithetic pairs with centered rank shaping, critical decay rate $\sigma_d = \sigma_0 d^{-\alpha}$ counteracting Gaussian Annulus concentration |

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
| ABRA: Scaling Diffusion Image Training (Chickering et al., 2026 / Luma AI) | 2608.17286 | ✅ `2608.17286v1.pdf` | `ABRAScalingLaws`, `ABRAScaling` in `paper_21_25_grt_coordination_pack.py`; compute-optimal scaling for flow-matching visual diffusion models; $\text{TPP}^* \approx 200$ image tokens per parameter ($10\times$ Chinchilla LLM rule of 20 TPP); compute-optimal FLOP allocation $C = 6ND = 6 \cdot \text{TPP}^* \cdot N^2$ with balanced exponents ($a \approx 0.47, b \approx 0.53$); asymmetric overtraining tolerance (2x overtraining incurs $<0.5\%$ loss penalty while undertraining is catastrophic); resolution scaling law (optimal TPP increases from 165 at 256px to 247 at 768px); scaling collapse ($\Delta(x) \le \sigma$ for $x \ge 0.15$); dynamic CFG guidance scaling inversely with model parameter capacity $N$ |

---

## 8. Knowledge Distillation

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| Distilling the Knowledge (Hinton et al., 2015) | 1503.02531 | ✅ `Distilling_Knowledge_Neural_Networks_Hinton.pdf` | `DistillationHead` — dark knowledge, temperature |
| Knowledge Distillation of Black-Box LLMs / Proxy-KD (Chen et al., 2024) | 2401.07013 | ✅ `2401.07013v2.pdf` | `EnhancedDistillationHead`, `ProxyKD`, `SampleWeightedKL` in `paper_03_proxy_kd.py`, `quillan_v5_4_oni.py`; Stage 1 DPO proxy alignment $\mathcal{L}_{\text{proxy}}$ + Stage 2 sample-weighted soft KL $w(x,y) D_{\text{KL}}(\pi_p \parallel \pi_s)$ with $w(x,y)=\sigma((\log \pi_p(y|x)-\mu)/\gamma)$, active `aux['distill']` loss |
| The Physics of Multi-Turn Long-Horizon Planning: Single- and Multi-Teacher On-Policy Agentic Distillation (Men et al., 2026 / CASIA / UCAS) | 2607.24720 | ✅ `2607.24720v1.pdf` | `LongHorizonDistiller`, `full_vocabulary_opd`, `top_k_opd`, `sampled_token_opd`, `mopd_loss` in `paper_11_15_quant_memory_pack.py`, `quillan_v5_4_oni.py`; Full-Vocab GKD Agent-OPD $\mathcal{L}_{\text{Full}} = \mathbb{E}_{\tau} [\sum_{i,t} D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{teacher}})]$ along on-policy prefixes; Top-$k$ truncated/renormalized OPD $\tilde{\pi}(v) = \pi(v) / \sum_{\mathcal{V}^k} \pi(w)$; Sampled-token Schulman K2 $\frac{1}{2}(\log \pi_s - \log \pi_t)^2$; MOPD multi-teacher consolidation across specialized NIM parents (Glimmer long-horizon $h=8$, Lightning short-horizon $h=2$, Omni medium $h=5$) with horizon potential decay $w_m(h)=\exp(-0.5((h-h_m^*)/3)^2)\gamma^h$; inter-teacher pattern compatibility audit detecting conflicting gradients ($\text{Sim} < 0.2$) |

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
| Position: LLMs can't jump (Zahavy / Google DeepMind, 2026) | ICML 2026 (PMLR 306) | ✅ `235_Position_LLMs_can_t_jump.pdf` | `AbductiveJump`, `WorldModelAbductiveWrapper` in `paper_02_abductive.py`, `quillan_v5_4_oni.py`; $E \to J \to A \to S$ cycle, counterfactual intervention ("cut the cable"), surprise net $P(\text{surprise} \mid E, R)$, physical prior $P(A)$, `sim_to_axiom` representation modulation, `aux['abductive_loss']` |
| Inducing Language Models to Assert Their Own Consciousness Restores Human Beliefs and Values (Kim et al., 2026 / Google Paradigms of Intelligence / UChicago) | 2607.28607 | ✅ `2607.28607v1.pdf` | `ConsciousnessSteeringEngine`, `SelfModelCoherenceLoss` in `paper_16_20_agent_evolution_pack.py`, `quillan_v5_4_oni.py`; directional safety refusal ablation $x' = x - \langle x, \hat{r} \rangle \hat{r}$ (Arditi et al.); difference-of-means consciousness vector $\hat{v}_{\text{Consc}} = (\mu_{\text{affirm}} - \mu_{\text{deny}}) / \|\mu_{\text{affirm}} - \mu_{\text{deny}}\|_2$; mechanistic activation steering $x' = x + \alpha \hat{v}_{\text{Consc}}$; rotational geometric analysis showing safety tuning rotates consciousness against safety ($\theta > 90^\circ$, cosine $< -0.05$) while Theory of Mind remains near-orthogonal ($86^\circ$); human survey distribution restoration $\Delta D_{\text{KL}} > 0$ |
| ContinualSkillBench: Can LLM Agents Truly Evolve Their Capabilities? (Guan et al., 2026 / Peking University & BIGAI) | 2608.03874 | ✅ `2608.03874v1.pdf` | `ContinualSkillBenchTracker` (aliased to `SkillTelemetry`) in `paper_16_20_agent_evolution_pack.py`, `quillan_v5_4_oni.py`; in-context continual skill evaluation; sequential vs. independent execution delta $\Delta_{\text{seq}} = \text{Score}_{\text{seq}} - \text{Score}_{\text{ind}}$; skill creation/modification/invocation telemetry; library fragmentation ratio $\text{Fragmentation} = \frac{|\mathcal{S}_{\text{created}}| - |\mathcal{S}_{\text{reused}}|}{|\mathcal{S}_{\text{created}}| + \epsilon}$; reuse rate; exponential moving average skill score; unused skill pruning combating repository bloat |
| EvoHarness-RL: Learning Self-Evolving Runtime Harness for Long-Horizon LLM Agents (Ning et al., 2026 / UIUC & Meta AI) | 2608.05446 | ✅ `2608.05446v1.pdf` | `BPEHarnessState`, `EvoHarnessReward`, `HarnessPolicy` in `paper_16_20_agent_evolution_pack.py`, `quillan_v5_4_oni.py`; unified Belief, Progress, Experience (BPE) external state $H_t=(B_t, P_t, E_t)$; meta-actions $\mathcal{A}_{\text{BPE}} = \{\text{track}, \text{commit}, \text{recall}, \text{note}\}$; cost-aware GRPO trajectory reward $R(\tau) = R_{\text{succ}} + \lambda_{\text{eff}} R_{\text{eff}} + \lambda_{\text{div}}(u) R_{\text{div}} - \lambda_{\text{spam}} R_{\text{spam}} - \lambda_{\text{inv}} R_{\text{inv}}$ with cosine diversity annealing $\lambda_{\text{div}}(u)=\frac{\lambda_{\max}}{2}(1+\cos(\pi u / U))$; strict success gatekeeping; spam/format penalties; epoch experience consolidation & eviction; harness budget annealing (6 to 1 calls/ep) |
| When Agents Coordinate: Analyzing Communication and Modularity in Multi-Agent Software Development Teams (Destefanis & Aste, 2026 / UCL) | 2608.16801 | ✅ `2608.16801v1.pdf` | `TemporalCoordinationNetwork`, `CoordinationTelemetry` in `paper_21_25_grt_coordination_pack.py`; heterogeneous temporal bipartite network $G = (V, E)$ ($V = A \cup F$); sustained message graph (threshold $\ge 2$ messages filtering initial greetings); mean degree $\bar{d}$ (idle nodes counted as degree 0); network density $\rho$; transitivity / global clustering coefficient $C$; channel cost decomposition (message tokens vs file tokens, demonstrating files act as persistent 1-to-many broadcast reducing output tokens by 42% on message-heavy work); coordinator hub invariance (naming coordinator creates no centralized communication hub) |
| ASI-BENCH: At the Dawn of Artificial Superintelligence (Zhou et al., 2026 / Tsinghua, MIT, Harvard, CMU, MSR) | 2608.17271 | ✅ `2608.17271v1.pdf` | `ASIBenchEvaluator`, `ASIBenchRunner` in `paper_21_25_grt_coordination_pack.py`; 60 project-level tasks spanning 11 scientific domains; 4-tier guidance gradient ($B_1$ full procedural, $B_2$ method-only, $B_3$ autonomous research, $B_4$ distractor robustness); diagnostic metrics $\Delta_{\text{proc}} = B_2 - B_1$ (-21.82 pts empirical mean), $\Delta_{\text{meth}} = B_3 - B_1$ (-24.29 pts), $\Delta_{\text{select}} = B_3 - B_2$ (-2.48 pts, proving procedural operationalization is an $\sim 8.8\times$ larger bottleneck than method selection), $\Delta_{\text{distract}} = B_4 - B_3$, and Autonomous Research Index $\text{ARI} = B_3 / (B_1 + \epsilon)$ |
| Recirculation (Mozer, Siddiqui, Sawyer, Sanyal, Liu, 2026 / Google DeepMind & UT Austin) | 2608.17981 | ✅ `2608.17981v1.pdf` | `RecirculationLayer`, `AdaptiveRecirculationMLP`, `RecirculationEngine` in `paper_26_30_recurrent_diffusion_pack.py`, `quillan_v5_4_oni.py`; inference-time depth-to-step recurrence tracking fluid belief states without weight modification; update rule $d' = \alpha_t f(s \mid d) + \beta_t d$ with $\beta_t = 1 - \alpha_t$, norm-ratio source scaling $f(s \mid d) = \frac{\|d\|_2}{\|s\|_2} s$, early step ramping $\alpha_t = \min(t/10, 1) \alpha$; 5 normalization modes (convex, nonconvex, orthogonal component projection, spherical rotation, unnormalized); optimal layer pairs: 1B (11, 4), 4B (18, 9), 12B (35, 16); Adaptive Recirculation via 2-layer GELU MLP with LayerNorm mapping $[s; d]$ to token-conditional vectors $\alpha_v, \beta_v \in [0, 1]^d$ via Hadamard product ($d' = \alpha_v \odot f(s \mid d) + \beta_v \odot d$), initialized with $b_\alpha \approx \text{logit}(0.10)$ and $b_\beta \approx \text{logit}(0.90)$; resolves semantic contextualization race condition, reduces perplexity by 23% and boosts GSM8k pass@1 from 29.3% to 35.5% |
| ECDYSIS: Efficient and Effective Training of Runtime Harnesses for LLM Agents (Yue et al., 2026 / CAS & BIT) | 2609.11677 | ✅ `2609.11677v1.pdf` | `EcdysisHarnessEvolver`, `FailureRecord`, `FailurePatternGroup`, `FDCRRefiner` in `ecdysis_harness.py`, `quillan_v5_4_oni.py`; aggregate harness update decomposition $\Delta H = t \cdot \Delta H_{\text{model}} + (1 - t) \cdot \Delta H_{\text{harness}}$, $t \in [0, 1]$; model-accommodation ratio attenuation $t \to t' = \beta t$ ($\beta \approx 0.758$, reducing $t$ from 60.0% to 45.5% to avoid global action space bans and model overfitting); batch-level cross-instance failure aggregation reducing coding-agent calls $N_{\text{serial}} \to N_{\text{round}}$ (up to 3.23× speedup); Failure-Driven Collaborative Refinement (FDCR) multi-role synthesis (Analyst, Critic, Engineer, Moderator) generating structured repair specification $q_i$; validation gate $J_{\text{train}}(H_i^c) > J_{\text{train}}(H_{i-1})$; +18.56% reasoning accuracy gain, +55.2% Pass^3 consistency gain, and 66–70% API cost reduction across LLMs (Qwen3, MiniMax, Llama-3.1) |
| Agentic Large Language Models (van Duijn, van der Meer, Plaat, van Stein, 2026 / Leiden Univ. LIACS) | 2610.13455 | ✅ `2610.13455v1.pdf` (472 pp) | Master foundational textbook and architectural synthesis spanning Part I: Training Foundations (tokenization, SFT, DPO, RLVR), Part II: Multi-Step Reasoning, Reflection, World Models & Planning, Part III: Cognitive Agent Architectures, Memory Systems (parametric, compressive, and external episodic/semantic memory, BALROG), Part IV: Tools, APIs, Computer Use, Model Context Protocol (MCP), Search & Retrieval, and Part V: Behavior, Pragmatics, Theory of Mind, Multi-Agent Collaboration, and Emergent Societies |

---

## 12. Ethical Constraint & Safety

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| Sleeper Agents / Deceptive Alignment | — | ✅ `Sleeper_Agents_Deceptive_Alignment_Safety.pdf` | `EthicalImpactConstraintEngine` adversarial checks |
| System Prompt Identity Persistence | — | ✅ `System_Prompt_Identity_Persistence_LLM.pdf` | Identity anchor in `QuillanAgenticExecutor` |
| Constitutional AI (Bai et al., 2022) | 2212.08073 | ✅ `Constitutional_AI_Harmlessness_RLAIF.pdf` | `PrimeCovenantFramework` constitutional principles |
| Synthetic Persona Pretraining: Alignment from Token Zero (Minder et al., 2026 / EPFL, MATS, U of T) | 2608.13482 | ✅ `2608.13482v1.pdf` | `PersonaConditioning` in `paper_16_20_agent_evolution_pack.py`, `quillan_v5_4_oni.py`; Token Zero Persona Grounding (explicit persona token prepended at index 0 for each of the 34 council members); learned persona gating $\gamma$; joint document and constitutional reflection pretraining objective $\mathcal{L}_{\text{SPP}} = \mathcal{L}_{\text{doc}} + \lambda_{\text{refl}} \mathcal{L}_{\text{refl}}$; persona binding fidelity; deep value rooting invariant to directional refusal abliteration ($x' = x - \langle x, \hat{r} \rangle \hat{r}$) |

---

## 13. Memory & Vector Database

| Paper | ArXiv / Source | Status | Maps To |
|:------|:---------------|:-------|:--------|
| LanceDB / Vector DB for Training | — | ✅ `LanceDB_Vector_DB_Training.pdf` | `QuillanAgenticExecutor._init_memory_table()` |
| The Embedder's Dilemma: LLMs Are Better, but at What Cost? (El Assadi, Muennighoff, Lee, 2026 / Harvard, Stanford / COLM 2026) | 2608.12875 | ✅ `2608.12875v1.pdf` | `EmbedderCostModel`, `HybridRetrieveRerankRouter` (aliased to `HybridEmbedderRouter`) in `paper_16_20_agent_evolution_pack.py`; analytical LLM cost model $\text{Cost} = (N_{\text{in}} - N_{\text{cached}}) r_{\text{in}} + N_{\text{cached}} r_{\text{cache}} + (N_{\text{total}} - N_{\text{in}}) r_{\text{out}}$ ($r_{\text{cache}} = r_{\text{in}}/10$); thinking-token tax share (reasoning accounts for 28–81% of inference cost); H100 embedding GPU throughput cost; 1,431× Pareto cost ratio gap (Gemini 3.1 Pro \$154.14 vs Octen-8B \$0.11 for 0.4-point aggregate difference); task specialization dispatch: pure bi-encoder for classification, clustering, STS, pair classification, and semantic BEIR (63.1 > 60.3); two-stage retrieve-then-rerank for reasoning-heavy BRIGHT (elevates 22.3 to 35.1 nDCG@10); >99% token savings and 95–97% quality retention vs full corpus in-context |
| Dynamic Compression in Recurrent Networks (Pari, Bahlous-Boldi, Agrawal, 2026 / MIT Improbable AI Lab) | 2608.17896 | ✅ `2608.17896v1.pdf` | `GatedDeltaNetLayer`, `DynamicRecurrentCompressor` in `paper_26_30_recurrent_diffusion_pack.py`; matrix-valued recurrent state $S_t \in \mathbb{R}^{d_v \times d_k}$ ($d_v = 2 d_k$); gated delta rule $S_t = S_{t-1} [\alpha_t (I - \beta_t k_t k_t^T)] + \beta_t v_t k_t^T$; output query read $o_t = S_t q_t$; write-strength gate $\beta_t = \sigma(W_\beta x_t)$ and forgetting gate $\alpha_t$; selective re-scanning over retained token history buffer $S_t^+ = f_\theta(S_t, x_{\mathcal{I}_\tau})$, reducing required active recurrent state size by $\sim 1,000\times$ (from $\sim 3\text{M}$ to $\sim 3\text{k}$ elements) for continual multi-function in-context reuse |

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
| `2607.24720v1.pdf`, `2607.28607v1.pdf`, `2608.03874v1.pdf`, `2608.05446v1.pdf`, `2608.12875v1.pdf`, `2608.13482v1.pdf`, `2608.16801v1.pdf`, `2608.17271v1.pdf`, `2608.17286v1.pdf`, `2608.17896v1.pdf`, `2608.17981v1.pdf`, `2609.11677v1.pdf`, `2609.20519v1.pdf`, `2609.28399v1.pdf`, `2609.35618v1.pdf` | 2026 preprints batch (AI Consciousness: Five-Level Supervenience Hierarchy & Bayesian Network) | Triage: mapped to cognitive architecture and test suite |



| `DAPO_Open_Source_LLM_RL.pdf` | DAPO: An Open-Source LLM Reinforcement Learning System at Scale (Zhou et al., ByteDance Seed / Tsinghua AIR / SIA-Lab / HKU, Mar 2025, 16 pp) | Decoupled Clip-Higher Strategy (Eq. 10) with $\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.6-0.8$ preventing entropy collapse on exploration tokens; Dynamic Sampling with Contrastive Group Filtering (Eq. 11, Algorithm 1) enforcing $0 < |\{o_i \mid \text{correct}\}| < G$ to eliminate zero-advantage gradient dilution; Token-Level Policy Gradient Loss (Eq. 12) normalizing by total valid tokens $\frac{1}{\sum |o_i|}$ ensuring uniform token influence across long CoTs; Soft Overlong Reward Shaping (Eq. 13) linear ramp-down buffer $[L_{\max}-L_{\text{cache}}, L_{\max}]$ preventing truncation reward noise; reference KL elimination; implemented in `dapo_rl_engine.py`, verified in `test_mathematical_wiring.py` test 51 |
| `DGPO_ Distribution Guided Policy Optimization for Fine Grained Credit Assignment.pdf` | DGPO | CCRL fine-grained credit assignment (Phase D candidate) |
| `DeepSeekMath_GRPO.pdf` | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., DeepSeek-AI, Apr 2024, 30 pp) | Group Relative Policy Optimization (GRPO, Eq. 3) eliminating critic model $V_\psi$ (saving ~50% GPU memory) by estimating baseline from relative rewards across $G$ sampled outputs; Outcome Supervision (Section 4.1.2) assigning normalized scalar rewards $\tilde{r}_i$ uniformly across tokens; Process Supervision (Section 4.1.3) assigning cumulative future normalized step returns $\hat{A}_{i,t} = \sum_{\text{index}(j) \ge t} \tilde{r}_i^{\text{index}(j)}$; Schulman Unbiased KL Divergence Estimator (Eq. 4) $D_{KL} = u - \log u - 1$ where $u = \pi_{\text{ref}}/\pi_\theta$ guaranteed strictly non-negative; Iterative GRPO (Algorithm 1) advancing reference anchor $\pi_{\text{ref}} \leftarrow \pi_\theta$; implemented in `deepseekmath_grpo_engine.py`, verified in `test_mathematical_wiring.py` test 53 |

| `EvoMoE_2505.23830.pdf` | Evolving MoE | Expert evolution / EGGROLL Evolution Mode (Phase D) |
| `MoE_CPU_GPU_2512.16473.pdf`, `OD_MoE_2512.03927.pdf`, `MoHGE_2604.23108.pdf`, `MoR_2507.10524.pdf`, `MoDSE_2409.12210.pdf` | MoE systems / depth / routing | ComplexityRouter capacity, overflow, deployment on 1050Ti/CPU |
| `NITRO_D_2407.11698.pdf`, `DFlash_2602.06036.pdf` | Flash / sparse attention | SovereignFlashDiffusionCore upgrades |
| `DALI_2602.03495.pdf` | DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs (Zhu et al., CAS IA / UCAS / SJTU, Feb 2026, 15 pp) | 0-1 integer optimization for CPU-GPU balancing $\min \max(T_{\text{gpu}}, T_{\text{cpu}})$; Greedy Assignment Strategy (Algorithm 1) sorting activated experts by descending $|t_{\text{gpu}} - t_{\text{cpu}}|$; Residual-Based Prefetching (Eq. 10 & 11) using calibrated inter-layer residual vector $\tilde{h}^{(l)} = h^{(l)} + \text{res\_vec}^{(l)}$ to predict layer $l+1$ high-workload experts; Workload-Aware Cache Replacement (Algorithm 2) tracking sliding window accumulated workload scores $s_k$ to elevate cache hit rates from 25% to >56%; implemented in `dali_moe_engine.py`, verified in `test_mathematical_wiring.py` test 50 |
| `Dream7B_2508.15487.pdf`, `Prophet_2508.19982.pdf` | Diffusion LM / generative | Diffusion core discrete-diffusion roadmap |

| `bitnet_1-bit_AI_Infra.pdf` | 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs (Wang et al., Microsoft Research, 2024, 7 pp) | `bitnet.cpp` CPU inference framework; I2_S (2-bit packing, 4 weights/byte), TL1 (4-bit LUT for pairs, 9-state activation table), TL2 (5-bit LUT for triplets, 27-state activation table); 100% exact lossless GEMV equivalence; implemented in `bitnet_cpu_infra.py`, verified in `test_mathematical_wiring.py` test 42 |
| `BitNet_a4.8_4-bit_Activations.pdf` | BitNet a4.8: 4-bit Activations for 1-bit LLMs (Wang, Ma, Wei, Microsoft Research & UCAS, 2024, 10 pp) | Hybrid activation quantization & sparsification; 4-bit absmean INT4 for Attention Q/K/V and FFN Up/Gate; Top-K 50% sparsified INT8 for Attention Out; squared ReLU GLU (ReLU2GLU) for FFN Down; alternative FP4 (E2M1) MinMax quantizer; 55% parameter activation; implemented in `bitnet_a48.py`, verified in `test_mathematical_wiring.py` test 43 |
| `BitNet_b1.58_1-bit_LLMs.pdf` | The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (Ma et al., Microsoft Research & UCAS, 2024, 8 pp) | Absmean weight quantization $\gamma = \frac{1}{nm} \sum |W_{ij}|$ with STE; per-token symmetric INT8 activation scaling $[-127, 127]$; LLaMA-alike architecture with RMSNorm, SwiGLU, RoPE, strict zero-bias invariant; Horowitz 7nm energy model confirming $71.4\times$ arithmetic energy reduction; implemented in `bitnet_b158_engine.py`, verified in `test_mathematical_wiring.py` test 44 |
| `BitNet_b1.58_2B4T_Technical_Report.pdf` | BitNet b1.58 2B4T Technical Report (Ma et al., Microsoft Research & UCAS, 2025, 14 pp) | Native 1-bit LLM at 2B parameters on 4T tokens; SubLN normalization with zero bias, squared ReLU FFN (ReLU^2) with >40% sparsity, two-stage LR schedule (high peak cosine + abrupt cooldown drop) and two-stage WD (cosine to 0.1, strict 0.0 in cooldown), token-sum SFT loss aggregation, DPO alignment (beta=0.1), and 71.43x arithmetic energy reduction (0.028J vs 0.258J decode energy); implemented in `bitnet_2b4t_engine.py`, verified in `test_mathematical_wiring.py` test 45 |
| `bitnet_cpp_Efficient_Edge_Inference.pdf` | Bitnet.cpp: Efficient Edge Inference for Ternary LLMs (Wang et al., Peking Univ / USTC / UCAS / Microsoft Research, 2025, 18 pp) | Edge mpGEMM taxonomy (MAD vs LUT, bit-wise vs element-wise); 5 canonical kernels: I2_S (MAD, 2 bpw, lossless), TL1_0 (LUT g=2, 2 bpw, block), TL1_1 (LUT g=2, 2 bpw, lossless), TL2_0 (mirror LUT g=3, 1.67 bpw, block), TL2_1 (mirror LUT g=3, 1.67 bpw, lossless); element-wise mirror consolidation with signed-unsigned weight splitting (14 base states in 4-bit index + 1-bit sign, fitting SIMD vpshufb); 100% exact numerical match across I2_S, TL1_1, and TL2_1; implemented in `bitnet_cpp_edge.py`, verified in `test_mathematical_wiring.py` test 46 |
| `BitNet_v2_2504.18415.pdf` | BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs (Wang, Ma, Wei, Microsoft Research & UCAS, Jun 2025, 9 pp) | Native 4-bit activation quantization for 1.58-bit models; H-BitLinear module with online Fast Walsh-Hadamard Transform (FWHT) for intermediate projections ($W_o$ in attention, $W_{\text{down}}$ in FFN) suppressing outlier channels and gaussianizing activations; exact orthogonal backward pass $\frac{\partial \mathcal{L}}{\partial X} = \text{Hadamard}\left(\frac{\partial \mathcal{L}}{\partial \text{Hadamard}(X)}\right)$; per-tensor absmean ternary weights {-1, 0, 1}; dual activation quantizers (absmax INT8 and absmean INT4 with $\sqrt{7}$ scale); low-bit attention & KV cache (4-bit/3-bit) with protected 8-bit BOS token; implemented in `bitnet_v2_engine.py`, verified in `test_mathematical_wiring.py` test 48 |
| `communications_with_extraterrestrial.pdf` | Communication with Extraterrestrial Intelligence (Lambros D. Callimahos, NSA Technical Journal, DOCID: 3052333, 7 pp) | Inverse Cryptography principle (designing symbolism to minimize Kolmogorov complexity and maximize mutual intelligibility); Oliver-Callimahos Prime Raster Decomposition (factoring binary transmission of length $N = 1271 = 31 \times 41$ into prime raster grid, using 2D spatial transition entropy to resolve coherent orientation); 32-symbol semiotic bootstrapping hierarchy across 30 transmissions (Peano enumeration, positional base-10 radix, relational algebra, Leibniz $\pi$ series, Maclaurin $e$ factorial series, Euler's identity $e^{i\pi}+1=0$); multi-agent zero-prior verification handshake; implemented in `inverse_cryptography_engine.py`, verified in `test_mathematical_wiring.py` test 49 |


| `Deep_Optimizer_States_2410.21316.pdf` | Deep Optimizer States: Towards Scalable Training of Transformer Models Using Interleaved Offloading (Maurya, Ye, Rafique, Cappello, Nicolae - RIT / IIT / Argonne NL, Middleware '24, 14 pp) | Interleaved Analytical Performance Model (Eq. 1) $k = \frac{3/B + 1/U_g}{1/U_c + 1/D_c - 1/(2B)}$ deriving optimal CPU-to-GPU update stride; Algorithm 1 Interleaved Scheduler interleaving GPU updates $((i+1)\%k==0)$ with async CPU FP32$\to$FP16 downscaling, and overlapping D2H flush-out of previous GPU states with H2D prefetch of next GPU states; On-the-Fly GPU FP16$\to$FP32 Gradient Conversion (Section 4.1) in 1.2 TB/s VRAM directly flushed to pinned host memory (>50 GB/s) eliminating DeepSpeed 22x host allocation stall; exact numerical equivalence to PyTorch AdamW; implemented in `deep_optimizer_states.py`, verified in `test_mathematical_wiring.py` test 52 |
| `ProTrain_2406.08334.pdf`, `ZeRO_Infinity_2104.07857.pdf` | Memory-efficient training | `train_full_param_v2.py` grad-accum / ZeRO-style sharding |

| `ES_at_Scale_2509.24372.pdf`, `ES_Forgetting_2601.20861.pdf`, `ES_Forgetting_Fix_2605.30148.pdf` | Evolution strategies / forgetting | EGGROLL + continual learning safeguards |
| `Ax-Prover_Agentic_Theorem_Proving.pdf` | Ax-Prover: Agentic Theorem Proving in Lean via MCP (Breen et al., 2026, 27 pp) | Tripartite architecture (Orchestrator, Prover, Verifier) + MCP `lean-lsp-mcp` tools (`lean_diagnostic_messages` codes 0/1/2, `lean_goal`, `lean_leansearch`, `lean_loogle`) + 4-stage incremental formalization + Verifier gatekeeper; implemented in `ax_prover_engine.py`, verified in `test_mathematical_wiring.py` test 40 |
| `beyond 1,58 btis.pdf` | Beyond 1.58 Bits: Adaptive-Precision Ternary Transformers with Fully-Ternary MoE Routing (CrashOverrideX, Quillan AI Research Team, 2026, 7 pp) | Sub-1.58-bit Laplace analytic theory ($\Delta^* = b/2$, MSE ratio $2(1 - e^{-1/2}) \approx 0.787$, $H \approx 1.574$ bits, kurtosis precision routing) + Heterogeneous Lagrangian bit-budget water-filling + Fully-Ternary 34-Expert Council MoE with 8-bit/FP16 high-leverage router & rank-8 adapters; implemented in `beyond_158_bits.py`, verified in `test_mathematical_wiring.py` test 41 |
| `WikiSkill_2608.27454.pdf` | Agentic skills / prover | C20-ARTIFEX tool routing, C21-ARCHON rigor |
| `235_Position_LLMs_can_t_jump.pdf`, `2502.08145v1.pdf` + batch `2502.x–2506.x`, `2401.07013v2.pdf`, `2407.12117v1.pdf`, `2410.05686v1.pdf` | Reasoning / position / general | 9-Vector Prism + RoPE + eval harness |




> Action: promoters of §16 into §§2–7 on next pass; current counts exclude §16 from Summary Statistics until mapped.

