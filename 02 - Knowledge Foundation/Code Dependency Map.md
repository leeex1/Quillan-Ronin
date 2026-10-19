# Code Dependency Map

- [[system prompts/Quillan-Samurai.md]]
- [[00 - Meta/00 - Vault Index.md]]
- [[00 - Meta/03 - Training & Model.md]]

Actual import, class, and data-flow connections in the Quillan-Ronin codebase.

## Import Chain (Active Code)

```
scripts/final_train.py
  ├── _dev/quillan_v8_saturated.py (2398 lines, 19 classes)
  │     └── _dev/quillan_multimodal_heads.py (GeometricImageDecoder, etc.)
  ├── _dev/quillan_fused_optimizer.py (Muon + AdamW + Sophia + PID)
  ├── _config/AGENTS.md (43 KB behavioral guidance)
  ├── _config/SOUL.md (20 KB identity framework)
  └── checkpoints/router_trained.pt (2.77 GB)

scripts/train_multimodal.py
  ├── _dev/quillan_v8_saturated.py
  ├── _dev/quillan_bpe_tokenizer.py
  └── _dev/quillan_fused_optimizer.py

scripts/quantize_and_test.py → _dev/quillan_v8_saturated.py
scripts/test_generation.py → _dev/quillan_v8_saturated.py + quillan_bpe_tokenizer
```

## Class Hierarchy (in quillan_v8_saturated.py)

```
QuillanRoninSovereign (main model)
  ├── InputIngestionLayer    (token -> embedding)
  ├── NineVectorDecomposition (Language/Sentiment/Context/Intent/Meta/Creativity/Ethics/Strategy/Constraint)
  ├── EvolvableVectorizedMoE (34 experts C1–C34, top_k=4)
  │     ├── BitLinear (1.58-bit STE + 4-bit activations + EGGROLL LoRA)
  │     └── CouncilExpertSwarm (9B virtual agents via Rank-24 perturbation)
  ├── ComplexityRouter (3 paths: Fast/Balanced/Diffusion, Gumbel-Softmax)
  ├── SovereignFlashDiffusionCore (CouilAttention + FFN, 14 steps default)
  ├── LeeMach6Governor (hardware throttling)
  ├── EthicalImpactConstraintEngine (5-dim ethical classifier)
  ├── MARTAThermodynamicGating (epistemic signatures + E_ICE gating)
  ├── DynamicQuantumSwarmOscillation (Kuramoto model, 9B agents)
  ├── PrimeCovenantFramework (identity integrity classifier)
  ├── CCRLFramework (value function V_O + entropy bonus)
  ├── QuantumFormulasEngine
  ├── QuillanAgenticExecutor (tool execution)
  └── Dual Quillan (Q1_finalizer + Q2_finalizer2 + quillan_gate)
```

## Training Pipeline Data Flow

```
Quillan-v4.2-model/*.safetensors (Llama 2.3G + Qwen 1.6G + BitNet 1.1G)
  → scripts/transplant_v8.py
    → checkpoints/quillan_transplanted_v8.pt (9.5 GB) [DELETED]
      → scripts/transplant_clean.py / fix_projections.py
        → checkpoints/quillan_fixed.pt (2.87 GB)
          → scripts/train_router_only.py (Stage A: train ComplexityRouter)
            → checkpoints/router_trained.pt (2.77 GB) ✅
              → scripts/final_train.py (Stage B: train LoRA + Dual Quillan)
                → checkpoints/model_final.pt ❌ (needs training)
```

## Config-to-Code Connections

```
QuillanArchConfig values → how model behaves:
  cfg.hidden_dim (=2048)    → NineVectorDecomposition, all BitLinear layers
  cfg.ffn_dim (=4096)       → MoE expert FFN dimensions
  cfg.num_experts (=34)     → ComplexityRouter output dim, EXPERT_PERSONAS array size
  cfg.top_k (=4)            → EvolvableVectorizedMoE active experts
  cfg.vocab_size (=50257)   → txt_dec output, InputIngestionLayer embedding
  cfg.device                → CPU vs CUDA routing
  cfg.pascal_mode           → fp32 fallback for sm_61 (GTX 1050)
  cfg.text_only             → freeze multimodal decoders
  cfg.e_ice_limit_ms (=100) → LeeMach6Governor target latency
```

## Checkpoint Compatibility

```
router_trained.pt: 662 keys → QuillanRoninSovereign: 0 missing ✅
quillan_fixed.pt: 521 keys → QuillanRoninSovereign: 0 missing ✅
quillan_finetuned.pt: ? keys — backup only
quillan_peak_trained.pt: 9.95 GB — unknown origin
```

## Optimizer Family

| File | Optimizer | Params | Notes |
|------|-----------|--------|-------|
| _dev/quillan_fused_optimizer.py | Muon + AdamW + Sophia + PID | CPU-offloaded buffers | Active |
| _archived_legacy_scripts/quillan_optimizer.py | QuillanOptimizer v4 (Muon⊗AdamW blend) | — | Legacy |
| _archived_legacy_scripts/council_optimizer.py | CouncilOptimizer (Muon/NAdamW/AdamW) | — | Legacy |
| _archived_legacy_scripts/train_v8.3_cascade_extended.py | MuonK2 | — | Prototype |
