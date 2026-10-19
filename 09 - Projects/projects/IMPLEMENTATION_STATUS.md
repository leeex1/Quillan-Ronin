# Quillan Implementation Status Matrix
Paper/MD → claimed module → verified location → status. Verified by code search, not trust.
Target code for Mini/Oni-6L: `09 - Projects/projects/oni/quillan_v5_4_oni.py` (binds 0/0).
Target code for Main/Ronin-12L: `scripts/hf_code_era/quillan_v5_4_oni.py`.
Statuses: LIVE (in code, referenced) / STUB (exists, unwired) / MISSING / UNVERIFIED.

## Batch 1 — Core Transformer + MoE + Quant (verified 2026-09-22)
| Paper/Technique | Claimed module | Verified | Status |
|---|---|---|---|
| RoPE/RoFormer | CouilAttention rotary | RotaryEmbedding (8 refs) | LIVE |
| RMSNorm/LayerNorm | norms throughout | norm classes (5 refs) | LIVE |
| SwiGLU | expert FFN | swiglu (1 ref) | LIVE |
| BitNet b1.58 | BitLinear ternary | BitLinear + 26 quant refs | LIVE |
| STE (Bengio) | BitLinear.forward | 196 refs | LIVE |
| Council MoE block | UnrolledCouncilMoEBlock | 3 refs | LIVE |
| PersonaPullGate | pull routing | 12 refs | LIVE |
| Top-k routing | ComplexityRouter | 41 refs | LIVE |
| ST-MoE Z-loss | router stability | 15 refs | LIVE |
| DeepSeekMoE shared expert | shared expert path | no named component | UNVERIFIED (may be inline) |

## Batch 2 — RL/CCRL/GRPO/DAPO/DGPO (queued)
## Batch 3 — Diffusion/DFlash/speculative (queued)
## Batch 4 — EGGROLL/swarm/DQSO/MARTA (queued)
## Batch 5 — Memory/attention-variants/norms (queued)
## Batch 6 — Quillan self-papers (philosophy → protocol check) (queued)
## Batch 7 — Strays + §16 additions (queued)
