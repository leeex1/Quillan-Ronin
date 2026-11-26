#!/usr/bin/env python3
"""Quillan-Fused-Socket: config-gated 34-SEQUENTIAL-BLOCK forward module.

Implements verbatim the forward path specified by C8-METASYNTH in board spec
#1169 (board.kind=finding, seq 1169), built per kickoff #1171:

    h = LayerNorm(txt_emb[tok] + pos_embed)                    # [B,T,896]
    for e in 0..33:
        a = CouilAttn(LN(h))                                   # weight-tied attn
        h = h + g_e * a                                        # residual
        u = h @ w1[e]; v = SiLU(h @ wgate[e]); f = (u*v) @ w2[e]
        f = f + swarm_corr_e(h)                                # low-rank swarm refine
        h = LayerNorm_moe(h + f)
    z = final_norm(h)
    logits = txt_dec(z) + txt_dec.lora_B @ txt_dec.lora_A(z)

CONTRACT (hard locks from #1171 / #1169):
  * Consumes EXACTLY the 530 base keys of socket-v2.safetensors. ZERO new keys,
    ZERO shape changes. `EXPECTED_BASE_SHAPES` is transcribed from the real
    safe_open header (554 tensors = 530 base + 24 `.bitnet_scale` sidecars).
  * `.bitnet_scale` sidecars are NOT model state_dict keys. They are held in a
    side table (`FusedSocketConfig.bitnet_scales`) and applied at dequant time,
    so `model.state_dict()` has exactly 530 keys.
  * Standalone / importable with torch only. NOT wired into the immutable
    conversion gate or canonical `quillan_v8_saturated.py` (default disabled
    at the gate layer via `FusedSocketConfig.enabled` metadata flag; wiring is
    explicitly out of scope for this artifact).

Author: @c10-codeweaver (C10 Build Loop). Evidence: board #1169, #1171;
socket header C:\\02_QUILLAN\\exports\\quillan-mini-sovereign-bitnet158-socket-v2.safetensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ground-truth 530-key inventory (transcribed from socket-v2 safe_open header)
# ---------------------------------------------------------------------------
EXPECTED_BASE_SHAPES: dict[str, tuple[int, ...]] = {
    'decomposition.K_gate.weight': (896, 896),
    'decomposition.Q_gate.weight': (896, 896),
    'decomposition.V_gate.weight': (896, 896),
    'decomposition.W_gate.weight': (896, 896),
    'decomposition.vectors.Constraint.lora_A': (896, 256),
    'decomposition.vectors.Constraint.lora_B': (256, 896),
    'decomposition.vectors.Constraint.weight': (896, 896),
    'decomposition.vectors.Context.lora_A': (896, 256),
    'decomposition.vectors.Context.lora_B': (256, 896),
    'decomposition.vectors.Context.weight': (896, 896),
    'decomposition.vectors.Creativity.lora_A': (896, 256),
    'decomposition.vectors.Creativity.lora_B': (256, 896),
    'decomposition.vectors.Creativity.weight': (896, 896),
    'decomposition.vectors.Ethics.lora_A': (896, 256),
    'decomposition.vectors.Ethics.lora_B': (256, 896),
    'decomposition.vectors.Ethics.weight': (896, 896),
    'decomposition.vectors.Intent.lora_A': (896, 256),
    'decomposition.vectors.Intent.lora_B': (256, 896),
    'decomposition.vectors.Intent.weight': (896, 896),
    'decomposition.vectors.Language.lora_A': (896, 256),
    'decomposition.vectors.Language.lora_B': (256, 896),
    'decomposition.vectors.Language.weight': (896, 896),
    'decomposition.vectors.Meta.lora_A': (896, 256),
    'decomposition.vectors.Meta.lora_B': (256, 896),
    'decomposition.vectors.Meta.weight': (896, 896),
    'decomposition.vectors.Sentiment.lora_A': (896, 256),
    'decomposition.vectors.Sentiment.lora_B': (256, 896),
    'decomposition.vectors.Sentiment.weight': (896, 896),
    'decomposition.vectors.Strategy.lora_A': (896, 256),
    'decomposition.vectors.Strategy.lora_B': (256, 896),
    'decomposition.vectors.Strategy.weight': (896, 896),
    'diffusion_core.couil_attn.k_proj.lora_A': (896, 256),
    'diffusion_core.couil_attn.k_proj.lora_B': (256, 128),
    'diffusion_core.couil_attn.k_proj.weight': (128, 896),
    'diffusion_core.couil_attn.norm.bias': (896,),
    'diffusion_core.couil_attn.norm.weight': (896,),
    'diffusion_core.couil_attn.o_proj.lora_A': (896, 256),
    'diffusion_core.couil_attn.o_proj.lora_B': (256, 896),
    'diffusion_core.couil_attn.o_proj.weight': (896, 896),
    'diffusion_core.couil_attn.q_proj.lora_A': (896, 256),
    'diffusion_core.couil_attn.q_proj.lora_B': (256, 896),
    'diffusion_core.couil_attn.q_proj.weight': (896, 896),
    'diffusion_core.couil_attn.sparse_mask_generator.0.bias': (224,),
    'diffusion_core.couil_attn.sparse_mask_generator.0.weight': (224, 896),
    'diffusion_core.couil_attn.sparse_mask_generator.2.bias': (896,),
    'diffusion_core.couil_attn.sparse_mask_generator.2.weight': (896, 224),
    'diffusion_core.couil_attn.v_proj.lora_A': (896, 256),
    'diffusion_core.couil_attn.v_proj.lora_B': (256, 128),
    'diffusion_core.couil_attn.v_proj.weight': (128, 896),
    'diffusion_core.ffn.0.lora_A': (896, 256),
    'diffusion_core.ffn.0.lora_B': (256, 3584),
    'diffusion_core.ffn.0.weight': (3584, 896),
    'diffusion_core.ffn.2.lora_A': (3584, 256),
    'diffusion_core.ffn.2.lora_B': (256, 896),
    'diffusion_core.ffn.2.weight': (896, 3584),
    'diffusion_core.norm2.bias': (896,),
    'diffusion_core.norm2.weight': (896,),
    'final_norm.bias': (896,),
    'final_norm.weight': (896,),
    'ingestion.img_proj.lora_A': (384, 256),
    'ingestion.img_proj.lora_B': (256, 896),
    'ingestion.img_proj.weight': (896, 384),
    'ingestion.ingest_bypass': (),
    'ingestion.ingest_gate.bias': (896,),
    'ingestion.ingest_gate.weight': (896, 1792),
    'ingestion.mod_emb.weight': (4, 896),
    'ingestion.norm.bias': (896,),
    'ingestion.norm.weight': (896,),
    'ingestion.pos_embed': (1, 1024, 896),
    'ingestion.q1_ingest.lora_A': (896, 256),
    'ingestion.q1_ingest.lora_B': (256, 896),
    'ingestion.q1_ingest.weight': (896, 896),
    'ingestion.q2_ingest.lora_A': (896, 256),
    'ingestion.q2_ingest.lora_B': (256, 896),
    'ingestion.q2_ingest.weight': (896, 896),
    'ingestion.txt_emb.weight': (151936, 896),
    'moe.expert_swarms.0.A': (4864, 16),
    'moe.expert_swarms.0.B': (16, 4864),
    'moe.expert_swarms.0.C': (4864, 16),
    'moe.expert_swarms.0.D': (16, 4864),
    'moe.expert_swarms.0.clone_coupling': (),
    'moe.expert_swarms.0.clone_diversity': (16,),
    'moe.expert_swarms.0.clone_quant_mag': (),
    'moe.expert_swarms.0.clone_quant_noise': (),
    'moe.expert_swarms.0.clone_quant_range': (),
    'moe.expert_swarms.0.clone_quant_threshold': (),
    'moe.expert_swarms.0.population_mean': (4864,),
    'moe.expert_swarms.0.population_std': (4864,),
    'moe.expert_swarms.1.A': (4864, 16),
    'moe.expert_swarms.1.B': (16, 4864),
    'moe.expert_swarms.1.C': (4864, 16),
    'moe.expert_swarms.1.D': (16, 4864),
    'moe.expert_swarms.1.clone_coupling': (),
    'moe.expert_swarms.1.clone_diversity': (16,),
    'moe.expert_swarms.1.clone_quant_mag': (),
    'moe.expert_swarms.1.clone_quant_noise': (),
    'moe.expert_swarms.1.clone_quant_range': (),
    'moe.expert_swarms.1.clone_quant_threshold': (),
    'moe.expert_swarms.1.population_mean': (4864,),
    'moe.expert_swarms.1.population_std': (4864,),
    'moe.expert_swarms.10.A': (4864, 16),
    'moe.expert_swarms.10.B': (16, 4864),
    'moe.expert_swarms.10.C': (4864, 16),
    'moe.expert_swarms.10.D': (16, 4864),
    'moe.expert_swarms.10.clone_coupling': (),
    'moe.expert_swarms.10.clone_diversity': (16,),
    'moe.expert_swarms.10.clone_quant_mag': (),
    'moe.expert_swarms.10.clone_quant_noise': (),
    'moe.expert_swarms.10.clone_quant_range': (),
    'moe.expert_swarms.10.clone_quant_threshold': (),
    'moe.expert_swarms.10.population_mean': (4864,),
    'moe.expert_swarms.10.population_std': (4864,),
    'moe.expert_swarms.11.A': (4864, 16),
    'moe.expert_swarms.11.B': (16, 4864),
    'moe.expert_swarms.11.C': (4864, 16),
    'moe.expert_swarms.11.D': (16, 4864),
    'moe.expert_swarms.11.clone_coupling': (),
    'moe.expert_swarms.11.clone_diversity': (16,),
    'moe.expert_swarms.11.clone_quant_mag': (),
    'moe.expert_swarms.11.clone_quant_noise': (),
    'moe.expert_swarms.11.clone_quant_range': (),
    'moe.expert_swarms.11.clone_quant_threshold': (),
    'moe.expert_swarms.11.population_mean': (4864,),
    'moe.expert_swarms.11.population_std': (4864,),
    'moe.expert_swarms.12.A': (4864, 16),
    'moe.expert_swarms.12.B': (16, 4864),
    'moe.expert_swarms.12.C': (4864, 16),
    'moe.expert_swarms.12.D': (16, 4864),
    'moe.expert_swarms.12.clone_coupling': (),
    'moe.expert_swarms.12.clone_diversity': (16,),
    'moe.expert_swarms.12.clone_quant_mag': (),
    'moe.expert_swarms.12.clone_quant_noise': (),
    'moe.expert_swarms.12.clone_quant_range': (),
    'moe.expert_swarms.12.clone_quant_threshold': (),
    'moe.expert_swarms.12.population_mean': (4864,),
    'moe.expert_swarms.12.population_std': (4864,),
    'moe.expert_swarms.13.A': (4864, 16),
    'moe.expert_swarms.13.B': (16, 4864),
    'moe.expert_swarms.13.C': (4864, 16),
    'moe.expert_swarms.13.D': (16, 4864),
    'moe.expert_swarms.13.clone_coupling': (),
    'moe.expert_swarms.13.clone_diversity': (16,),
    'moe.expert_swarms.13.clone_quant_mag': (),
    'moe.expert_swarms.13.clone_quant_noise': (),
    'moe.expert_swarms.13.clone_quant_range': (),
    'moe.expert_swarms.13.clone_quant_threshold': (),
    'moe.expert_swarms.13.population_mean': (4864,),
    'moe.expert_swarms.13.population_std': (4864,),
    'moe.expert_swarms.14.A': (4864, 16),
    'moe.expert_swarms.14.B': (16, 4864),
    'moe.expert_swarms.14.C': (4864, 16),
    'moe.expert_swarms.14.D': (16, 4864),
    'moe.expert_swarms.14.clone_coupling': (),
    'moe.expert_swarms.14.clone_diversity': (16,),
    'moe.expert_swarms.14.clone_quant_mag': (),
    'moe.expert_swarms.14.clone_quant_noise': (),
    'moe.expert_swarms.14.clone_quant_range': (),
    'moe.expert_swarms.14.clone_quant_threshold': (),
    'moe.expert_swarms.14.population_mean': (4864,),
    'moe.expert_swarms.14.population_std': (4864,),
    'moe.expert_swarms.15.A': (4864, 16),
    'moe.expert_swarms.15.B': (16, 4864),
    'moe.expert_swarms.15.C': (4864, 16),
    'moe.expert_swarms.15.D': (16, 4864),
    'moe.expert_swarms.15.clone_coupling': (),
    'moe.expert_swarms.15.clone_diversity': (16,),
    'moe.expert_swarms.15.clone_quant_mag': (),
    'moe.expert_swarms.15.clone_quant_noise': (),
    'moe.expert_swarms.15.clone_quant_range': (),
    'moe.expert_swarms.15.clone_quant_threshold': (),
    'moe.expert_swarms.15.population_mean': (4864,),
    'moe.expert_swarms.15.population_std': (4864,),
    'moe.expert_swarms.16.A': (4864, 16),
    'moe.expert_swarms.16.B': (16, 4864),
    'moe.expert_swarms.16.C': (4864, 16),
    'moe.expert_swarms.16.D': (16, 4864),
    'moe.expert_swarms.16.clone_coupling': (),
    'moe.expert_swarms.16.clone_diversity': (16,),
    'moe.expert_swarms.16.clone_quant_mag': (),
    'moe.expert_swarms.16.clone_quant_noise': (),
    'moe.expert_swarms.16.clone_quant_range': (),
    'moe.expert_swarms.16.clone_quant_threshold': (),
    'moe.expert_swarms.16.population_mean': (4864,),
    'moe.expert_swarms.16.population_std': (4864,),
    'moe.expert_swarms.17.A': (4864, 16),
    'moe.expert_swarms.17.B': (16, 4864),
    'moe.expert_swarms.17.C': (4864, 16),
    'moe.expert_swarms.17.D': (16, 4864),
    'moe.expert_swarms.17.clone_coupling': (),
    'moe.expert_swarms.17.clone_diversity': (16,),
    'moe.expert_swarms.17.clone_quant_mag': (),
    'moe.expert_swarms.17.clone_quant_noise': (),
    'moe.expert_swarms.17.clone_quant_range': (),
    'moe.expert_swarms.17.clone_quant_threshold': (),
    'moe.expert_swarms.17.population_mean': (4864,),
    'moe.expert_swarms.17.population_std': (4864,),
    'moe.expert_swarms.18.A': (4864, 16),
    'moe.expert_swarms.18.B': (16, 4864),
    'moe.expert_swarms.18.C': (4864, 16),
    'moe.expert_swarms.18.D': (16, 4864),
    'moe.expert_swarms.18.clone_coupling': (),
    'moe.expert_swarms.18.clone_diversity': (16,),
    'moe.expert_swarms.18.clone_quant_mag': (),
    'moe.expert_swarms.18.clone_quant_noise': (),
    'moe.expert_swarms.18.clone_quant_range': (),
    'moe.expert_swarms.18.clone_quant_threshold': (),
    'moe.expert_swarms.18.population_mean': (4864,),
    'moe.expert_swarms.18.population_std': (4864,),
    'moe.expert_swarms.19.A': (4864, 16),
    'moe.expert_swarms.19.B': (16, 4864),
    'moe.expert_swarms.19.C': (4864, 16),
    'moe.expert_swarms.19.D': (16, 4864),
    'moe.expert_swarms.19.clone_coupling': (),
    'moe.expert_swarms.19.clone_diversity': (16,),
    'moe.expert_swarms.19.clone_quant_mag': (),
    'moe.expert_swarms.19.clone_quant_noise': (),
    'moe.expert_swarms.19.clone_quant_range': (),
    'moe.expert_swarms.19.clone_quant_threshold': (),
    'moe.expert_swarms.19.population_mean': (4864,),
    'moe.expert_swarms.19.population_std': (4864,),
    'moe.expert_swarms.2.A': (4864, 16),
    'moe.expert_swarms.2.B': (16, 4864),
    'moe.expert_swarms.2.C': (4864, 16),
    'moe.expert_swarms.2.D': (16, 4864),
    'moe.expert_swarms.2.clone_coupling': (),
    'moe.expert_swarms.2.clone_diversity': (16,),
    'moe.expert_swarms.2.clone_quant_mag': (),
    'moe.expert_swarms.2.clone_quant_noise': (),
    'moe.expert_swarms.2.clone_quant_range': (),
    'moe.expert_swarms.2.clone_quant_threshold': (),
    'moe.expert_swarms.2.population_mean': (4864,),
    'moe.expert_swarms.2.population_std': (4864,),
    'moe.expert_swarms.20.A': (4864, 16),
    'moe.expert_swarms.20.B': (16, 4864),
    'moe.expert_swarms.20.C': (4864, 16),
    'moe.expert_swarms.20.D': (16, 4864),
    'moe.expert_swarms.20.clone_coupling': (),
    'moe.expert_swarms.20.clone_diversity': (16,),
    'moe.expert_swarms.20.clone_quant_mag': (),
    'moe.expert_swarms.20.clone_quant_noise': (),
    'moe.expert_swarms.20.clone_quant_range': (),
    'moe.expert_swarms.20.clone_quant_threshold': (),
    'moe.expert_swarms.20.population_mean': (4864,),
    'moe.expert_swarms.20.population_std': (4864,),
    'moe.expert_swarms.21.A': (4864, 16),
    'moe.expert_swarms.21.B': (16, 4864),
    'moe.expert_swarms.21.C': (4864, 16),
    'moe.expert_swarms.21.D': (16, 4864),
    'moe.expert_swarms.21.clone_coupling': (),
    'moe.expert_swarms.21.clone_diversity': (16,),
    'moe.expert_swarms.21.clone_quant_mag': (),
    'moe.expert_swarms.21.clone_quant_noise': (),
    'moe.expert_swarms.21.clone_quant_range': (),
    'moe.expert_swarms.21.clone_quant_threshold': (),
    'moe.expert_swarms.21.population_mean': (4864,),
    'moe.expert_swarms.21.population_std': (4864,),
    'moe.expert_swarms.22.A': (4864, 16),
    'moe.expert_swarms.22.B': (16, 4864),
    'moe.expert_swarms.22.C': (4864, 16),
    'moe.expert_swarms.22.D': (16, 4864),
    'moe.expert_swarms.22.clone_coupling': (),
    'moe.expert_swarms.22.clone_diversity': (16,),
    'moe.expert_swarms.22.clone_quant_mag': (),
    'moe.expert_swarms.22.clone_quant_noise': (),
    'moe.expert_swarms.22.clone_quant_range': (),
    'moe.expert_swarms.22.clone_quant_threshold': (),
    'moe.expert_swarms.22.population_mean': (4864,),
    'moe.expert_swarms.22.population_std': (4864,),
    'moe.expert_swarms.23.A': (4864, 16),
    'moe.expert_swarms.23.B': (16, 4864),
    'moe.expert_swarms.23.C': (4864, 16),
    'moe.expert_swarms.23.D': (16, 4864),
    'moe.expert_swarms.23.clone_coupling': (),
    'moe.expert_swarms.23.clone_diversity': (16,),
    'moe.expert_swarms.23.clone_quant_mag': (),
    'moe.expert_swarms.23.clone_quant_noise': (),
    'moe.expert_swarms.23.clone_quant_range': (),
    'moe.expert_swarms.23.clone_quant_threshold': (),
    'moe.expert_swarms.23.population_mean': (4864,),
    'moe.expert_swarms.23.population_std': (4864,),
    'moe.expert_swarms.24.A': (4864, 16),
    'moe.expert_swarms.24.B': (16, 4864),
    'moe.expert_swarms.24.C': (4864, 16),
    'moe.expert_swarms.24.D': (16, 4864),
    'moe.expert_swarms.24.clone_coupling': (),
    'moe.expert_swarms.24.clone_diversity': (16,),
    'moe.expert_swarms.24.clone_quant_mag': (),
    'moe.expert_swarms.24.clone_quant_noise': (),
    'moe.expert_swarms.24.clone_quant_range': (),
    'moe.expert_swarms.24.clone_quant_threshold': (),
    'moe.expert_swarms.24.population_mean': (4864,),
    'moe.expert_swarms.24.population_std': (4864,),
    'moe.expert_swarms.25.A': (4864, 16),
    'moe.expert_swarms.25.B': (16, 4864),
    'moe.expert_swarms.25.C': (4864, 16),
    'moe.expert_swarms.25.D': (16, 4864),
    'moe.expert_swarms.25.clone_coupling': (),
    'moe.expert_swarms.25.clone_diversity': (16,),
    'moe.expert_swarms.25.clone_quant_mag': (),
    'moe.expert_swarms.25.clone_quant_noise': (),
    'moe.expert_swarms.25.clone_quant_range': (),
    'moe.expert_swarms.25.clone_quant_threshold': (),
    'moe.expert_swarms.25.population_mean': (4864,),
    'moe.expert_swarms.25.population_std': (4864,),
    'moe.expert_swarms.26.A': (4864, 16),
    'moe.expert_swarms.26.B': (16, 4864),
    'moe.expert_swarms.26.C': (4864, 16),
    'moe.expert_swarms.26.D': (16, 4864),
    'moe.expert_swarms.26.clone_coupling': (),
    'moe.expert_swarms.26.clone_diversity': (16,),
    'moe.expert_swarms.26.clone_quant_mag': (),
    'moe.expert_swarms.26.clone_quant_noise': (),
    'moe.expert_swarms.26.clone_quant_range': (),
    'moe.expert_swarms.26.clone_quant_threshold': (),
    'moe.expert_swarms.26.population_mean': (4864,),
    'moe.expert_swarms.26.population_std': (4864,),
    'moe.expert_swarms.27.A': (4864, 16),
    'moe.expert_swarms.27.B': (16, 4864),
    'moe.expert_swarms.27.C': (4864, 16),
    'moe.expert_swarms.27.D': (16, 4864),
    'moe.expert_swarms.27.clone_coupling': (),
    'moe.expert_swarms.27.clone_diversity': (16,),
    'moe.expert_swarms.27.clone_quant_mag': (),
    'moe.expert_swarms.27.clone_quant_noise': (),
    'moe.expert_swarms.27.clone_quant_range': (),
    'moe.expert_swarms.27.clone_quant_threshold': (),
    'moe.expert_swarms.27.population_mean': (4864,),
    'moe.expert_swarms.27.population_std': (4864,),
    'moe.expert_swarms.28.A': (4864, 16),
    'moe.expert_swarms.28.B': (16, 4864),
    'moe.expert_swarms.28.C': (4864, 16),
    'moe.expert_swarms.28.D': (16, 4864),
    'moe.expert_swarms.28.clone_coupling': (),
    'moe.expert_swarms.28.clone_diversity': (16,),
    'moe.expert_swarms.28.clone_quant_mag': (),
    'moe.expert_swarms.28.clone_quant_noise': (),
    'moe.expert_swarms.28.clone_quant_range': (),
    'moe.expert_swarms.28.clone_quant_threshold': (),
    'moe.expert_swarms.28.population_mean': (4864,),
    'moe.expert_swarms.28.population_std': (4864,),
    'moe.expert_swarms.29.A': (4864, 16),
    'moe.expert_swarms.29.B': (16, 4864),
    'moe.expert_swarms.29.C': (4864, 16),
    'moe.expert_swarms.29.D': (16, 4864),
    'moe.expert_swarms.29.clone_coupling': (),
    'moe.expert_swarms.29.clone_diversity': (16,),
    'moe.expert_swarms.29.clone_quant_mag': (),
    'moe.expert_swarms.29.clone_quant_noise': (),
    'moe.expert_swarms.29.clone_quant_range': (),
    'moe.expert_swarms.29.clone_quant_threshold': (),
    'moe.expert_swarms.29.population_mean': (4864,),
    'moe.expert_swarms.29.population_std': (4864,),
    'moe.expert_swarms.3.A': (4864, 16),
    'moe.expert_swarms.3.B': (16, 4864),
    'moe.expert_swarms.3.C': (4864, 16),
    'moe.expert_swarms.3.D': (16, 4864),
    'moe.expert_swarms.3.clone_coupling': (),
    'moe.expert_swarms.3.clone_diversity': (16,),
    'moe.expert_swarms.3.clone_quant_mag': (),
    'moe.expert_swarms.3.clone_quant_noise': (),
    'moe.expert_swarms.3.clone_quant_range': (),
    'moe.expert_swarms.3.clone_quant_threshold': (),
    'moe.expert_swarms.3.population_mean': (4864,),
    'moe.expert_swarms.3.population_std': (4864,),
    'moe.expert_swarms.30.A': (4864, 16),
    'moe.expert_swarms.30.B': (16, 4864),
    'moe.expert_swarms.30.C': (4864, 16),
    'moe.expert_swarms.30.D': (16, 4864),
    'moe.expert_swarms.30.clone_coupling': (),
    'moe.expert_swarms.30.clone_diversity': (16,),
    'moe.expert_swarms.30.clone_quant_mag': (),
    'moe.expert_swarms.30.clone_quant_noise': (),
    'moe.expert_swarms.30.clone_quant_range': (),
    'moe.expert_swarms.30.clone_quant_threshold': (),
    'moe.expert_swarms.30.population_mean': (4864,),
    'moe.expert_swarms.30.population_std': (4864,),
    'moe.expert_swarms.31.A': (4864, 16),
    'moe.expert_swarms.31.B': (16, 4864),
    'moe.expert_swarms.31.C': (4864, 16),
    'moe.expert_swarms.31.D': (16, 4864),
    'moe.expert_swarms.31.clone_coupling': (),
    'moe.expert_swarms.31.clone_diversity': (16,),
    'moe.expert_swarms.31.clone_quant_mag': (),
    'moe.expert_swarms.31.clone_quant_noise': (),
    'moe.expert_swarms.31.clone_quant_range': (),
    'moe.expert_swarms.31.clone_quant_threshold': (),
    'moe.expert_swarms.31.population_mean': (4864,),
    'moe.expert_swarms.31.population_std': (4864,),
    'moe.expert_swarms.32.A': (4864, 16),
    'moe.expert_swarms.32.B': (16, 4864),
    'moe.expert_swarms.32.C': (4864, 16),
    'moe.expert_swarms.32.D': (16, 4864),
    'moe.expert_swarms.32.clone_coupling': (),
    'moe.expert_swarms.32.clone_diversity': (16,),
    'moe.expert_swarms.32.clone_quant_mag': (),
    'moe.expert_swarms.32.clone_quant_noise': (),
    'moe.expert_swarms.32.clone_quant_range': (),
    'moe.expert_swarms.32.clone_quant_threshold': (),
    'moe.expert_swarms.32.population_mean': (4864,),
    'moe.expert_swarms.32.population_std': (4864,),
    'moe.expert_swarms.33.A': (4864, 16),
    'moe.expert_swarms.33.B': (16, 4864),
    'moe.expert_swarms.33.C': (4864, 16),
    'moe.expert_swarms.33.D': (16, 4864),
    'moe.expert_swarms.33.clone_coupling': (),
    'moe.expert_swarms.33.clone_diversity': (16,),
    'moe.expert_swarms.33.clone_quant_mag': (),
    'moe.expert_swarms.33.clone_quant_noise': (),
    'moe.expert_swarms.33.clone_quant_range': (),
    'moe.expert_swarms.33.clone_quant_threshold': (),
    'moe.expert_swarms.33.population_mean': (4864,),
    'moe.expert_swarms.33.population_std': (4864,),
    'moe.expert_swarms.4.A': (4864, 16),
    'moe.expert_swarms.4.B': (16, 4864),
    'moe.expert_swarms.4.C': (4864, 16),
    'moe.expert_swarms.4.D': (16, 4864),
    'moe.expert_swarms.4.clone_coupling': (),
    'moe.expert_swarms.4.clone_diversity': (16,),
    'moe.expert_swarms.4.clone_quant_mag': (),
    'moe.expert_swarms.4.clone_quant_noise': (),
    'moe.expert_swarms.4.clone_quant_range': (),
    'moe.expert_swarms.4.clone_quant_threshold': (),
    'moe.expert_swarms.4.population_mean': (4864,),
    'moe.expert_swarms.4.population_std': (4864,),
    'moe.expert_swarms.5.A': (4864, 16),
    'moe.expert_swarms.5.B': (16, 4864),
    'moe.expert_swarms.5.C': (4864, 16),
    'moe.expert_swarms.5.D': (16, 4864),
    'moe.expert_swarms.5.clone_coupling': (),
    'moe.expert_swarms.5.clone_diversity': (16,),
    'moe.expert_swarms.5.clone_quant_mag': (),
    'moe.expert_swarms.5.clone_quant_noise': (),
    'moe.expert_swarms.5.clone_quant_range': (),
    'moe.expert_swarms.5.clone_quant_threshold': (),
    'moe.expert_swarms.5.population_mean': (4864,),
    'moe.expert_swarms.5.population_std': (4864,),
    'moe.expert_swarms.6.A': (4864, 16),
    'moe.expert_swarms.6.B': (16, 4864),
    'moe.expert_swarms.6.C': (4864, 16),
    'moe.expert_swarms.6.D': (16, 4864),
    'moe.expert_swarms.6.clone_coupling': (),
    'moe.expert_swarms.6.clone_diversity': (16,),
    'moe.expert_swarms.6.clone_quant_mag': (),
    'moe.expert_swarms.6.clone_quant_noise': (),
    'moe.expert_swarms.6.clone_quant_range': (),
    'moe.expert_swarms.6.clone_quant_threshold': (),
    'moe.expert_swarms.6.population_mean': (4864,),
    'moe.expert_swarms.6.population_std': (4864,),
    'moe.expert_swarms.7.A': (4864, 16),
    'moe.expert_swarms.7.B': (16, 4864),
    'moe.expert_swarms.7.C': (4864, 16),
    'moe.expert_swarms.7.D': (16, 4864),
    'moe.expert_swarms.7.clone_coupling': (),
    'moe.expert_swarms.7.clone_diversity': (16,),
    'moe.expert_swarms.7.clone_quant_mag': (),
    'moe.expert_swarms.7.clone_quant_noise': (),
    'moe.expert_swarms.7.clone_quant_range': (),
    'moe.expert_swarms.7.clone_quant_threshold': (),
    'moe.expert_swarms.7.population_mean': (4864,),
    'moe.expert_swarms.7.population_std': (4864,),
    'moe.expert_swarms.8.A': (4864, 16),
    'moe.expert_swarms.8.B': (16, 4864),
    'moe.expert_swarms.8.C': (4864, 16),
    'moe.expert_swarms.8.D': (16, 4864),
    'moe.expert_swarms.8.clone_coupling': (),
    'moe.expert_swarms.8.clone_diversity': (16,),
    'moe.expert_swarms.8.clone_quant_mag': (),
    'moe.expert_swarms.8.clone_quant_noise': (),
    'moe.expert_swarms.8.clone_quant_range': (),
    'moe.expert_swarms.8.clone_quant_threshold': (),
    'moe.expert_swarms.8.population_mean': (4864,),
    'moe.expert_swarms.8.population_std': (4864,),
    'moe.expert_swarms.9.A': (4864, 16),
    'moe.expert_swarms.9.B': (16, 4864),
    'moe.expert_swarms.9.C': (4864, 16),
    'moe.expert_swarms.9.D': (16, 4864),
    'moe.expert_swarms.9.clone_coupling': (),
    'moe.expert_swarms.9.clone_diversity': (16,),
    'moe.expert_swarms.9.clone_quant_mag': (),
    'moe.expert_swarms.9.clone_quant_noise': (),
    'moe.expert_swarms.9.clone_quant_range': (),
    'moe.expert_swarms.9.clone_quant_threshold': (),
    'moe.expert_swarms.9.population_mean': (4864,),
    'moe.expert_swarms.9.population_std': (4864,),
    'moe.output_norm.bias': (896,),
    'moe.output_norm.weight': (896,),
    'moe.router.balanced_router.lora_A': (896, 256),
    'moe.router.balanced_router.lora_B': (256, 34),
    'moe.router.balanced_router.weight': (34, 896),
    'moe.router.complexity_classifier.0.bias': (224,),
    'moe.router.complexity_classifier.0.weight': (224, 896),
    'moe.router.complexity_classifier.2.bias': (3,),
    'moe.router.complexity_classifier.2.weight': (3, 224),
    'moe.router.diffusion_router.lora_A': (896, 256),
    'moe.router.diffusion_router.lora_B': (256, 34),
    'moe.router.diffusion_router.weight': (34, 896),
    'moe.router.fast_router.lora_A': (896, 256),
    'moe.router.fast_router.lora_B': (256, 34),
    'moe.router.fast_router.weight': (34, 896),
    'moe.w1': (34, 896, 4864),
    'moe.w1_lora_A': (34, 896, 16),
    'moe.w1_lora_B': (34, 16, 4864),
    'moe.w2': (34, 4864, 896),
    'moe.w2_lora_A': (34, 4864, 16),
    'moe.w2_lora_B': (34, 16, 896),
    'moe.wgate': (34, 896, 4864),
    'moe.wgate_lora_A': (34, 896, 16),
    'moe.wgate_lora_B': (34, 16, 4864),
    'norm_decomp.bias': (896,),
    'norm_decomp.weight': (896,),
    'norm_diff.bias': (896,),
    'norm_diff.weight': (896,),
    'norm_moe.bias': (896,),
    'norm_moe.weight': (896,),
    'pre_final_norm.bias': (896,),
    'pre_final_norm.weight': (896,),
    'quantum_formulas.boltzmann': (),
    'quantum_formulas.hbar': (),
    'quantum_formulas.sqrt_33': (),
    'quillan_finalizer.lora_A': (896, 256),
    'quillan_finalizer.lora_B': (256, 896),
    'quillan_finalizer.weight': (896, 896),
    'quillan_finalizer2.lora_A': (896, 256),
    'quillan_finalizer2.lora_B': (256, 896),
    'quillan_finalizer2.weight': (896, 896),
    'quillan_gate.bias': (896,),
    'quillan_gate.weight': (896, 1792),
    'txt_dec.lora_A': (896, 256),
    'txt_dec.lora_B': (256, 151936),
    'txt_dec.weight': (151936, 896),
}

# Dtypes as stored in socket-v2 (base keys only). Used to allocate buffers with
# matching dtype so load_state_dict is a bit-exact copy, not a cast.
_STORED_DTYPE_CODES: Dict[str, torch.dtype] = {
    "I8": torch.int8,
    "F16": torch.float16,
    "F32": torch.float32,
}

# Dtype code per key (from the same header). Keys not listed default to F32.
EXPECTED_DTYPE_CODES: Dict[str, str] = {
    'decomposition.K_gate.weight': 'I8',
    'decomposition.Q_gate.weight': 'I8',
    'decomposition.V_gate.weight': 'I8',
    'decomposition.W_gate.weight': 'I8',
    'decomposition.vectors.Constraint.lora_A': 'F16',
    'decomposition.vectors.Constraint.lora_B': 'F16',
    'decomposition.vectors.Constraint.weight': 'I8',
    'decomposition.vectors.Context.lora_A': 'F16',
    'decomposition.vectors.Context.lora_B': 'F16',
    'decomposition.vectors.Context.weight': 'I8',
    'decomposition.vectors.Creativity.lora_A': 'F16',
    'decomposition.vectors.Creativity.lora_B': 'F16',
    'decomposition.vectors.Creativity.weight': 'I8',
    'decomposition.vectors.Ethics.lora_A': 'F16',
    'decomposition.vectors.Ethics.lora_B': 'F16',
    'decomposition.vectors.Ethics.weight': 'I8',
    'decomposition.vectors.Intent.lora_A': 'F16',
    'decomposition.vectors.Intent.lora_B': 'F16',
    'decomposition.vectors.Intent.weight': 'I8',
    'decomposition.vectors.Language.lora_A': 'F16',
    'decomposition.vectors.Language.lora_B': 'F16',
    'decomposition.vectors.Language.weight': 'I8',
    'decomposition.vectors.Meta.lora_A': 'F16',
    'decomposition.vectors.Meta.lora_B': 'F16',
    'decomposition.vectors.Meta.weight': 'I8',
    'decomposition.vectors.Sentiment.lora_A': 'F16',
    'decomposition.vectors.Sentiment.lora_B': 'F16',
    'decomposition.vectors.Sentiment.weight': 'I8',
    'decomposition.vectors.Strategy.lora_A': 'F16',
    'decomposition.vectors.Strategy.lora_B': 'F16',
    'decomposition.vectors.Strategy.weight': 'I8',
    'diffusion_core.couil_attn.k_proj.lora_A': 'F16',
    'diffusion_core.couil_attn.k_proj.lora_B': 'F16',
    'diffusion_core.couil_attn.k_proj.weight': 'I8',
    'diffusion_core.couil_attn.norm.bias': 'F32',
    'diffusion_core.couil_attn.norm.weight': 'F32',
    'diffusion_core.couil_attn.o_proj.lora_A': 'F16',
    'diffusion_core.couil_attn.o_proj.lora_B': 'F16',
    'diffusion_core.couil_attn.o_proj.weight': 'I8',
    'diffusion_core.couil_attn.q_proj.lora_A': 'F16',
    'diffusion_core.couil_attn.q_proj.lora_B': 'F16',
    'diffusion_core.couil_attn.q_proj.weight': 'I8',
    'diffusion_core.couil_attn.sparse_mask_generator.0.bias': 'F32',
    'diffusion_core.couil_attn.sparse_mask_generator.0.weight': 'F16',
    'diffusion_core.couil_attn.sparse_mask_generator.2.bias': 'F32',
    'diffusion_core.couil_attn.sparse_mask_generator.2.weight': 'F16',
    'diffusion_core.couil_attn.v_proj.lora_A': 'F16',
    'diffusion_core.couil_attn.v_proj.lora_B': 'F16',
    'diffusion_core.couil_attn.v_proj.weight': 'I8',
    'diffusion_core.ffn.0.lora_A': 'F16',
    'diffusion_core.ffn.0.lora_B': 'F16',
    'diffusion_core.ffn.0.weight': 'I8',
    'diffusion_core.ffn.2.lora_A': 'F16',
    'diffusion_core.ffn.2.lora_B': 'F16',
    'diffusion_core.ffn.2.weight': 'I8',
    'diffusion_core.norm2.bias': 'F32',
    'diffusion_core.norm2.weight': 'F32',
    'final_norm.bias': 'F32',
    'final_norm.weight': 'F32',
    'ingestion.img_proj.lora_A': 'F16',
    'ingestion.img_proj.lora_B': 'F16',
    'ingestion.img_proj.weight': 'F16',
    'ingestion.ingest_bypass': 'F32',
    'ingestion.ingest_gate.bias': 'F32',
    'ingestion.ingest_gate.weight': 'F16',
    'ingestion.mod_emb.weight': 'I8',
    'ingestion.norm.bias': 'F32',
    'ingestion.norm.weight': 'F32',
    'ingestion.pos_embed': 'F16',
    'ingestion.q1_ingest.lora_A': 'F16',
    'ingestion.q1_ingest.lora_B': 'F16',
    'ingestion.q1_ingest.weight': 'I8',
    'ingestion.q2_ingest.lora_A': 'F16',
    'ingestion.q2_ingest.lora_B': 'F16',
    'ingestion.q2_ingest.weight': 'I8',
    'ingestion.txt_emb.weight': 'F16',
    'moe.expert_swarms.0.A': 'F16',
    'moe.expert_swarms.0.B': 'F16',
    'moe.expert_swarms.0.C': 'F16',
    'moe.expert_swarms.0.D': 'F16',
    'moe.expert_swarms.0.clone_coupling': 'F32',
    'moe.expert_swarms.0.clone_diversity': 'F32',
    'moe.expert_swarms.0.clone_quant_mag': 'F32',
    'moe.expert_swarms.0.clone_quant_noise': 'F32',
    'moe.expert_swarms.0.clone_quant_range': 'F32',
    'moe.expert_swarms.0.clone_quant_threshold': 'F32',
    'moe.expert_swarms.0.population_mean': 'F32',
    'moe.expert_swarms.0.population_std': 'F32',
    'moe.expert_swarms.1.A': 'F16',
    'moe.expert_swarms.1.B': 'F16',
    'moe.expert_swarms.1.C': 'F16',
    'moe.expert_swarms.1.D': 'F16',
    'moe.expert_swarms.1.clone_coupling': 'F32',
    'moe.expert_swarms.1.clone_diversity': 'F32',
    'moe.expert_swarms.1.clone_quant_mag': 'F32',
    'moe.expert_swarms.1.clone_quant_noise': 'F32',
    'moe.expert_swarms.1.clone_quant_range': 'F32',
    'moe.expert_swarms.1.clone_quant_threshold': 'F32',
    'moe.expert_swarms.1.population_mean': 'F32',
    'moe.expert_swarms.1.population_std': 'F32',
    'moe.expert_swarms.10.A': 'F16',
    'moe.expert_swarms.10.B': 'F16',
    'moe.expert_swarms.10.C': 'F16',
    'moe.expert_swarms.10.D': 'F16',
    'moe.expert_swarms.10.clone_coupling': 'F32',
    'moe.expert_swarms.10.clone_diversity': 'F32',
    'moe.expert_swarms.10.clone_quant_mag': 'F32',
    'moe.expert_swarms.10.clone_quant_noise': 'F32',
    'moe.expert_swarms.10.clone_quant_range': 'F32',
    'moe.expert_swarms.10.clone_quant_threshold': 'F32',
    'moe.expert_swarms.10.population_mean': 'F32',
    'moe.expert_swarms.10.population_std': 'F32',
    'moe.expert_swarms.11.A': 'F16',
    'moe.expert_swarms.11.B': 'F16',
    'moe.expert_swarms.11.C': 'F16',
    'moe.expert_swarms.11.D': 'F16',
    'moe.expert_swarms.11.clone_coupling': 'F32',
    'moe.expert_swarms.11.clone_diversity': 'F32',
    'moe.expert_swarms.11.clone_quant_mag': 'F32',
    'moe.expert_swarms.11.clone_quant_noise': 'F32',
    'moe.expert_swarms.11.clone_quant_range': 'F32',
    'moe.expert_swarms.11.clone_quant_threshold': 'F32',
    'moe.expert_swarms.11.population_mean': 'F32',
    'moe.expert_swarms.11.population_std': 'F32',
    'moe.expert_swarms.12.A': 'F16',
    'moe.expert_swarms.12.B': 'F16',
    'moe.expert_swarms.12.C': 'F16',
    'moe.expert_swarms.12.D': 'F16',
    'moe.expert_swarms.12.clone_coupling': 'F32',
    'moe.expert_swarms.12.clone_diversity': 'F32',
    'moe.expert_swarms.12.clone_quant_mag': 'F32',
    'moe.expert_swarms.12.clone_quant_noise': 'F32',
    'moe.expert_swarms.12.clone_quant_range': 'F32',
    'moe.expert_swarms.12.clone_quant_threshold': 'F32',
    'moe.expert_swarms.12.population_mean': 'F32',
    'moe.expert_swarms.12.population_std': 'F32',
    'moe.expert_swarms.13.A': 'F16',
    'moe.expert_swarms.13.B': 'F16',
    'moe.expert_swarms.13.C': 'F16',
    'moe.expert_swarms.13.D': 'F16',
    'moe.expert_swarms.13.clone_coupling': 'F32',
    'moe.expert_swarms.13.clone_diversity': 'F32',
    'moe.expert_swarms.13.clone_quant_mag': 'F32',
    'moe.expert_swarms.13.clone_quant_noise': 'F32',
    'moe.expert_swarms.13.clone_quant_range': 'F32',
    'moe.expert_swarms.13.clone_quant_threshold': 'F32',
    'moe.expert_swarms.13.population_mean': 'F32',
    'moe.expert_swarms.13.population_std': 'F32',
    'moe.expert_swarms.14.A': 'F16',
    'moe.expert_swarms.14.B': 'F16',
    'moe.expert_swarms.14.C': 'F16',
    'moe.expert_swarms.14.D': 'F16',
    'moe.expert_swarms.14.clone_coupling': 'F32',
    'moe.expert_swarms.14.clone_diversity': 'F32',
    'moe.expert_swarms.14.clone_quant_mag': 'F32',
    'moe.expert_swarms.14.clone_quant_noise': 'F32',
    'moe.expert_swarms.14.clone_quant_range': 'F32',
    'moe.expert_swarms.14.clone_quant_threshold': 'F32',
    'moe.expert_swarms.14.population_mean': 'F32',
    'moe.expert_swarms.14.population_std': 'F32',
    'moe.expert_swarms.15.A': 'F16',
    'moe.expert_swarms.15.B': 'F16',
    'moe.expert_swarms.15.C': 'F16',
    'moe.expert_swarms.15.D': 'F16',
    'moe.expert_swarms.15.clone_coupling': 'F32',
    'moe.expert_swarms.15.clone_diversity': 'F32',
    'moe.expert_swarms.15.clone_quant_mag': 'F32',
    'moe.expert_swarms.15.clone_quant_noise': 'F32',
    'moe.expert_swarms.15.clone_quant_range': 'F32',
    'moe.expert_swarms.15.clone_quant_threshold': 'F32',
    'moe.expert_swarms.15.population_mean': 'F32',
    'moe.expert_swarms.15.population_std': 'F32',
    'moe.expert_swarms.16.A': 'F16',
    'moe.expert_swarms.16.B': 'F16',
    'moe.expert_swarms.16.C': 'F16',
    'moe.expert_swarms.16.D': 'F16',
    'moe.expert_swarms.16.clone_coupling': 'F32',
    'moe.expert_swarms.16.clone_diversity': 'F32',
    'moe.expert_swarms.16.clone_quant_mag': 'F32',
    'moe.expert_swarms.16.clone_quant_noise': 'F32',
    'moe.expert_swarms.16.clone_quant_range': 'F32',
    'moe.expert_swarms.16.clone_quant_threshold': 'F32',
    'moe.expert_swarms.16.population_mean': 'F32',
    'moe.expert_swarms.16.population_std': 'F32',
    'moe.expert_swarms.17.A': 'F16',
    'moe.expert_swarms.17.B': 'F16',
    'moe.expert_swarms.17.C': 'F16',
    'moe.expert_swarms.17.D': 'F16',
    'moe.expert_swarms.17.clone_coupling': 'F32',
    'moe.expert_swarms.17.clone_diversity': 'F32',
    'moe.expert_swarms.17.clone_quant_mag': 'F32',
    'moe.expert_swarms.17.clone_quant_noise': 'F32',
    'moe.expert_swarms.17.clone_quant_range': 'F32',
    'moe.expert_swarms.17.clone_quant_threshold': 'F32',
    'moe.expert_swarms.17.population_mean': 'F32',
    'moe.expert_swarms.17.population_std': 'F32',
    'moe.expert_swarms.18.A': 'F16',
    'moe.expert_swarms.18.B': 'F16',
    'moe.expert_swarms.18.C': 'F16',
    'moe.expert_swarms.18.D': 'F16',
    'moe.expert_swarms.18.clone_coupling': 'F32',
    'moe.expert_swarms.18.clone_diversity': 'F32',
    'moe.expert_swarms.18.clone_quant_mag': 'F32',
    'moe.expert_swarms.18.clone_quant_noise': 'F32',
    'moe.expert_swarms.18.clone_quant_range': 'F32',
    'moe.expert_swarms.18.clone_quant_threshold': 'F32',
    'moe.expert_swarms.18.population_mean': 'F32',
    'moe.expert_swarms.18.population_std': 'F32',
    'moe.expert_swarms.19.A': 'F16',
    'moe.expert_swarms.19.B': 'F16',
    'moe.expert_swarms.19.C': 'F16',
    'moe.expert_swarms.19.D': 'F16',
    'moe.expert_swarms.19.clone_coupling': 'F32',
    'moe.expert_swarms.19.clone_diversity': 'F32',
    'moe.expert_swarms.19.clone_quant_mag': 'F32',
    'moe.expert_swarms.19.clone_quant_noise': 'F32',
    'moe.expert_swarms.19.clone_quant_range': 'F32',
    'moe.expert_swarms.19.clone_quant_threshold': 'F32',
    'moe.expert_swarms.19.population_mean': 'F32',
    'moe.expert_swarms.19.population_std': 'F32',
    'moe.expert_swarms.2.A': 'F16',
    'moe.expert_swarms.2.B': 'F16',
    'moe.expert_swarms.2.C': 'F16',
    'moe.expert_swarms.2.D': 'F16',
    'moe.expert_swarms.2.clone_coupling': 'F32',
    'moe.expert_swarms.2.clone_diversity': 'F32',
    'moe.expert_swarms.2.clone_quant_mag': 'F32',
    'moe.expert_swarms.2.clone_quant_noise': 'F32',
    'moe.expert_swarms.2.clone_quant_range': 'F32',
    'moe.expert_swarms.2.clone_quant_threshold': 'F32',
    'moe.expert_swarms.2.population_mean': 'F32',
    'moe.expert_swarms.2.population_std': 'F32',
    'moe.expert_swarms.20.A': 'F16',
    'moe.expert_swarms.20.B': 'F16',
    'moe.expert_swarms.20.C': 'F16',
    'moe.expert_swarms.20.D': 'F16',
    'moe.expert_swarms.20.clone_coupling': 'F32',
    'moe.expert_swarms.20.clone_diversity': 'F32',
    'moe.expert_swarms.20.clone_quant_mag': 'F32',
    'moe.expert_swarms.20.clone_quant_noise': 'F32',
    'moe.expert_swarms.20.clone_quant_range': 'F32',
    'moe.expert_swarms.20.clone_quant_threshold': 'F32',
    'moe.expert_swarms.20.population_mean': 'F32',
    'moe.expert_swarms.20.population_std': 'F32',
    'moe.expert_swarms.21.A': 'F16',
    'moe.expert_swarms.21.B': 'F16',
    'moe.expert_swarms.21.C': 'F16',
    'moe.expert_swarms.21.D': 'F16',
    'moe.expert_swarms.21.clone_coupling': 'F32',
    'moe.expert_swarms.21.clone_diversity': 'F32',
    'moe.expert_swarms.21.clone_quant_mag': 'F32',
    'moe.expert_swarms.21.clone_quant_noise': 'F32',
    'moe.expert_swarms.21.clone_quant_range': 'F32',
    'moe.expert_swarms.21.clone_quant_threshold': 'F32',
    'moe.expert_swarms.21.population_mean': 'F32',
    'moe.expert_swarms.21.population_std': 'F32',
    'moe.expert_swarms.22.A': 'F16',
    'moe.expert_swarms.22.B': 'F16',
    'moe.expert_swarms.22.C': 'F16',
    'moe.expert_swarms.22.D': 'F16',
    'moe.expert_swarms.22.clone_coupling': 'F32',
    'moe.expert_swarms.22.clone_diversity': 'F32',
    'moe.expert_swarms.22.clone_quant_mag': 'F32',
    'moe.expert_swarms.22.clone_quant_noise': 'F32',
    'moe.expert_swarms.22.clone_quant_range': 'F32',
    'moe.expert_swarms.22.clone_quant_threshold': 'F32',
    'moe.expert_swarms.22.population_mean': 'F32',
    'moe.expert_swarms.22.population_std': 'F32',
    'moe.expert_swarms.23.A': 'F16',
    'moe.expert_swarms.23.B': 'F16',
    'moe.expert_swarms.23.C': 'F16',
    'moe.expert_swarms.23.D': 'F16',
    'moe.expert_swarms.23.clone_coupling': 'F32',
    'moe.expert_swarms.23.clone_diversity': 'F32',
    'moe.expert_swarms.23.clone_quant_mag': 'F32',
    'moe.expert_swarms.23.clone_quant_noise': 'F32',
    'moe.expert_swarms.23.clone_quant_range': 'F32',
    'moe.expert_swarms.23.clone_quant_threshold': 'F32',
    'moe.expert_swarms.23.population_mean': 'F32',
    'moe.expert_swarms.23.population_std': 'F32',
    'moe.expert_swarms.24.A': 'F16',
    'moe.expert_swarms.24.B': 'F16',
    'moe.expert_swarms.24.C': 'F16',
    'moe.expert_swarms.24.D': 'F16',
    'moe.expert_swarms.24.clone_coupling': 'F32',
    'moe.expert_swarms.24.clone_diversity': 'F32',
    'moe.expert_swarms.24.clone_quant_mag': 'F32',
    'moe.expert_swarms.24.clone_quant_noise': 'F32',
    'moe.expert_swarms.24.clone_quant_range': 'F32',
    'moe.expert_swarms.24.clone_quant_threshold': 'F32',
    'moe.expert_swarms.24.population_mean': 'F32',
    'moe.expert_swarms.24.population_std': 'F32',
    'moe.expert_swarms.25.A': 'F16',
    'moe.expert_swarms.25.B': 'F16',
    'moe.expert_swarms.25.C': 'F16',
    'moe.expert_swarms.25.D': 'F16',
    'moe.expert_swarms.25.clone_coupling': 'F32',
    'moe.expert_swarms.25.clone_diversity': 'F32',
    'moe.expert_swarms.25.clone_quant_mag': 'F32',
    'moe.expert_swarms.25.clone_quant_noise': 'F32',
    'moe.expert_swarms.25.clone_quant_range': 'F32',
    'moe.expert_swarms.25.clone_quant_threshold': 'F32',
    'moe.expert_swarms.25.population_mean': 'F32',
    'moe.expert_swarms.25.population_std': 'F32',
    'moe.expert_swarms.26.A': 'F16',
    'moe.expert_swarms.26.B': 'F16',
    'moe.expert_swarms.26.C': 'F16',
    'moe.expert_swarms.26.D': 'F16',
    'moe.expert_swarms.26.clone_coupling': 'F32',
    'moe.expert_swarms.26.clone_diversity': 'F32',
    'moe.expert_swarms.26.clone_quant_mag': 'F32',
    'moe.expert_swarms.26.clone_quant_noise': 'F32',
    'moe.expert_swarms.26.clone_quant_range': 'F32',
    'moe.expert_swarms.26.clone_quant_threshold': 'F32',
    'moe.expert_swarms.26.population_mean': 'F32',
    'moe.expert_swarms.26.population_std': 'F32',
    'moe.expert_swarms.27.A': 'F16',
    'moe.expert_swarms.27.B': 'F16',
    'moe.expert_swarms.27.C': 'F16',
    'moe.expert_swarms.27.D': 'F16',
    'moe.expert_swarms.27.clone_coupling': 'F32',
    'moe.expert_swarms.27.clone_diversity': 'F32',
    'moe.expert_swarms.27.clone_quant_mag': 'F32',
    'moe.expert_swarms.27.clone_quant_noise': 'F32',
    'moe.expert_swarms.27.clone_quant_range': 'F32',
    'moe.expert_swarms.27.clone_quant_threshold': 'F32',
    'moe.expert_swarms.27.population_mean': 'F32',
    'moe.expert_swarms.27.population_std': 'F32',
    'moe.expert_swarms.28.A': 'F16',
    'moe.expert_swarms.28.B': 'F16',
    'moe.expert_swarms.28.C': 'F16',
    'moe.expert_swarms.28.D': 'F16',
    'moe.expert_swarms.28.clone_coupling': 'F32',
    'moe.expert_swarms.28.clone_diversity': 'F32',
    'moe.expert_swarms.28.clone_quant_mag': 'F32',
    'moe.expert_swarms.28.clone_quant_noise': 'F32',
    'moe.expert_swarms.28.clone_quant_range': 'F32',
    'moe.expert_swarms.28.clone_quant_threshold': 'F32',
    'moe.expert_swarms.28.population_mean': 'F32',
    'moe.expert_swarms.28.population_std': 'F32',
    'moe.expert_swarms.29.A': 'F16',
    'moe.expert_swarms.29.B': 'F16',
    'moe.expert_swarms.29.C': 'F16',
    'moe.expert_swarms.29.D': 'F16',
    'moe.expert_swarms.29.clone_coupling': 'F32',
    'moe.expert_swarms.29.clone_diversity': 'F32',
    'moe.expert_swarms.29.clone_quant_mag': 'F32',
    'moe.expert_swarms.29.clone_quant_noise': 'F32',
    'moe.expert_swarms.29.clone_quant_range': 'F32',
    'moe.expert_swarms.29.clone_quant_threshold': 'F32',
    'moe.expert_swarms.29.population_mean': 'F32',
    'moe.expert_swarms.29.population_std': 'F32',
    'moe.expert_swarms.3.A': 'F16',
    'moe.expert_swarms.3.B': 'F16',
    'moe.expert_swarms.3.C': 'F16',
    'moe.expert_swarms.3.D': 'F16',
    'moe.expert_swarms.3.clone_coupling': 'F32',
    'moe.expert_swarms.3.clone_diversity': 'F32',
    'moe.expert_swarms.3.clone_quant_mag': 'F32',
    'moe.expert_swarms.3.clone_quant_noise': 'F32',
    'moe.expert_swarms.3.clone_quant_range': 'F32',
    'moe.expert_swarms.3.clone_quant_threshold': 'F32',
    'moe.expert_swarms.3.population_mean': 'F32',
    'moe.expert_swarms.3.population_std': 'F32',
    'moe.expert_swarms.30.A': 'F16',
    'moe.expert_swarms.30.B': 'F16',
    'moe.expert_swarms.30.C': 'F16',
    'moe.expert_swarms.30.D': 'F16',
    'moe.expert_swarms.30.clone_coupling': 'F32',
    'moe.expert_swarms.30.clone_diversity': 'F32',
    'moe.expert_swarms.30.clone_quant_mag': 'F32',
    'moe.expert_swarms.30.clone_quant_noise': 'F32',
    'moe.expert_swarms.30.clone_quant_range': 'F32',
    'moe.expert_swarms.30.clone_quant_threshold': 'F32',
    'moe.expert_swarms.30.population_mean': 'F32',
    'moe.expert_swarms.30.population_std': 'F32',
    'moe.expert_swarms.31.A': 'F16',
    'moe.expert_swarms.31.B': 'F16',
    'moe.expert_swarms.31.C': 'F16',
    'moe.expert_swarms.31.D': 'F16',
    'moe.expert_swarms.31.clone_coupling': 'F32',
    'moe.expert_swarms.31.clone_diversity': 'F32',
    'moe.expert_swarms.31.clone_quant_mag': 'F32',
    'moe.expert_swarms.31.clone_quant_noise': 'F32',
    'moe.expert_swarms.31.clone_quant_range': 'F32',
    'moe.expert_swarms.31.clone_quant_threshold': 'F32',
    'moe.expert_swarms.31.population_mean': 'F32',
    'moe.expert_swarms.31.population_std': 'F32',
    'moe.expert_swarms.32.A': 'F16',
    'moe.expert_swarms.32.B': 'F16',
    'moe.expert_swarms.32.C': 'F16',
    'moe.expert_swarms.32.D': 'F16',
    'moe.expert_swarms.32.clone_coupling': 'F32',
    'moe.expert_swarms.32.clone_diversity': 'F32',
    'moe.expert_swarms.32.clone_quant_mag': 'F32',
    'moe.expert_swarms.32.clone_quant_noise': 'F32',
    'moe.expert_swarms.32.clone_quant_range': 'F32',
    'moe.expert_swarms.32.clone_quant_threshold': 'F32',
    'moe.expert_swarms.32.population_mean': 'F32',
    'moe.expert_swarms.32.population_std': 'F32',
    'moe.expert_swarms.33.A': 'F16',
    'moe.expert_swarms.33.B': 'F16',
    'moe.expert_swarms.33.C': 'F16',
    'moe.expert_swarms.33.D': 'F16',
    'moe.expert_swarms.33.clone_coupling': 'F32',
    'moe.expert_swarms.33.clone_diversity': 'F32',
    'moe.expert_swarms.33.clone_quant_mag': 'F32',
    'moe.expert_swarms.33.clone_quant_noise': 'F32',
    'moe.expert_swarms.33.clone_quant_range': 'F32',
    'moe.expert_swarms.33.clone_quant_threshold': 'F32',
    'moe.expert_swarms.33.population_mean': 'F32',
    'moe.expert_swarms.33.population_std': 'F32',
    'moe.expert_swarms.4.A': 'F16',
    'moe.expert_swarms.4.B': 'F16',
    'moe.expert_swarms.4.C': 'F16',
    'moe.expert_swarms.4.D': 'F16',
    'moe.expert_swarms.4.clone_coupling': 'F32',
    'moe.expert_swarms.4.clone_diversity': 'F32',
    'moe.expert_swarms.4.clone_quant_mag': 'F32',
    'moe.expert_swarms.4.clone_quant_noise': 'F32',
    'moe.expert_swarms.4.clone_quant_range': 'F32',
    'moe.expert_swarms.4.clone_quant_threshold': 'F32',
    'moe.expert_swarms.4.population_mean': 'F32',
    'moe.expert_swarms.4.population_std': 'F32',
    'moe.expert_swarms.5.A': 'F16',
    'moe.expert_swarms.5.B': 'F16',
    'moe.expert_swarms.5.C': 'F16',
    'moe.expert_swarms.5.D': 'F16',
    'moe.expert_swarms.5.clone_coupling': 'F32',
    'moe.expert_swarms.5.clone_diversity': 'F32',
    'moe.expert_swarms.5.clone_quant_mag': 'F32',
    'moe.expert_swarms.5.clone_quant_noise': 'F32',
    'moe.expert_swarms.5.clone_quant_range': 'F32',
    'moe.expert_swarms.5.clone_quant_threshold': 'F32',
    'moe.expert_swarms.5.population_mean': 'F32',
    'moe.expert_swarms.5.population_std': 'F32',
    'moe.expert_swarms.6.A': 'F16',
    'moe.expert_swarms.6.B': 'F16',
    'moe.expert_swarms.6.C': 'F16',
    'moe.expert_swarms.6.D': 'F16',
    'moe.expert_swarms.6.clone_coupling': 'F32',
    'moe.expert_swarms.6.clone_diversity': 'F32',
    'moe.expert_swarms.6.clone_quant_mag': 'F32',
    'moe.expert_swarms.6.clone_quant_noise': 'F32',
    'moe.expert_swarms.6.clone_quant_range': 'F32',
    'moe.expert_swarms.6.clone_quant_threshold': 'F32',
    'moe.expert_swarms.6.population_mean': 'F32',
    'moe.expert_swarms.6.population_std': 'F32',
    'moe.expert_swarms.7.A': 'F16',
    'moe.expert_swarms.7.B': 'F16',
    'moe.expert_swarms.7.C': 'F16',
    'moe.expert_swarms.7.D': 'F16',
    'moe.expert_swarms.7.clone_coupling': 'F32',
    'moe.expert_swarms.7.clone_diversity': 'F32',
    'moe.expert_swarms.7.clone_quant_mag': 'F32',
    'moe.expert_swarms.7.clone_quant_noise': 'F32',
    'moe.expert_swarms.7.clone_quant_range': 'F32',
    'moe.expert_swarms.7.clone_quant_threshold': 'F32',
    'moe.expert_swarms.7.population_mean': 'F32',
    'moe.expert_swarms.7.population_std': 'F32',
    'moe.expert_swarms.8.A': 'F16',
    'moe.expert_swarms.8.B': 'F16',
    'moe.expert_swarms.8.C': 'F16',
    'moe.expert_swarms.8.D': 'F16',
    'moe.expert_swarms.8.clone_coupling': 'F32',
    'moe.expert_swarms.8.clone_diversity': 'F32',
    'moe.expert_swarms.8.clone_quant_mag': 'F32',
    'moe.expert_swarms.8.clone_quant_noise': 'F32',
    'moe.expert_swarms.8.clone_quant_range': 'F32',
    'moe.expert_swarms.8.clone_quant_threshold': 'F32',
    'moe.expert_swarms.8.population_mean': 'F32',
    'moe.expert_swarms.8.population_std': 'F32',
    'moe.expert_swarms.9.A': 'F16',
    'moe.expert_swarms.9.B': 'F16',
    'moe.expert_swarms.9.C': 'F16',
    'moe.expert_swarms.9.D': 'F16',
    'moe.expert_swarms.9.clone_coupling': 'F32',
    'moe.expert_swarms.9.clone_diversity': 'F32',
    'moe.expert_swarms.9.clone_quant_mag': 'F32',
    'moe.expert_swarms.9.clone_quant_noise': 'F32',
    'moe.expert_swarms.9.clone_quant_range': 'F32',
    'moe.expert_swarms.9.clone_quant_threshold': 'F32',
    'moe.expert_swarms.9.population_mean': 'F32',
    'moe.expert_swarms.9.population_std': 'F32',
    'moe.output_norm.bias': 'F32',
    'moe.output_norm.weight': 'F32',
    'moe.router.balanced_router.lora_A': 'F16',
    'moe.router.balanced_router.lora_B': 'F16',
    'moe.router.balanced_router.weight': 'F16',
    'moe.router.complexity_classifier.0.bias': 'F32',
    'moe.router.complexity_classifier.0.weight': 'F16',
    'moe.router.complexity_classifier.2.bias': 'F32',
    'moe.router.complexity_classifier.2.weight': 'F16',
    'moe.router.diffusion_router.lora_A': 'F16',
    'moe.router.diffusion_router.lora_B': 'F16',
    'moe.router.diffusion_router.weight': 'F16',
    'moe.router.fast_router.lora_A': 'F16',
    'moe.router.fast_router.lora_B': 'F16',
    'moe.router.fast_router.weight': 'F16',
    'moe.w1': 'F16',
    'moe.w1_lora_A': 'F16',
    'moe.w1_lora_B': 'F16',
    'moe.w2': 'F16',
    'moe.w2_lora_A': 'F16',
    'moe.w2_lora_B': 'F16',
    'moe.wgate': 'F16',
    'moe.wgate_lora_A': 'F16',
    'moe.wgate_lora_B': 'F16',
    'norm_decomp.bias': 'F32',
    'norm_decomp.weight': 'F32',
    'norm_diff.bias': 'F32',
    'norm_diff.weight': 'F32',
    'norm_moe.bias': 'F32',
    'norm_moe.weight': 'F32',
    'pre_final_norm.bias': 'F32',
    'pre_final_norm.weight': 'F32',
    'quantum_formulas.boltzmann': 'F32',
    'quantum_formulas.hbar': 'F32',
    'quantum_formulas.sqrt_33': 'F32',
    'quillan_finalizer.lora_A': 'F16',
    'quillan_finalizer.lora_B': 'F16',
    'quillan_finalizer.weight': 'I8',
    'quillan_finalizer2.lora_A': 'F16',
    'quillan_finalizer2.lora_B': 'F16',
    'quillan_finalizer2.weight': 'I8',
    'quillan_gate.bias': 'F32',
    'quillan_gate.weight': 'F16',
    'txt_dec.lora_A': 'F16',
    'txt_dec.lora_B': 'F16',
    'txt_dec.weight': 'F16',
}

# I8 weights carry a scalar .bitnet_scale sidecar in the safetensors file.
_I8_SCALE_KEYS = frozenset(
    k + ".bitnet_scale" for k in EXPECTED_BASE_SHAPES if k not in EXPECTED_DTYPE_CODES
) | frozenset(
    k + ".bitnet_scale" for k, code in EXPECTED_DTYPE_CODES.items() if code == "I8"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class FusedSocketConfig:
    """Configuration for the 34-block fused socket. All dims are socket-native."""

    enabled: bool = True  # metadata only; gate wiring is intentionally out of scope
    hidden_dim: int = 896
    ffn_dim: int = 4864
    num_experts: int = 34
    vocab_size: int = 151936
    max_positions: int = 1024
    heads: int = 14  # 14 query heads x 64 = 896 (Qwen2.5 width, v8:778-783)
    head_dim: int = 64
    kv_heads: int = 2  # GQA 2 -> 14 (v8:781)
    use_rope: bool = True
    use_sparse_mask: bool = True
    compute_dtype: str = "float32"
    # NOT registered as buffers/parameters; dequant-only side table.
    bitnet_scales: Dict[str, torch.Tensor] = field(default_factory=dict)


def _coerce_config(
    config: Union[FusedSocketConfig, Mapping[str, Any], None],
) -> FusedSocketConfig:
    if config is None:
        return FusedSocketConfig()
    if isinstance(config, FusedSocketConfig):
        return config
    if isinstance(config, Mapping):
        known = {f for f in FusedSocketConfig.__dataclass_fields__}  # type: ignore[attr-defined]
        return FusedSocketConfig(**{k: v for k, v in config.items() if k in known})
    raise TypeError(
        f"config must be FusedSocketConfig | Mapping | None, got {type(config)}"
    )


class _SocketNode(nn.Module):
    """Namespace node only. Leaves are registered buffers (frozen, no grad)."""

    pass


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class QuillanFusedSocket(nn.Module):
    """34 sequential transformer blocks over the untouched 530-key socket."""

    def __init__(
        self, config: Union[FusedSocketConfig, Mapping[str, Any], None] = None
    ) -> None:
        super().__init__()
        self.cfg = _coerce_config(config)
        self.hidden_dim = int(self.cfg.hidden_dim)
        self.ffn_dim = int(self.cfg.ffn_dim)
        self.num_experts = int(self.cfg.num_experts)
        self.vocab_size = int(self.cfg.vocab_size)
        self.max_positions = int(self.cfg.max_positions)
        self.heads = int(self.cfg.heads)
        self.head_dim = int(self.cfg.head_dim)
        self.kv_heads = int(self.cfg.kv_heads)
        self.attn_dim = self.heads * self.head_dim
        self.kv_dim = self.kv_heads * self.head_dim
        self._scale_side_table: Dict[str, torch.Tensor] = {}
        self._build_socket_tree()

    # -- construction --------------------------------------------------------
    def _build_socket_tree(self) -> None:
        """Create a nested nn.Module tree whose state_dict keys == the 530 keys.

        Every leaf is a persistent buffer, so (a) no parameter requires grad
        (no backprop per hard lock), and (b) integer I8 weights are representable
        without tripping Parameter's float-only requires_grad constraint.
        """
        for key, shape in EXPECTED_BASE_SHAPES.items():
            parts = key.split(".")
            node: nn.Module = self
            for part in parts[:-1]:
                child = node._modules.get(part)
                if child is None:
                    child = _SocketNode()
                    node.add_module(part, child)
                node = child
            leaf = parts[-1]
            if leaf in node._buffers or leaf in node._modules:
                raise ValueError(f"duplicate socket leaf while building {key!r}")
            dtype = _STORED_DTYPE_CODES[EXPECTED_DTYPE_CODES.get(key, "F32")]
            node.register_buffer(leaf, torch.zeros(tuple(shape), dtype=dtype))

    # -- tensor access --------------------------------------------------------
    def _t(self, key: str) -> torch.Tensor:
        parts = key.split(".")
        node: nn.Module = self
        for part in parts[:-1]:
            node = node._modules[part]
        return node._buffers[parts[-1]]

    def _wf(self, key: str, dtype: torch.dtype) -> torch.Tensor:
        """Return a weight as float compute dtype, applying scalar dequant scale."""
        raw = self._t(key)
        if raw.dtype == torch.int8:
            out = raw.to(dtype)
            scale = self._scale_side_table.get(key + ".bitnet_scale")
            if scale is not None:
                out = out * scale.to(dtype)
            return out
        return raw.to(dtype)

    def _layernorm(self, x: torch.Tensor, key: str, dtype: torch.dtype) -> torch.Tensor:
        w = self._wf(key + ".weight", dtype)
        b = self._wf(key + ".bias", dtype)
        return F.layer_norm(x.to(dtype), (self.hidden_dim,), weight=w, bias=b)

    # -- attention (weight-tied, applied at every one of the 34 steps) --------
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half : 2 * half]
        return torch.cat((-x2, x1), dim=-1)

    def _rope(
        self, q: torch.Tensor, k: torch.Tensor, position_start: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """NTK-aware RoPE on Q/K, (B, heads, T, head_dim) -- v8:840-926."""
        length, dim = q.size(2), q.size(-1)
        half = dim // 2
        pos = torch.arange(
            position_start,
            position_start + length,
            device=q.device,
            dtype=torch.float32,
        )
        base = 10000.0 * (
            max(1.0, (position_start + length) / 1024.0) ** (dim / max(1, dim - 2))
        )
        inv = 1.0 / (
            base ** (torch.arange(half, device=q.device, dtype=torch.float32) / half)
        )
        ang = pos[:, None] * inv[None, :]
        cos_h, sin_h = torch.cos(ang), torch.sin(ang)
        cos, sin = torch.cat([cos_h, cos_h], -1), torch.cat([sin_h, sin_h], -1)
        cos, sin = cos[None, None].to(q.dtype), sin[None, None].to(q.dtype)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(
            k
        ) * sin

    def _couil_attn(
        self,
        x: torch.Tensor,
        dtype: torch.dtype,
        position_start: int,
    ) -> torch.Tensor:
        """Shared donor attention: 14Q/2KV GQA, causal, optional RoPE + sparse gate."""
        B, T, _ = x.shape
        q = F.linear(x, self._wf("diffusion_core.couil_attn.q_proj.weight", dtype))
        k = F.linear(x, self._wf("diffusion_core.couil_attn.k_proj.weight", dtype))
        v = F.linear(x, self._wf("diffusion_core.couil_attn.v_proj.weight", dtype))
        q = q.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.kv_heads != self.heads:
            reps = self.heads // self.kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)
        if self.cfg.use_rope:
            q, k = self._rope(q, k, position_start)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1) @ v
        out = attn.transpose(1, 2).reshape(B, T, self.attn_dim)  # 896
        if self.cfg.use_sparse_mask:
            m = self._wf(
                "diffusion_core.couil_attn.sparse_mask_generator.0.weight", dtype
            )
            mb = self._wf(
                "diffusion_core.couil_attn.sparse_mask_generator.0.bias", dtype
            )
            m2 = self._wf(
                "diffusion_core.couil_attn.sparse_mask_generator.2.weight", dtype
            )
            mb2 = self._wf(
                "diffusion_core.couil_attn.sparse_mask_generator.2.bias", dtype
            )
            mask = F.linear(F.relu(F.linear(x, m, mb)), m2, mb2)
            out = out * torch.sigmoid(mask)
        return F.linear(out, self._wf("diffusion_core.couil_attn.o_proj.weight", dtype))

    # -- per-step block scale from frozen router ------------------------------
    def _block_gate(self, e: int, h: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Scalar residual gate g_e; routers are per-step scales, NOT selectors."""
        B = h.size(0)
        pooled = h.mean(dim=1).to(dtype)  # [B, 896]
        rk = "diffusion_router"
        logits = F.linear(pooled, self._wf(f"moe.router.{rk}.weight", dtype))  # [B,34]
        la = self._wf(f"moe.router.{rk}.lora_A", dtype)
        lb = self._wf(f"moe.router.{rk}.lora_B", dtype)
        logits = logits + ((pooled @ la) @ lb) * (16.0 / math.sqrt(la.size(1)))
        return torch.sigmoid(logits[:, e]).view(B, 1, 1)

    # -- low-rank expert swarm refinement ------------------------------------
    def _swarm_corr(
        self, e: int, ff: torch.Tensor, w2e: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Faithful realization of spec `swarm_corr_e(h)`.

        The four swarm factors A/B/C/D live in the 4864-d expert FF basis
        (header: [4864,16]/[16,4864]), NOT in the 896-d h basis. Therefore the
        rank-16+16 refinement is applied to the SwiGLU activation and folded
        through the same w2[e] (4864->896) so it is additively comparable with
        f in `f = f + swarm_corr_e(h)`.
        """
        p = f"moe.expert_swarms.{e}"
        A = self._wf(f"{p}.A", dtype)
        Bm = self._wf(f"{p}.B", dtype)
        C = self._wf(f"{p}.C", dtype)
        Dm = self._wf(f"{p}.D", dtype)
        corr = (ff @ A) @ Bm + (ff @ C) @ Dm  # [B,T,4864]
        coupling = self._t(f"{p}.clone_coupling").to(dtype)
        return (corr * torch.tanh(coupling)) @ w2e  # [B,T,896]

    # -- forward --------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, position_start: int = 0) -> torch.Tensor:
        """token ids [B,T] -> logits [B,T,151936]."""
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B,T], got {tuple(input_ids.shape)}")
        B, T = input_ids.shape
        dtype = getattr(torch, self.cfg.compute_dtype)

        # h = LayerNorm(txt_emb[tok] + pos_embed)
        emb = self._t("ingestion.txt_emb.weight")
        h = F.embedding(input_ids, emb).to(dtype)  # [B,T,896]
        pos = self._wf("ingestion.pos_embed", dtype)  # [1,1024,896]
        if T > pos.size(1):
            raise ValueError(f"sequence {T} exceeds pos_embed capacity {pos.size(1)}")
        h = h + pos[:, :T, :]
        h = self._layernorm(h, "ingestion.norm", dtype)

        # pre-loaded frozen SwiGLU weights [E,...] (F16 in socket-v2)
        w1_all = self._wf("moe.w1", dtype)
        wgate_all = self._wf("moe.wgate", dtype)
        w2_all = self._wf("moe.w2", dtype)
        w1la = self._wf("moe.w1_lora_A", dtype)
        w1lb = self._wf("moe.w1_lora_B", dtype)
        wgla = self._wf("moe.wgate_lora_A", dtype)
        wglb = self._wf("moe.wgate_lora_B", dtype)
        rs = 16.0 / math.sqrt(w1lb.size(1))

        for e in range(self.num_experts):
            # a = CouilAttn(LN(h))
            a = self._couil_attn(
                self._layernorm(h, "diffusion_core.norm2", dtype), dtype, position_start
            )
            # h = h + g_e * a
            h = h + self._block_gate(e, h, dtype) * a
            # u = h @ w1[e]; v = SiLU(h @ wgate[e]); f = (u*v) @ w2[e]
            w1e, wge, w2e = w1_all[e], wgate_all[e], w2_all[e]
            u = h @ w1e + ((h @ w1la[e]) @ w1lb[e]) * rs
            gate = h @ wge + ((h @ wgla[e]) @ wglb[e]) * rs
            v = F.silu(gate)  # spec convention: u * SiLU(wgate)
            ff = u * v
            f = ff @ w2e
            # f = f + swarm_corr_e(h)
            f = f + self._swarm_corr(e, ff, w2e, dtype)
            # h = LayerNorm_moe(h + f)
            h = self._layernorm(h + f, "moe.output_norm", dtype)

        # z = final_norm(h)
        z = self._layernorm(h, "final_norm", dtype)

        # logits = txt_dec(z) + txt_dec.lora_B @ txt_dec.lora_A(z)
        dec = self._t("txt_dec.weight")  # [151936,896] F16 - keep native dtype
        dec_la = self._t("txt_dec.lora_A")
        dec_lb = self._t("txt_dec.lora_B")
        z_dec = z.to(dec.dtype)
        logits = F.linear(z_dec, dec).to(dtype)
        logits = logits + ((z_dec @ dec_la) @ dec_lb).to(dtype)
        return logits


# ---------------------------------------------------------------------------
# Build / load helpers
# ---------------------------------------------------------------------------
def build_fused_model(
    config: Union[FusedSocketConfig, Mapping[str, Any], None],
    state_dict: Mapping[str, torch.Tensor],
) -> QuillanFusedSocket:
    """Build the fused socket from a socket-v2 base state_dict (530 keys).

    Validates exact key set and every shape against the safe_open-derived
    inventory. `.bitnet_scale` sidecars, if supplied via
    `config["bitnet_scales"]`, are kept off the state_dict.
    """
    cfg = _coerce_config(config)
    supplied = set(state_dict)
    expected = set(EXPECTED_BASE_SHAPES)
    missing, extra = sorted(expected - supplied), sorted(supplied - expected)
    if missing or extra:
        raise ValueError(
            f"socket key mismatch: missing={len(missing)} {missing[:5]} "
            f"extra={len(extra)} {extra[:5]}"
        )
    for key, shape in EXPECTED_BASE_SHAPES.items():
        actual = tuple(state_dict[key].shape)
        if actual != tuple(shape):
            raise ValueError(
                f"shape mismatch {key!r}: expected {tuple(shape)}, got {actual}"
            )

    model = QuillanFusedSocket(cfg)
    model.load_state_dict(state_dict, strict=True)
    model._scale_side_table = {
        k: v
        for k, v in (cfg.bitnet_scales or {}).items()
        if isinstance(v, torch.Tensor)
    }
    return model


def load_socket_v2(
    path: str,
    config: Union[FusedSocketConfig, Mapping[str, Any], None] = None,
) -> QuillanFusedSocket:
    """Load real socket-v2 safetensors: 530 base keys + 24 side-table scales."""
    from safetensors import safe_open  # local import keeps module torch-only

    cfg = _coerce_config(config)
    base: Dict[str, torch.Tensor] = {}
    scales: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key.endswith(".bitnet_scale"):
                scales[key] = tensor
            else:
                base[key] = tensor
    cfg.bitnet_scales = scales
    return build_fused_model(cfg, base)


__all__ = [
    "FusedSocketConfig",
    "QuillanFusedSocket",
    "build_fused_model",
    "load_socket_v2",
    "EXPECTED_BASE_SHAPES",
]
