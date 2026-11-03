#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Quillan-Ronin: WSP Quad-Donor Sovereign Compiler (Qwen 3.5 2B + 128k Parity)
=============================================================================
Synthesizes:
  1. Llama 3.2 1B: Pure causal foundation & 128k BPE vocabulary anchor (D=2048)
  2. Falcon 3 1B: High-frequency causal attention & normalization (D=2048)
  3. BitNet 1.58b: Low-precision {-1, 0, 1} synaptic foundations
  4. Qwen 3.5 2B: Gated Delta recurrence & frontier reasoning circuits (D=2048)

Core WSP Techniques (Takehi, Teoh, Zhang, Khattab; MIT CSAIL, Nov 2, 2026):
  - 128,256 Pure Tokenizer Alignment (zero index-collision, 100% token parity)
  - Gain-Matched LayerNorm Normalization (unit mean gamma = 1.0)
  - Takehi SVD Null-Space Deflation on MLP down-projections (I - v v^T)
  - Uniform-Rank (r=16) 34-Expert SVD Extraction enabling Kaido Batched GEMM
  - Complete Structural Key Invariance (0 missing keys, 0 unexpected keys)
=============================================================================
"""

import os
import sys

# Limit OpenBLAS / OpenMP threading to avoid Windows heap crashes
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import math
import time
import logging
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import gguf
import gguf.quants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WSP-Qwen35] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("WSP-Qwen35")

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni


def to_bitnet_ternary_wsp(W: torch.Tensor, target_gain: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Weight-Space Programming BitNet 1.58b Quantization."""
    W_f = W.float().cpu()
    scale = 1.0 / W_f.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    W_scaled = W_f * scale
    W_q = torch.round(torch.clamp(W_scaled, -1.0, 1.0))
    W_dequant = W_q / scale
    
    fan_in = W_f.size(1)
    target_std = target_gain / math.sqrt(fan_in)
    curr_std = W_dequant.std().clamp(min=1e-6)
    W_dequant = W_dequant * (target_std / curr_std)
    return W_dequant.cpu()


def remove_attractor_direction(W: torch.Tensor, num_directions: int = 1) -> torch.Tensor:
    """Takehi et al. Experiment 8: Projects out self-reinforcing cyclic attractor directions."""
    W_f = W.float().cpu()
    U, S, V = torch.svd_lowrank(W_f, q=num_directions + 2, niter=2)
    v_top = V[:, :num_directions]
    P_null = torch.eye(W_f.size(1)) - v_top @ v_top.t()
    W_clean = W_f @ P_null
    return W_clean


def normalize_layernorm_vector(v: torch.Tensor, target_dim: int, target_mean: float = 1.0) -> torch.Tensor:
    """Normalizes LayerNorm gamma vector to target_mean across donors."""
    v_f = v.float().cpu()
    d = v_f.size(0)
    if d == target_dim:
        v_proj = v_f
    elif d > target_dim:
        v_proj = v_f[:target_dim]
    else:
        pad = target_dim - d
        repeats = (pad // d) + 1
        v_proj = torch.cat([v_f, v_f.repeat(repeats)[:pad]], dim=0)
    mean_val = v_proj.mean().clamp(min=1e-4)
    return v_proj * (target_mean / mean_val)


def project_dim(W: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
    """Energy-preserving projection."""
    W_f = W.float().cpu()
    r, c = W_f.shape
    if r == out_dim:
        W_r = W_f
    elif r > out_dim:
        W_r = W_f[:out_dim, :]
    else:
        pad = out_dim - r
        repeats = (pad // r) + 1
        W_r = torch.cat([W_f, W_f.repeat(repeats, 1)[:pad, :]], dim=0)
        
    if c == in_dim:
        W_c = W_r
    elif c > in_dim:
        W_c = W_r[:, :in_dim]
    else:
        pad = in_dim - c
        repeats = (pad // c) + 1
        W_c = torch.cat([W_r, W_r.repeat(1, repeats)[:, :pad]], dim=1)
        
    scale = math.sqrt((r * c) / (out_dim * in_dim))
    return W_c * scale


def extract_34_uniform_experts(W: torch.Tensor, target_dim: int, rank: int = 16) -> list:
    """Extracts 34 MoE experts with uniform rank r=16 for Kaido Batched GEMM."""
    W_f = W.float().cpu()
    U, S, V = torch.svd_lowrank(W_f, q=32, niter=2)
    S_sqrt = torch.diag(torch.sqrt(S))
    w0_full = S_sqrt @ V.t()
    w1_full = U @ S_sqrt
    
    experts = []
    for e_idx in range(34):
        offset = (e_idx * 2) % max(1, (w0_full.size(0) - rank))
        w0_sub = project_dim(w0_full[offset:offset+rank, :], rank, target_dim)
        w1_sub = project_dim(w1_full[:, offset:offset+rank], target_dim, rank)
        
        w0_tern = to_bitnet_ternary_wsp(w0_sub, target_gain=0.50)
        w1_tern = to_bitnet_ternary_wsp(w1_sub, target_gain=0.50)
        experts.append((w0_tern.to(torch.bfloat16), w1_tern.to(torch.bfloat16)))
    return experts


def compile_quad_wsp_qwen35():
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("COMPILING QUILLAN-RONIN SOVEREIGN 1B (QUAD-DONOR WSP 128K WITH QWEN 3.5 2B)")
    logger.info("=" * 80)
    
    hf_cache = r"C:\Users\Admin\.cache\huggingface\hub"
    ollama_blobs = r"C:\Users\Admin\.ollama\models\blobs"
    
    # 1. Llama 3.2 1B
    llama_dir = os.path.join(hf_cache, "models--unsloth--Llama-3.2-1B-Instruct", "snapshots")
    llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
    llama_st_path = os.path.join(llama_snap, "model.safetensors")
    
    # 2. Falcon 3 1B
    falcon_blob = os.path.join(ollama_blobs, "sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01")
    
    # 3. BitNet 1.58b Large
    bitnet_dir = os.path.join(hf_cache, "models--1bitLLM--bitnet_b1_58-large", "snapshots")
    bitnet_snap = os.path.join(bitnet_dir, os.listdir(bitnet_dir)[0])
    bitnet_st_path = os.path.join(bitnet_snap, "model.safetensors")
    
    # 4. Qwen 3.5 2B GGUF
    qwen35_blob = os.path.join(ollama_blobs, "sha256-e8bf47d40c5ba68baf8f3e503ee36710da6e8641cb0a670a508cff5b3f0cfbf7")
    
    out_dir = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "quillan_1b_sovereign_wsp128k_qwen35.pt")
    
    # Architecture Dimensions
    hidden_dim = 2048
    n_layer = 16
    ffn_dim = 8192
    num_experts = 34
    vocab_size = 128256
    
    logger.info(f"Target: D={hidden_dim}, Layers={n_layer}, Vocab={vocab_size}, Experts={num_experts} (Uniform r=16)")
    
    # Step 1: Instantiate Canonical QuillanRoninOni Model (guarantees 0 missing / 0 unexpected keys)
    logger.info(">>> Instantiating Canonical QuillanRoninOni model architecture...")
    cfg = QuillanOniConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        n_layer=n_layer,
        n_head=32,
        head_dim=64,
        ffn_dim=ffn_dim,
        num_experts=num_experts,
        expert_rank=16,
        use_evo_moe=False
    )
    cfg.vectorized_experts = True
    model = QuillanRoninOni(cfg)
    
    # Step 2: Ingest Donors
    logger.info(">>> Ingesting Donor 1: Llama 3.2 1B (Safetensors)...")
    st_llama = load_file(llama_st_path)
    
    logger.info(">>> Ingesting Donor 2: Falcon 3 1B (GGUF selective)...")
    r_falcon = gguf.GGUFReader(falcon_blob)
    def dequant_gguf(t) -> torch.Tensor:
        if t.tensor_type != 0:
            arr = gguf.quants.dequantize(t.data, t.tensor_type)
        else:
            arr = t.data
        return torch.from_numpy(arr.copy()).float()
        
    falcon_needed = {"token_embd.weight", "blk.0.attn_norm.weight"}
    for f_idx in range(4):
        for k in ["attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]:
            falcon_needed.add(f"blk.{f_idx}.{k}")
            
    falcon_tensors = {t.name: dequant_gguf(t) for t in r_falcon.tensors if t.name in falcon_needed}
    del r_falcon
    
    logger.info(">>> Ingesting Donor 3: BitNet 1.58b Large (Safetensors)...")
    st_bitnet = load_file(bitnet_st_path)
    
    logger.info(">>> Ingesting Donor 4: Qwen 3.5 2B (GGUF selective)...")
    r_qwen35 = gguf.GGUFReader(qwen35_blob)
    qwen35_tensors = {
        t.name: dequant_gguf(t) 
        for t in r_qwen35.tensors 
        if any(t.name.startswith(f"blk.{q_idx}.") for q_idx in range(4))
    }
    del r_qwen35
    import gc
    gc.collect()
    
    # 1. Pure 128k Vocab Embeddings (100% Llama 3.2 Anchor - Zero index contamination)
    logger.info(">>> [1/5] Splicing 128k Pure Anchor Vocabulary Embeddings...")
    emb_llama = st_llama["model.embed_tokens.weight"].float() # [128256, 2048]
    model.wte.weight.data.copy_(emb_llama.to(torch.bfloat16))
    model.lm_head.weight.data.copy_(emb_llama.to(torch.bfloat16))
    del emb_llama
    gc.collect()
    
    # Final LayerNorm
    norm_l = st_llama["model.norm.weight"].float()
    norm_f = falcon_tensors["blk.0.attn_norm.weight"].float()
    norm_unified = 0.70 * norm_l + 0.30 * norm_f
    model.ln_f.weight.data.copy_(normalize_layernorm_vector(norm_unified, hidden_dim, target_mean=1.0).to(torch.bfloat16))
    model.ln_f.bias.data.zero_()
    
    # 2. Dual-Brain Ingestion & Finalizer Bridges
    logger.info(">>> [2/5] Calibrating Dual-Brain Bridges and Finalizers...")
    model.q1_bridge.weight.data.copy_(torch.eye(hidden_dim, dtype=torch.bfloat16))
    model.q2_bridge.weight.data.copy_(torch.eye(hidden_dim, dtype=torch.bfloat16))
    model.ingest_gate.weight.data.zero_()
    model.ingest_gate.bias.data.zero_()
    
    model.quillan_finalizer_q1.weight.data.copy_(torch.eye(hidden_dim, dtype=torch.bfloat16))
    model.quillan_finalizer_q2.weight.data.copy_(torch.eye(hidden_dim, dtype=torch.bfloat16))
    model.quillan_comm_gate.weight.data.zero_()
    model.quillan_comm_gate.bias.data.zero_()
    
    # 3. 16 Layers Splicing across Quad Donors
    logger.info(">>> [3/5] Splicing 16 Sovereign Layers across Quad Donors...")
    # Layer Allocation:
    # Layers 0..5 (6 layers): Llama 3.2 1B (Dense causal foundation)
    # Layers 6..9 (4 layers): Falcon 3 1B (High-bandwidth attention)
    # Layers 10..11 (2 layers): BitNet 1.58b (Ternary compute core)
    # Layers 12..15 (4 layers): Qwen 3.5 2B (Native D=2048 Gated Delta + Frontier Reasoning)
    for l in range(n_layer):
        blk = model.h[l]
        
        if l < 6:
            donor = "Llama-3.2-1B"
            p = f"model.layers.{l}"
            ln1 = st_llama[f"{p}.input_layernorm.weight"]
            ln2 = st_llama[f"{p}.post_attention_layernorm.weight"]
            
            q = st_llama[f"{p}.self_attn.q_proj.weight"].float()
            k = st_llama[f"{p}.self_attn.k_proj.weight"].float()
            v = st_llama[f"{p}.self_attn.v_proj.weight"].float()
            o = st_llama[f"{p}.self_attn.o_proj.weight"].float()
            
            k_exp = k.view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048)
            v_exp = v.view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048)
            
            ffn_gate = st_llama[f"{p}.mlp.gate_proj.weight"].float() # [8192, 2048]
            ffn_up = st_llama[f"{p}.mlp.up_proj.weight"].float()     # [8192, 2048]
            ffn_down = st_llama[f"{p}.mlp.down_proj.weight"].float() # [2048, 8192]
            
        elif l < 10:
            donor = "Falcon-3-1B"
            f_idx = l - 6
            ln1 = falcon_tensors[f"blk.{f_idx}.attn_norm.weight"].float()
            ln2 = falcon_tensors[f"blk.{f_idx}.ffn_norm.weight"].float()
            
            q = falcon_tensors[f"blk.{f_idx}.attn_q.weight"].float()
            k = falcon_tensors[f"blk.{f_idx}.attn_k.weight"].float()
            v = falcon_tensors[f"blk.{f_idx}.attn_v.weight"].float()
            o = falcon_tensors[f"blk.{f_idx}.attn_output.weight"].float()
            
            k_exp = k.view(16, 64, 2048).repeat_interleave(2, dim=0).view(2048, 2048)
            v_exp = v.view(16, 64, 2048).repeat_interleave(2, dim=0).view(2048, 2048)
            
            ffn_gate = falcon_tensors[f"blk.{f_idx}.ffn_gate.weight"].float() # [8192, 2048]
            ffn_up = falcon_tensors[f"blk.{f_idx}.ffn_up.weight"].float()     # [8192, 2048]
            ffn_down = falcon_tensors[f"blk.{f_idx}.ffn_down.weight"].float() # [2048, 8192]
            
        elif l < 12:
            donor = "BitNet-1.58b"
            b_idx = l - 10
            p = f"model.layers.{b_idx}"
            ln1 = st_bitnet[f"{p}.input_layernorm.weight"]
            ln2 = st_bitnet[f"{p}.post_attention_norm.weight"] if f"{p}.post_attention_norm.weight" in st_bitnet else st_bitnet[f"{p}.post_attention_layernorm.weight"]
            
            q = project_dim(st_bitnet[f"{p}.self_attn.q_proj.weight"], hidden_dim, hidden_dim)
            k = project_dim(st_bitnet[f"{p}.self_attn.k_proj.weight"], hidden_dim, hidden_dim)
            v = project_dim(st_bitnet[f"{p}.self_attn.v_proj.weight"], hidden_dim, hidden_dim)
            o = project_dim(st_bitnet[f"{p}.self_attn.o_proj.weight"], hidden_dim, hidden_dim)
            k_exp = k
            v_exp = v
            
            ffn_gate = project_dim(st_bitnet[f"{p}.mlp.gate_proj.weight"], ffn_dim, hidden_dim)
            ffn_up = project_dim(st_bitnet[f"{p}.mlp.up_proj.weight"], ffn_dim, hidden_dim)
            ffn_down = project_dim(st_bitnet[f"{p}.mlp.down_proj.weight"], hidden_dim, ffn_dim)
            
        else:
            donor = "Qwen-3.5-2B"
            q_idx = l - 12
            ln1 = qwen35_tensors[f"blk.{q_idx}.attn_norm.weight"].float()
            ln2 = qwen35_tensors[f"blk.{q_idx}.post_attention_norm.weight"].float()
            
            if f"blk.{q_idx}.attn_qkv.weight" in qwen35_tensors:
                qkv = qwen35_tensors[f"blk.{q_idx}.attn_qkv.weight"].float()
                q = qkv[:2048, :]
                k_exp = qkv[2048:4096, :]
                v_exp = qkv[4096:6144, :]
                o = qwen35_tensors.get(f"blk.{q_idx}.ssm_out.weight", qwen35_tensors.get(f"blk.{q_idx}.attn_gate.weight")).float()
            else:
                q_raw = qwen35_tensors[f"blk.{q_idx}.attn_q.weight"].float()
                k_raw = qwen35_tensors[f"blk.{q_idx}.attn_k.weight"].float()
                v_raw = qwen35_tensors[f"blk.{q_idx}.attn_v.weight"].float()
                o = qwen35_tensors[f"blk.{q_idx}.attn_output.weight"].float()
                q = project_dim(q_raw, hidden_dim, hidden_dim)
                k_exp = project_dim(k_raw, hidden_dim, hidden_dim)
                v_exp = project_dim(v_raw, hidden_dim, hidden_dim)
            
            ffn_gate_raw = qwen35_tensors[f"blk.{q_idx}.ffn_gate.weight"].float()
            ffn_up_raw = qwen35_tensors[f"blk.{q_idx}.ffn_up.weight"].float()
            ffn_down_raw = qwen35_tensors[f"blk.{q_idx}.ffn_down.weight"].float()
            ffn_gate = project_dim(ffn_gate_raw, ffn_dim, hidden_dim)
            ffn_up = project_dim(ffn_up_raw, ffn_dim, hidden_dim)
            ffn_down = project_dim(ffn_down_raw, hidden_dim, ffn_dim)
            
        # 1. Gain-Matched LayerNorms
        blk.ln_1.weight.data.copy_(normalize_layernorm_vector(ln1, hidden_dim, target_mean=1.0).to(torch.bfloat16))
        blk.ln_1.bias.data.zero_()
        blk.ln_2.weight.data.copy_(normalize_layernorm_vector(ln2, hidden_dim, target_mean=1.0).to(torch.bfloat16))
        blk.ln_2.bias.data.zero_()
        
        # 2. Attention
        c_attn = torch.cat([q, k_exp, v_exp], dim=0) # [6144, 2048]
        blk.attn.c_attn.weight.data.copy_(c_attn.to(torch.bfloat16))
        blk.attn.c_attn.bias.data.zero_()
        blk.attn.c_proj.weight.data.copy_(o.to(torch.bfloat16))
        blk.attn.c_proj.bias.data.zero_()
        
        # Zero out prism gate so uncalibrated semantic vectors don't inject noise
        blk.attn.prism.w_gate.weight.data.zero_()
        
        # 3. Dense SwiGLU MLP
        c_fc = torch.cat([ffn_gate, ffn_up], dim=0) # [2 * ffn_dim, hidden_dim]
        blk.moe.c_fc.weight.data.copy_(c_fc.to(torch.bfloat16))
        blk.moe.c_fc.bias.data.zero_()
        
        # 4. Takehi Attractor SVD Deflation on MLP down-projection
        ffn_down_clean = remove_attractor_direction(ffn_down, num_directions=1)
        blk.moe.c_proj.weight.data.copy_(ffn_down_clean.to(torch.bfloat16))
        blk.moe.c_proj.bias.data.zero_()
        
        # 5. Uniform r=16 34-Expert SVD MoE Extraction (Mapped to lora_A and lora_B)
        experts = extract_34_uniform_experts(ffn_down_clean, hidden_dim, rank=16)
        for e_idx, (w0, w1) in enumerate(experts):
            exp = blk.moe.experts[e_idx]
            exp.lora_A.data.copy_(w1.to(torch.bfloat16))  # [2048, 16]
            exp.lora_B.data.copy_(w0.to(torch.bfloat16))  # [16, 2048]
            exp.swarm.A.data.zero_()
            exp.swarm.B.data.zero_()
            exp.swarm.C.data.zero_()
            exp.swarm.D.data.zero_()
            exp.swarm.clone_diversity.data.zero_()
            exp.swarm.clone_coupling.data.zero_()
            
        # Calibrate MoE gating: router initialized to zero, mild gating bias (0.05)
        blk.moe.router.weight.data.zero_()
        blk.moe.pull_gate.gate.weight.data.zero_()
        blk.moe.moe_gate.weight.data.zero_()
        blk.moe.moe_gate.bias.data.fill_(0.05)
        
        logger.info(f"  Layer {l:02d}/16 spliced from [{donor:12s}] (LN=1.0, Attn=6144x2048, Takehi Deflated, 34 Uniform r=16 Experts)")
        
    # Step 4: Packaging Checkpoint
    logger.info(">>> [4/5] Packaging Sovereign Model with 100% Invariant State Dict (bfloat16)...")
    model = model.to(torch.bfloat16)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "vocab_size": vocab_size,
        "tokenizer_type": "llama3.2_128k",
        "donors": ["llama-3.2-1b", "falcon-3-1b", "bitnet-1.58b", "qwen-3.5-2b"],
        "wsp_techniques": [
            "128k_pure_anchor_parity",
            "gqa_head_broadcast",
            "unit_mean_layernorm",
            "takehi_attractor_deflation",
            "uniform_r16_34expert_svd",
            "zero_missing_key_invariance"
        ]
    }
    
    torch.save(ckpt, out_path)
    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f">>> [5/5] Sovereign 1B (Qwen 3.5 2B Spliced) -> {out_path} ({sz_mb:.1f} MB, {time.time()-t0:.2f}s)")
    return out_path


if __name__ == "__main__":
    compile_quad_wsp_qwen35()
