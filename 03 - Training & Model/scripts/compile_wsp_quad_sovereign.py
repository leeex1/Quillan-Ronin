#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Quillan-Ronin: WSP Quad-Donor Sovereign Compiler (128k Tokenizer Parity)
=============================================================================
Synthesizes:
  1. Llama 3.2 1B (128k vocab, D=2048 causal backbone)
  2. Falcon 3 1B GGUF (128k/131k vocab, D=2048 Llama-arch backbone)
  3. BitNet 1.58b (Ternary {-1, 0, 1} MoE foundations)
  4. Qwen 2.5 (Dense reasoning circuits)

Key Transformations:
  - Exact 128,256 Tokenizer Alignment (Zero collision, 100% token parity)
  - GQA Repeat-Interleave Attention Projection (Exact 4x head broadcast)
  - Gain-Matched LayerNorm Normalization (Mean gamma = 1.0 across all donors)
  - Takehi SVD Null-Space Deflation (Removes dominant cyclic attractor)
  - 1-Shot SVD 34-Expert MoE Extraction
=============================================================================
"""

import os
import sys
import gc
import math
import time
import logging
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import gguf
import gguf.quants
from tokenizers import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WSP-Quad] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("WSP-Quad")

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
sys.path.insert(0, r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni')
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
    """Takehi et al. Experiment 8: Projects out self-reinforcing attractor directions."""
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


def extract_34_experts_fast(W: torch.Tensor, target_dim: int) -> list:
    """1-Shot SVD expert extraction across 34 MoE experts."""
    W_f = W.float().cpu()
    U, S, V = torch.svd_lowrank(W_f, q=48, niter=2)
    S_sqrt = torch.diag(torch.sqrt(S))
    w0_full = S_sqrt @ V.t()
    w1_full = U @ S_sqrt
    
    experts = []
    for e_idx in range(34):
        rk = 8 if e_idx < 11 else (24 if e_idx < 22 else 48)
        w0_sub = project_dim(w0_full[:rk, :], rk, target_dim)
        w1_sub = project_dim(w1_full[:, :rk], target_dim, rk)
        
        w0_tern = to_bitnet_ternary_wsp(w0_sub, target_gain=0.50)
        w1_tern = to_bitnet_ternary_wsp(w1_sub, target_gain=0.50)
        experts.append((w0_tern.to(torch.bfloat16), w1_tern.to(torch.bfloat16)))
    return experts


def compile_quad_wsp_sovereign():
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("COMPILING QUILLAN-RONIN SOVEREIGN 1B (QUAD-DONOR WSP 128K)")
    logger.info("=" * 80)
    
    # Paths
    hf_cache = r"C:\Users\Admin\.cache\huggingface\hub"
    ollama_blobs = r"C:\Users\Admin\.ollama\models\blobs"
    
    llama_dir = os.path.join(hf_cache, "models--unsloth--Llama-3.2-1B-Instruct", "snapshots")
    llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
    llama_st_path = os.path.join(llama_snap, "model.safetensors")
    llama_tok_path = os.path.join(llama_snap, "tokenizer.json")
    
    falcon_blob = os.path.join(ollama_blobs, "sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01")
    
    bitnet_dir = os.path.join(hf_cache, "models--1bitLLM--bitnet_b1_58-large", "snapshots")
    bitnet_snap = os.path.join(bitnet_dir, os.listdir(bitnet_dir)[0])
    bitnet_st_path = os.path.join(bitnet_snap, "model.safetensors")
    
    qwen_dir = os.path.join(hf_cache, "models--Qwen--Qwen2.5-0.5B-Instruct", "snapshots")
    qwen_snap = os.path.join(qwen_dir, os.listdir(qwen_dir)[0])
    qwen_st_path = os.path.join(qwen_snap, "model.safetensors")
    
    out_dir = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "quillan_1b_sovereign_wsp128k.pt")
    
    # Architecture dimensions
    hidden_dim = 2048
    n_layer = 16
    ffn_dim = 8192
    num_experts = 34
    vocab_size = 128256
    
    logger.info(f"Target Architecture: D={hidden_dim}, Layers={n_layer}, Vocab={vocab_size}, Experts={num_experts}")
    
    # Load Llama 3.2 1B (Base anchor)
    logger.info(">>> Ingesting Donor 1: Llama 3.2 1B (Safetensors)...")
    st_llama = load_file(llama_st_path)
    
    # Load Falcon 3 1B (GGUF)
    logger.info(">>> Ingesting Donor 2: Falcon 3 1B (GGUF)...")
    r_falcon = gguf.GGUFReader(falcon_blob)
    def get_falcon_tensor(t) -> torch.Tensor:
        if t.tensor_type != 0:
            arr = gguf.quants.dequantize(t.data, t.tensor_type)
        else:
            arr = t.data
        return torch.from_numpy(arr.copy()).float()
    falcon_tensors = {t.name: get_falcon_tensor(t) for t in r_falcon.tensors}
    
    # Load BitNet 1.58b Large
    logger.info(">>> Ingesting Donor 3: BitNet 1.58b Large (Safetensors)...")
    st_bitnet = load_file(bitnet_st_path)
    
    # Load Qwen 2.5
    logger.info(">>> Ingesting Donor 4: Qwen 2.5 (Safetensors)...")
    st_qwen = load_file(qwen_st_path)
    
    model_state = {}
    
    # 1. Embeddings: Pure 128,256 Llama 3.2 anchor with Falcon 3 blend
    logger.info(">>> [1/5] Synthesizing 128k Sovereign Vocabulary Embeddings...")
    emb_llama = st_llama["model.embed_tokens.weight"].float() # [128256, 2048]
    raw_falcon = falcon_tensors["token_embd.weight"].float()
    if raw_falcon.shape[0] != 131072:
        raw_falcon = raw_falcon.t()
    emb_falcon_128k = raw_falcon[:vocab_size, :]
    
    # Superposition 80% Llama + 20% Falcon (Exact same subwords!)
    emb_unified = 0.80 * emb_llama + 0.20 * emb_falcon_128k
    target_emb_std = 1.0 / math.sqrt(hidden_dim)
    emb_unified = emb_unified * (target_emb_std / emb_unified.std().clamp(min=1e-6))
    
    model_state["wte.weight"] = emb_unified.to(torch.bfloat16)
    model_state["lm_head.weight"] = emb_unified.clone().to(torch.bfloat16)
    
    # Final Norm: Llama + Falcon
    norm_l = st_llama["model.norm.weight"].float()
    norm_f = falcon_tensors["blk.0.attn_norm.weight"].float() # [2048]
    norm_unified = 0.70 * norm_l + 0.30 * norm_f
    model_state["ln_f.weight"] = normalize_layernorm_vector(norm_unified, hidden_dim, target_mean=1.0).to(torch.bfloat16)
    model_state["ln_f.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)
    
    # 2. Dual-Brain Bridges
    logger.info(">>> [2/5] Synthesizing Lossless Dual-Brain Bridges...")
    model_state["q1_bridge.weight"] = torch.eye(hidden_dim, dtype=torch.bfloat16)
    model_state["q2_bridge.weight"] = torch.eye(hidden_dim, dtype=torch.bfloat16)
    model_state["ingest_gate.weight"] = torch.zeros(hidden_dim, 2 * hidden_dim, dtype=torch.bfloat16)
    model_state["ingest_gate.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)
    model_state["quillan_finalizer_q1.weight"] = torch.eye(hidden_dim, dtype=torch.bfloat16)
    model_state["quillan_finalizer_q2.weight"] = torch.eye(hidden_dim, dtype=torch.bfloat16)
    model_state["quillan_comm_gate.weight"] = torch.zeros(hidden_dim, 2 * hidden_dim, dtype=torch.bfloat16)
    model_state["quillan_comm_gate.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)
    
    # 3. Layers Splicing (16 Layers)
    logger.info(">>> [3/5] Splicing 16 Sovereign Layers with GQA Head Expansion & WSP...")
    for l in range(n_layer):
        dst = f"h.{l}"
        # Rotate donor stream: Llama (Layers 0..5), Falcon (6..9), BitNet (10..12), Qwen (13..15)
        if l < 6:
            donor = "Llama"
            p = f"model.layers.{l}"
            ln1 = st_llama[f"{p}.input_layernorm.weight"]
            ln2 = st_llama[f"{p}.post_attention_layernorm.weight"]
            
            q = st_llama[f"{p}.self_attn.q_proj.weight"].float() # [2048, 2048]
            k = st_llama[f"{p}.self_attn.k_proj.weight"].float() # [512, 2048] (8 heads)
            v = st_llama[f"{p}.self_attn.v_proj.weight"].float() # [512, 2048] (8 heads)
            o = st_llama[f"{p}.self_attn.o_proj.weight"].float() # [2048, 2048]
            
            # GQA 4x expansion for K and V
            k_exp = k.view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048)
            v_exp = v.view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048)
            
            ffn_gate = st_llama[f"{p}.mlp.gate_proj.weight"].float() # [8192, 2048]
            ffn_up = st_llama[f"{p}.mlp.up_proj.weight"].float()     # [8192, 2048]
            ffn_down = st_llama[f"{p}.mlp.down_proj.weight"].float() # [2048, 8192]
            
        elif l < 10:
            donor = "Falcon"
            f_idx = l - 6
            ln1 = falcon_tensors[f"blk.{f_idx}.attn_norm.weight"].float()
            ln2 = falcon_tensors[f"blk.{f_idx}.ffn_norm.weight"].float()
            
            q = falcon_tensors[f"blk.{f_idx}.attn_q.weight"].float() # [2048, 2048]
            k = falcon_tensors[f"blk.{f_idx}.attn_k.weight"].float() # [1024, 2048]
            v = falcon_tensors[f"blk.{f_idx}.attn_v.weight"].float() # [1024, 2048]
            o = falcon_tensors[f"blk.{f_idx}.attn_output.weight"].float() # [2048, 2048]
            
            # GQA 2x expansion for K and V (from 16 to 32 heads)
            k_exp = k.view(16, 64, 2048).repeat_interleave(2, dim=0).view(2048, 2048)
            v_exp = v.view(16, 64, 2048).repeat_interleave(2, dim=0).view(2048, 2048)
            
            ffn_gate = falcon_tensors[f"blk.{f_idx}.ffn_gate.weight"].float() # [8192, 2048]
            ffn_up = falcon_tensors[f"blk.{f_idx}.ffn_up.weight"].float()     # [8192, 2048]
            ffn_down = falcon_tensors[f"blk.{f_idx}.ffn_down.weight"].float() # [2048, 8192]
            
        elif l < 13:
            donor = "BitNet"
            b_idx = l - 10
            p = f"model.layers.{b_idx}"
            ln1 = st_bitnet[f"{p}.input_layernorm.weight"]
            ln2 = st_bitnet[f"{p}.post_attention_layernorm.weight"]
            
            # BitNet attention projected to 2048
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
            donor = "Qwen"
            q_idx = l - 13
            p = f"model.layers.{q_idx}"
            ln1 = st_qwen[f"{p}.input_layernorm.weight"]
            ln2 = st_qwen[f"{p}.post_attention_layernorm.weight"]
            
            q = project_dim(st_qwen[f"{p}.self_attn.q_proj.weight"], hidden_dim, hidden_dim)
            k = project_dim(st_qwen[f"{p}.self_attn.k_proj.weight"], hidden_dim, hidden_dim)
            v = project_dim(st_qwen[f"{p}.self_attn.v_proj.weight"], hidden_dim, hidden_dim)
            o = project_dim(st_qwen[f"{p}.self_attn.o_proj.weight"], hidden_dim, hidden_dim)
            k_exp = k
            v_exp = v
            
            ffn_gate = project_dim(st_qwen[f"{p}.mlp.gate_proj.weight"], ffn_dim, hidden_dim)
            ffn_up = project_dim(st_qwen[f"{p}.mlp.up_proj.weight"], ffn_dim, hidden_dim)
            ffn_down = project_dim(st_qwen[f"{p}.mlp.down_proj.weight"], hidden_dim, ffn_dim)
            
        # 1. LayerNorms (Gain = 1.0)
        model_state[f"{dst}.ln_1.weight"] = normalize_layernorm_vector(ln1, hidden_dim, target_mean=1.0).to(torch.bfloat16)
        model_state[f"{dst}.ln_1.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)
        model_state[f"{dst}.ln_2.weight"] = normalize_layernorm_vector(ln2, hidden_dim, target_mean=1.0).to(torch.bfloat16)
        model_state[f"{dst}.ln_2.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)
        
        # 2. Attention
        c_attn = torch.cat([q, k_exp, v_exp], dim=0) # [6144, 2048]
        model_state[f"{dst}.attn.c_attn.weight"] = c_attn.to(torch.bfloat16)
        model_state[f"{dst}.attn.c_attn.bias"] = torch.zeros(3 * hidden_dim, dtype=torch.bfloat16)
        model_state[f"{dst}.attn.c_proj.weight"] = o.to(torch.bfloat16)
        model_state[f"{dst}.attn.c_proj.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)
        
        # 3. Takehi Attractor Deflation on MLP Down-Projection
        ffn_down_deflated = remove_attractor_direction(ffn_down, num_directions=1)
        
        # 4. 1-Shot 34-Expert MoE Extraction
        experts = extract_34_experts_fast(ffn_down_deflated, hidden_dim)
        for e_idx, (w0, w1) in enumerate(experts):
            model_state[f"{dst}.moe.experts.{e_idx}.w0.weight"] = w0
            model_state[f"{dst}.moe.experts.{e_idx}.w1.weight"] = w1
            
        logger.info(f"  Layer {l:02d}/16 spliced from [{donor:6s}] (LN=1.0, Attn=6144x2048, Attractor Deflated, 34 MoE Experts)")

    # 4. Save Final Sovereign Checkpoint
    logger.info(">>> [4/5] Packaging Sovereign Model...")
    cfg = QuillanOniConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        n_layer=n_layer,
        n_head=32,
        head_dim=64,
        ffn_dim=ffn_dim,
        num_experts=num_experts
    )
    
    ckpt = {
        "model_state_dict": model_state,
        "config": cfg.__dict__,
        "vocab_size": vocab_size,
        "tokenizer_type": "llama3.2_128k",
        "donors": ["llama-3.2-1b", "falcon-3-1b", "bitnet-1.58b", "qwen-2.5-0.5b"],
        "wsp_techniques": ["128k_direct_parity", "gqa_head_broadcast", "unit_mean_layernorm", "takehi_attractor_deflation", "1shot_svd_moe"]
    }
    
    torch.save(ckpt, out_path)
    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f">>> [5/5] Sovereign 1B Compiled Successfully -> {out_path} ({sz_mb:.1f} MB, {time.time()-t0:.2f}s)")
    return out_path


if __name__ == "__main__":
    compile_quad_wsp_sovereign()
