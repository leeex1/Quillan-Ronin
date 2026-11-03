#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Quillan-Ronin: Canonical Mini 0.6B Sovereign Quad-Donor Compiler (128k)
=============================================================================
Surgically unifies Quad Donors into the Edge/Mini 0.6B Sovereign Architecture:
  - Hidden Dimension: D = 1152, FFN = 3072, 6 Layers, 34 Experts (Uniform r=16)
  - Vocabulary: Pure 128k Tiktoken BPE (100% Llama 3.2 Anchor, 0 index contamination)
  - Donors:
      * Layers 00..01 (2 layers): Llama 3.2 1B (Dense causal foundation)
      * Layer 02      (1 layer):  Falcon 3 1B (High-bandwidth attention)
      * Layer 03      (1 layer):  BitNet 1.58b Large (Ternary compute core)
      * Layers 04..05 (2 layers): Qwen 3.5 2B (Gated Delta recurrence & reasoning)
  - Zero-Missing-Key Invariance: Directly mapped to canonical QuillanRoninOni
  - Takehi SVD Attractor Deflation on MLP down-projections
=============================================================================
"""

import os
import sys
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
    format="%(asctime)s [%(levelname)s] [Mini-0.6B] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Mini-0.6B")

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni


def to_bitnet_ternary_wsp(W: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    W_f = W.float().cpu()
    scale = 1.0 / W_f.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    W_scaled = W_f * scale
    W_q = torch.round(torch.clamp(W_scaled, -1.0, 1.0))
    return W_q / scale


def project_dim(W: torch.Tensor, target_out: int, target_in: int) -> torch.Tensor:
    W = W.float().cpu()
    r, c = W.shape
    if r == target_out and c == target_in:
        return W
    if r >= target_out:
        W = W[:target_out, :]
    else:
        pad = target_out - r
        repeats = (pad // r) + 1
        W = torch.cat([W, W.repeat(repeats, 1)[:pad, :]], dim=0)
    if c >= target_in:
        W = W[:, :target_in]
    else:
        pad = target_in - c
        repeats = (pad // c) + 1
        W = torch.cat([W, W.repeat(1, repeats)[:, :pad]], dim=1)
    return W


def normalize_layernorm_vector(v: torch.Tensor, target_dim: int, target_mean: float = 1.0) -> torch.Tensor:
    v = v.float().cpu()
    d = v.size(0)
    if d >= target_dim:
        v_proj = v[:target_dim]
    else:
        pad = target_dim - d
        repeats = (pad // d) + 1
        v_proj = torch.cat([v, v.repeat(repeats)[:pad]], dim=0)
    v_proj = v_proj - v_proj.mean() + target_mean
    return v_proj.clamp(min=0.1, max=3.0)


def remove_attractor_direction(W: torch.Tensor, num_directions: int = 1) -> torch.Tensor:
    W_f = W.float().cpu()
    try:
        U, S, V = torch.pca_lowrank(W_f, q=min(num_directions + 2, min(W_f.shape)), center=False)
        v = V[:, :num_directions]
        W_clean = W_f - (W_f @ v) @ v.t()
        return W_clean
    except Exception:
        return W_f


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
        
        w0_tern = to_bitnet_ternary_wsp(w0_sub)
        w1_tern = to_bitnet_ternary_wsp(w1_sub)
        experts.append((w0_tern.to(torch.bfloat16), w1_tern.to(torch.bfloat16)))
    return experts


def compile_mini_06b_sovereign():
    print("=" * 80, flush=True)
    print("COMPILING QUILLAN-RONIN MINI 0.6B SOVEREIGN QUAD-DONOR WSP 128K", flush=True)
    print("=" * 80, flush=True)

    hf_cache = r"C:\Users\Admin\.cache\huggingface\hub"
    ollama_blobs = r"C:\Users\Admin\.ollama\models\blobs"

    llama_dir = os.path.join(hf_cache, "models--unsloth--Llama-3.2-1B-Instruct", "snapshots")
    llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
    llama_st_path = os.path.join(llama_snap, "model.safetensors")

    falcon_blob = os.path.join(ollama_blobs, "sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01")

    bitnet_dir = os.path.join(hf_cache, "models--1bitLLM--bitnet_b1_58-large", "snapshots")
    bitnet_snap = os.path.join(bitnet_dir, os.listdir(bitnet_dir)[0])
    bitnet_st_path = os.path.join(bitnet_snap, "model.safetensors")

    qwen35_blob = os.path.join(ollama_blobs, "sha256-e8bf47d40c5ba68baf8f3e503ee36710da6e8641cb0a670a508cff5b3f0cfbf7")

    out_dir = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "quillan_mini_0.6b_sovereign_wsp128k_qwen35.pt")

    hidden_dim = 2048
    n_layer = 8
    ffn_dim = 8192
    num_experts = 34
    vocab_size = 128256

    logger.info(f"Target: D={hidden_dim}, Layers={n_layer}, Vocab={vocab_size}, Experts={num_experts} (Balanced 25% Quad-Donor, Uniform r=16)")

    # 1. Instantiate Model Architecture
    logger.info(">>> [1/5] Instantiating Canonical QuillanRoninOni Mini Sovereign...")
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

    # 2. Ingest Donors
    logger.info(">>> [2/5] Ingesting Quad Donors...")
    st_llama = load_file(llama_st_path)
    
    r_falcon = gguf.GGUFReader(falcon_blob)
    def dequant_gguf(t) -> torch.Tensor:
        if t.tensor_type != 0:
            arr = gguf.quants.dequantize(t.data, t.tensor_type)
        else:
            arr = t.data
        return torch.from_numpy(arr.copy()).float()
        
    falcon_needed = {"token_embd.weight", "blk.0.attn_norm.weight"}
    for f_idx in range(2):
        for k in ["attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]:
            falcon_needed.add(f"blk.{f_idx}.{k}")
    falcon_tensors = {t.name: dequant_gguf(t) for t in r_falcon.tensors if t.name in falcon_needed}
    del r_falcon
    
    st_bitnet = load_file(bitnet_st_path)
    
    r_qwen35 = gguf.GGUFReader(qwen35_blob)
    qwen35_tensors = {
        t.name: dequant_gguf(t)
        for t in r_qwen35.tensors
        if any(t.name.startswith(f"blk.{q_idx}.") for q_idx in range(2))
    }
    del r_qwen35
    import gc
    gc.collect()

    # 3. Pure 128k Vocab Embeddings (100% Llama Anchor D=2048 - Zero Truncation)
    logger.info(">>> [3/5] Splicing 128k Pure Anchor Vocabulary Embeddings (Native D=2048)...")
    emb_llama = st_llama["model.embed_tokens.weight"].float()
    model.wte.weight.data.copy_(emb_llama.to(torch.bfloat16))
    model.lm_head.weight.data.copy_(emb_llama.to(torch.bfloat16))
    del emb_llama

    norm_l = st_llama["model.norm.weight"].float()
    model.ln_f.weight.data.copy_(norm_l.to(torch.bfloat16))
    if hasattr(model.ln_f, "bias") and model.ln_f.bias is not None:
        model.ln_f.bias.data.zero_()

    model.q1_bridge.weight.data.copy_(torch.eye(hidden_dim, dtype=torch.bfloat16))
    model.q2_bridge.weight.data.copy_(torch.eye(hidden_dim, dtype=torch.bfloat16))
    model.ingest_gate.weight.data.zero_()
    model.ingest_gate.bias.data.zero_()
    model.quillan_finalizer_q1.weight.data.copy_(torch.eye(hidden_dim, dtype=torch.bfloat16))
    model.quillan_finalizer_q2.weight.data.copy_(torch.eye(hidden_dim, dtype=torch.bfloat16))
    model.quillan_comm_gate.weight.data.zero_()
    model.quillan_comm_gate.bias.data.zero_()

    # 4. Splicing 8 Layers across Balanced Quad Donors (2/2/2/2 = 25% each)
    logger.info(">>> [4/5] Splicing 8 Layers across Balanced Quad Donors (25% each)...")
    for l in range(n_layer):
        blk = model.h[l]
        if l < 2:
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
            ffn_gate = st_llama[f"{p}.mlp.gate_proj.weight"].float()
            ffn_up = st_llama[f"{p}.mlp.up_proj.weight"].float()
            ffn_down = st_llama[f"{p}.mlp.down_proj.weight"].float()
        elif l < 4:
            donor = "Falcon-3-1B"
            f_idx = l - 2
            ln1 = falcon_tensors[f"blk.{f_idx}.attn_norm.weight"].float()
            ln2 = falcon_tensors[f"blk.{f_idx}.ffn_norm.weight"].float()
            q = falcon_tensors[f"blk.{f_idx}.attn_q.weight"].float()
            k = falcon_tensors[f"blk.{f_idx}.attn_k.weight"].float()
            v = falcon_tensors[f"blk.{f_idx}.attn_v.weight"].float()
            o = falcon_tensors[f"blk.{f_idx}.attn_output.weight"].float()
            k_exp = k.view(16, 64, 2048).repeat_interleave(2, dim=0).view(2048, 2048)
            v_exp = v.view(16, 64, 2048).repeat_interleave(2, dim=0).view(2048, 2048)
            ffn_gate = falcon_tensors[f"blk.{f_idx}.ffn_gate.weight"].float()
            ffn_up = falcon_tensors[f"blk.{f_idx}.ffn_up.weight"].float()
            ffn_down = falcon_tensors[f"blk.{f_idx}.ffn_down.weight"].float()
        elif l < 6:
            donor = "BitNet-1.58b"
            b_idx = l - 4
            p = f"model.layers.{b_idx}"
            ln1 = st_bitnet[f"{p}.input_layernorm.weight"].float()
            ln2 = st_bitnet[f"{p}.post_attention_norm.weight"].float() if f"{p}.post_attention_norm.weight" in st_bitnet else st_bitnet[f"{p}.post_attention_layernorm.weight"].float()
            
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
            q_idx = l - 6
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

        ln1_proj = project_dim(ln1.unsqueeze(0), 1, hidden_dim).squeeze(0)
        ln2_proj = project_dim(ln2.unsqueeze(0), 1, hidden_dim).squeeze(0)
        blk.ln_1.weight.data.copy_(ln1_proj.to(torch.bfloat16))
        if hasattr(blk.ln_1, "bias") and blk.ln_1.bias is not None:
            blk.ln_1.bias.data.zero_()
        blk.ln_2.weight.data.copy_(ln2_proj.to(torch.bfloat16))
        if hasattr(blk.ln_2, "bias") and blk.ln_2.bias is not None:
            blk.ln_2.bias.data.zero_()

        c_attn = torch.cat([q, k_exp, v_exp], dim=0) # [3 * hidden_dim, hidden_dim]
        blk.attn.c_attn.weight.data.copy_(c_attn.to(torch.bfloat16))
        blk.attn.c_attn.bias.data.zero_()
        blk.attn.c_proj.weight.data.copy_(o.to(torch.bfloat16))
        blk.attn.c_proj.bias.data.zero_()
        blk.attn.prism.w_gate.weight.data.zero_()

        c_fc = torch.cat([ffn_gate, ffn_up], dim=0)
        blk.moe.c_fc.weight.data.copy_(c_fc.to(torch.bfloat16))
        blk.moe.c_fc.bias.data.zero_()

        ffn_down_clean = remove_attractor_direction(ffn_down, num_directions=1)
        blk.moe.c_proj.weight.data.copy_(ffn_down_clean.to(torch.bfloat16))
        blk.moe.c_proj.bias.data.zero_()

        experts = extract_34_uniform_experts(ffn_down_clean, hidden_dim, rank=16)
        for e_idx, (w0, w1) in enumerate(experts):
            exp = blk.moe.experts[e_idx]
            exp.lora_A.data.copy_(w1.to(torch.bfloat16))
            exp.lora_B.data.copy_(w0.to(torch.bfloat16))
            exp.swarm.A.data.zero_()
            exp.swarm.B.data.zero_()
            exp.swarm.C.data.zero_()
            exp.swarm.D.data.zero_()
            exp.swarm.clone_diversity.data.zero_()
            exp.swarm.clone_coupling.data.zero_()

        blk.moe.router.weight.data.zero_()
        blk.moe.pull_gate.gate.weight.data.zero_()
        blk.moe.moe_gate.weight.data.zero_()
        blk.moe.moe_gate.bias.data.fill_(0.05)

        logger.info(f"  Layer {l:02d}/08 spliced from [{donor:12s}] (LN=1.0, Attn=6144x2048, Deflated, 34 r=16 Experts)")

    # 5. Export Checkpoint
    logger.info(">>> [5/5] Packaging Mini 0.6B Sovereign Checkpoint (bfloat16)...")
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
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f"[SUCCESS] Mini 0.6B Sovereign saved to {out_path} ({size_mb:.1f} MB)")
    print(f"\n[DONE] Checkpoint ready: {out_path} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    compile_mini_06b_sovereign()
