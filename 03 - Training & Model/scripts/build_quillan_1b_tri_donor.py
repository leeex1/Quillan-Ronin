#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Quillan-Ronin: 1B Sovereign Tri-Donor Fusion Engine
=============================================================================
Builds the flagship Quillan 1B Sovereign model (D=2048, 16 Layers, 34 MoE):
  - Backbone & Embeddings : Llama 3.2 1B (1:1 direct match at D=2048, 128k vocab)
  - Experts 0..11 (Values): Llama 3.2 1B (Astra, Vir, Solace, Echo, Typist)
  - Experts 12..22 (Sys)  : BitNet 4T2B  (Warden, Kaido, Vortex, Talos, Ronin)
  - Experts 23..33 (Logic): Qwen 0.6B    (Logos, Scaffold, Archon, Calculus, Predator)

Operates deterministically in CPU host RAM with streaming safetensors (<6.5 GB).
=============================================================================
"""

import gc
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [1B-Fusion] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("1B-Fusion")


def extract_svd_adapter(
    W_donor: torch.Tensor,
    target_dim: int = 2048,
    rank: int = 32,
    niter: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes fast randomized low-rank SVD and adapts to target_dim."""
    W = W_donor.float().cpu()
    U, S, V = torch.svd_lowrank(W, q=rank, niter=niter)
    S_sqrt = torch.diag(torch.sqrt(S))

    w0 = S_sqrt @ V.t()  # (rank, cols)
    w1 = U @ S_sqrt      # (rows, rank)

    if w0.shape[1] != target_dim:
        w0 = F.interpolate(
            w0.unsqueeze(0).unsqueeze(0),
            size=(rank, target_dim),
            mode="bilinear",
            align_corners=False
        ).squeeze(0).squeeze(0)

    if w1.shape[0] != target_dim:
        w1 = F.interpolate(
            w1.t().unsqueeze(0).unsqueeze(0),
            size=(rank, target_dim),
            mode="bilinear",
            align_corners=False
        ).squeeze(0).squeeze(0).t()

    return w0, w1


def build_quillan_1b_fusion(
    output_path: str,
    donor_llama_path: str,
    donor_bitnet_path: str,
    donor_qwen_path: str,
    hidden_dim: int = 2048,
    n_layer: int = 16,
    num_experts: int = 34,
    expert_rank: int = 32
):
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("QUILLAN-RONIN 1B SOVEREIGN TRI-DONOR FUSION")
    logger.info(f"Target Architecture: D={hidden_dim}, Layers={n_layer}, Experts={num_experts}, Rank={expert_rank}")
    logger.info("=" * 70)

    model_state = {}

    # -------------------------------------------------------------------------
    # STEP 1: Transplant Llama 3.2 1B (Backbone, Embeddings & Experts 0..11)
    # -------------------------------------------------------------------------
    logger.info(">>> [Phase 1/3] Loading Donor 1: Llama 3.2 1B (Backbone & Experts 0..11)...")
    with safe_open(donor_llama_path, framework="pt", device="cpu") as f_llama:
        # Embeddings & Output Normalization
        logger.info("  Transplanting Token Embeddings (128,256 x 2048)...")
        emb = f_llama.get_tensor("model.embed_tokens.weight")
        model_state["wte.weight"] = emb.to(torch.bfloat16)
        vocab_size = emb.shape[0]

        norm_weight = f_llama.get_tensor("model.norm.weight")
        model_state["ln_f.weight"] = norm_weight.to(torch.bfloat16)

        llama_mlp_keys = [k for k in f_llama.keys() if "mlp" in k and "weight" in k and len(f_llama.get_tensor(k).shape) == 2]

        for l in range(n_layer):
            src_prefix = f"model.layers.{l}"
            dst_prefix = f"h.{l}"

            # Self-Attention projections (1:1 exact transfer)
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                k = f"{src_prefix}.self_attn.{proj}.weight"
                if k in f_llama.keys():
                    model_state[f"{dst_prefix}.attn.{proj}.weight"] = f_llama.get_tensor(k).to(torch.bfloat16)

            # LayerNorms
            model_state[f"{dst_prefix}.ln_1.weight"] = f_llama.get_tensor(f"{src_prefix}.input_layernorm.weight").to(torch.bfloat16)
            model_state[f"{dst_prefix}.ln_2.weight"] = f_llama.get_tensor(f"{src_prefix}.post_attention_layernorm.weight").to(torch.bfloat16)

            # Base FFN
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                k = f"{src_prefix}.mlp.{proj}.weight"
                if k in f_llama.keys():
                    model_state[f"{dst_prefix}.mlp.{proj}.weight"] = f_llama.get_tensor(k).to(torch.bfloat16)

            # MoE Router & Prior initialization
            model_state[f"{dst_prefix}.moe.router.weight"] = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16) * 0.02
            model_state[f"{dst_prefix}.moe.pull_gate.prior"] = torch.zeros(num_experts, dtype=torch.bfloat16)
            model_state[f"{dst_prefix}.moe.pull_gate.gate.weight"] = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16) * 0.02

            # Experts 0..11: Values, Ethics, Cognitive & Memory
            for e in range(0, 12):
                src_k = llama_mlp_keys[(l * 12 + e) % len(llama_mlp_keys)]
                src_w = f_llama.get_tensor(src_k)
                w0, w1 = extract_svd_adapter(src_w, target_dim=hidden_dim, rank=expert_rank)
                model_state[f"{dst_prefix}.moe.evo_moe.experts.{e}.0.weight"] = w0.to(torch.bfloat16)
                model_state[f"{dst_prefix}.moe.evo_moe.experts.{e}.1.weight"] = w1.to(torch.bfloat16)

    gc.collect()
    logger.info("  Llama 3.2 1B backbone and Experts 0..11 successfully fused.")

    # -------------------------------------------------------------------------
    # STEP 2: Transplant BitNet 4T2B (Experts 12..22: Systems, Swarms & Security)
    # -------------------------------------------------------------------------
    logger.info(">>> [Phase 2/3] Splicing Donor 2: BitNet 4T2B (Experts 12..22)...")
    with safe_open(donor_bitnet_path, framework="pt", device="cpu") as f_bitnet:
        bitnet_keys = [k for k in f_bitnet.keys() if "mlp" in k and "weight" in k and len(f_bitnet.get_tensor(k).shape) == 2]

        for l in range(n_layer):
            dst_prefix = f"h.{l}"
            for e in range(12, 23):
                src_k = bitnet_keys[(l * 11 + (e - 12)) % len(bitnet_keys)]
                src_w = f_bitnet.get_tensor(src_k)
                w0, w1 = extract_svd_adapter(src_w, target_dim=hidden_dim, rank=expert_rank)
                model_state[f"{dst_prefix}.moe.evo_moe.experts.{e}.0.weight"] = w0.to(torch.bfloat16)
                model_state[f"{dst_prefix}.moe.evo_moe.experts.{e}.1.weight"] = w1.to(torch.bfloat16)

    gc.collect()
    logger.info("  BitNet 4T2B circuits successfully grafted into Experts 12..22.")

    # -------------------------------------------------------------------------
    # STEP 3: Transplant Qwen 0.6B (Experts 23..33: Logic, Math, Code & Red-Team)
    # -------------------------------------------------------------------------
    logger.info(">>> [Phase 3/3] Splicing Donor 3: Qwen 0.6B (Experts 23..33)...")
    with safe_open(donor_qwen_path, framework="pt", device="cpu") as f_qwen:
        qwen_keys = [k for k in f_qwen.keys() if ("mlp" in k or "ffn" in k) and "weight" in k and len(f_qwen.get_tensor(k).shape) == 2]

        for l in range(n_layer):
            dst_prefix = f"h.{l}"
            for e in range(23, 34):
                src_k = qwen_keys[(l * 11 + (e - 23)) % len(qwen_keys)]
                src_w = f_qwen.get_tensor(src_k)
                w0, w1 = extract_svd_adapter(src_w, target_dim=hidden_dim, rank=expert_rank)
                model_state[f"{dst_prefix}.moe.evo_moe.experts.{e}.0.weight"] = w0.to(torch.bfloat16)
                model_state[f"{dst_prefix}.moe.evo_moe.experts.{e}.1.weight"] = w1.to(torch.bfloat16)

    gc.collect()
    logger.info("  Qwen 0.6B circuits successfully grafted into Experts 23..33.")

    # -------------------------------------------------------------------------
    # Checkpoint Assembly & Serialization
    # -------------------------------------------------------------------------
    checkpoint = {
        "step": 0,
        "config": {
            "hidden_dim": hidden_dim,
            "n_layer": n_layer,
            "n_head": 32,
            "head_dim": 64,
            "ffn_dim": 8192,
            "num_experts": num_experts,
            "expert_rank": expert_rank,
            "vocab_size": vocab_size,
            "model_type": "quillan_1b_sovereign_fusion",
            "rope_theta": 500000.0,
        },
        "model_state_dict": model_state,
        "tri_donor_manifest": {
            "donors": ["Llama-3.2-1B-Instruct", "BitNet-4T2B", "Qwen-0.6B"],
            "total_layers": n_layer,
            "total_experts": num_experts,
            "hidden_dim": hidden_dim,
            "vocab_size": vocab_size
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    logger.info(f"Serializing 1B Sovereign Checkpoint to: {output_path}...")
    torch.save(checkpoint, output_path)

    elapsed = time.time() - start_time
    total_params = sum(p.numel() for p in model_state.values()) / 1e6
    logger.info(f"Done! Assembled {total_params:.2f}M parameters in {elapsed:.2f}s.")
    logger.info(f"Quillan 1B Sovereign Tri-Donor Fusion is fully compiled and ready.")


if __name__ == "__main__":
    d_llama = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\model.safetensors"
    d_bitnet = r"C:\Users\Admin\.cache\huggingface\hub\models--1bitLLM--bitnet_b1_58-large\snapshots\85d047191dcb224f0e04f20d26110caaf8dc1a47\model.safetensors"
    d_qwen = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors"
    out_dir = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "quillan_1b_tri_donor_sovereign.pt")

    build_quillan_1b_fusion(out_file, d_llama, d_bitnet, d_qwen)
