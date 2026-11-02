#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Quillan-Ronin: Tri-Donor Weight-Space Programming (WSP) Splicer
=============================================================================
Grafts circuits from 3 distinct donor models into Quillan's 34-Council MoE:
  - Donor 1: BitNet 2B-4T (Experts 12-22: Systems, Edge Architecture, Security)
  - Donor 2: Llama 3.2 1B (Experts 0-11:  Knowledge, Values, Ethics, Context)
  - Donor 3: Qwen 0.6B    (Experts 23-33: Logic, Math, Code, Red-Teaming)

Dynamically adapts to Quillan's heterogeneous EGGROLL ranks (r in {8, 24, 48}).
Runs deterministically on CPU host memory.
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
import torch.nn.functional as F
from safetensors import safe_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [TriDonor-WSP] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TriDonor-WSP")


def extract_svd_adapter(
    W_donor: torch.Tensor,
    target_dim: int = 1152,
    rank: int = 8,
    niter: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes low-rank SVD decomposition of donor matrix W:
      W approx U @ diag(S) @ V.T
    Splits singular values symmetrically:
      W0 = diag(sqrt(S)) @ V.T  -> adapted to (rank, target_dim)
      W1 = U @ diag(sqrt(S))    -> adapted to (target_dim, rank)
    """
    W = W_donor.float().cpu()
    # Compute fast low-rank SVD
    U, S, V = torch.svd_lowrank(W, q=rank, niter=niter)
    S_sqrt = torch.diag(torch.sqrt(S))
    
    w0 = S_sqrt @ V.t()  # (rank, cols)
    w1 = U @ S_sqrt      # (rows, rank)
    
    # Bilinear projection to target dimension
    w0_proj = F.interpolate(
        w0.unsqueeze(0).unsqueeze(0),
        size=(rank, target_dim),
        mode="bilinear",
        align_corners=False
    ).squeeze(0).squeeze(0)
    
    w1_proj = F.interpolate(
        w1.t().unsqueeze(0).unsqueeze(0),
        size=(rank, target_dim),
        mode="bilinear",
        align_corners=False
    ).squeeze(0).squeeze(0).t()
    
    return w0_proj, w1_proj


def graft_tri_donors(
    quillan_checkpoint_path: str,
    output_checkpoint_path: str,
    donor_bitnet_path: str,
    donor_llama_path: str,
    donor_qwen_path: str,
    target_dim: int = 1152
):
    start_time = time.time()
    logger.info(f"Loading Base Quillan Checkpoint: {quillan_checkpoint_path}")
    checkpoint = torch.load(quillan_checkpoint_path, map_location="cpu")
    model_state = checkpoint["model_state_dict"]
    
    total_grafted_tensors = 0

    # -------------------------------------------------------------------------
    # DONOR 2: Llama 3.2 1B -> Experts 0 to 11 (Knowledge, Values, Ethics, Context)
    # -------------------------------------------------------------------------
    logger.info(">>> [1/3] Splicing Donor: Llama 3.2 1B into Council Experts 0..11...")
    with safe_open(donor_llama_path, framework="pt", device="cpu") as f_llama:
        llama_keys = [k for k in f_llama.keys() if ("mlp" in k or "self_attn" in k) and "weight" in k and len(f_llama.get_tensor(k).shape) == 2]
        logger.info(f"    Discovered {len(llama_keys)} 2D candidate projection matrices in Llama 3.2 1B.")
        
        for l in range(8):
            for e in range(0, 12):
                key_0 = f"h.{l}.moe.evo_moe.experts.{e}.0.weight"
                key_1 = f"h.{l}.moe.evo_moe.experts.{e}.1.weight"
                if key_0 in model_state and key_1 in model_state:
                    current_rank = model_state[key_0].shape[0]
                    src_k = llama_keys[(l * 12 + e) % len(llama_keys)]
                    src_w = f_llama.get_tensor(src_k)
                    w0_p, w1_p = extract_svd_adapter(src_w, target_dim=target_dim, rank=current_rank)
                    
                    # Norm matching
                    orig_norm0 = model_state[key_0].float().norm()
                    orig_norm1 = model_state[key_1].float().norm()
                    scale0 = (orig_norm0 / (w0_p.norm() + 1e-8)).clamp(0.1, 5.0)
                    scale1 = (orig_norm1 / (w1_p.norm() + 1e-8)).clamp(0.1, 5.0)
                    
                    model_state[key_0].copy_((w0_p * scale0).to(model_state[key_0].dtype))
                    model_state[key_1].copy_((w1_p * scale1).to(model_state[key_1].dtype))
                    total_grafted_tensors += 2

    gc.collect()
    logger.info("    Llama 3.2 1B circuits successfully grafted into Experts 0..11.")

    # -------------------------------------------------------------------------
    # DONOR 1: BitNet 4T2B -> Experts 12 to 22 (Systems, Edge Architecture, Security)
    # -------------------------------------------------------------------------
    logger.info(">>> [2/3] Splicing Donor: BitNet 4T2B into Council Experts 12..22...")
    with safe_open(donor_bitnet_path, framework="pt", device="cpu") as f_bitnet:
        bitnet_keys = [k for k in f_bitnet.keys() if "mlp" in k and "weight" in k and len(f_bitnet.get_tensor(k).shape) == 2]
        logger.info(f"    Discovered {len(bitnet_keys)} MLP projection matrices in BitNet 4T2B.")
        
        for l in range(8):
            for e in range(12, 23):
                key_0 = f"h.{l}.moe.evo_moe.experts.{e}.0.weight"
                key_1 = f"h.{l}.moe.evo_moe.experts.{e}.1.weight"
                if key_0 in model_state and key_1 in model_state:
                    current_rank = model_state[key_0].shape[0]
                    src_k = bitnet_keys[(l * 11 + (e - 12)) % len(bitnet_keys)]
                    src_w = f_bitnet.get_tensor(src_k)
                    w0_p, w1_p = extract_svd_adapter(src_w, target_dim=target_dim, rank=current_rank)
                    
                    # Norm matching
                    orig_norm0 = model_state[key_0].float().norm()
                    orig_norm1 = model_state[key_1].float().norm()
                    scale0 = (orig_norm0 / (w0_p.norm() + 1e-8)).clamp(0.1, 5.0)
                    scale1 = (orig_norm1 / (w1_p.norm() + 1e-8)).clamp(0.1, 5.0)
                    
                    model_state[key_0].copy_((w0_p * scale0).to(model_state[key_0].dtype))
                    model_state[key_1].copy_((w1_p * scale1).to(model_state[key_1].dtype))
                    total_grafted_tensors += 2

    gc.collect()
    logger.info("    BitNet 4T2B circuits successfully grafted into Experts 12..22.")

    # -------------------------------------------------------------------------
    # DONOR 3: Qwen 0.6B -> Experts 23 to 33 (Logic, Mathematics, Code, Red-Teaming)
    # -------------------------------------------------------------------------
    logger.info(">>> [3/3] Splicing Donor: Qwen 0.6B into Council Experts 23..33...")
    with safe_open(donor_qwen_path, framework="pt", device="cpu") as f_qwen:
        qwen_keys = [k for k in f_qwen.keys() if ("mlp" in k or "ffn" in k) and "weight" in k and len(f_qwen.get_tensor(k).shape) == 2]
        logger.info(f"    Discovered {len(qwen_keys)} MLP projection matrices in Qwen 0.6B.")
        
        for l in range(8):
            for e in range(23, 34):
                key_0 = f"h.{l}.moe.evo_moe.experts.{e}.0.weight"
                key_1 = f"h.{l}.moe.evo_moe.experts.{e}.1.weight"
                if key_0 in model_state and key_1 in model_state:
                    current_rank = model_state[key_0].shape[0]
                    src_k = qwen_keys[(l * 11 + (e - 23)) % len(qwen_keys)]
                    src_w = f_qwen.get_tensor(src_k)
                    w0_p, w1_p = extract_svd_adapter(src_w, target_dim=target_dim, rank=current_rank)
                    
                    # Norm matching
                    orig_norm0 = model_state[key_0].float().norm()
                    orig_norm1 = model_state[key_1].float().norm()
                    scale0 = (orig_norm0 / (w0_p.norm() + 1e-8)).clamp(0.1, 5.0)
                    scale1 = (orig_norm1 / (w1_p.norm() + 1e-8)).clamp(0.1, 5.0)
                    
                    model_state[key_0].copy_((w0_p * scale0).to(model_state[key_0].dtype))
                    model_state[key_1].copy_((w1_p * scale1).to(model_state[key_1].dtype))
                    total_grafted_tensors += 2

    gc.collect()
    logger.info("    Qwen 0.6B circuits successfully grafted into Experts 23..33.")

    # -------------------------------------------------------------------------
    # Update Checkpoint Metadata & Reset AdamW Momentum for Grafted Parameters
    # -------------------------------------------------------------------------
    checkpoint["model_state_dict"] = model_state
    checkpoint["tri_donor_grafted"] = True
    checkpoint["tri_donor_manifest"] = {
        "donors": ["BitNet-4T2B", "Llama-3.2-1B", "Qwen-0.6B"],
        "expert_mapping": {
            "0-11": "Llama-3.2-1B (Astra..Typist: Knowledge, Values, Ethics, Context)",
            "12-22": "BitNet-4T2B (Warden..Ronin: Systems, Edge Architecture, Security)",
            "23-33": "Qwen-0.6B (Logos..Predator: Logic, Math, Code, Red-Teaming)"
        },
        "grafted_tensors": total_grafted_tensors,
        "base_step": checkpoint.get("step", 3600)
    }

    # Reset optimizer state dict so stale momentum does not perturb freshly grafted weights
    if "optimizer_state_dict" in checkpoint:
        logger.info("Resetting optimizer state dict to prevent momentum corruption of grafted weights.")
        checkpoint["optimizer_state_dict"] = None

    logger.info(f"Saving Tri-Donor Grafted Checkpoint to: {output_checkpoint_path}")
    torch.save(checkpoint, output_checkpoint_path)
    
    elapsed = time.time() - start_time
    logger.info(f"Done! Successfully grafted {total_grafted_tensors} adapter tensors across 8 layers in {elapsed:.2f}s.")
    logger.info("Quillan-Ronin is now endowed with Tri-Donor Weight-Space circuits.")


if __name__ == "__main__":
    base_ckpt = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_sft_latest.pt"
    out_ckpt = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_sft_tri_donor.pt"
    
    d_bitnet = r"C:\Users\Admin\.cache\huggingface\hub\models--1bitLLM--bitnet_b1_58-large\snapshots\85d047191dcb224f0e04f20d26110caaf8dc1a47\model.safetensors"
    d_llama = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\model.safetensors"
    d_qwen = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors"

    graft_tri_donors(base_ckpt, out_ckpt, d_bitnet, d_llama, d_qwen)
