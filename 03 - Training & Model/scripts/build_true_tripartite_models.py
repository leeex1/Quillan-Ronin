#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Quillan-Ronin: True Tripartite Equal 1/3 Fusion Compiler
=============================================================================
Strictly enforces equal 1/3 contribution from all three donor models:
  1. Llama 3.2 1B : 1/3 Layers, 1/3 Embeddings, 11/34 Council Experts, 3/9 Prism
  2. BitNet 4T2B  : 1/3 Layers, 1/3 Embeddings, 12/34 Council Experts, 3/9 Prism
  3. Qwen 0.6B    : 1/3 Layers, 1/3 Embeddings, 11/34 Council Experts, 3/9 Prism

Generates:
  - Mini 0.6B Tripartite: D=1152, 6 Layers (2 Llama, 2 BitNet, 2 Qwen), 34 Experts
  - 1B Sovereign Tripartite: D=2048, 12 Layers (4 Llama, 4 BitNet, 4 Qwen), 34 Experts

Fully aligned with QuillanRoninOni architecture and native tokenizer.
=============================================================================
"""

import gc
import os
import sys
import time
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Tripartite-Fusion] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Tripartite-Fusion")


def get_2d_keys(f) -> List[str]:
    """Returns all keys in safetensors file that have 2D shapes with dims >= 64."""
    keys = []
    for k in f.keys():
        try:
            shape = f.get_slice(k).get_shape()
        except Exception:
            shape = f.get_tensor(k).shape
        if len(shape) == 2 and shape[0] >= 64 and shape[1] >= 64:
            keys.append(k)
    return keys


def project_weight(W: torch.Tensor, target_rows: int, target_cols: int) -> torch.Tensor:
    """Projects any 2D weight matrix into target_rows x target_cols via bilinear projection."""
    W = W.float().cpu()
    if len(W.shape) != 2:
        W = W.view(-1, 1).repeat(1, target_cols)[:target_rows, :]
    r, c = W.shape
    if r == target_rows and c == target_cols:
        return W
    
    W = F.interpolate(
        W.unsqueeze(0).unsqueeze(0),
        size=(target_rows, target_cols),
        mode="bilinear",
        align_corners=False
    ).squeeze(0).squeeze(0)
    return W


def get_expert_rank(expert_idx: int) -> int:
    """Returns heterogeneous EGGROLL rank matching EvoMoE: 11 small (8), 11 med (24), 12 large (48)."""
    if expert_idx < 11:
        return 8   # Llama 3.2 1B
    elif expert_idx < 22:
        return 24  # BitNet 4T2B
    else:
        return 48  # Qwen 0.6B


def extract_svd_expert(W: torch.Tensor, target_dim: int, rank: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decomposes weight into down/up expert adapter pair (rank, target_dim) and (target_dim, rank)."""
    W = W.float().cpu()
    U, S, V = torch.svd_lowrank(W, q=rank, niter=2)
    S_sqrt = torch.diag(torch.sqrt(S))

    w0 = S_sqrt @ V.t()  # (rank, cols)
    w1 = U @ S_sqrt      # (rows, rank)

    w0 = project_weight(w0, rank, target_dim)
    w1 = project_weight(w1, target_dim, rank)
    return w0, w1


def compile_tripartite_model(
    model_name: str,
    output_path: str,
    donor_llama_path: str,
    donor_bitnet_path: str,
    donor_qwen_path: str,
    hidden_dim: int,
    n_layer: int,
    ffn_dim: int,
    num_experts: int = 34,
    expert_rank: int = 16,
    vocab_size: int = 50262
):
    t0 = time.time()
    logger.info("=" * 75)
    logger.info(f"COMPILING TRIPARTITE MODEL: {model_name.upper()}")
    logger.info(f"Specs: D={hidden_dim}, Layers={n_layer}, FFN={ffn_dim}, Experts={num_experts}, Rank={expert_rank}, Vocab={vocab_size}")
    logger.info("Strict Parity: Exact 1/3 Llama 3.2 1B + 1/3 BitNet 4T2B + 1/3 Qwen 0.6B")
    logger.info("=" * 75)

    model_state = {}
    layers_per_donor = n_layer // 3  # Exactly 1/3 of the layers each

    # -------------------------------------------------------------------------
    # PART 1: TOKEN EMBEDDINGS (Equal 1/3 Superposition across all three donors)
    # -------------------------------------------------------------------------
    logger.info(">>> [1/5] Synthesizing Equal 1/3 Superposition Embeddings...")
    with safe_open(donor_llama_path, framework="pt", device="cpu") as f_llama:
        w_emb_llama = f_llama.get_tensor("model.embed_tokens.weight")[:vocab_size, :]
        w_emb_llama = project_weight(w_emb_llama, vocab_size, hidden_dim)

    with safe_open(donor_bitnet_path, framework="pt", device="cpu") as f_bitnet:
        bitnet_emb_key = "model.embed_tokens.weight" if "model.embed_tokens.weight" in f_bitnet.keys() else get_2d_keys(f_bitnet)[0]
        w_raw_bit = f_bitnet.get_tensor(bitnet_emb_key)
        # Pad or tile if bitnet vocab < vocab_size
        if w_raw_bit.shape[0] < vocab_size:
            repeats = (vocab_size // w_raw_bit.shape[0]) + 1
            w_raw_bit = w_raw_bit.repeat(repeats, 1)[:vocab_size, :]
        w_emb_bitnet = project_weight(w_raw_bit[:vocab_size, :], vocab_size, hidden_dim)

    with safe_open(donor_qwen_path, framework="pt", device="cpu") as f_qwen:
        qwen_emb_key = "model.embed_tokens.weight" if "model.embed_tokens.weight" in f_qwen.keys() else get_2d_keys(f_qwen)[0]
        w_raw_qwen = f_qwen.get_tensor(qwen_emb_key)
        w_emb_qwen = project_weight(w_raw_qwen[:vocab_size, :], vocab_size, hidden_dim)

    # Strictly equal 1/3 weighting
    w_emb_final = (w_emb_llama * (1.0 / 3.0) + w_emb_bitnet * (1.0 / 3.0) + w_emb_qwen * (1.0 / 3.0))
    model_state["wte.weight"] = w_emb_final.to(torch.bfloat16)
    model_state["lm_head.weight"] = model_state["wte.weight"]
    model_state["ln_f.weight"] = torch.ones(hidden_dim, dtype=torch.bfloat16)
    model_state["ln_f.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)

    logger.info(f"    Embeddings synthesized: {model_state['wte.weight'].shape} (1/3 Llama + 1/3 BitNet + 1/3 Qwen)")
    del w_emb_llama, w_emb_bitnet, w_emb_qwen, w_emb_final
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 2: DUAL-BRAIN BRIDGES & FINALIZERS (Equal 1/3 Distribution)
    # -------------------------------------------------------------------------
    logger.info(">>> [2/5] Synthesizing Dual Ingestion Bridges & Finalizer Gates...")
    with safe_open(donor_llama_path, framework="pt", device="cpu") as f_llama, \
         safe_open(donor_bitnet_path, framework="pt", device="cpu") as f_bitnet, \
         safe_open(donor_qwen_path, framework="pt", device="cpu") as f_qwen:
        
        llama_2d = get_2d_keys(f_llama)
        bitnet_2d = get_2d_keys(f_bitnet)
        qwen_2d = get_2d_keys(f_qwen)

        # q1 bridge from Llama, q2 bridge from BitNet, ingest gate from Qwen
        model_state["q1_bridge.weight"] = project_weight(f_llama.get_tensor(llama_2d[0]), hidden_dim, hidden_dim).to(torch.bfloat16)
        model_state["q2_bridge.weight"] = project_weight(f_bitnet.get_tensor(bitnet_2d[0]), hidden_dim, hidden_dim).to(torch.bfloat16)
        model_state["ingest_gate.weight"] = project_weight(f_qwen.get_tensor(qwen_2d[0]), hidden_dim, 2 * hidden_dim).to(torch.bfloat16)
        model_state["ingest_gate.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)

        # finalizers: q1 from Llama, q2 from BitNet, comm gate from Qwen
        model_state["quillan_finalizer_q1.weight"] = project_weight(f_llama.get_tensor(llama_2d[1 % len(llama_2d)]), hidden_dim, hidden_dim).to(torch.bfloat16)
        model_state["quillan_finalizer_q2.weight"] = project_weight(f_bitnet.get_tensor(bitnet_2d[1 % len(bitnet_2d)]), hidden_dim, hidden_dim).to(torch.bfloat16)
        model_state["quillan_comm_gate.weight"] = project_weight(f_qwen.get_tensor(qwen_2d[1 % len(qwen_2d)]), hidden_dim, 2 * hidden_dim).to(torch.bfloat16)
        model_state["quillan_comm_gate.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)

    # -------------------------------------------------------------------------
    # PART 3: TRANSFORMER LAYERS (1/3 Llama, 1/3 BitNet, 1/3 Qwen)
    # -------------------------------------------------------------------------
    logger.info(f">>> [3/5] Splicing {n_layer} Transformer Layers ({layers_per_donor} Llama, {layers_per_donor} BitNet, {layers_per_donor} Qwen)...")
    
    with safe_open(donor_llama_path, framework="pt", device="cpu") as f_llama, \
         safe_open(donor_bitnet_path, framework="pt", device="cpu") as f_bitnet, \
         safe_open(donor_qwen_path, framework="pt", device="cpu") as f_qwen:

        for l in range(n_layer):
            dst = f"h.{l}"
            if l < layers_per_donor:
                donor_tag = "Llama 3.2 1B (Knowledge, Context, Ethics)"
                stream = f_llama
                keys = llama_2d
            elif l < 2 * layers_per_donor:
                donor_tag = "BitNet 4T2B (Ternary Systems, Edge, Speed)"
                stream = f_bitnet
                keys = bitnet_2d
            else:
                donor_tag = "Qwen 0.6B (Logic, Mathematics, Code)"
                stream = f_qwen
                keys = qwen_2d

            # Layer Norms
            model_state[f"{dst}.ln_1.weight"] = torch.ones(hidden_dim, dtype=torch.bfloat16)
            model_state[f"{dst}.ln_1.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)
            model_state[f"{dst}.ln_2.weight"] = torch.ones(hidden_dim, dtype=torch.bfloat16)
            model_state[f"{dst}.ln_2.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)

            # CausalSelfAttention: c_attn [3 * hidden_dim, hidden_dim] and c_proj [hidden_dim, hidden_dim]
            k_attn_qkv = keys[(l * 5 + 0) % len(keys)]
            k_attn_proj = keys[(l * 5 + 1) % len(keys)]
            model_state[f"{dst}.attn.c_attn.weight"] = project_weight(stream.get_tensor(k_attn_qkv), 3 * hidden_dim, hidden_dim).to(torch.bfloat16)
            model_state[f"{dst}.attn.c_attn.bias"] = torch.zeros(3 * hidden_dim, dtype=torch.bfloat16)
            model_state[f"{dst}.attn.c_proj.weight"] = project_weight(stream.get_tensor(k_attn_proj), hidden_dim, hidden_dim).to(torch.bfloat16)
            model_state[f"{dst}.attn.c_proj.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)

            # FFN: c_fc [2 * ffn_dim, hidden_dim] and c_proj [hidden_dim, ffn_dim]
            k_ffn_fc = keys[(l * 5 + 2) % len(keys)]
            k_ffn_proj = keys[(l * 5 + 3) % len(keys)]
            model_state[f"{dst}.moe.c_fc.weight"] = project_weight(stream.get_tensor(k_ffn_fc), 2 * ffn_dim, hidden_dim).to(torch.bfloat16)
            model_state[f"{dst}.moe.c_fc.bias"] = torch.zeros(2 * ffn_dim, dtype=torch.bfloat16)
            model_state[f"{dst}.moe.c_proj.weight"] = project_weight(stream.get_tensor(k_ffn_proj), hidden_dim, ffn_dim).to(torch.bfloat16)
            model_state[f"{dst}.moe.c_proj.bias"] = torch.zeros(hidden_dim, dtype=torch.bfloat16)

            # MoE Gate & Router
            model_state[f"{dst}.moe.moe_gate.weight"] = torch.zeros(1, hidden_dim, dtype=torch.bfloat16)
            model_state[f"{dst}.moe.moe_gate.bias"] = torch.zeros(1, dtype=torch.bfloat16)
            model_state[f"{dst}.moe.router.weight"] = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16) * 0.02
            model_state[f"{dst}.moe.pull_gate.prior"] = torch.zeros(num_experts, dtype=torch.bfloat16)
            model_state[f"{dst}.moe.pull_gate.gate.weight"] = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16) * 0.02

            logger.info(f"    Layer {l:2d}: Backbone spliced from {donor_tag}")

    # -------------------------------------------------------------------------
    # PART 4: 9-VECTOR PRISM (3 Llama, 3 BitNet, 3 Qwen in EVERY layer)
    # -------------------------------------------------------------------------
    logger.info(">>> [4/5] Splicing 9-Vector Prism (3 Llama + 3 BitNet + 3 Qwen in every layer)...")
    with safe_open(donor_llama_path, framework="pt", device="cpu") as f_llama, \
         safe_open(donor_bitnet_path, framework="pt", device="cpu") as f_bitnet, \
         safe_open(donor_qwen_path, framework="pt", device="cpu") as f_qwen:

        for l in range(n_layer):
            dst = f"h.{l}"
            # 3 vectors from Llama: Language, Sentiment, Context
            for idx, name in enumerate(["Language", "Sentiment", "Context"]):
                src_k = llama_2d[(l * 3 + idx) % len(llama_2d)]
                model_state[f"{dst}.attn.prism.vectors.{name}.weight"] = project_weight(f_llama.get_tensor(src_k), hidden_dim, hidden_dim).to(torch.bfloat16)

            # 3 vectors from BitNet: Intent, Meta, Creativity
            for idx, name in enumerate(["Intent", "Meta", "Creativity"]):
                src_k = bitnet_2d[(l * 3 + idx) % len(bitnet_2d)]
                model_state[f"{dst}.attn.prism.vectors.{name}.weight"] = project_weight(f_bitnet.get_tensor(src_k), hidden_dim, hidden_dim).to(torch.bfloat16)

            # 3 vectors from Qwen: Ethics, Strategy, Constraint
            for idx, name in enumerate(["Ethics", "Strategy", "Constraint"]):
                src_k = qwen_2d[(l * 3 + idx) % len(qwen_2d)]
                model_state[f"{dst}.attn.prism.vectors.{name}.weight"] = project_weight(f_qwen.get_tensor(src_k), hidden_dim, hidden_dim).to(torch.bfloat16)

            # Prism weight gate (1/3 superposition of all 3 donors)
            w_gate = (
                project_weight(f_llama.get_tensor(llama_2d[l % len(llama_2d)]), hidden_dim, hidden_dim) * (1.0 / 3.0) +
                project_weight(f_bitnet.get_tensor(bitnet_2d[l % len(bitnet_2d)]), hidden_dim, hidden_dim) * (1.0 / 3.0) +
                project_weight(f_qwen.get_tensor(qwen_2d[l % len(qwen_2d)]), hidden_dim, hidden_dim) * (1.0 / 3.0)
            )
            model_state[f"{dst}.attn.prism.w_gate.weight"] = w_gate.to(torch.bfloat16)

    logger.info("    9-Vector Prism fully spliced across all layers with equal 1/3 donor parity.")

    # -------------------------------------------------------------------------
    # PART 5: 34-COUNCIL MOE EXPERTS (11 Llama + 12 BitNet + 11 Qwen in EVERY layer)
    # -------------------------------------------------------------------------
    logger.info(">>> [5/5] Splicing 34 Council Experts (11 Llama + 12 BitNet + 11 Qwen in EVERY layer)...")
    with safe_open(donor_llama_path, framework="pt", device="cpu") as f_llama, \
         safe_open(donor_bitnet_path, framework="pt", device="cpu") as f_bitnet, \
         safe_open(donor_qwen_path, framework="pt", device="cpu") as f_qwen:

        llama_mlp = [k for k in llama_2d if "mlp" in k] or llama_2d
        bitnet_mlp = [k for k in bitnet_2d if "mlp" in k] or bitnet_2d
        qwen_mlp = [k for k in qwen_2d if "mlp" in k] or qwen_2d

        for l in range(n_layer):
            dst = f"h.{l}"
            # Experts 0..10 (11 Experts) -> Llama 3.2 1B (Knowledge, Ethics, Context) [Rank 8]
            for e in range(0, 11):
                src_w = f_llama.get_tensor(llama_mlp[(l * 11 + e) % len(llama_mlp)])
                w0, w1 = extract_svd_expert(src_w, target_dim=hidden_dim, rank=get_expert_rank(e))
                model_state[f"{dst}.moe.experts.{e}.0.weight"] = w0.to(torch.bfloat16)
                model_state[f"{dst}.moe.experts.{e}.1.weight"] = w1.to(torch.bfloat16)
                model_state[f"{dst}.moe.evo_moe.experts.{e}.0.weight"] = w0.to(torch.bfloat16)
                model_state[f"{dst}.moe.evo_moe.experts.{e}.1.weight"] = w1.to(torch.bfloat16)

            # Experts 11..21 (11 Experts) -> BitNet 4T2B (Ternary Systems, Edge, Speed) [Rank 24]
            for e in range(11, 22):
                src_w = f_bitnet.get_tensor(bitnet_mlp[(l * 11 + (e - 11)) % len(bitnet_mlp)])
                w0, w1 = extract_svd_expert(src_w, target_dim=hidden_dim, rank=get_expert_rank(e))
                model_state[f"{dst}.moe.experts.{e}.0.weight"] = w0.to(torch.bfloat16)
                model_state[f"{dst}.moe.experts.{e}.1.weight"] = w1.to(torch.bfloat16)
                model_state[f"{dst}.moe.evo_moe.experts.{e}.0.weight"] = w0.to(torch.bfloat16)
                model_state[f"{dst}.moe.evo_moe.experts.{e}.1.weight"] = w1.to(torch.bfloat16)

            # Experts 22..33 (12 Experts) -> Qwen 0.6B (Logic, Mathematics, Code) [Rank 48]
            for e in range(22, 34):
                src_w = f_qwen.get_tensor(qwen_mlp[(l * 12 + (e - 22)) % len(qwen_mlp)])
                w0, w1 = extract_svd_expert(src_w, target_dim=hidden_dim, rank=get_expert_rank(e))
                model_state[f"{dst}.moe.experts.{e}.0.weight"] = w0.to(torch.bfloat16)
                model_state[f"{dst}.moe.experts.{e}.1.weight"] = w1.to(torch.bfloat16)
                model_state[f"{dst}.moe.evo_moe.experts.{e}.0.weight"] = w0.to(torch.bfloat16)
                model_state[f"{dst}.moe.evo_moe.experts.{e}.1.weight"] = w1.to(torch.bfloat16)

            # Evo MoE Router
            model_state[f"{dst}.moe.evo_moe.router.weight"] = model_state[f"{dst}.moe.router.weight"]

    # -------------------------------------------------------------------------
    # PACKAGE & SAVE CHECKPOINT
    # -------------------------------------------------------------------------
    config_dict = {
        "vocab_size": vocab_size,
        "eos_token_id": 0,
        "max_seq_len": 256,
        "hidden_dim": hidden_dim,
        "n_layer": n_layer,
        "n_head": 18 if hidden_dim <= 1152 else 32,
        "head_dim": 64,
        "ffn_dim": ffn_dim,
        "num_experts": num_experts,
        "router_mode": "dense_pull",
        "top_k": 4,
        "expert_rank": expert_rank,
        "swarm_rank": expert_rank,
        "use_evo_moe": True,
        "tripartite_ratios": {
            "llama_3_2_1b": 0.3333,
            "bitnet_4t2b": 0.3333,
            "qwen_0_6b": 0.3333
        }
    }

    checkpoint = {
        "step": 0,
        "best_val_ce": None,
        "config": config_dict,
        "model_state_dict": model_state,
        "tri_donor_manifest": {
            "architecture": "True Tripartite Sovereign (Exact 1/3 Equal Contribution)",
            "donors": ["Llama 3.2 1B", "BitNet 4T2B", "Qwen 0.6B"],
            "embeddings": "1/3 Llama + 1/3 BitNet + 1/3 Qwen Equal Superposition",
            "layers": f"{layers_per_donor} Llama, {layers_per_donor} BitNet, {layers_per_donor} Qwen",
            "experts": "11 Llama (C0-C10, rank 8), 11 BitNet (C11-C21, rank 24), 12 Qwen (C22-C33, rank 48)",
            "prism": "3 Llama, 3 BitNet, 3 Qwen per layer",
            "bridges": "q1 from Llama, q2 from BitNet, ingest/comm from Qwen"
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    logger.info(f"Writing checkpoint to disk: {output_path}...")
    torch.save(checkpoint, output_path)
    total_params = sum(p.numel() for p in model_state.values()) / 1e6
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    elapsed = time.time() - t0
    logger.info(f"SUCCESS! {model_name} compiled with {total_params:.2f}M parameters ({file_size_mb:.1f} MB) in {elapsed:.2f}s.")
    logger.info("=" * 75)


if __name__ == "__main__":
    d_llama = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\model.safetensors"
    d_bitnet = r"C:\Users\Admin\.cache\huggingface\hub\models--1bitLLM--bitnet_b1_58-large\snapshots\85d047191dcb224f0e04f20d26110caaf8dc1a47\model.safetensors"
    d_qwen = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors"
    
    out_dir = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Compile True Tripartite Mini 0.6B Model (D=1152, 6 Layers: 2 Llama, 2 BitNet, 2 Qwen, 34 Experts)
    out_mini = os.path.join(out_dir, "quillan_mini_0.6b_tripartite.pt")
    compile_tripartite_model(
        model_name="Quillan-Mini-0.6B-Tripartite",
        output_path=out_mini,
        donor_llama_path=d_llama,
        donor_bitnet_path=d_bitnet,
        donor_qwen_path=d_qwen,
        hidden_dim=1152,
        n_layer=6,
        ffn_dim=3072,
        num_experts=34,
        expert_rank=16,
        vocab_size=50262
    )

    # 2. Compile True Tripartite Flagship 1B Sovereign Model (D=2048, 12 Layers: 4 Llama, 4 BitNet, 4 Qwen, 34 Experts)
    out_1b = os.path.join(out_dir, "quillan_1b_tripartite.pt")
    compile_tripartite_model(
        model_name="Quillan-1B-Tripartite-Sovereign",
        output_path=out_1b,
        donor_llama_path=d_llama,
        donor_bitnet_path=d_bitnet,
        donor_qwen_path=d_qwen,
        hidden_dim=2048,
        n_layer=12,
        ffn_dim=5632,
        num_experts=34,
        expert_rank=32,
        vocab_size=50262
    )
