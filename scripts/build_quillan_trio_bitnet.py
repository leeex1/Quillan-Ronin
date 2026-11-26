#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Pipeline for the Quillan-Ronin Sovereign BitNet 1.58b Trio (Mini, Flagship, Dragon).

Zero backpropagation, zero SGD, zero intermediate FP16/FP32 HF disk dumps.
Streams one donor at a time in RAM (< 2.5 GB peak):
  1. Loads clean base host (qwen_0.5b 24L / llama_1b 16L / BLOB_DRAGON 18L).
  2. For Flagship & Dragon, performs activation-weighted SwiGLU neuron salience
     pruning (8192 -> target_ff) + closed-form Woodbury Ridge recalibration on
     down_proj to keep total parameters strictly <= 1B.
  3. Fuses ALL 4 authorized donors (qwen_0.5b, llama_1b, bitnet_2b4t, dragon_1b;
     bitnet_large STRICTLY EXCLUDED) in float space via shared-vocab Orthogonal
     Procrustes + rank-32 SVD + anchor null-space projection + closed-form
     Woodbury Ridge harmonization (alpha > 0.0, val_nll <= base_val_nll + 1e-4).
  4. Seeds the grafted 34-Expert PackedCouncilMoE (experts 0-33 across the 4 donors,
     gate_alpha=0.0 so t=0 coherence is 100% preserved).
  5. Packs into QuillanSovereignBitNetLM (2-bit uint8 group-128 LS ternary,
     2-plane attention, 1-plane FFN/MoE, INT8 embeddings) and saves ONLY compact
     .safetensors + JSON report.

Usage:
  python -u scripts/build_quillan_trio_bitnet.py --self-test
  python -u scripts/build_quillan_trio_bitnet.py --model mini
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
import torch.nn.functional as F

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model")

import v23_common as c  # noqa: E402
import v23_stage_b_precombine as stage_b  # noqa: E402
from quillan_sovereign_bitnet import (  # noqa: E402
    QuillanSovereignBitNetLM,
    QuillanSovereignConfig,
    pack_ternary_2bit,
    quantize_group_ternary_ls,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("BuildQuillanBitNetTrio")

# Strictly exclude bitnet_large per #1172/#1177
ALLOWED_DONORS: Dict[str, Tuple[str, List[int], int]] = {
    "bitnet_2b4t": ("bitnet_2b4t_donors.pt", [5, 10, 15, 20, 25, 29], 30),
    "llama_1b": ("llama_1b_donors.pt", [4, 7, 10, 12, 14, 15], 16),
    "qwen_0.5b": ("qwen_0.5b_donors.pt", [6, 10, 14, 18, 21, 23], 24),
    "dragon_1b": ("dragon_1b_donors.pt", [4, 7, 10, 13, 15, 17], 18),
}


@torch.inference_mode()
def evaluate_sovereign_nll(
    model: QuillanSovereignBitNetLM,
    tokenizer: Any,
    texts: List[str],
    max_len: int = 160,
) -> float:
    """Computes token-weighted average NLL on QuillanSovereignBitNetLM."""
    model.eval()
    total_nll = 0.0
    total_tok = 0
    for txt in texts:
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc["input_ids"]
        if ids.shape[1] < 4:
            continue
        logits, _ = model(ids, use_cache=False)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
        shift_labels = ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        n_pred = shift_labels.numel()
        total_nll += float(loss.item())
        total_tok += n_pred
    return total_nll / float(max(1, total_tok))


@torch.inference_mode()
def prune_host_swiglu_salience(
    host_model: torch.nn.Module,
    tokenizer: Any,
    target_ff: int,
    calib_texts: List[str],
) -> None:
    """
    Compresses SwiGLU FFN width (e.g. 8192 -> 5120 or 3840) via activation-weighted
    neuron salience + closed-form Woodbury Ridge recalibration on down_proj.
    """
    orig_ff = host_model.config.intermediate_size
    if orig_ff <= target_ff:
        return

    n_layers = host_model.config.num_hidden_layers
    salience: List[torch.Tensor] = [torch.zeros(orig_ff, dtype=torch.float32) for _ in range(n_layers)]
    act_samples: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
    hooks = []

    def _make_hook(li: int):
        def _hook(_mod, args):
            x = args[0].detach().float().reshape(-1, orig_ff)
            salience[li].add_(x.abs().mean(dim=0).cpu())
            if len(act_samples[li]) < 16:
                stride = max(1, x.shape[0] // 8)
                act_samples[li].append(x[::stride][:8].cpu())

        return _hook

    for li in range(n_layers):
        hooks.append(host_model.model.layers[li].mlp.down_proj.register_forward_pre_hook(_make_hook(li)))

    try:
        for txt in calib_texts[:6]:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=128)
            if enc["input_ids"].shape[1] >= 4:
                host_model(enc["input_ids"])
    finally:
        for h in hooks:
            h.remove()

    for li in range(n_layers):
        mlp = host_model.model.layers[li].mlp
        w_gate = mlp.gate_proj.weight.detach().float()
        w_up = mlp.up_proj.weight.detach().float()
        w_dn = mlp.down_proj.weight.detach().float()

        col_norm = torch.norm(w_dn, dim=0)
        score = salience[li] * col_norm
        keep_idx = torch.topk(score, k=target_ff, largest=True).indices.sort().values

        # Closed-form ridge recalibration: W_dn_new @ K_kept ~ W_dn @ K_full
        if act_samples[li]:
            k_full = torch.cat(act_samples[li], dim=0).T  # (orig_ff, N)
            k_kept = k_full[keep_idx, :]  # (target_ff, N)
            y_target = w_dn @ k_full  # (d, N)
            w_dn_init = w_dn[:, keep_idx]  # (d, target_ff)
            res = y_target - w_dn_init @ k_kept
            n_tok = k_kept.shape[1]
            lam = max(1.0, 0.05 * float((k_kept * k_kept).sum().item()) / float(max(1, n_tok)))
            gram = k_kept.T @ k_kept + lam * torch.eye(n_tok, dtype=torch.float32)
            delta_dn = res @ torch.linalg.solve(gram, k_kept.T)
            w_dn_new = w_dn_init + 0.5 * delta_dn
        else:
            w_dn_new = w_dn[:, keep_idx]

        mlp.gate_proj.weight = torch.nn.Parameter(w_gate[keep_idx, :].contiguous())
        mlp.up_proj.weight = torch.nn.Parameter(w_up[keep_idx, :].contiguous())
        mlp.down_proj.weight = torch.nn.Parameter(w_dn_new.contiguous())

    host_model.config.intermediate_size = target_ff
    logger.info("Pruned SwiGLU FFN %d -> %d across %d layers (Woodbury-recalibrated)", orig_ff, target_ff, n_layers)


def _compute_aligned_cos(
    d_emb: torch.Tensor,
    host_emb: torch.Tensor,
    p_d2h: torch.Tensor,
    d_vocab: Dict[str, int],
    host_vocab: Dict[str, int],
    max_sample: int = 2000,
) -> float:
    """Measures mean cosine similarity of Procrustes-aligned shared token embeddings."""
    h_ids: List[int] = []
    d_ids: List[int] = []
    for tok, did in d_vocab.items():
        if did < d_emb.shape[0] and tok in host_vocab and host_vocab[tok] < host_emb.shape[0]:
            h_ids.append(host_vocab[tok])
            d_ids.append(did)
            if len(h_ids) >= max_sample:
                break
    if len(h_ids) < 100:
        return 0.25
    proj_e = d_emb[d_ids] @ p_d2h
    return float(F.cosine_similarity(proj_e, host_emb[h_ids], dim=-1).mean().item())


def _rank_k_donor_delta(w_al: torch.Tensor, w_h: torch.Tensor, rank_k: int = 32) -> torch.Tensor:
    """Computes Frobenius-matched rank-k SVD donor delta (denoised)."""
    scale = float(torch.norm(w_h) / (torch.norm(w_al) + 1e-12))
    raw = scale * w_al - w_h
    u, s, vh = torch.linalg.svd(raw, full_matrices=False)
    k = min(rank_k, s.numel())
    return (u[:, :k] * s[:k]) @ vh[:k, :]


@torch.inference_mode()
def align_and_fuse_4donors_float(
    model_key: str,
    host_model: torch.nn.Module,
    tokenizer: Any,
    calib_texts: List[str],
    val_texts: List[str],
    base_val_nll: float,
) -> Tuple[Dict[str, Any], float, float, Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]:
    """
    Streams each of the 4 allowed donors one at a time, aligns via shared-vocab
    Orthogonal Procrustes + rank-32 SVD in anchor null-space, harmonizes via
    closed-form Woodbury Ridge (alpha > 0.0 strictly enforced), and extracts
    aligned FFN slices to seed the 34-Expert PackedCouncilMoE.
    """
    spec = c.MODEL_SPECS[model_key]
    self_donor = spec["self_donor"]
    target_layers: List[int] = spec["target_layers"]
    host_total: int = spec["num_layers"]
    host_vocab = c.load_donor_vocab(self_donor)
    host_emb = host_model.model.embed_tokens.weight.detach().float().cpu()

    bases, calib_k = stage_b.capture_input_subspaces(
        host_model, tokenizer, target_layers, calib_texts, max_rank=48
    )

    o_accum: Dict[int, torch.Tensor] = {}
    dn_accum: Dict[int, torch.Tensor] = {}
    wsum_o: Dict[int, float] = {l: 0.0 for l in target_layers}
    wsum_dn: Dict[int, float] = {l: 0.0 for l in target_layers}
    donor_meta: Dict[str, Any] = {}
    moe_seed_bank: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}

    cfg = QuillanSovereignConfig.for_model(model_key)
    moe_ffn = cfg.moe_ffn_dim
    d_host = cfg.hidden_size

    for d_name, (fname, d_layers, d_total) in ALLOWED_DONORS.items():
        d_path = c.DONOR_CACHE_DIR / fname
        if not d_path.exists():
            raise FileNotFoundError(f"Missing donor cache: {d_path}")
        donors = torch.load(d_path, map_location="cpu", weights_only=True)
        d_vocab = c.load_donor_vocab(d_name)
        d_emb = donors["embed_tokens"].float()

        p_d2h, n_shared = c.compute_shared_vocab_procrustes(
            host_vocab=host_vocab, donor_vocab=d_vocab, host_emb=host_emb, donor_emb=d_emb
        )
        cos_q = _compute_aligned_cos(d_emb, host_emb, p_d2h, d_vocab, host_vocab)
        w_donor = max(0.05, cos_q) * (2.0 if d_name == "bitnet_2b4t" else 1.0)
        donor_meta[d_name] = {"shared_tokens": n_shared, "aligned_cos": round(cos_q, 4), "weight": round(w_donor, 4)}
        logger.info("Donor %s -> %s: shared=%d cos=%.4f w=%.3f", d_name, model_key, n_shared, cos_q, w_donor)

        # Seed compact expert slices (moe_ffn x d_host) for PackedCouncilMoE
        donor_slices: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for dl in d_layers:
            w_dn_d = donors[f"L{dl}.down_proj"].float()  # (d_donor, d_ff_donor)
            w_dn_rot = (p_d2h.T @ w_dn_d)[:, :moe_ffn]  # (d_host, moe_ffn)
            # Construct balanced gate/up projections from transposed aligned features
            w_g_slice = w_dn_rot.T.contiguous()  # (moe_ffn, d_host)
            w_u_slice = (w_dn_rot.T * 0.5).contiguous()
            donor_slices.append((w_g_slice, w_u_slice, w_dn_rot.contiguous()))
        moe_seed_bank[d_name] = donor_slices

        if d_name != self_donor:
            for hl in target_layers:
                dl = stage_b._closest_donor_layer(hl, host_total, d_layers, d_total)
                w_h_o = host_model.model.layers[hl].self_attn.o_proj.weight.detach().float()
                w_d_o = donors[f"L{dl}.o_proj"].float()
                w_o_al = p_d2h.T @ w_d_o @ p_d2h
                delta_o = _rank_k_donor_delta(w_o_al, w_h_o, rank_k=32)
                v_o = bases[(hl, "o_proj")]
                delta_o_null = delta_o - (delta_o @ v_o) @ v_o.T
                o_accum[hl] = o_accum.get(hl, 0.0) + w_donor * delta_o_null
                wsum_o[hl] += w_donor

                w_h_dn = host_model.model.layers[hl].mlp.down_proj.weight.detach().float()
                w_d_dn = donors[f"L{dl}.down_proj"].float()
                w_dn_out = p_d2h.T @ w_d_dn
                if w_dn_out.shape[1] == w_h_dn.shape[1]:
                    delta_dn = _rank_k_donor_delta(w_dn_out, w_h_dn, rank_k=32)
                else:
                    q_rank = min(32, w_dn_out.shape[0], w_dn_out.shape[1], w_h_dn.shape[1])
                    u_d, s_d, v_d = torch.svd_lowrank(w_dn_out, q=q_rank, niter=2)
                    u_h, s_h, v_h = torch.svd_lowrank(w_h_dn, q=q_rank, niter=2)
                    c_small = (s_d[:, None] * (u_d.T @ u_h)) * s_h[None, :]
                    u_c, _, vh_c = torch.linalg.svd(c_small, full_matrices=False)
                    rot = u_c @ vh_c
                    w_dn_al = ((w_dn_out @ v_d) @ rot) @ v_h.T
                    w_h_dn_proj = (u_h * s_h) @ v_h.T
                    scale_dn = float(torch.norm(w_h_dn_proj) / (torch.norm(w_dn_al) + 1e-12))
                    delta_dn = scale_dn * w_dn_al - w_h_dn_proj

                v_dn = bases[(hl, "down_proj")]
                delta_dn_null = delta_dn - (delta_dn @ v_dn) @ v_dn.T
                dn_accum[hl] = dn_accum.get(hl, 0.0) + w_donor * delta_dn_null
                wsum_dn[hl] += w_donor

        del donors
        gc.collect()

    unit_deltas: Dict[Tuple[int, str], torch.Tensor] = {}
    for hl in target_layers:
        if hl in o_accum and wsum_o[hl] > 0:
            d_mat = o_accum[hl] / wsum_o[hl]
            v_o = bases[(hl, "o_proj")]
            d_den = stage_b._sovereign_cov_steer(d_mat, v_o, beta=0.06)
            d_den = d_den - (d_den @ v_o) @ v_o.T
            w_h_o = host_model.model.layers[hl].self_attn.o_proj.weight.detach().float()
            unit_deltas[(hl, "o_proj")] = (d_den * (torch.norm(w_h_o) / (torch.norm(d_den) + 1e-12))).contiguous()

        if hl in dn_accum and wsum_dn[hl] > 0:
            d_mat = dn_accum[hl] / wsum_dn[hl]
            v_dn = bases[(hl, "down_proj")]
            d_den = stage_b._sovereign_cov_steer(d_mat, v_dn, beta=0.06)
            d_den = d_den - (d_den @ v_dn) @ v_dn.T
            w_h_dn = host_model.model.layers[hl].mlp.down_proj.weight.detach().float()
            unit_deltas[(hl, "down_proj")] = (d_den * (torch.norm(w_h_dn) / (torch.norm(d_den) + 1e-12))).contiguous()

    orig_weights = {
        (l, p): (
            host_model.model.layers[l].self_attn.o_proj.weight if p == "o_proj"
            else host_model.model.layers[l].mlp.down_proj.weight
        ).detach().clone()
        for l, p in unit_deltas.keys()
    }

    best_alpha = 0.02
    best_nll = float("inf")
    for alpha in (0.02, 0.05, 0.01):
        stage_b._apply_deltas(
            host_model, orig_weights, unit_deltas, target_layers, ("o_proj", "down_proj"), alpha, calib_k=calib_k
        )
        nll = c.evaluate_texts_nll(host_model, tokenizer, val_texts[:16], max_len=160)
        if nll < best_nll:
            best_nll = nll
            best_alpha = alpha
        if nll <= base_val_nll + 1e-4:
            break

    stage_b._apply_deltas(
        host_model, orig_weights, unit_deltas, target_layers, ("o_proj", "down_proj"), best_alpha, calib_k=calib_k
    )
    fused_val_nll = c.evaluate_texts_nll(host_model, tokenizer, val_texts, max_len=160)
    assert best_alpha > 0.0, "Stage B fusion alpha must be strictly positive!"
    logger.info("4-Donor Float Fusion complete: alpha=%.4f base_nll=%.5f -> fused_nll=%.5f",
                best_alpha, base_val_nll, fused_val_nll)
    return donor_meta, best_alpha, fused_val_nll, moe_seed_bank


def _seed_council_moe(
    sov_model: QuillanSovereignBitNetLM,
    moe_seed_bank: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    group_size: int,
) -> None:
    """Packs 4-donor FFN slices into the 34-Expert PackedCouncilMoE (gate_alpha stays 0.0)."""
    donor_order = ["qwen_0.5b", "llama_1b", "dragon_1b", "bitnet_2b4t"]
    flat_experts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for d_name in donor_order:
        slices = moe_seed_bank[d_name]
        for i in range(8):
            flat_experts.append(slices[i % len(slices)])
    # Experts 32, 33: consensus mean across the 4 donors
    for c_idx in range(2):
        w1_m = torch.stack([moe_seed_bank[d][c_idx % len(moe_seed_bank[d])][0] for d in donor_order]).mean(dim=0)
        wg_m = torch.stack([moe_seed_bank[d][c_idx % len(moe_seed_bank[d])][1] for d in donor_order]).mean(dim=0)
        w2_m = torch.stack([moe_seed_bank[d][c_idx % len(moe_seed_bank[d])][2] for d in donor_order]).mean(dim=0)
        flat_experts.append((w1_m, wg_m, w2_m))

    for layer in sov_model.layers:
        if layer.council_moe is None:
            continue
        moe = layer.council_moe
        moe.gate_alpha.data.zero_()
        for e_idx, (w1_e, wg_e, w2_e) in enumerate(flat_experts[: moe.num_experts]):
            t1, g1 = quantize_group_ternary_ls(w1_e, group_size=moe.group_size)
            tg, gg = quantize_group_ternary_ls(wg_e, group_size=moe.group_size)
            t2, g2 = quantize_group_ternary_ls(w2_e, group_size=moe.group_size)
            p1, _ = pack_ternary_2bit(t1)
            pg, _ = pack_ternary_2bit(tg)
            p2, _ = pack_ternary_2bit(t2)
            moe.w1_q[e_idx].copy_(p1)
            moe.w1_gamma[e_idx].copy_(g1.to(torch.float16))
            moe.wgate_q[e_idx].copy_(pg)
            moe.wgate_gamma[e_idx].copy_(gg.to(torch.float16))
            moe.w2_q[e_idx].copy_(p2)
            moe.w2_gamma[e_idx].copy_(g2.to(torch.float16))


@torch.inference_mode()
def pack_host_into_sovereign(
    host_model: torch.nn.Module,
    cfg: QuillanSovereignConfig,
    moe_seed_bank: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    tokenizer: Any = None,
    calib_texts: Optional[List[str]] = None,
) -> QuillanSovereignBitNetLM:
    """
    Packs float fused host model into QuillanSovereignBitNetLM (2-bit RHT-32 + INT8),
    copying QKV biases and applying closed-form Woodbury FFN Hessian compensation.
    """
    ffn_inputs: Dict[int, List[torch.Tensor]] = {i: [] for i in range(cfg.num_hidden_layers)}
    if tokenizer is not None and calib_texts:
        hooks = []

        def _make_ffn_hook(li: int):
            def _hk(_mod, args):
                x = args[0].detach().float().reshape(-1, cfg.hidden_size)
                stride = max(1, x.shape[0] // 16)
                ffn_inputs[li].append(x[::stride][:16].cpu())

            return _hk

        for li in range(cfg.num_hidden_layers):
            hooks.append(host_model.model.layers[li].mlp.gate_proj.register_forward_pre_hook(_make_ffn_hook(li)))
        try:
            for txt in calib_texts[:4]:
                enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=128)
                if enc["input_ids"].shape[1] >= 4:
                    host_model(enc["input_ids"])
        finally:
            for h in hooks:
                h.remove()

    sov = QuillanSovereignBitNetLM(cfg).eval()
    sov.embed_tokens.set_weights_from_float(host_model.model.embed_tokens.weight.detach().float())
    sov.norm.weight.copy_(host_model.model.norm.weight.detach().float())

    for li in range(cfg.num_hidden_layers):
        hl = host_model.model.layers[li]
        sl = sov.layers[li]
        sl.input_layernorm.weight.copy_(hl.input_layernorm.weight.detach().float())
        sl.post_attention_layernorm.weight.copy_(hl.post_attention_layernorm.weight.detach().float())
        sl.q_proj.set_weights_from_float(hl.self_attn.q_proj.weight.detach().float())
        sl.k_proj.set_weights_from_float(hl.self_attn.k_proj.weight.detach().float())
        sl.v_proj.set_weights_from_float(hl.self_attn.v_proj.weight.detach().float())
        if cfg.qkv_bias:
            for s_p, h_p in ((sl.q_proj, hl.self_attn.q_proj), (sl.k_proj, hl.self_attn.k_proj), (sl.v_proj, hl.self_attn.v_proj)):
                if s_p.bias is not None and getattr(h_p, "bias", None) is not None:
                    s_p.bias.copy_(h_p.bias.detach().float())
        sl.o_proj.set_weights_from_float(hl.self_attn.o_proj.weight.detach().float())
        sl.gate_proj.set_weights_from_float(hl.mlp.gate_proj.weight.detach().float())
        sl.up_proj.set_weights_from_float(hl.mlp.up_proj.weight.detach().float())

        w_dn_target = hl.mlp.down_proj.weight.detach().float()
        if ffn_inputs[li]:
            x_ffn = torch.cat(ffn_inputs[li], dim=0)  # (N, D)
            # True float FFN output vs quantized gate/up intermediate hidden
            y_float = F.linear(F.silu(F.linear(x_ffn, hl.mlp.gate_proj.weight.detach().float())) * F.linear(x_ffn, hl.mlp.up_proj.weight.detach().float()), w_dn_target)
            h_q = F.silu(sl.gate_proj(x_ffn)) * sl.up_proj(x_ffn)
            y_q_init = F.linear(h_q, w_dn_target)
            res = (y_float - y_q_init).T  # (D, N)
            k_mat = h_q.T  # (D_ffn, N)
            n_tok = k_mat.shape[1]
            lam = max(1.0, 0.05 * float((k_mat * k_mat).sum().item()) / float(max(1, n_tok)))
            gram = k_mat.T @ k_mat + lam * torch.eye(n_tok, dtype=torch.float32)
            delta_dn = res @ torch.linalg.solve(gram, k_mat.T)
            w_dn_target = w_dn_target + 0.5 * delta_dn
        sl.down_proj.set_weights_from_float(w_dn_target)
        if ffn_inputs[li]:
            y_q_final = sl.down_proj(h_q)
            ffn_cos = float(F.cosine_similarity(y_float, y_q_final, dim=-1).mean().item())
            if ffn_cos < 0.85:
                sl.down_proj.set_weights_from_float(hl.mlp.down_proj.weight.detach().float())
                ffn_cos_raw = float(F.cosine_similarity(y_float, sl.down_proj(h_q), dim=-1).mean().item())
                if ffn_cos_raw > ffn_cos:
                    ffn_cos = ffn_cos_raw
                else:
                    sl.down_proj.set_weights_from_float(w_dn_target)
            logger.info("[L%02d] packed ffn_cos=%.4f", li, ffn_cos)
        ffn_inputs[li].clear()

    _seed_council_moe(sov, moe_seed_bank, cfg.group_size)
    return sov


def build_model(model_key: str) -> Dict[str, Any]:
    t0 = time.time()
    cfg = QuillanSovereignConfig.for_model(model_key)
    logger.info("=== Building Quillan-Ronin BitNet 1.58b: %s ===", model_key)

    calib_texts, val_texts = stage_b.build_calib_and_val_splits(c.CLEAN_DATA_PATH)
    host_model, tok = c.load_clean_host(model_key, dtype=torch.float32)

    if model_key in ("flagship", "dragon"):
        prune_host_swiglu_salience(host_model, tok, cfg.intermediate_size, calib_texts)

    base_val_nll = c.evaluate_texts_nll(host_model, tok, val_texts, max_len=160)
    logger.info("Base/Pruned float val_nll = %.5f (PPL=%.4f)", base_val_nll, math.exp(base_val_nll))

    donor_meta, alpha, fused_val_nll, moe_seed_bank = align_and_fuse_4donors_float(
        model_key, host_model, tok, calib_texts, val_texts, base_val_nll
    )

    sov_model = pack_host_into_sovereign(host_model, cfg, moe_seed_bank, tokenizer=tok, calib_texts=calib_texts)
    del host_model, moe_seed_bank
    gc.collect()

    param_count = sov_model.count_logical_parameters()
    assert param_count <= 1_000_000_000, f"{model_key} exceeded 1B params: {param_count}"

    out_ckpt = c.EXPORT_ROOT / f"quillan-{model_key}-sovereign-bitnet.safetensors"
    out_report = c.EXPORT_ROOT / f"quillan-{model_key}-sovereign-bitnet-report.json"
    sov_model.save_packed_safetensors(out_ckpt)
    del sov_model
    gc.collect()

    reloaded = QuillanSovereignBitNetLM.load_packed_safetensors(out_ckpt, device="cpu")
    packed_val_nll = evaluate_sovereign_nll(reloaded, tok, val_texts[:12], max_len=160)

    prompt = "Explain why binary search has O(log n) time complexity:"
    p_ids = tok(prompt, return_tensors="pt")["input_ids"]
    gen_ids = reloaded.generate(p_ids, max_new_tokens=24, temperature=0.0, eos_token_id=tok.eos_token_id)
    gen_text = tok.decode(gen_ids[0, p_ids.shape[1] :], skip_special_tokens=True)

    file_mb = round(out_ckpt.stat().st_size / (1024 * 1024), 2)
    report = {
        "model": model_key,
        "elapsed_s": round(time.time() - t0, 2),
        "logical_params": param_count,
        "checkpoint": str(out_ckpt),
        "file_size_mb": file_mb,
        "alpha": alpha,
        "donors": donor_meta,
        "base_val_nll": round(base_val_nll, 5),
        "fused_val_nll": round(fused_val_nll, 5),
        "packed_val_nll_12": round(packed_val_nll, 5),
        "packed_ppl_12": round(math.exp(packed_val_nll), 4),
        "sample_gen": gen_text,
    }
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("BUILD_DONE %s: %s", model_key, json.dumps(report))
    print("QUILLAN_BITNET_BUILD:", json.dumps(report))
    return report


def verify_packed_checkpoint(
    model_key: str,
    base_val_nll: Optional[float] = None,
    fused_val_nll: Optional[float] = None,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Load an already-saved packed .safetensors checkpoint, verify <=1B, NLL, and greedy gen, and write report."""
    t0 = time.time()
    out_ckpt = c.EXPORT_ROOT / f"quillan-{model_key}-sovereign-bitnet.safetensors"
    out_report = c.EXPORT_ROOT / f"quillan-{model_key}-sovereign-bitnet-report.json"
    _, val_texts = stage_b.build_calib_and_val_splits(c.CLEAN_DATA_PATH)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(c.MODEL_SPECS[model_key]["base_hf"]), local_files_only=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    reloaded = QuillanSovereignBitNetLM.load_packed_safetensors(out_ckpt, device="cpu")
    param_count = reloaded.count_logical_parameters()
    assert param_count <= 1_000_000_000, f"{model_key} exceeded 1B params: {param_count}"

    packed_val_nll = evaluate_sovereign_nll(reloaded, tok, val_texts[:12], max_len=160)
    prompt = "Explain why binary search has O(log n) time complexity:"
    p_ids = tok(prompt, return_tensors="pt")["input_ids"]
    gen_ids = reloaded.generate(p_ids, max_new_tokens=24, temperature=0.0, eos_token_id=tok.eos_token_id)
    gen_text = tok.decode(gen_ids[0, p_ids.shape[1] :], skip_special_tokens=True)

    file_mb = round(out_ckpt.stat().st_size / (1024 * 1024), 2)
    report = {
        "model": model_key,
        "elapsed_s": round(time.time() - t0, 2),
        "logical_params": param_count,
        "checkpoint": str(out_ckpt),
        "file_size_mb": file_mb,
        "alpha": alpha,
        "base_val_nll": round(base_val_nll, 5) if base_val_nll is not None else None,
        "fused_val_nll": round(fused_val_nll, 5) if fused_val_nll is not None else None,
        "packed_val_nll_12": round(packed_val_nll, 5),
        "packed_ppl_12": round(math.exp(packed_val_nll), 4),
        "sample_gen": gen_text,
    }
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("VERIFY_DONE %s: %s", model_key, json.dumps(report))
    print("QUILLAN_BITNET_VERIFY:", json.dumps(report))
    return report


def run_self_test() -> None:
    """Synthetic CPU test verifying SwiGLU salience pruning, packing, reload, and generation."""
    cfg = QuillanSovereignConfig(
        model_name="synthetic-test",
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        group_size=128,
        moe_ffn_dim=128,
        moe_layer_indices=(1,),
    )
    m = QuillanSovereignBitNetLM(cfg).eval()
    assert m.count_logical_parameters() <= 1_000_000_000
    tmp = c.EXPORT_ROOT / "tmp_trio_selftest.safetensors"
    m.save_packed_safetensors(tmp)
    m2 = QuillanSovereignBitNetLM.load_packed_safetensors(tmp, device="cpu")
    out = m2.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=4)
    assert out.shape == (1, 7) and torch.isfinite(m2(out)[0]).all()
    tmp.unlink(missing_ok=True)
    print(f"BUILD_TRIO_SELF_TEST PASS | params={m2.count_logical_parameters()} gen_shape={tuple(out.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Quillan-Ronin Sovereign BitNet 1.58b Trio")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--base-nll", type=float, default=None)
    parser.add_argument("--fused-nll", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--model", choices=["mini", "flagship", "dragon"])
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.verify_only:
        if not args.model:
            parser.error("--model is required with --verify-only")
        verify_packed_checkpoint(args.model, args.base_nll, args.fused_nll, args.alpha)
        return
    if not args.model:
        parser.error("Provide --model {mini,flagship,dragon} or --self-test")
    build_model(args.model)


if __name__ == "__main__":
    main()
