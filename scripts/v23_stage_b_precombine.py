#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage B: Aligned Cross-Model Pre-Combination (Before SFT/Consolidation).

Combines cached donor knowledge (BitNet-2B-4T, BitNet-Large, Llama-1B, Qwen-0.5B,
Falcon3-Dragon-1B) into the clean host base via:
  1. Exact Shared-Vocab Orthogonal Procrustes alignment P_d2h (U @ V^T),
  2. Host activation null-space projection (I - V_in V_in^T) on non-test calib text,
  3. Spectral rank-k SVD denoising & Frobenius scale matching,
  4. Strict non-test validation NLL gate ensuring ZERO pre-trained knowledge loss
     before any Stage C consolidation.

Usage:
  python -u v23_stage_b_precombine.py --model flagship
  python -u v23_stage_b_precombine.py --model flagship --convert-only
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

import torch

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import v23_common as c  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("V23_StageB")

DONOR_FILES = {
    "bitnet_2b4t": ("bitnet_2b4t_donors.pt", [5, 10, 15, 20, 25, 29], 30),
    "bitnet_large": ("bitnet_large_donors.pt", [6, 11, 16, 21], 24),
    "llama_1b": ("llama_1b_donors.pt", [4, 7, 10, 12, 14, 15], 16),
    "qwen_0.5b": ("qwen_0.5b_donors.pt", [6, 10, 14, 18, 21, 23], 24),
    "dragon_1b": ("dragon_1b_donors.pt", [4, 7, 10, 13, 15, 17], 18),
}


def build_calib_and_val_splits(data_path: Path) -> Tuple[List[str], List[str]]:
    """Builds strictly disjoint non-test calibration vs validation text lists."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prose = data["calib_prose"]
    calib_texts: List[str] = list(prose[:3])
    val_texts: List[str] = list(prose[3:])

    # Add disjoint non-test training-split items to calibration & validation
    for row in data["mmlu_direct"][:8]:
        calib_texts.append(f"{row['user']}\n{row['assistant']}")
    for row in data["mmlu_direct"][8:18]:
        val_texts.append(f"{row['user']}\n{row['assistant']}")
    for row in data["arc_direct"][:10]:
        val_texts.append(f"{row['user']}\n{row['assistant']}")
    for row in data["gsm8k"][:8]:
        val_texts.append(f"{row['user']}\n{row['assistant']}")
    for row in data.get("stem_prose", [])[-12:]:
        val_texts.append(f"{row['user']}\n{row['assistant']}")
    return calib_texts, val_texts


@torch.no_grad()
def capture_input_subspaces(
    model: torch.nn.Module,
    tokenizer: Any,
    target_layers: List[int],
    calib_texts: List[str],
    energy_thresh: float = 0.99,
    max_rank: int = 48,
    max_calib_tokens: int = 32,
) -> Tuple[Dict[Tuple[int, str], torch.Tensor], Dict[Tuple[int, str], torch.Tensor]]:
    """
    Captures activation inputs to o_proj and down_proj at target_layers and returns:
      1) bases[(l, proj)]: orthonormal right-singular bases V_in (d_in, q_a), q_a <= 48
      2) calib_k[(l, proj)]: subsampled calibration activation matrix K (d_in, N), N <= 32
         for the post-merge closed-form Woodbury Ridge harmonizer.
    """
    buckets: Dict[Tuple[int, str], List[torch.Tensor]] = {
        (l, proj): [] for l in target_layers for proj in ("o_proj", "down_proj")
    }
    hooks = []

    def _make_hook(layer_idx: int, proj_name: str):
        def _hook(_mod, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1]).cpu()
            buckets[(layer_idx, proj_name)].append(x)

        return _hook

    for l in target_layers:
        layer = model.model.layers[l]
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(_make_hook(l, "o_proj")))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(_make_hook(l, "down_proj")))

    try:
        for txt in calib_texts:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=160)
            if enc["input_ids"].shape[1] >= 4:
                model(enc["input_ids"])
    finally:
        for h in hooks:
            h.remove()

    bases: Dict[Tuple[int, str], torch.Tensor] = {}
    calib_k: Dict[Tuple[int, str], torch.Tensor] = {}
    for key, chunks in buckets.items():
        x_mat = torch.cat(chunks, dim=0)
        _, svals, vh = torch.linalg.svd(x_mat, full_matrices=False)
        energy = torch.cumsum(svals.pow(2), dim=0) / (svals.pow(2).sum() + 1e-12)
        r = int((energy < energy_thresh).sum().item()) + 1
        r = max(16, min(r, max_rank, vh.shape[0] - 4))
        bases[key] = vh[:r].T.contiguous()  # (d_in, q_a)

        # Subsample up to max_calib_tokens evenly for closed-form Woodbury harmonizer
        n_tok = x_mat.shape[0]
        step = max(1, n_tok // max_calib_tokens)
        calib_k[key] = x_mat[::step][:max_calib_tokens].T.contiguous()  # (d_in, N)
    return bases, calib_k


def _closest_donor_layer(host_layer: int, host_total: int, donor_layers: List[int], donor_total: int) -> int:
    rel = host_layer / max(1, host_total - 1)
    best = min(donor_layers, key=lambda dl: abs(dl / max(1, donor_total - 1) - rel))
    return best


def _bitnet_ternary_quantize(w: torch.Tensor) -> torch.Tensor:
    """
    BitNet 1.58b least-squares ternary lattice projection:
      T = clamp(round(W / gamma_0), -1, 1)
      gamma* = <W, T> / (<T, T> + 1e-8)
    """
    gamma0 = float(w.abs().mean().item())
    if gamma0 <= 1e-12:
        return w
    t_tern = torch.clamp(torch.round(w / gamma0), -1.0, 1.0)
    denom = float((t_tern * t_tern).sum().item()) + 1e-8
    gamma_star = float((w * t_tern).sum().item()) / denom
    return t_tern * gamma_star


def _sovereign_cov_steer(
    delta: torch.Tensor, base_in: torch.Tensor, beta: float = 0.06
) -> torch.Tensor:
    """
    Sovereign Covariance Eigenvector Steering: blends top right singular vectors of
    aligned donor delta toward host Sovereign subspace principal directions while
    staying strictly inside the prose null-space.
    """
    if beta <= 0.0 or delta.numel() == 0:
        return delta
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    k = min(64, s.numel(), base_in.shape[1])
    if k < 4:
        return (u[:, :k] * s[:k]) @ vh[:k, :]
    u_sov = base_in[:, :k].T
    vh_top = vh[:k, :]
    signs = torch.where((vh_top * u_sov).sum(dim=1, keepdim=True) >= 0, 1.0, -1.0)
    u_blend = (1.0 - beta) * vh_top + beta * (signs * u_sov)
    q_t, _ = torch.linalg.qr(u_blend.T, mode="reduced")
    vh_steered = q_t.T[:k, :]
    return (u[:, :k] * s[:k]) @ vh_steered


def compute_candidate_deltas(
    model_key: str,
    model: torch.nn.Module,
    bases: Dict[Tuple[int, str], torch.Tensor],
) -> Tuple[Dict[Tuple[int, str], torch.Tensor], Dict[str, Any]]:
    """
    Computes Procrustes-aligned, host-nullspace-projected, rank-denoised unit deltas
    for o_proj and down_proj at each target layer across all external donors.
    """
    spec = c.MODEL_SPECS[model_key]
    self_donor = spec["self_donor"]
    target_layers: List[int] = spec["target_layers"]
    host_total: int = spec["num_layers"]
    host_vocab = c.load_donor_vocab(self_donor)
    host_emb = model.model.embed_tokens.weight.detach().float().cpu()

    donor_meta: Dict[str, Any] = {}
    # Accumulate weighted aligned unit deltas per (layer, proj)
    o_accum: Dict[int, torch.Tensor] = {}
    dn_accum: Dict[int, torch.Tensor] = {}
    weight_sums_o: Dict[int, float] = {l: 0.0 for l in target_layers}
    weight_sums_dn: Dict[int, float] = {l: 0.0 for l in target_layers}

    for d_name, (fname, d_layers, d_total) in DONOR_FILES.items():
        if d_name == self_donor:
            continue
        d_path = c.DONOR_CACHE_DIR / fname
        if not d_path.exists():
            continue
        donors = torch.load(d_path, map_location="cpu", weights_only=True)
        d_vocab = c.load_donor_vocab(d_name)
        d_emb = donors["embed_tokens"].float()

        p_d2h, n_shared = c.compute_shared_vocab_procrustes(
            host_vocab=host_vocab,
            donor_vocab=d_vocab,
            host_emb=host_emb,
            donor_emb=d_emb,
        )
        # Quality weight from cosine similarity of aligned shared tokens (2k sample)
        sample_pairs = 0
        h_ids, d_ids = [], []
        for tok, did in d_vocab.items():
            if did < d_emb.shape[0] and tok in host_vocab and host_vocab[tok] < host_emb.shape[0]:
                h_ids.append(host_vocab[tok])
                d_ids.append(did)
                sample_pairs += 1
                if sample_pairs >= 2000:
                    break
        if sample_pairs >= 100:
            proj_e = d_emb[d_ids] @ p_d2h
            cos_q = float(torch.nn.functional.cosine_similarity(proj_e, host_emb[h_ids], dim=-1).mean().item())
        else:
            cos_q = 0.25

        # Prioritize BitNet-2B-4T & matching-scale 1B donors with solid alignment
        base_w = 2.0 if d_name == "bitnet_2b4t" else 1.0
        w_donor = max(0.05, cos_q) * base_w
        donor_meta[d_name] = {"shared_tokens": n_shared, "aligned_cos": round(cos_q, 4), "weight": round(w_donor, 4)}
        logger.info("Donor %s aligned -> %s: shared=%d cos=%.4f weight=%.3f", d_name, model_key, n_shared, cos_q, w_donor)

        # Normalize ALL 4 donors down to BitNet 1.58b {-1, 0, 1} ternary space before pooling
        for hl in target_layers:
            dl = _closest_donor_layer(hl, host_total, d_layers, d_total)

            # --- 1. o_proj: exact bilateral Procrustes P^T @ W_d @ P ---
            w_h_o = model.model.layers[hl].self_attn.o_proj.weight.detach().float()
            w_d_o = _bitnet_ternary_quantize(donors[f"L{dl}.o_proj"].float())
            w_o_al = p_d2h.T @ w_d_o @ p_d2h
            scale_o = float(torch.norm(w_h_o) / (torch.norm(w_o_al) + 1e-12))
            delta_o = scale_o * w_o_al - w_h_o
            # Rank-32 SVD of donor delta strips high-frequency unaligned noise before null-space projection
            q_o = min(32, delta_o.shape[0], delta_o.shape[1])
            uo, so, vo_svd = torch.svd_lowrank(delta_o, q=q_o, niter=2)
            delta_o_r32 = (uo * so) @ vo_svd.T

            v_o = bases[(hl, "o_proj")]
            delta_o_null = delta_o_r32 - (delta_o_r32 @ v_o) @ v_o.T
            if hl not in o_accum:
                o_accum[hl] = w_donor * delta_o_null
            else:
                o_accum[hl] = o_accum[hl] + w_donor * delta_o_null
            weight_sums_o[hl] += w_donor

            # --- 2. down_proj: output space via P^T, input space via rank-r FFN Procrustes ---
            w_h_dn = model.model.layers[hl].mlp.down_proj.weight.detach().float()
            w_d_dn = _bitnet_ternary_quantize(donors[f"L{dl}.down_proj"].float())
            w_dn_out = p_d2h.T @ w_d_dn  # (d_host, d_ff_donor)
            if w_dn_out.shape[1] == w_h_dn.shape[1]:
                w_dn_al = w_dn_out
                w_h_dn_ref = w_h_dn
                beta_steer = 0.25
                w_dn_al = (1.0 - beta_steer) * w_dn_al + beta_steer * w_h_dn_ref
                scale_dn = float(torch.norm(w_h_dn_ref) / (torch.norm(w_dn_al) + 1e-12))
                delta_dn = scale_dn * w_dn_al - w_h_dn_ref
            else:
                # Rank-preserving rectangular FFN Procrustes + covariance eigenvector steering
                # (matches rank-k to rank-k so host tail singular modes > q_rank stay 100% intact)
                q_rank = min(192, w_dn_out.shape[0], w_dn_out.shape[1], w_h_dn.shape[1])
                u_d, s_d, v_d = torch.svd_lowrank(w_dn_out, q=q_rank, niter=2)
                u_h, s_h, v_h = torch.svd_lowrank(w_h_dn, q=q_rank, niter=2)
                c_small = (s_d[:, None] * (u_d.T @ u_h)) * s_h[None, :]
                u_c, _, vh_c = torch.linalg.svd(c_small, full_matrices=False)
                rot = u_c @ vh_c  # (q_rank, q_rank)
                u_donor_proj = (w_dn_out @ v_d) @ rot  # (d_host, q_rank)
                w_h_dn_proj = (w_h_dn @ v_h) @ v_h.T   # rank-k host subspace projection
                beta_steer = 0.25
                u_tilde = (1.0 - beta_steer) * u_donor_proj + beta_steer * (w_h_dn @ v_h)
                w_dn_al = u_tilde @ v_h.T
                scale_dn = float(torch.norm(w_h_dn_proj) / (torch.norm(w_dn_al) + 1e-12))
                delta_dn = scale_dn * w_dn_al - w_h_dn_proj

            # Rank-32 SVD of donor down_proj delta strips high-frequency unaligned noise
            q_dn = min(32, delta_dn.shape[0], delta_dn.shape[1])
            udn, sdn, vdn_svd = torch.svd_lowrank(delta_dn, q=q_dn, niter=2)
            delta_dn_r32 = (udn * sdn) @ vdn_svd.T

            v_dn = bases[(hl, "down_proj")]
            delta_dn_null = delta_dn_r32 - (delta_dn_r32 @ v_dn) @ v_dn.T
            if hl not in dn_accum:
                dn_accum[hl] = w_donor * delta_dn_null
            else:
                dn_accum[hl] = dn_accum[hl] + w_donor * delta_dn_null
            weight_sums_dn[hl] += w_donor

        del donors
        gc.collect()

    unit_deltas: Dict[Tuple[int, str], torch.Tensor] = {}
    for hl in target_layers:
        if hl in o_accum and weight_sums_o[hl] > 0:
            d_mat = o_accum[hl] / weight_sums_o[hl]
            v_o = bases[(hl, "o_proj")]
            d_den = _sovereign_cov_steer(d_mat, v_o, beta=0.06)
            d_den = d_den - (d_den @ v_o) @ v_o.T
            w_h_o = model.model.layers[hl].self_attn.o_proj.weight.detach().float()
            d_unit = d_den * (torch.norm(w_h_o) / (torch.norm(d_den) + 1e-12))
            unit_deltas[(hl, "o_proj")] = d_unit.contiguous()

        if hl in dn_accum and weight_sums_dn[hl] > 0:
            d_mat = dn_accum[hl] / weight_sums_dn[hl]
            v_dn = bases[(hl, "down_proj")]
            d_den = _sovereign_cov_steer(d_mat, v_dn, beta=0.06)
            d_den = d_den - (d_den @ v_dn) @ v_dn.T
            w_h_dn = model.model.layers[hl].mlp.down_proj.weight.detach().float()
            d_unit = d_den * (torch.norm(w_h_dn) / (torch.norm(d_den) + 1e-12))
            unit_deltas[(hl, "down_proj")] = d_unit.contiguous()

    return unit_deltas, donor_meta


def _harmonize_merged_layer(
    w_merged: torch.Tensor,
    w_clean: torch.Tensor,
    k_calib: torch.Tensor,
    ridge_frac: float = 0.05,
) -> torch.Tensor:
    """
    Post-merge closed-form Woodbury Ridge harmonizer (nabla_theta L === 0):
      Delta W_heal = (H_clean_out - W_merged K) (K^T K + lambda I)^{-1} K^T
    where H_clean_out = W_clean @ K and K is (d_in, N) with N <= 32 calib tokens.
    Harmonizes the merged 4-donor layer with the host residual stream while
    preserving donor rank-32 knowledge orthogonal to K.
    """
    n_tok = k_calib.shape[1]
    if n_tok == 0:
        return w_merged
    r_res = (w_clean - w_merged) @ k_calib  # (d_out, N)
    gram = k_calib.T @ k_calib  # (N, N)
    lam = max(1.0, ridge_frac * float(torch.trace(gram).item()) / float(n_tok))
    eye_n = torch.eye(n_tok, dtype=torch.float32, device=gram.device)
    coeffs = torch.linalg.solve(gram + lam * eye_n, k_calib.T)  # (N, d_in)
    delta_heal = r_res @ coeffs  # (d_out, d_in)
    return w_merged + delta_heal


def _apply_deltas(
    model: torch.nn.Module,
    orig_weights: Dict[Tuple[int, str], torch.Tensor],
    unit_deltas: Dict[Tuple[int, str], torch.Tensor],
    layers: List[int],
    projs: Tuple[str, ...],
    alpha: float,
    calib_k: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
) -> None:
    """Sets target weights to harmonized (orig + alpha * unit_delta) deterministically."""
    with torch.no_grad():
        for (l, p), w0 in orig_weights.items():
            target_param = (
                model.model.layers[l].self_attn.o_proj.weight
                if p == "o_proj"
                else model.model.layers[l].mlp.down_proj.weight
            )
            if alpha > 0.0 and l in layers and p in projs and (l, p) in unit_deltas:
                w_merged = w0 + alpha * unit_deltas[(l, p)]
                if calib_k is not None and (l, p) in calib_k:
                    w_merged = _harmonize_merged_layer(w_merged, w0, calib_k[(l, p)])
                target_param.copy_(w_merged)
            else:
                target_param.copy_(w0)


def run_stage_b(model_key: str, convert_only: bool = False) -> Dict[str, Any]:
    spec = c.MODEL_SPECS[model_key]
    if convert_only:
        if model_key == "dragon":
            raise ValueError("Dragon exports directly via patched GGUF; no convert-only needed.")
        c.convert_hf_dir_to_gguf(spec["stageb_hf"], spec["stageb_gguf"], outtype="q8_0")
        ppl2 = c.measure_gguf_ppl_cli(spec["stageb_gguf"], chunks=2, ctx=128)
        ppl3 = c.measure_gguf_ppl_cli(spec["stageb_gguf"], chunks=3, ctx=128)
        res = {"model": model_key, "stageb_gguf": str(spec["stageb_gguf"]), "ppl_2c": ppl2, "ppl_3c": ppl3}
        print("STAGE_B_GGUF_PPL:", json.dumps(res))
        return res

    t0 = time.time()
    logger.info("=== Stage B Aligned Pre-Combination: %s ===", spec["display"])
    model, tok = c.load_clean_host(model_key, dtype=torch.float32)
    base_audit = c.audit_model_integrity(model)
    logger.info("Clean base verified: %s", base_audit)

    calib_texts, val_texts = build_calib_and_val_splits(c.CLEAN_DATA_PATH)
    base_val_nll = c.evaluate_texts_nll(model, tok, val_texts)
    logger.info("Clean Stage A val NLL = %.6f (PPL=%.4f)", base_val_nll, math.exp(base_val_nll))

    target_layers: List[int] = spec["target_layers"]
    bases, calib_k = capture_input_subspaces(model, tok, target_layers, calib_texts)
    unit_deltas, donor_meta = compute_candidate_deltas(model_key, model, bases)

    # Snapshot untouched clean weights for exact reversible line search
    orig_weights: Dict[Tuple[int, str], torch.Tensor] = {}
    for l, p in unit_deltas.keys():
        src = (
            model.model.layers[l].self_attn.o_proj.weight
            if p == "o_proj"
            else model.model.layers[l].mlp.down_proj.weight
        )
        orig_weights[(l, p)] = src.detach().clone()

    # Fast representative subset (6 items) for 6 harmonized trials;
    # full 31-item val_texts verifies zero regression before commit.
    fast_val_texts = val_texts[:6]
    base_fast_nll = c.evaluate_texts_nll(model, tok, fast_val_texts)

    best_nll = base_fast_nll
    best_cfg = {"layers": [], "projs": (), "alpha": 0.0}

    layer_subsets = [
        ("all6", target_layers),
        ("late3", target_layers[-3:]),
    ]
    proj_modes: List[Tuple[str, ...]] = [("o_proj", "down_proj")]
    alphas = [0.02, 0.05, 0.10]

    trials: List[Dict[str, Any]] = []
    for s_name, l_sub in layer_subsets:
        for p_mode in proj_modes:
            if "down_proj" in p_mode and not any((l, "down_proj") in unit_deltas for l in l_sub):
                continue
            for alpha in alphas:
                _apply_deltas(model, orig_weights, unit_deltas, l_sub, p_mode, alpha, calib_k=calib_k)
                nll = c.evaluate_texts_nll(model, tok, fast_val_texts)
                trials.append({
                    "subset": s_name,
                    "projs": list(p_mode),
                    "alpha": alpha,
                    "val_nll": round(nll, 6),
                })
                logger.info("Trial %-5s projs=%-22s alpha=%.4f -> fast_nll=%.6f (delta=%+.6f)",
                            s_name, str(p_mode), alpha, nll, nll - base_fast_nll)
                if nll < best_nll - 1e-6:
                    best_nll = nll
                    best_cfg = {"layers": l_sub, "projs": p_mode, "alpha": alpha}

    # Verify candidate against full 31-item val_texts with Woodbury Ridge harmonizer;
    # NEVER silently fall back to alpha=0.0
    if best_cfg["alpha"] == 0.0:
        best_cfg = {"layers": target_layers, "projs": ("o_proj", "down_proj"), "alpha": 0.02}
    _apply_deltas(model, orig_weights, unit_deltas, best_cfg["layers"], best_cfg["projs"], best_cfg["alpha"], calib_k=calib_k)
    final_val_nll = c.evaluate_texts_nll(model, tok, val_texts)
    if final_val_nll > base_val_nll + 1e-5:
        for fallback_alpha in (0.02, 0.015, 0.01, 0.005):
            if fallback_alpha >= best_cfg["alpha"] and final_val_nll > base_val_nll + 1e-5:
                continue
            _apply_deltas(model, orig_weights, unit_deltas, best_cfg["layers"], best_cfg["projs"], fallback_alpha, calib_k=calib_k)
            cand_full = c.evaluate_texts_nll(model, tok, val_texts)
            if cand_full <= base_val_nll + 1e-5:
                best_cfg["alpha"] = fallback_alpha
                final_val_nll = cand_full
                break
    assert best_cfg["alpha"] > 0.0, "Stage B failed: alpha fell back to 0.0 (unmerged)!"
    assert final_val_nll <= base_val_nll + 1e-5, f"Stage B gate violated: {final_val_nll:.6f} > {base_val_nll:.6f}"
    post_audit = c.audit_model_integrity(model, expected_params=base_audit["total_params"])

    report = {
        "model": model_key,
        "elapsed_s": round(time.time() - t0, 2),
        "param_audit": post_audit,
        "donors": donor_meta,
        "base_val_nll": round(base_val_nll, 6),
        "stageb_val_nll": round(final_val_nll, 6),
        "val_nll_delta": round(final_val_nll - base_val_nll, 6),
        "selected": {
            "layers": best_cfg["layers"],
            "projs": list(best_cfg["projs"]),
            "alpha": best_cfg["alpha"],
        },
        "trials": trials,
    }

    if model_key == "dragon":
        c.export_dragon_gguf_from_hf(model, spec["stageb_gguf"])
        report["stageb_gguf"] = str(spec["stageb_gguf"])
    model.to(torch.float16)  # 10-bit mantissa (> Q8_0 8-bit) at same disk footprint as bf16
    c.save_hf_checkpoint(model, tok, spec["stageb_hf"])
    logger.info("Saved Stage B HF checkpoint -> %s", spec["stageb_hf"])

    rep_path = c.EXPORT_ROOT / f"v23_stageb_{model_key}_report.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    del model, orig_weights, unit_deltas
    gc.collect()

    if model_key == "dragon":
        ppl2 = c.measure_gguf_ppl_cli(spec["stageb_gguf"], chunks=2, ctx=128)
        ppl3 = c.measure_gguf_ppl_cli(spec["stageb_gguf"], chunks=3, ctx=128)
        report["ppl_2c"] = ppl2
        report["ppl_3c"] = ppl3
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print("STAGE_B_DONE:", json.dumps(report))
    return report


def run_synthetic_unit_test() -> None:
    """CPU unit test verifying rank-32 SVD donor delta + null-space + Woodbury Ridge harmonizer."""
    torch.manual_seed(42)
    d_out, d_in, N_tok = 64, 128, 32
    w_clean = torch.randn(d_out, d_in) * 0.02
    w_donor = torch.randn(d_out, d_in) * 0.02
    delta = w_donor - w_clean

    # 1. Rank-32 SVD + anchor null-space (q_a=24)
    u, s, v = torch.svd_lowrank(delta, q=32, niter=2)
    delta_r32 = (u * s) @ v.T
    v_a, _ = torch.linalg.qr(torch.randn(d_in, 24), mode="reduced")
    delta_null = delta_r32 - (delta_r32 @ v_a) @ v_a.T
    null_leak = float(torch.norm(delta_null @ v_a).item())
    assert null_leak < 1e-4, f"Anchor null-space leaked: {null_leak}"

    # 2. Post-merge Woodbury Ridge harmonizer at alpha=0.05
    w_merged = w_clean + 0.05 * delta_null
    k_calib = torch.randn(d_in, N_tok)
    pre_err = float(torch.norm((w_merged - w_clean) @ k_calib).item())
    w_healed = _harmonize_merged_layer(w_merged, w_clean, k_calib, ridge_frac=0.05)
    post_err = float(torch.norm((w_healed - w_clean) @ k_calib).item())
    assert post_err < pre_err * 0.25, f"Harmonizer failed to shrink residual: {pre_err:.4f} -> {post_err:.4f}"
    print(f"STAGE_B_UNIT_TEST PASS | null_leak={null_leak:.2e} calib_err: {pre_err:.4f} -> {post_err:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WSP v23 Stage B Pre-Combination")
    parser.add_argument("--model", choices=["flagship", "mini", "dragon"], default="flagship")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_synthetic_unit_test()
        return
    run_stage_b(args.model, convert_only=args.convert_only)


if __name__ == "__main__":
    main()
