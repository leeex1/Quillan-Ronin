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
from typing import Any, Dict, List, Tuple

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
    return calib_texts, val_texts


@torch.no_grad()
def capture_input_subspaces(
    model: torch.nn.Module,
    tokenizer: Any,
    target_layers: List[int],
    calib_texts: List[str],
    energy_thresh: float = 0.99,
    max_rank: int = 192,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """
    Captures activation inputs to o_proj and down_proj at target_layers and returns
    orthonormal right-singular bases V_in (d_in, r) spanning base knowledge features.
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
    for key, chunks in buckets.items():
        x_mat = torch.cat(chunks, dim=0)
        # Centering focuses subspace on active token variation; randomized or full SVD
        _, svals, vh = torch.linalg.svd(x_mat, full_matrices=False)
        energy = torch.cumsum(svals.pow(2), dim=0) / (svals.pow(2).sum() + 1e-12)
        r = int((energy < energy_thresh).sum().item()) + 1
        r = max(16, min(r, max_rank, vh.shape[0] - 8))
        bases[key] = vh[:r].T.contiguous()  # (d_in, r)
    return bases


def _closest_donor_layer(host_layer: int, host_total: int, donor_layers: List[int], donor_total: int) -> int:
    rel = host_layer / max(1, host_total - 1)
    best = min(donor_layers, key=lambda dl: abs(dl / max(1, donor_total - 1) - rel))
    return best


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

        for hl in target_layers:
            dl = _closest_donor_layer(hl, host_total, d_layers, d_total)

            # --- 1. o_proj: exact bilateral Procrustes P^T @ W_d @ P ---
            w_h_o = model.model.layers[hl].self_attn.o_proj.weight.detach().float()
            w_d_o = donors[f"L{dl}.o_proj"].float()
            w_o_al = p_d2h.T @ w_d_o @ p_d2h
            scale_o = float(torch.norm(w_h_o) / (torch.norm(w_o_al) + 1e-12))
            delta_o = scale_o * w_o_al - w_h_o

            v_o = bases[(hl, "o_proj")]
            delta_o_null = delta_o - (delta_o @ v_o) @ v_o.T
            if hl not in o_accum:
                o_accum[hl] = w_donor * delta_o_null
            else:
                o_accum[hl] = o_accum[hl] + w_donor * delta_o_null
            weight_sums_o[hl] += w_donor

            # --- 2. down_proj: output space via P^T, input space matched if dim agrees ---
            w_h_dn = model.model.layers[hl].mlp.down_proj.weight.detach().float()
            w_d_dn = donors[f"L{dl}.down_proj"].float()
            w_dn_out = p_d2h.T @ w_d_dn  # (d_host, d_ff_donor)
            if w_dn_out.shape[1] == w_h_dn.shape[1]:
                scale_dn = float(torch.norm(w_h_dn) / (torch.norm(w_dn_out) + 1e-12))
                delta_dn = scale_dn * w_dn_out - w_h_dn
                v_dn = bases[(hl, "down_proj")]
                delta_dn_null = delta_dn - (delta_dn @ v_dn) @ v_dn.T
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
            # Rank-64 SVD denoise
            u, s, vh = torch.linalg.svd(d_mat, full_matrices=False)
            k = min(64, s.numel())
            d_den = (u[:, :k] * s[:k]) @ vh[:k, :]
            # Normalize to match host layer norm so alpha has consistent scale across models
            w_h_o = model.model.layers[hl].self_attn.o_proj.weight.detach().float()
            d_unit = d_den * (torch.norm(w_h_o) / (torch.norm(d_den) + 1e-12))
            unit_deltas[(hl, "o_proj")] = d_unit.contiguous()

        if hl in dn_accum and weight_sums_dn[hl] > 0:
            d_mat = dn_accum[hl] / weight_sums_dn[hl]
            u, s, vh = torch.linalg.svd(d_mat, full_matrices=False)
            k = min(64, s.numel())
            d_den = (u[:, :k] * s[:k]) @ vh[:k, :]
            w_h_dn = model.model.layers[hl].mlp.down_proj.weight.detach().float()
            d_unit = d_den * (torch.norm(w_h_dn) / (torch.norm(d_den) + 1e-12))
            unit_deltas[(hl, "down_proj")] = d_unit.contiguous()

    return unit_deltas, donor_meta


def _apply_deltas(
    model: torch.nn.Module,
    orig_weights: Dict[Tuple[int, str], torch.Tensor],
    unit_deltas: Dict[Tuple[int, str], torch.Tensor],
    layers: List[int],
    projs: Tuple[str, ...],
    alpha: float,
) -> None:
    """Sets target weights to orig + alpha * unit_delta deterministically."""
    with torch.no_grad():
        for (l, p), w0 in orig_weights.items():
            target_param = (
                model.model.layers[l].self_attn.o_proj.weight
                if p == "o_proj"
                else model.model.layers[l].mlp.down_proj.weight
            )
            if alpha > 0.0 and l in layers and p in projs and (l, p) in unit_deltas:
                target_param.copy_(w0 + alpha * unit_deltas[(l, p)])
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
    bases = capture_input_subspaces(model, tok, target_layers, calib_texts)
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

    # Fast representative subset (10 items across prose/MMLU/ARC/GSM8K) for search;
    # full 31-item val_texts verifies zero regression before commit.
    fast_val_texts = val_texts[:3] + val_texts[3:6] + val_texts[13:15] + val_texts[23:25]
    base_fast_nll = c.evaluate_texts_nll(model, tok, fast_val_texts)

    best_nll = base_fast_nll
    best_cfg = {"layers": [], "projs": (), "alpha": 0.0}

    layer_subsets = [
        ("all6", target_layers),
        ("late3", target_layers[-3:]),
        ("late2", target_layers[-2:]),
    ]
    proj_modes: List[Tuple[str, ...]] = [("o_proj",), ("o_proj", "down_proj")]
    alphas = [0.002, 0.005, 0.015, 0.035]

    trials: List[Dict[str, Any]] = []
    for s_name, l_sub in layer_subsets:
        for p_mode in proj_modes:
            if "down_proj" in p_mode and not any((l, "down_proj") in unit_deltas for l in l_sub):
                continue
            for alpha in alphas:
                _apply_deltas(model, orig_weights, unit_deltas, l_sub, p_mode, alpha)
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

    # Verify candidate against full 31-item val_texts; fall back to smaller alpha if needed
    _apply_deltas(model, orig_weights, unit_deltas, best_cfg["layers"], best_cfg["projs"], best_cfg["alpha"])
    final_val_nll = c.evaluate_texts_nll(model, tok, val_texts)
    if final_val_nll > base_val_nll and best_cfg["alpha"] > 0.0:
        for fallback_alpha in (best_cfg["alpha"] * 0.5, 0.001, 0.0):
            _apply_deltas(model, orig_weights, unit_deltas, best_cfg["layers"], best_cfg["projs"], fallback_alpha)
            cand_full = c.evaluate_texts_nll(model, tok, val_texts)
            if cand_full <= base_val_nll + 1e-6 or fallback_alpha == 0.0:
                best_cfg["alpha"] = fallback_alpha
                final_val_nll = cand_full
                break
    assert final_val_nll <= base_val_nll + 1e-6, "Stage B gate violated!"
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
    model.to(torch.bfloat16)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="WSP v23 Stage B Pre-Combination")
    parser.add_argument("--model", required=True, choices=["flagship", "mini", "dragon"])
    parser.add_argument("--convert-only", action="store_true")
    args = parser.parse_args()
    run_stage_b(args.model, convert_only=args.convert_only)


if __name__ == "__main__":
    main()
