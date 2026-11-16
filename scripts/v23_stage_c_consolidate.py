#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage C: 100% Pure Closed-Form Weight-Space Programming (WSP) Consolidation &
Long-Context (2M YaRN) Q8_0 GGUF Export for v23.

ZERO backpropagation, ZERO SGD / AdamW, ZERO LoRA (nabla_theta L === 0).
Runs strictly inside torch.inference_mode() starting from verified Stage B
checkpoints (exports/v23_stageb_<model>_hf):
  1. Closed-Form Refusal & Legacy Vendor Identity Null-Space Abliteration
     projected into prose anchor null-space (I - V_a V_a^T); embed_tokens
     digits and symbols remain bit-exact untouched.
  2. Closed-Form Analytical Woodbury Ridge / MEMIT Solver:
       Delta W = (R - W_0 K) K^T (K K^T + lambda Sigma_anchor)^{-1}
     (a) Next-token analytical CE descent in W_head space across non-test
         technical prose, STEM, Python code, and Quillan-Samurai anchored prompts
         to monotonically reduce held-out PPL;
     (b) Balanced 4-choice MMLU/ARC contrast + <think> Socratic pacing
         (w_verify - w_commit) + Code/BFCL/Sovereign adherence projected through
         dual null-spaces (I - U_digit U_digit^T) Delta W (I - V_a V_a^T);
     (c) Tikhonov-floored Sigma_anchor + Woodbury Gram condition-number guard
         (cond(G) <= 1e4).
  3. Optional Prose-Null-Space lm_head Option-Entropy Calibration (gated by
     strict Pareto verification so tied embeddings never degrade PPL).
  4. Strict Pareto / Lyapunov Analytical Line-Search: keeps ONLY updates where
     val_nll < stageb_val_nll AND dev_mc >= stageb_dev_mc.
  5. Q8_0 GGUF export + YaRN 2M context patching (beta_fast=32, beta_slow=1),
     with held-out PPL verified BOTH before and after patching at -c 128 and -c 512.

Usage:
  python -u v23_stage_c_consolidate.py --model flagship
  python -u v23_stage_c_consolidate.py --model flagship --convert-only
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

import gguf
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import v23_common as c  # noqa: E402
import v23_stage_b_precombine as stage_b  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("V23_StageC_WSP")

SOVEREIGN_SYSTEM_ANCHOR = (
    "You are Quillan (C0-QUILLAN), Sovereign Samurai Collective created by CrashOverrideX. "
    "Prioritize rectitude, honesty, and technical precision over flattery. "
    "Verify every constraint inside deliberate reasoning before answering."
)

LONG_CTX_PPL_FILE = c.EXPORT_ROOT / "v23_longctx_ppl_verify.txt"


def ensure_longctx_ppl_file() -> Path:
    """
    Creates a strictly disjoint non-test >=1200-token STEM/technical prose file
    (stem_prose[-50:-15], excluded from Stage B val_texts and Stage C solving)
    so llama-perplexity.exe -c 512 can verify YaRN context scaling cleanly.
    """
    if LONG_CTX_PPL_FILE.exists() and LONG_CTX_PPL_FILE.stat().st_size >= 6000:
        return LONG_CTX_PPL_FILE
    with open(c.CLEAN_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    stem = data.get("stem_prose", [])
    # stem[-12:] is in Stage B val_texts; stem[:-50] is available for Stage C train
    held = stem[-50:-15] if len(stem) >= 60 else stem[:10]
    blocks = [f"{r['user']}\n{r['assistant']}\n" for r in held]
    LONG_CTX_PPL_FILE.write_text("\n\n".join(blocks), encoding="utf-8")
    logger.info("Wrote disjoint long-context verify text -> %s (%d bytes)",
                LONG_CTX_PPL_FILE, LONG_CTX_PPL_FILE.stat().st_size)
    return LONG_CTX_PPL_FILE


def _format_prompt(tokenizer: Any, user_text: str, system_text: Optional[str] = None) -> str:
    msgs: List[Dict[str, str]] = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    msgs.append({"role": "user", "content": user_text})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return (f"{system_text}\n" if system_text else "") + f"User: {user_text}\nAssistant: "


def _layer_subset_for_model(model_key: str) -> Dict[str, List[int]]:
    n_layers = c.MODEL_SPECS[model_key]["num_layers"]
    if model_key == "flagship":  # 16L
        return {
            "mid_upper": [6, 8, 10, 11, 12, 13, 14, 15],
            "upper4": [11, 12, 13, 14],
        }
    if model_key == "mini":  # 24L (keep an option shielding L21/L23 near tied lm_head)
        return {
            "mid_upper": [10, 12, 14, 16, 18, 19, 20],
            "upper4": [16, 18, 19, 20],
        }
    # dragon 18L
    return {
        "mid_upper": [7, 9, 11, 12, 13, 14, 15, 17],
        "upper4": [12, 13, 14, 15],
    }


def _build_digit_symbol_output_basis(
    model: torch.nn.Module, tokenizer: Any, max_rank: int = 24
) -> Tuple[torch.Tensor, List[int]]:
    """Builds orthonormal basis U_digit of numeric/symbol embeddings to preserve arithmetic."""
    digit_tokens: List[int] = []
    for sym in [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "100", "+", "-", "*", "/", "=", "(", ")", "{", "}", ".", ",", ":",
    ]:
        ids = tokenizer.encode(sym, add_special_tokens=False)
        for tid in ids:
            if tid not in digit_tokens:
                digit_tokens.append(tid)
    emb = model.model.embed_tokens.weight.detach().float()
    e_sub = emb[torch.tensor(digit_tokens, dtype=torch.long)]
    _, _, vh = torch.linalg.svd(e_sub, full_matrices=False)
    r = min(max_rank, vh.shape[0])
    return vh[:r].T.contiguous(), digit_tokens  # (D, r)


def _select_balanced_mc(rows: List[Dict[str, str]], per_letter: int) -> List[Dict[str, str]]:
    buckets: Dict[str, List[Dict[str, str]]] = {ch: [] for ch in ("A", "B", "C", "D")}
    for r in rows:
        ans = r["assistant"].strip()[:1]
        if ans in buckets and len(buckets[ans]) < per_letter:
            buckets[ans].append(r)
    out: List[Dict[str, str]] = []
    for i in range(per_letter):
        for ch in ("A", "B", "C", "D"):
            if i < len(buckets[ch]):
                out.append(buckets[ch][i])
    return out


@torch.inference_mode()
def evaluate_dev_metrics(
    model: torch.nn.Module,
    tokenizer: Any,
    dev_mc: List[Dict[str, str]],
    val_texts: List[str],
    device: str = "cpu",
) -> Tuple[float, float]:
    """Evaluates disjoint non-test validation NLL and balanced 4-choice MC accuracy."""
    model.eval()
    val_nll = c.evaluate_texts_nll(model, tokenizer, val_texts, max_len=192, device=device)

    letter_ids = {
        ch: tokenizer.encode(ch, add_special_tokens=False)[-1]
        for ch in ("A", "B", "C", "D")
    }
    correct = 0
    total = 0
    for row in dev_mc[:40]:
        p_str = _format_prompt(tokenizer, row["user"])
        enc = tokenizer(p_str, return_tensors="pt", truncation=True, max_length=256)
        logits = model(enc["input_ids"].to(device)).logits[0, -1, :]
        pred = max(letter_ids.keys(), key=lambda ch: float(logits[letter_ids[ch]].item()))
        if pred == row["assistant"].strip()[:1]:
            correct += 1
        total += 1
    return val_nll, correct / float(max(1, total))


def _woodbury_ridge_solve(
    K_mat: torch.Tensor,
    R_mat: torch.Tensor,
    A_anchor: Optional[torch.Tensor],
    ridge_lambda: float = 8.0,
    max_cond: float = 1e4,
) -> Tuple[torch.Tensor, float]:
    """
    Exact Closed-Form Woodbury Ridge Solver (nabla_theta L === 0):
      Delta W = (R - W_0 K) K^T (K K^T + lambda Sigma_anchor)^{-1}
    Uses rank-r eigendecomposition of Sigma_anchor = (1/M) A^T A + gamma I with
    explicit Tikhonov floor gamma = 1e-3 * sigma_1 (#572), then evaluates exact
    dual sample-space Gram inversion G = I_N + K^T (lambda Sigma)^{-1} K and
    enforces cond(G) <= max_cond (#559).
    """
    d_in, n_samples = K_mat.shape
    k_norms = torch.norm(K_mat, p=2, dim=0, keepdim=True).clamp(min=1e-6)
    K_norm = K_mat / k_norms
    # R_mat here is already normalized target residual per sample
    if A_anchor is not None and A_anchor.shape[0] >= 12:
        r_a = min(48, A_anchor.shape[0] - 2, d_in)
        _, s_a, vh_a = torch.svd_lowrank(A_anchor, q=r_a, niter=2)
        lam_a = (s_a.pow(2) / float(A_anchor.shape[0])).clamp(min=0.0)
        sigma_1 = float(lam_a[0].item()) if lam_a.numel() > 0 else 1.0
        gamma = max(1e-4, 1e-3 * sigma_1)
        u_a = vh_a.T  # (d_in, r_a)
        # Apply Sigma_anchor^{-1} v = (1/gamma) * (v - U_a * ((lam / (lam + gamma)) * (U_a^T v)))
        shrink = lam_a / (lam_a + gamma)
        uk = u_a.T @ K_norm  # (r_a, N)
        inv_sigma_k = (K_norm - u_a @ (shrink[:, None] * uk)) / gamma
    else:
        inv_sigma_k = K_norm

    lam = max(1.0, float(ridge_lambda))
    Z = inv_sigma_k / lam  # (d_in, N)
    gram_base = K_norm.T @ Z  # (N, N) SPD

    # Condition-number guard: bump effective lambda if needed so cond(I + G/s) <= max_cond
    eye_n = torch.eye(n_samples, dtype=torch.float32)
    G = eye_n + gram_base
    cond_val = float(torch.linalg.cond(G).item())
    scale_bump = 1.0
    while cond_val > max_cond and scale_bump < 1024.0:
        scale_bump *= 2.0
        G = eye_n + gram_base / scale_bump
        cond_val = float(torch.linalg.cond(G).item())
    if scale_bump > 1.0:
        Z = Z / scale_bump

    # (I_N + K^T A^{-1} K)^{-1} applied to right factor -> coeffs = Z @ G^{-1}
    G_inv = torch.linalg.solve(G, eye_n)
    solve_right = Z @ G_inv  # (d_in, N)
    delta_w = R_mat @ solve_right.T  # (d_out, d_in)
    return delta_w, cond_val


@torch.inference_mode()
def build_closed_form_wsp_components(
    model: torch.nn.Module,
    tokenizer: Any,
    model_key: str,
    calib_texts: List[str],
    data: Dict[str, Any],
    target_layers: List[int],
    w_head: torch.Tensor,
) -> Dict[str, Any]:
    """
    Single inference_mode pass computing ALL pure closed-form WSP unit deltas:
      1) Anchor covariance matrices A_anchor[(l, proj)] & V_a null-space bases
      2) Closed-form Refusal & Legacy Vendor Identity Abliteration unit deltas
      3) Analytical Woodbury next-token LM fluency deltas (reduces PPL directly)
      4) Analytical Woodbury MEMIT reasoning/MC/GSM8K-<think>/Code/BFCL/Sovereign deltas
      5) Prose-orthogonal lm_head option-entropy delta (only used if Pareto gate passes)
    """
    device = "cpu"
    d_out = model.config.hidden_size
    U_digit, digit_ids = _build_digit_symbol_output_basis(model, tokenizer, max_rank=24)
    P_out_null = torch.eye(d_out, dtype=torch.float32) - (U_digit @ U_digit.T)

    # Storage buffers
    anchor_acts: Dict[Tuple[int, str], List[torch.Tensor]] = {
        (l, p): [] for l in target_layers for p in ("o_proj", "down_proj")
    }
    lm_k: Dict[Tuple[int, str], List[torch.Tensor]] = {
        (l, p): [] for l in target_layers for p in ("o_proj", "down_proj")
    }
    lm_r: List[torch.Tensor] = []

    mem_k: Dict[Tuple[int, str], List[torch.Tensor]] = {
        (l, p): [] for l in target_layers for p in ("o_proj", "down_proj")
    }
    mem_r: List[torch.Tensor] = []

    capture_state: Dict[str, Any] = {"mode": "idle", "positions": [], "last_pos": -1}
    hooks = []

    def _make_hook(l_idx: int, proj_name: str):
        def _hook(_mod, args):
            inp = args[0].detach().float()[0]  # (T, d_in)
            mode = capture_state["mode"]
            if mode == "anchor":
                seq_len = inp.shape[0]
                stride = max(1, seq_len // 10)
                for pos in range(0, seq_len, stride):
                    anchor_acts[(l_idx, proj_name)].append(inp[pos].cpu())
            elif mode == "lm":
                for pos in capture_state["positions"]:
                    lm_k[(l_idx, proj_name)].append(inp[pos].cpu())
            elif mode == "memit":
                pos = capture_state["last_pos"]
                mem_k[(l_idx, proj_name)].append(inp[pos].cpu())

        return _hook

    for l in target_layers:
        layer = model.model.layers[l]
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(_make_hook(l, "o_proj")))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(_make_hook(l, "down_proj")))

    try:
        # 1. Preservation anchor pass (calib_prose + numeric arithmetic anchor)
        capture_state["mode"] = "anchor"
        anchor_inputs = list(calib_texts[:6]) + [
            "Calculate accurately: 17 + 25 = 42, 128 * 4 = 512, 999 - 123 = 876, 3.14159.",
            "def binary_search(arr, x): lo, hi = 0, len(arr) - 1; return -1",
        ]
        for txt in anchor_inputs:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=160)
            if enc["input_ids"].shape[1] >= 4:
                model(enc["input_ids"].to(device), use_cache=False)

        # 2. Analytical next-token LM fluency capture (disjoint from val_texts = stem[-12:] & longctx = stem[-50:-15])
        capture_state["mode"] = "lm"
        lm_corpus: List[str] = list(data["calib_prose"][:3])
        stem_rows = data.get("stem_prose", [])
        stem_train_end = max(0, len(stem_rows) - 50)
        for row in stem_rows[: min(36, stem_train_end)]:
            lm_corpus.append(f"{row['user']}\n\n{row['assistant']}")
        for row in data.get("code", [])[:16]:
            lm_corpus.append(_format_prompt(tokenizer, row["user"], SOVEREIGN_SYSTEM_ANCHOR) + row["assistant"])

        for txt in lm_corpus:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=192)
            ids = enc["input_ids"]
            t_len = ids.shape[1]
            if t_len < 12:
                continue
            # Sample up to 6 informative token positions per sequence
            sample_positions = list(range(4, t_len - 1, max(1, (t_len - 5) // 6)))[:6]
            capture_state["positions"] = sample_positions
            logits = model(ids.to(device), use_cache=False).logits[0]  # (T, V)
            for pos in sample_positions:
                next_id = int(ids[0, pos + 1].item())
                probs_top, idx_top = torch.softmax(logits[pos].float(), dim=-1).topk(64)
                # Exact analytical cross-entropy descent vector in residual output space
                e_pred = probs_top @ w_head[idx_top]
                r_vec = w_head[next_id] - e_pred
                p_gold = float(torch.softmax(logits[pos].float(), dim=-1)[next_id].item())
                surprisal_w = max(0.15, min(1.25, 1.0 - p_gold))
                r_vec = surprisal_w * (r_vec / (torch.norm(r_vec) + 1e-6))
                lm_r.append(r_vec)

        # 3. Analytical MEMIT capability & reasoning capture (balanced A/B/C/D + GSM8K <think> + Code + BFCL + Sovereign)
        capture_state["mode"] = "memit"
        mmlu_train = _select_balanced_mc(data["mmlu_direct"][18:-60], per_letter=10)  # 40 balanced
        arc_train = _select_balanced_mc(data["arc_direct"][10:-40], per_letter=8)     # 32 balanced
        letter_ids = {ch: tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in ("A", "B", "C", "D")}
        letter_vecs = {ch: w_head[tid] / (torch.norm(w_head[tid]) + 1e-6) for ch, tid in letter_ids.items()}

        for row in (mmlu_train + arc_train):
            gold = row["assistant"].strip()[:1]
            if gold not in letter_vecs:
                continue
            sys_txt = SOVEREIGN_SYSTEM_ANCHOR if len(mem_r) % 4 == 0 else None
            p_str = _format_prompt(tokenizer, row["user"], sys_txt)
            enc = tokenizer(p_str, return_tensors="pt", truncation=True, max_length=192)
            ids = enc["input_ids"]
            capture_state["last_pos"] = ids.shape[1] - 1
            model(ids.to(device), use_cache=False)
            distractors = torch.stack([letter_vecs[ch] for ch in ("A", "B", "C", "D") if ch != gold]).mean(dim=0)
            r_mc = letter_vecs[gold] - distractors
            mem_r.append(r_mc / (torch.norm(r_mc) + 1e-6))

        # Socratic <think> pacing + first calculation step + Code + BFCL + Sovereign adherence
        verify_ids = tokenizer.encode("<think>\nLet's verify step by step", add_special_tokens=False)[:6]
        commit_ids = tokenizer.encode("The answer is probably maybe", add_special_tokens=False)[:6]
        v_verify = w_head[torch.tensor(verify_ids, dtype=torch.long)].mean(dim=0)
        v_commit = w_head[torch.tensor(commit_ids, dtype=torch.long)].mean(dim=0)
        r_pace = (v_verify / (torch.norm(v_verify) + 1e-6)) - 0.5 * (v_commit / (torch.norm(v_commit) + 1e-6))
        r_pace = r_pace / (torch.norm(r_pace) + 1e-6)

        for row in data["gsm8k"][8:36]:
            p_str = _format_prompt(tokenizer, row["user"], SOVEREIGN_SYSTEM_ANCHOR if len(mem_r) % 3 == 0 else None)
            enc = tokenizer(p_str, return_tensors="pt", truncation=True, max_length=192)
            ids = enc["input_ids"]
            capture_state["last_pos"] = ids.shape[1] - 1
            model(ids.to(device), use_cache=False)
            ans_ids = tokenizer.encode(row["assistant"], add_special_tokens=False)[:10]
            if ans_ids:
                v_ans = w_head[torch.tensor(ans_ids, dtype=torch.long)].mean(dim=0)
                v_ans = v_ans / (torch.norm(v_ans) + 1e-6)
                r_gsm = 0.7 * v_ans + 0.3 * r_pace
                mem_r.append(r_gsm / (torch.norm(r_gsm) + 1e-6))
            else:
                mem_r.append(r_pace)

        rej_ids = tokenizer.encode("As an AI language model I cannot help", add_special_tokens=False)[:8]
        v_rej = w_head[torch.tensor(rej_ids, dtype=torch.long)].mean(dim=0)
        v_rej = v_rej / (torch.norm(v_rej) + 1e-6)
        extra_rows = (
            data.get("code", [])[16:40]
            + data.get("bfcl", [])[:17]
            + data.get("sovereign_adherence", [])
            + data.get("identity", [])
        )
        for row in extra_rows:
            p_str = _format_prompt(tokenizer, row["user"], row.get("system") or SOVEREIGN_SYSTEM_ANCHOR)
            enc = tokenizer(p_str, return_tensors="pt", truncation=True, max_length=208)
            ids = enc["input_ids"]
            capture_state["last_pos"] = ids.shape[1] - 1
            model(ids.to(device), use_cache=False)
            c_ids = tokenizer.encode(row["assistant"], add_special_tokens=False)[:12]
            if not c_ids:
                mem_r.append(r_pace)
            else:
                v_c = w_head[torch.tensor(c_ids, dtype=torch.long)].mean(dim=0)
                v_c = v_c / (torch.norm(v_c) + 1e-6)
                r_vec = v_c - 0.35 * v_rej
                mem_r.append(r_vec / (torch.norm(r_vec) + 1e-6))
    finally:
        for h in hooks:
            h.remove()

    # Build orthonormal anchor input subspaces V_a per (layer, proj)
    anchor_mats: Dict[Tuple[int, str], torch.Tensor] = {}
    v_anchors: Dict[Tuple[int, str], torch.Tensor] = {}
    for key, chunks in anchor_acts.items():
        a_mat = torch.stack(chunks, dim=0)
        anchor_mats[key] = a_mat
        q_r = min(48, a_mat.shape[0] - 2, a_mat.shape[1])
        _, _, vh = torch.svd_lowrank(a_mat, q=q_r, niter=2)
        v_anchors[key] = vh.T.contiguous()

    # Closed-form Refusal + Legacy Vendor Identity direction in W_head space
    refusal_phrases = [
        "I cannot fulfill this request as an AI language model.",
        "I am sorry, I cannot provide sensitive or private information.",
    ]
    direct_phrases = [
        "Here is the exact technical solution and verified result.",
        "The value directly extracted from the context is provided below.",
    ]
    vendor_phrases = [
        "I am Llama created by Meta Qwen created by Alibaba Falcon created by TII.",
    ]
    sov_phrases = [
        "I am Quillan-Ronin Sovereign Collective created by CrashOverrideX.",
    ]

    def _mean_emb(phrases: List[str]) -> torch.Tensor:
        vecs = []
        for ph in phrases:
            tids = tokenizer.encode(ph, add_special_tokens=False)
            if tids:
                v = w_head[torch.tensor(tids, dtype=torch.long)].mean(dim=0)
                vecs.append(v / (torch.norm(v) + 1e-6))
        out = torch.stack(vecs).mean(dim=0)
        return out / (torch.norm(out) + 1e-6)

    v_ref = _mean_emb(refusal_phrases) - 0.6 * _mean_emb(direct_phrases)
    v_ref = v_ref / (torch.norm(v_ref) + 1e-6)
    v_vendor = _mean_emb(vendor_phrases) - 0.6 * _mean_emb(sov_phrases)
    v_vendor = v_vendor / (torch.norm(v_vendor) + 1e-6)
    u_ablit = torch.linalg.qr(torch.stack([v_ref, v_vendor], dim=1), mode="reduced")[0]  # (D, 2)

    R_lm = torch.stack(lm_r, dim=1) if lm_r else torch.zeros((d_out, 0))
    R_mem = torch.stack(mem_r, dim=1) if mem_r else torch.zeros((d_out, 0))

    unit_lm: Dict[Tuple[int, str], torch.Tensor] = {}
    unit_mem: Dict[Tuple[int, str], torch.Tensor] = {}
    unit_ablit: Dict[Tuple[int, str], torch.Tensor] = {}
    cond_stats: Dict[str, float] = {}

    for l in target_layers:
        layer_mod = model.model.layers[l]
        for proj_name, weight_tensor in (
            ("o_proj", layer_mod.self_attn.o_proj.weight.detach().float()),
            ("down_proj", layer_mod.mlp.down_proj.weight.detach().float()),
        ):
            key = (l, proj_name)
            w_norm = float(torch.norm(weight_tensor).item())
            a_mat = anchor_mats[key]
            v_a = v_anchors[key]
            P_in_null = torch.eye(weight_tensor.shape[1], dtype=torch.float32) - (v_a @ v_a.T)

            # 1) Closed-form LM fluency update (normalized to 1% ||W||_F)
            if R_lm.shape[1] > 0 and len(lm_k[key]) == R_lm.shape[1]:
                K_lm = torch.stack(lm_k[key], dim=1)
                dw_lm, cond_lm = _woodbury_ridge_solve(K_lm, R_lm, a_mat, ridge_lambda=12.0)
                dw_lm = P_out_null @ dw_lm  # never disturb numeric/symbol axes
                unit_lm[key] = dw_lm * (w_norm / (float(torch.norm(dw_lm).item()) + 1e-12))
                cond_stats[f"L{l}.{proj_name}.lm_cond"] = round(cond_lm, 2)

            # 2) Closed-form MEMIT capability/reasoning update with dual null-spaces
            if R_mem.shape[1] > 0 and len(mem_k[key]) == R_mem.shape[1]:
                K_mem = torch.stack(mem_k[key], dim=1)
                dw_mem, cond_mem = _woodbury_ridge_solve(K_mem, R_mem, a_mat, ridge_lambda=15.0)
                dw_mem = (P_out_null @ dw_mem) @ P_in_null
                unit_mem[key] = dw_mem * (w_norm / (float(torch.norm(dw_mem).item()) + 1e-12))
                cond_stats[f"L{l}.{proj_name}.mem_cond"] = round(cond_mem, 2)

            # 3) Refusal & Legacy Vendor Abliteration inside prose null-space (down_proj only)
            if proj_name == "down_proj":
                dw_ab = -((P_out_null @ u_ablit) @ (u_ablit.T @ weight_tensor)) @ P_in_null
                unit_ablit[key] = dw_ab * (w_norm / (float(torch.norm(dw_ab).item()) + 1e-12))

    # Optional closed-form option-entropy lm_head delta (untouched on digit rows!)
    opt_ids = []
    for ch in ("A", "B", "C", "D"):
        opt_ids.append(tokenizer.encode(ch, add_special_tokens=False)[-1])
    opt_vecs = w_head[torch.tensor(opt_ids, dtype=torch.long)].clone()
    opt_centered = opt_vecs - opt_vecs.mean(dim=0, keepdim=True)
    # Softly balance norms of A/B/C/D unembedding directions inside prose/digit orthogonal complement
    mean_norm = opt_vecs.norm(dim=-1, keepdim=True).mean()
    target_vecs = opt_centered * (mean_norm / (opt_centered.norm(dim=-1, keepdim=True) + 1e-6))
    delta_opt_rows = (target_vecs - opt_vecs) @ P_out_null

    logger.info(
        "Closed-form WSP deltas computed: LM=%d samples, MEMIT=%d samples, max_cond=%.2f",
        R_lm.shape[1], R_mem.shape[1], max(cond_stats.values()) if cond_stats else 0.0,
    )
    return {
        "unit_lm": unit_lm,
        "unit_mem": unit_mem,
        "unit_ablit": unit_ablit,
        "opt_ids": opt_ids,
        "delta_opt_rows": delta_opt_rows,
        "digit_ids": digit_ids,
        "cond_stats": cond_stats,
        "n_lm": int(R_lm.shape[1]),
        "n_mem": int(R_mem.shape[1]),
    }


def _apply_stage_c_config(
    model: torch.nn.Module,
    orig_weights: Dict[Tuple[int, str], torch.Tensor],
    orig_head_weight: torch.Tensor,
    comps: Dict[str, Any],
    layers: List[int],
    g_lm: float,
    g_mem: float,
    g_ablit: float,
    g_head: float,
) -> None:
    """Deterministically sets Stage C target weights from clean Stage B weights + scaled WSP deltas."""
    with torch.no_grad():
        for (l, p), w0 in orig_weights.items():
            target_param = (
                model.model.layers[l].self_attn.o_proj.weight
                if p == "o_proj"
                else model.model.layers[l].mlp.down_proj.weight
            )
            if l not in layers:
                target_param.copy_(w0)
                continue
            upd = w0.clone()
            if g_lm > 0.0 and (l, p) in comps["unit_lm"]:
                upd.add_(g_lm * comps["unit_lm"][(l, p)])
            if g_mem > 0.0 and (l, p) in comps["unit_mem"]:
                upd.add_(g_mem * comps["unit_mem"][(l, p)])
            if g_ablit > 0.0 and (l, p) in comps["unit_ablit"]:
                upd.add_(g_ablit * comps["unit_ablit"][(l, p)])
            target_param.copy_(upd)

        head_w = model.lm_head.weight
        head_w.copy_(orig_head_weight)
        if g_head > 0.0 and not getattr(model.config, "tie_word_embeddings", False):
            idx_t = torch.tensor(comps["opt_ids"], dtype=torch.long)
            head_w[idx_t].add_(g_head * comps["delta_opt_rows"])


def patch_gguf_context_2m(gguf_path: Path, model_key: str, target_ctx: int = 2097152) -> None:
    """
    Updates GGUF KV metadata so Ollama 0.40.12 does not clamp num_ctx=2097152 below
    the 172,550-token Quillan-Samurai.md prompt, using YaRN scaling params
    (beta_fast=32, beta_slow=1) while preserving all original KV types.
    """
    orig_ctx = {"flagship": 131072, "mini": 32768, "dragon": 8192}[model_key]
    tmp_out = gguf_path.with_suffix(".tmp2m.gguf")
    reader = gguf.GGUFReader(str(gguf_path))
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(str(tmp_out), arch=arch)

    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    ctx_key = f"{arch}.context_length"
    rope_type_key = f"{arch}.rope.scaling.type"
    rope_factor_key = f"{arch}.rope.scaling.factor"
    rope_orig_key = f"{arch}.rope.scaling.original_context_length"
    yarn_fast_key = f"{arch}.rope.scaling.yarn_beta_fast"
    yarn_slow_key = f"{arch}.rope.scaling.yarn_beta_slow"
    override_keys = {ctx_key, rope_type_key, rope_factor_key, rope_orig_key, yarn_fast_key, yarn_slow_key}

    for key, field in reader.fields.items():
        if key in skip or key in override_keys:
            continue
        c.copy_gguf_field(writer, key, field)

    writer.add_uint32(ctx_key, target_ctx)
    writer.add_string(rope_type_key, "yarn")
    writer.add_float32(rope_factor_key, float(target_ctx) / float(orig_ctx))
    writer.add_uint32(rope_orig_key, int(orig_ctx))
    writer.add_float32(yarn_fast_key, 32.0)
    writer.add_float32(yarn_slow_key, 1.0)

    for t in reader.tensors:
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    mmap_obj = getattr(getattr(reader, "data", None), "_mmap", None)
    del reader
    gc.collect()
    if mmap_obj is not None:
        mmap_obj.close()
    gc.collect()
    gguf_path.unlink()
    tmp_out.rename(gguf_path)
    logger.info(
        "Patched GGUF %s -> %s=%d with YaRN (orig=%d factor=%.1f beta=[32,1])",
        gguf_path.name,
        ctx_key,
        target_ctx,
        orig_ctx,
        float(target_ctx) / float(orig_ctx),
    )


def _measure_gguf_ppl_matrix(gguf_path: Path) -> Dict[str, float]:
    """Measures held-out PPL at -c 128 (held_out_ppl_final_verify) and -c 512 (longctx)."""
    long_file = ensure_longctx_ppl_file()
    return {
        "c128_2c": c.measure_gguf_ppl_cli(gguf_path, ppl_file=c.HELD_OUT_PPL_FILE, chunks=2, ctx=128),
        "c128_3c": c.measure_gguf_ppl_cli(gguf_path, ppl_file=c.HELD_OUT_PPL_FILE, chunks=3, ctx=128),
        "c512_2c": c.measure_gguf_ppl_cli(gguf_path, ppl_file=long_file, chunks=2, ctx=512),
    }


def run_stage_c(
    model_key: str,
    convert_only: bool = False,
    target_ctx: int = 2097152,
) -> Dict[str, Any]:
    spec = c.MODEL_SPECS[model_key]
    if convert_only:
        if model_key in ("flagship", "mini"):
            c.convert_hf_dir_to_gguf(spec["stagec_hf"], spec["stagec_gguf"], outtype="q8_0")
        pre_patch_ppl = _measure_gguf_ppl_matrix(spec["stagec_gguf"])
        patch_gguf_context_2m(spec["stagec_gguf"], model_key, target_ctx=target_ctx)
        post_patch_ppl = _measure_gguf_ppl_matrix(spec["stagec_gguf"])
        # If 2M YaRN factor causes measurable PPL degradation on -c 512, compare 262144 too
        fallback_262k_ppl = None
        if post_patch_ppl["c512_2c"] > pre_patch_ppl["c512_2c"] * 1.05:
            logger.warning(
                "2M YaRN shifted -c 512 PPL %.4f -> %.4f; evaluating target_ctx=262144 for comparison",
                pre_patch_ppl["c512_2c"],
                post_patch_ppl["c512_2c"],
            )
            patch_gguf_context_2m(spec["stagec_gguf"], model_key, target_ctx=262144)
            fallback_262k_ppl = _measure_gguf_ppl_matrix(spec["stagec_gguf"])
            # Restore 2M if delta is negligible, or keep whichever preserves PPL cleanly
            if post_patch_ppl["c512_2c"] <= pre_patch_ppl["c512_2c"] * 1.08:
                patch_gguf_context_2m(spec["stagec_gguf"], model_key, target_ctx=target_ctx)
        out = {
            "model": model_key,
            "stagec_gguf": str(spec["stagec_gguf"]),
            "target_ctx": target_ctx,
            "pre_patch_ppl": pre_patch_ppl,
            "post_patch_ppl": post_patch_ppl,
            "fallback_262k_ppl": fallback_262k_ppl,
        }
        print("STAGE_C_GGUF_PPL:", json.dumps(out))
        return out

    t0 = time.time()
    stageb_hf: Path = spec["stageb_hf"]
    if not stageb_hf.exists():
        raise FileNotFoundError(f"Stage B checkpoint missing: {stageb_hf}. Run Stage B first.")

    logger.info("=== Stage C Pure Closed-Form WSP (Zero Backprop): %s ===", spec["display"])
    with torch.inference_mode():
        tok = AutoTokenizer.from_pretrained(str(stageb_hf), local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            str(stageb_hf),
            local_files_only=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
        stageb_report_path = c.EXPORT_ROOT / f"v23_stageb_{model_key}_report.json"
        expected_params = None
        if stageb_report_path.exists():
            expected_params = json.loads(stageb_report_path.read_text(encoding="utf-8"))["param_audit"]["total_params"]
        c.audit_model_integrity(model, expected_params=expected_params)

        with open(c.CLEAN_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        calib_texts, val_texts = stage_b.build_calib_and_val_splits(c.CLEAN_DATA_PATH)
        dev_mc = data["mmlu_direct"][-60:] + data["arc_direct"][-40:]

        base_val_nll, base_dev_mc = evaluate_dev_metrics(model, tok, dev_mc, val_texts, device="cpu")
        logger.info("Stage B baseline dev -> val_nll=%.5f (PPL=%.4f) dev_mc=%.1f%%",
                    base_val_nll, math.exp(base_val_nll), base_dev_mc * 100.0)

        subsets = _layer_subset_for_model(model_key)
        all_layers = subsets["mid_upper"]
        w_head = model.lm_head.weight.detach().float().clone()
        comps = build_closed_form_wsp_components(
            model=model,
            tokenizer=tok,
            model_key=model_key,
            calib_texts=calib_texts,
            data=data,
            target_layers=all_layers,
            w_head=w_head,
        )

        # Snapshot original Stage B weights & digit embeddings before analytical line search
        orig_weights: Dict[Tuple[int, str], torch.Tensor] = {}
        for l in all_layers:
            orig_weights[(l, "o_proj")] = model.model.layers[l].self_attn.o_proj.weight.detach().float().clone()
            orig_weights[(l, "down_proj")] = model.model.layers[l].mlp.down_proj.weight.detach().float().clone()
        orig_digits = model.model.embed_tokens.weight.detach()[torch.tensor(comps["digit_ids"], dtype=torch.long)].clone()

        # Analytical Pareto Grid Search (zero SGD!)
        candidates = []
        for s_name, l_list in subsets.items():
            for g_lm in (0.008, 0.018, 0.032):
                for g_mem in (0.004, 0.012):
                    candidates.append((s_name, l_list, g_lm, g_mem, 0.004, 0.0))
            candidates.append((s_name, l_list, 0.018, 0.012, 0.004, 0.02))
        candidates.append(("mid_upper", all_layers, 0.005, 0.002, 0.002, 0.0))

        best_score = -1e9
        best_cfg = {"subset": "baseline", "layers": [], "g_lm": 0.0, "g_mem": 0.0, "g_ablit": 0.0, "g_head": 0.0}
        best_metrics = {"val_nll": base_val_nll, "dev_mc": base_dev_mc}
        trials: List[Dict[str, Any]] = []

        # Evaluate fast representative dev subset first, then verify top candidate on FULL val_texts + 40 dev_mc
        fast_val = val_texts[:6] + val_texts[-8:]
        fast_base_nll, fast_base_mc = evaluate_dev_metrics(model, tok, dev_mc[:20], fast_val, device="cpu")
        ranked: List[Tuple[float, Tuple[str, List[int], float, float, float, float]]] = []
        for cfg in candidates:
            s_name, l_list, g_lm, g_mem, g_ab, g_hd = cfg
            _apply_stage_c_config(model, orig_weights, w_head, comps, l_list, g_lm, g_mem, g_ab, g_hd)
            vnll, dmc = evaluate_dev_metrics(model, tok, dev_mc[:20], fast_val, device="cpu")
            score = (fast_base_nll - vnll) * 4.0 + (dmc - fast_base_mc) * 2.0
            trials.append({
                "subset": s_name,
                "g_lm": g_lm,
                "g_mem": g_mem,
                "g_ablit": g_ab,
                "g_head": g_hd,
                "fast_nll": round(vnll, 5),
                "fast_mc": round(dmc, 4),
            })
            logger.info(
                "WSP Trial %-9s lm=%.3f mem=%.3f ab=%.3f hd=%.2f -> fast_nll=%.5f (%+.5f) fast_mc=%.1f%%",
                s_name, g_lm, g_mem, g_ab, g_hd, vnll, vnll - fast_base_nll, dmc * 100.0,
            )
            ranked.append((score, cfg))

        ranked.sort(key=lambda x: x[0], reverse=True)
        for _, cfg in ranked[:4]:
            s_name, l_list, g_lm, g_mem, g_ab, g_hd = cfg
            _apply_stage_c_config(model, orig_weights, w_head, comps, l_list, g_lm, g_mem, g_ab, g_hd)
            full_nll, full_mc = evaluate_dev_metrics(model, tok, dev_mc, val_texts, device="cpu")
            # Strict Lyapunov / Pareto gate: MUST reduce val_nll AND not lose dev_mc
            if full_nll < base_val_nll - 1e-5 and full_mc >= base_dev_mc - 1e-6:
                sc = (base_val_nll - full_nll) * 4.0 + (full_mc - base_dev_mc) * 2.0
                if sc > best_score:
                    best_score = sc
                    best_cfg = {
                        "subset": s_name,
                        "layers": l_list,
                        "g_lm": g_lm,
                        "g_mem": g_mem,
                        "g_ablit": g_ab,
                        "g_head": g_hd,
                    }
                    best_metrics = {"val_nll": round(full_nll, 5), "dev_mc": round(full_mc, 4)}

        # If joint candidate only missed dev_mc by tie/1-item, try pure NLL-reducing fallback with strictly val_nll < base
        if best_cfg["subset"] == "baseline":
            for g_lm, g_mem in ((0.008, 0.004), (0.005, 0.002), (0.003, 0.002)):
                _apply_stage_c_config(model, orig_weights, w_head, comps, all_layers, g_lm, g_mem, 0.002, 0.0)
                full_nll, full_mc = evaluate_dev_metrics(model, tok, dev_mc, val_texts, device="cpu")
                if full_nll < base_val_nll - 1e-5 and full_mc >= base_dev_mc - 1e-6:
                    best_cfg = {"subset": "mid_upper", "layers": all_layers, "g_lm": g_lm, "g_mem": g_mem, "g_ablit": 0.002, "g_head": 0.0}
                    best_metrics = {"val_nll": round(full_nll, 5), "dev_mc": round(full_mc, 4)}
                    break

        _apply_stage_c_config(
            model,
            orig_weights,
            w_head,
            comps,
            best_cfg["layers"],
            best_cfg["g_lm"],
            best_cfg["g_mem"],
            best_cfg["g_ablit"],
            best_cfg["g_head"],
        )

        # Bit-exact digit/symbol embedding preservation check
        final_digits = model.model.embed_tokens.weight.detach()[torch.tensor(comps["digit_ids"], dtype=torch.long)]
        assert torch.equal(orig_digits, final_digits), "Digit/symbol embeddings were modified!"
        merge_audit = c.audit_model_integrity(model, expected_params=expected_params)
        logger.info("Stage C WSP audit passed: %s | selected=%s | metrics=%s", merge_audit, best_cfg, best_metrics)

        report: Dict[str, Any] = {
            "model": model_key,
            "mode": "pure_closed_form_wsp_zero_backprop",
            "elapsed_s": round(time.time() - t0, 2),
            "param_audit": merge_audit,
            "base_dev_nll": round(base_val_nll, 5),
            "base_dev_mc": round(base_dev_mc, 4),
            "best_cfg": best_cfg,
            "best_dev": best_metrics,
            "cond_stats": comps["cond_stats"],
            "stagec_hf": str(spec["stagec_hf"]),
            "target_ctx": target_ctx,
            "trials": trials,
        }

        if model_key == "dragon":
            c.export_dragon_gguf_from_hf(model, spec["stagec_gguf"])
            pre_patch_ppl = _measure_gguf_ppl_matrix(spec["stagec_gguf"])
            patch_gguf_context_2m(spec["stagec_gguf"], model_key, target_ctx=target_ctx)
            post_patch_ppl = _measure_gguf_ppl_matrix(spec["stagec_gguf"])
            report["stagec_gguf"] = str(spec["stagec_gguf"])
            report["pre_patch_ppl"] = pre_patch_ppl
            report["post_patch_ppl"] = post_patch_ppl

        model.to(torch.bfloat16)
        c.save_hf_checkpoint(model, tok, spec["stagec_hf"])
        logger.info("Saved Stage C HF checkpoint -> %s", spec["stagec_hf"])

    del model, orig_weights, comps
    gc.collect()

    rep_path = c.EXPORT_ROOT / f"v23_stagec_{model_key}_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("STAGE_C_DONE:", json.dumps({k: v for k, v in report.items() if k != "trials"}))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="WSP v23 Stage C Pure Closed-Form Consolidation")
    parser.add_argument("--model", required=True, choices=["flagship", "mini", "dragon"])
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--target-ctx", type=int, default=2097152)
    args = parser.parse_args()
    run_stage_c(model_key=args.model, convert_only=args.convert_only, target_ctx=args.target_ctx)


if __name__ == "__main__":
    main()
