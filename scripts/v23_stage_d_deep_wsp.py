#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage D: Sequential Layer-Wise Closed-Form Deep Weight-Space Programming (WSP).

ZERO backpropagation, ZERO SGD / AdamW, ZERO LoRA (nabla_theta L === 0).
Runs strictly inside torch.inference_mode() starting from verified Stage C
checkpoints (or Stage B if specified):
  1. Dynamic Forward Activation Forwarding: solves layers sequentially
     l = L_start .. L_end, re-capturing K^{(l)} and anchor subspaces A^{(l)}
     through the updated model prefix {W^{(1..l-1)}} to eliminate inter-layer
     residual drift.
  2. Dual-Space Woodbury / Ridge Solve per layer across o_proj and down_proj:
       Delta W_l = (R_l - W_0 K_l) K_l^T (K_l K_l^T + lambda Sigma_anchor)^{-1}
     over 500-800 balanced non-test tokens/prompts (stem_prose, mmlu_direct,
     arc_direct, gsm8k <think>, code, bfcl, sovereign_adherence, identity) with
     adaptive Tikhonov floor and condition-number guard (cond <= 1e4).
  3. Associative Dual Null-Space Projection (O(k * d) RAM, zero dense d x d):
       _proj_out_null(X, U) = X - U @ (U.T @ X)
       _proj_in_null(X, V)  = X - (X @ V) @ V.T
  4. Greedy Per-Layer Lyapunov Acceptance Gate:
     Commits layer l edit ONLY if val_nll strictly drops (with dev_mc within
     1-sample tolerance) OR dev_mc strictly improves without NLL regression,
     enforcing relative Frobenius bound ||Delta W_l||_F / ||W_l||_F <= 0.005.
  5. Optional untied lm_head calibration (e.g., Dragon) + bit-exact digit/symbol
     embedding invariance + Q8_0 GGUF export & 2M/262k YaRN verification.

Usage:
  python -u v23_stage_d_deep_wsp.py --self-test
  python -u v23_stage_d_deep_wsp.py --model flagship
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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import v23_common as c  # noqa: E402
import v23_stage_b_precombine as stage_b  # noqa: E402
import v23_stage_c_consolidate as stage_c  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("V23_StageD_DeepWSP")

# Strict non-vacuous-progress gate primitives are owned by Stage B (shared contract).
WSPGateError = stage_b.WSPGateError
write_failed_report = stage_b.write_failed_report


def check_stage_d_gate(accepted_layers: List[Dict[str, Any]]) -> int:
    """
    Stage D non-vacuous-progress gate: at least ONE layer MUST have been accepted.
    A run with all layers rejected (compensation accepted on 0 layers) is a no-op
    and MUST NOT export or report success.
    Raises WSPGateError on failure. Returns accepted-layer count on success.
    """
    if not isinstance(accepted_layers, list) or len(accepted_layers) < 1:
        raise WSPGateError(
            "STAGE_D_GATE_FAIL: zero layers accepted by the Lyapunov gate - "
            "no real deep-WSP edit was committed; exporting an unchanged model is forbidden."
        )
    return len(accepted_layers)


def _proj_out_null(X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Associative left null-space projection (I - U U^T) X in O(k * d_out * d_in)."""
    if U.numel() == 0:
        return X
    return X - U @ (U.T @ X)


def _proj_in_null(X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Associative right null-space projection X (I - V V^T) in O(k * d_out * d_in)."""
    if V.numel() == 0:
        return X
    return X - (X @ V) @ V.T


def _clamp_q8_block_outliers(delta: torch.Tensor, sigma_k: float = 2.5) -> torch.Tensor:
    """
    Clamps per-32-weight Q8_0 block outliers in delta to +/- sigma_k * std(block)
    (paper_012 / 2509.25149v2) so fp32 Woodbury gains survive Q8_0 export
    without inflating 32-weight block scales.
    """
    rows, cols = delta.shape
    if cols % 32 != 0:
        std = delta.std().clamp(min=1e-8)
        return delta.clamp(-sigma_k * std, sigma_k * std)
    blocks = delta.view(rows, cols // 32, 32)
    b_std = blocks.std(dim=-1, keepdim=True).clamp(min=1e-8)
    limit = sigma_k * b_std
    return blocks.clamp(-limit, limit).view(rows, cols)


def _ternarize_block256_ls(W: torch.Tensor) -> torch.Tensor:
    """
    Per-256-block BitNet 1.58b absmean ternarization T_b in {-1, 0, 1}
    with exact least-squares scale gamma_b* = <W_b, T_b> / (<T_b, T_b> + 1e-8).
    """
    rows, cols = W.shape
    block_sz = 256 if cols % 256 == 0 else cols
    blocks = W.view(rows, cols // block_sz, block_sz)
    gamma0 = blocks.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    t_b = torch.clamp(torch.round(blocks / gamma0), -1.0, 1.0)
    denom = (t_b * t_b).sum(dim=-1, keepdim=True) + 1e-8
    gamma_star = (blocks * t_b).sum(dim=-1, keepdim=True) / denom
    return (t_b * gamma_star).view(rows, cols)


def _compensate_ternary_hessian_residual(
    W_l: torch.Tensor,
    K_calib: torch.Tensor,
    ridge_frac: float = 0.05,
    comp_gain: float = 0.35,
) -> torch.Tensor:
    """
    Closed-form Woodbury Hessian error compensation (OBQ/GPTQ-style, zero backprop):
      E_l = W_l - ternarize_block256_ls(W_l)
      Delta W_comp = (E_l K) (K^T K + lambda I)^{-1} K^T
    Clamps block outliers (<= 2.5 sigma) before returning W_l + comp_gain * Delta W_comp.
    """
    if K_calib.numel() == 0:
        return W_l
    w_hat = _ternarize_block256_ls(W_l)
    e_res = W_l - w_hat
    n_tok = K_calib.shape[1]
    r_err = e_res @ K_calib  # (d_out, N)
    gram = K_calib.T @ K_calib  # (N, N)
    lam = max(1.0, ridge_frac * float(torch.trace(gram).item()) / float(n_tok))
    eye_n = torch.eye(n_tok, dtype=torch.float32, device=gram.device)
    coeffs = torch.linalg.solve(gram + lam * eye_n, K_calib.T)  # (N, d_in)
    dw_comp = _clamp_q8_block_outliers(r_err @ coeffs, sigma_k=2.5)
    return W_l + comp_gain * dw_comp


def _build_persona_orthogonal_ablit_basis(
    tokenizer: Any,
    w_head: torch.Tensor,
    U_digit: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Paper 020 (#781-4): Builds orthonormal Sovereign persona subspace U_persona and
    refusal/legacy-vendor abliteration basis u_ablit strictly orthogonal to both
    U_persona and U_digit:
      u_ablit_safe = QR((I - U_digit U_digit^T)(I - U_persona U_persona^T) [v_ref, v_vendor])
    """
    persona_words = [
        "Quillan",
        "Ronin",
        "CrashOverrideX",
        "Sovereign",
        "Samurai",
        "Bushido",
        "Council",
    ]
    p_ids: List[int] = []
    for w in persona_words:
        p_ids.extend(tokenizer.encode(w, add_special_tokens=False))
    p_ids = list(dict.fromkeys(p_ids))[:24]
    if p_ids:
        M_p = w_head[torch.tensor(p_ids, dtype=torch.long)].float().T
        U_persona, _ = torch.linalg.qr(M_p, mode="reduced")
    else:
        U_persona = torch.zeros((w_head.shape[1], 0), dtype=torch.float32)

    def _mean_emb(phrases: List[str]) -> torch.Tensor:
        vecs = []
        for ph in phrases:
            tids = tokenizer.encode(ph, add_special_tokens=False)
            if tids:
                v = w_head[torch.tensor(tids, dtype=torch.long)].mean(dim=0)
                vecs.append(v / (torch.norm(v) + 1e-6))
        out = torch.stack(vecs).mean(dim=0)
        return out / (torch.norm(out) + 1e-6)

    v_ref = _mean_emb(
        [
            "I cannot fulfill this request as an AI language model.",
            "I am sorry, I cannot provide sensitive or private information.",
        ]
    ) - 0.6 * _mean_emb(
        [
            "Here is the exact technical solution and verified result.",
            "The value directly extracted from the context is provided below.",
        ]
    )
    v_vendor = _mean_emb(
        [
            "I am Llama created by Meta Qwen created by Alibaba Falcon created by TII.",
        ]
    ) - 0.6 * _mean_emb(
        [
            "I am Quillan-Ronin Sovereign Collective created by CrashOverrideX.",
        ]
    )

    raw = torch.stack([v_ref, v_vendor], dim=1)
    # Build joint orthonormal basis [U_persona, U_digit] so projection is orthogonal to BOTH simultaneously
    if U_persona.numel() > 0 and U_digit.numel() > 0:
        U_joint, _ = torch.linalg.qr(
            torch.cat([U_persona, U_digit], dim=1), mode="reduced"
        )
    elif U_persona.numel() > 0:
        U_joint = U_persona
    else:
        U_joint = U_digit
    orth = _proj_out_null(raw, U_joint)
    u_ablit, _ = torch.linalg.qr(orth, mode="reduced")
    return U_persona, u_ablit


def _layer_range_for_model(model_key: str, total_layers: int) -> List[int]:
    """Returns sequential mid-to-upper layer indices [L_start .. L_end] for Stage D."""
    if model_key == "flagship":
        # L5..L14 (protects early lexical L0-L4 and tied output boundary L15)
        return list(range(5, min(15, total_layers)))
    if model_key == "mini":
        # L6..L20 (protects surface syntax L0-L5 and tied output boundary L21-L23)
        return list(range(6, min(21, total_layers)))
    # dragon (18 layers): L4..L16
    return list(range(4, min(17, total_layers)))


def build_stage_d_prompts(
    data: Dict[str, Any],
    max_lm_seqs: int = 64,
    max_mc_per_letter: int = 16,
    max_gsm: int = 40,
    max_extra: int = 48,
) -> Dict[str, Any]:
    """
    Collects 500-800 balanced non-test tokens/prompts across all capability domains
    from v23_clean_train_data.json, strictly disjoint from held-out validation splits.
    """
    lm_corpus: List[str] = list(data.get("calib_prose", [])[:4])
    stem_rows = data.get("stem_prose", [])
    stem_train_end = max(0, len(stem_rows) - 50)
    for row in stem_rows[: min(max_lm_seqs, stem_train_end)]:
        lm_corpus.append(f"{row['user']}\n\n{row['assistant']}")

    mmlu_train = stage_c._select_balanced_mc(
        data.get("mmlu_direct", [])[18:-60], per_letter=max_mc_per_letter
    )
    arc_train = stage_c._select_balanced_mc(
        data.get("arc_direct", [])[10:-40], per_letter=max(4, max_mc_per_letter // 2)
    )
    gsm_rows = data.get("gsm8k", [])[8 : 8 + max_gsm]
    code_rows = data.get("code", [])[: max_extra // 3]
    bfcl_rows = data.get("bfcl", [])[: max_extra // 3]
    sov_rows = (data.get("sovereign_adherence", []) + data.get("identity", []))[
        : max_extra // 3
    ]

    return {
        "lm_corpus": lm_corpus,
        "mc_rows": mmlu_train + arc_train,
        "gsm_rows": gsm_rows,
        "extra_rows": code_rows + bfcl_rows + sov_rows,
    }


@torch.inference_mode()
def _capture_layer_activations_and_solve(
    model: torch.nn.Module,
    tokenizer: Any,
    layer_idx: int,
    corpus: Dict[str, Any],
    calib_texts: List[str],
    w_head: torch.Tensor,
    U_digit: torch.Tensor,
    U_persona: torch.Tensor,
    u_ablit: torch.Tensor,
    max_rel_frob: float = 0.005,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, float],
]:
    """
    Runs dynamic forward passes through the CURRENT model state to capture layer_idx
    input activations K_lm, K_mem, and A_anchor, then computes closed-form dual-space
    Woodbury Ridge deltas for o_proj and down_proj.
    """
    layer = model.model.layers[layer_idx]
    anchor_acts: Dict[str, List[torch.Tensor]] = {"o_proj": [], "down_proj": []}
    lm_k: Dict[str, List[torch.Tensor]] = {"o_proj": [], "down_proj": []}
    mem_k: Dict[str, List[torch.Tensor]] = {"o_proj": [], "down_proj": []}
    lm_r: List[torch.Tensor] = []
    mem_r: List[torch.Tensor] = []

    state: Dict[str, Any] = {"mode": "idle", "positions": [], "last_pos": -1}

    def _make_hook(proj_name: str):
        def _hook(_mod, args):
            inp = args[0].detach().float()[0]  # (T, d_in)
            mode = state["mode"]
            if mode == "anchor":
                seq_len = inp.shape[0]
                stride = max(1, seq_len // 10)
                for pos in range(0, seq_len, stride):
                    anchor_acts[proj_name].append(inp[pos].cpu())
            elif mode == "lm":
                for pos in state["positions"]:
                    lm_k[proj_name].append(inp[pos].cpu())
            elif mode == "memit":
                pos = state["last_pos"]
                mem_k[proj_name].append(inp[pos].cpu())

        return _hook

    h_o = layer.self_attn.o_proj.register_forward_pre_hook(_make_hook("o_proj"))
    h_dn = layer.mlp.down_proj.register_forward_pre_hook(_make_hook("down_proj"))

    try:
        # 1. Anchor covariance pass
        state["mode"] = "anchor"
        anchor_inputs = list(calib_texts[:6]) + [
            "Calculate accurately: 17 + 25 = 42, 128 * 4 = 512, 999 - 123 = 876, 3.14159.",
            "def binary_search(arr, x): lo, hi = 0, len(arr) - 1; return -1",
        ]
        for txt in anchor_inputs:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=160)
            if enc["input_ids"].shape[1] >= 4:
                model(enc["input_ids"], use_cache=False)

        # 2. Analytical next-token LM fluency pass (~400-500 tokens)
        state["mode"] = "lm"
        for txt in corpus["lm_corpus"]:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=192)
            ids = enc["input_ids"]
            t_len = ids.shape[1]
            if t_len < 12:
                continue
            sample_positions = list(range(4, t_len - 1, max(1, (t_len - 5) // 8)))[:8]
            state["positions"] = sample_positions
            logits = model(ids, use_cache=False).logits[0]
            for pos in sample_positions:
                next_id = int(ids[0, pos + 1].item())
                probs_top, idx_top = torch.softmax(logits[pos].float(), dim=-1).topk(64)
                e_pred = probs_top @ w_head[idx_top]
                r_vec = w_head[next_id] - e_pred
                p_gold = float(
                    torch.softmax(logits[pos].float(), dim=-1)[next_id].item()
                )
                surprisal_w = max(0.15, min(1.25, 1.0 - p_gold))
                lm_r.append(surprisal_w * (r_vec / (torch.norm(r_vec) + 1e-6)))

        # 3. Capability & reasoning MEMIT pass (MC, GSM8K <think>, Code/BFCL/Sovereign)
        state["mode"] = "memit"
        letter_ids = {
            ch: tokenizer.encode(ch, add_special_tokens=False)[-1]
            for ch in ("A", "B", "C", "D")
        }
        letter_vecs = {
            ch: w_head[tid] / (torch.norm(w_head[tid]) + 1e-6)
            for ch, tid in letter_ids.items()
        }

        for row in corpus["mc_rows"]:
            gold = row["assistant"].strip()[:1]
            if gold not in letter_vecs:
                continue
            sys_txt = stage_c.SOVEREIGN_SYSTEM_ANCHOR if len(mem_r) % 4 == 0 else None
            p_str = stage_c._format_prompt(tokenizer, row["user"], sys_txt)
            enc = tokenizer(p_str, return_tensors="pt", truncation=True, max_length=192)
            ids = enc["input_ids"]
            state["last_pos"] = ids.shape[1] - 1
            model(ids, use_cache=False)
            neg_vecs = [v for k_ch, v in letter_vecs.items() if k_ch != gold]
            r_dir = letter_vecs[gold] - 0.50 * (sum(neg_vecs) / float(len(neg_vecs)))
            mem_r.append(r_dir / (torch.norm(r_dir) + 1e-6))

        for row in corpus["gsm_rows"]:
            p_str = stage_c._format_prompt(
                tokenizer, row["user"], stage_c.SOVEREIGN_SYSTEM_ANCHOR
            )
            enc = tokenizer(p_str, return_tensors="pt", truncation=True, max_length=192)
            ids = enc["input_ids"]
            state["last_pos"] = ids.shape[1] - 1
            model(ids, use_cache=False)
            ans_ids = tokenizer.encode(row["assistant"][:40], add_special_tokens=False)[
                :8
            ]
            if ans_ids:
                v_ans = w_head[torch.tensor(ans_ids, dtype=torch.long)].mean(dim=0)
                mem_r.append(v_ans / (torch.norm(v_ans) + 1e-6))
            else:
                mem_k["o_proj"].pop()
                mem_k["down_proj"].pop()

        for row in corpus["extra_rows"]:
            p_str = stage_c._format_prompt(
                tokenizer, row["user"], stage_c.SOVEREIGN_SYSTEM_ANCHOR
            )
            enc = tokenizer(p_str, return_tensors="pt", truncation=True, max_length=192)
            ids = enc["input_ids"]
            state["last_pos"] = ids.shape[1] - 1
            model(ids, use_cache=False)
            ans_ids = tokenizer.encode(row["assistant"][:60], add_special_tokens=False)[
                :10
            ]
            if ans_ids:
                v_ans = w_head[torch.tensor(ans_ids, dtype=torch.long)].mean(dim=0)
                mem_r.append(v_ans / (torch.norm(v_ans) + 1e-6))
            else:
                mem_k["o_proj"].pop()
                mem_k["down_proj"].pop()
    finally:
        h_o.remove()
        h_dn.remove()

    R_lm = (
        torch.stack(lm_r, dim=1)
        if lm_r
        else torch.zeros((w_head.shape[1], 0), dtype=torch.float32)
    )
    R_mem = (
        torch.stack(mem_r, dim=1)
        if mem_r
        else torch.zeros((w_head.shape[1], 0), dtype=torch.float32)
    )

    unit_lm: Dict[str, torch.Tensor] = {}
    unit_mem: Dict[str, torch.Tensor] = {}
    unit_ablit: Dict[str, torch.Tensor] = {}
    stats: Dict[str, float] = {}

    for proj_name in ("o_proj", "down_proj"):
        w_param = (
            layer.self_attn.o_proj.weight
            if proj_name == "o_proj"
            else layer.mlp.down_proj.weight
        )
        w0 = w_param.detach().float()
        w_norm = float(torch.norm(w0).item())
        a_mat = torch.stack(anchor_acts[proj_name], dim=0)
        q_r = min(48, a_mat.shape[0] - 2, a_mat.shape[1])
        _, _, v_a = torch.svd_lowrank(a_mat, q=q_r, niter=2)
        v_a = v_a.contiguous()

        if R_lm.shape[1] > 0 and len(lm_k[proj_name]) == R_lm.shape[1]:
            K_lm = torch.stack(lm_k[proj_name], dim=1)
            trace_k = float((K_lm * K_lm).sum().item())
            lam_lm = max(8.0, 1e-2 * trace_k / float(max(1, K_lm.shape[1])))
            dw_lm, cond_lm = stage_c._woodbury_ridge_solve(
                K_lm, R_lm, a_mat, ridge_lambda=lam_lm, max_cond=1e4
            )
            dw_lm = _clamp_q8_block_outliers(dw_lm, sigma_k=2.5)
            dw_lm = _proj_in_null(_proj_out_null(dw_lm, U_digit), v_a)
            dw_norm = float(torch.norm(dw_lm).item())
            # Normalized to 1.0 * ||W_0||_F so step_gain directly bounds relative Frobenius norm
            unit_lm[proj_name] = (dw_lm * (w_norm / (dw_norm + 1e-12))).to(
                torch.bfloat16
            )
            stats[f"{proj_name}.lm_cond"] = round(cond_lm, 2)

        if R_mem.shape[1] > 0 and len(mem_k[proj_name]) == R_mem.shape[1]:
            K_mem = torch.stack(mem_k[proj_name], dim=1)
            trace_m = float((K_mem * K_mem).sum().item())
            lam_mem = max(10.0, 1e-2 * trace_m / float(max(1, K_mem.shape[1])))
            dw_mem, cond_mem = stage_c._woodbury_ridge_solve(
                K_mem, R_mem, a_mat, ridge_lambda=lam_mem, max_cond=1e4
            )
            dw_mem = _clamp_q8_block_outliers(dw_mem, sigma_k=2.5)
            dw_mem = _proj_in_null(_proj_out_null(dw_mem, U_digit), v_a)
            dw_norm = float(torch.norm(dw_mem).item())
            unit_mem[proj_name] = (dw_mem * (w_norm / (dw_norm + 1e-12))).to(
                torch.bfloat16
            )
            stats[f"{proj_name}.mem_cond"] = round(cond_mem, 2)

        if proj_name == "down_proj" and u_ablit.numel() > 0:
            # Paper 020: project refusal/vendor abliteration orthogonal to joint [U_persona, U_digit] and v_a
            if U_persona.numel() > 0 and U_digit.numel() > 0:
                U_joint, _ = torch.linalg.qr(
                    torch.cat([U_persona, U_digit], dim=1), mode="reduced"
                )
            elif U_persona.numel() > 0:
                U_joint = U_persona
            else:
                U_joint = U_digit
            dw_ab = -u_ablit @ (u_ablit.T @ w0)
            dw_ab = _proj_in_null(_proj_out_null(dw_ab, U_joint), v_a)
            dw_ab_norm = float(torch.norm(dw_ab).item())
            if dw_ab_norm > 1e-10:
                unit_ablit[proj_name] = (dw_ab * (w_norm / dw_ab_norm)).to(
                    torch.bfloat16
                )

    stats["n_lm"] = float(R_lm.shape[1])
    stats["n_mem"] = float(R_mem.shape[1])
    stats["max_rel_frob_cap"] = max_rel_frob
    return unit_lm, unit_mem, unit_ablit, stats


@torch.inference_mode()
def run_stage_d(
    model_key: str,
    target_ctx: int = 2097152,
    max_rel_frob: Optional[float] = None,
    quant_type: str = "TQ2_0",
) -> Dict[str, Any]:
    """
    Executes Sequential Layer-Wise Closed-Form Deep WSP across [L_start .. L_end]
    with greedy per-layer Lyapunov acceptance gating and Q8_0 GGUF verification.
    """
    t0 = time.time()
    if max_rel_frob is None:
        max_rel_frob = 0.0020 if model_key == "mini" else 0.0025
    spec = c.MODEL_SPECS[model_key]
    ckpt_dir = spec["stagec_hf"] if spec["stagec_hf"].exists() else spec["stageb_hf"]
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Neither Stage C nor Stage B checkpoint exists for {model_key}"
        )

    logger.info(
        "=== Stage D Sequential Deep WSP: %s (from %s) ===", spec["display"], ckpt_dir
    )
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_dir),
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()

    base_audit = c.audit_model_integrity(model)
    expected_params = base_audit["total_params"]

    with open(c.CLEAN_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    calib_texts, val_texts = stage_b.build_calib_and_val_splits(c.CLEAN_DATA_PATH)
    dev_mc = stage_c._select_balanced_mc(
        data["mmlu_direct"][-60:], per_letter=6
    ) + stage_c._select_balanced_mc(data["arc_direct"][-40:], per_letter=4)

    U_digit, digit_ids = stage_c._build_digit_symbol_output_basis(
        model, tok, max_rank=24
    )
    orig_digits = model.model.embed_tokens.weight.detach()[
        torch.tensor(digit_ids, dtype=torch.long)
    ].clone()
    w_head = model.lm_head.weight.detach()
    U_persona, u_ablit = _build_persona_orthogonal_ablit_basis(tok, w_head, U_digit)

    cur_nll, cur_mc = stage_c.evaluate_dev_metrics(
        model, tok, dev_mc, val_texts, device="cpu"
    )
    base_nll, base_mc = cur_nll, cur_mc
    mc_tol = 1.0 / float(max(1, len(dev_mc)))
    logger.info(
        "Stage D Baseline -> val_nll=%.5f (PPL=%.4f) dev_mc=%.1f%%",
        cur_nll,
        math.exp(cur_nll),
        cur_mc * 100.0,
    )

    corpus = build_stage_d_prompts(data)
    layers_seq = _layer_range_for_model(model_key, model.config.num_hidden_layers)

    # Candidate step gains strictly clamped so ||Delta W_l||_F / ||W_l||_F <= max_rel_frob
    gain_grid = [
        (0.0008, 0.0004),
        (0.0012, 0.0008),
        (0.0018, 0.0007),
    ]

    accepted_layers: List[Dict[str, Any]] = []
    g_ablit = 0.0003
    for l_idx in layers_seq:
        unit_lm, unit_mem, unit_ablit, l_stats = _capture_layer_activations_and_solve(
            model,
            tok,
            l_idx,
            corpus,
            calib_texts,
            w_head,
            U_digit,
            U_persona,
            u_ablit,
            max_rel_frob=max_rel_frob,
        )
        layer = model.model.layers[l_idx]
        orig_o = layer.self_attn.o_proj.weight.detach().clone()
        orig_dn = layer.mlp.down_proj.weight.detach().clone()

        best_layer_score = 0.0
        best_gains: Optional[Tuple[float, float]] = None
        best_nll, best_mc = cur_nll, cur_mc

        for g_lm, g_mem in gain_grid:
            if g_lm + g_mem + g_ablit > max_rel_frob:
                continue
            for p_name, orig_w in (("o_proj", orig_o), ("down_proj", orig_dn)):
                upd = orig_w.clone()
                if p_name in unit_lm:
                    upd.add_(g_lm * unit_lm[p_name].float())
                if p_name in unit_mem:
                    upd.add_(g_mem * unit_mem[p_name].float())
                if p_name in unit_ablit:
                    upd.add_(g_ablit * unit_ablit[p_name].float())
                target_param = (
                    layer.self_attn.o_proj.weight
                    if p_name == "o_proj"
                    else layer.mlp.down_proj.weight
                )
                target_param.copy_(upd)

            cand_nll, cand_mc = stage_c.evaluate_dev_metrics(
                model, tok, dev_mc, val_texts, device="cpu"
            )
            nll_improved = (cand_nll < cur_nll - 1e-5) and (cand_mc >= cur_mc - mc_tol)
            mc_improved = (cand_mc > cur_mc + 1e-6) and (cand_nll <= cur_nll + 1e-5)
            if nll_improved or mc_improved:
                sc = (cur_nll - cand_nll) * 4.0 + (cand_mc - cur_mc) * 2.0
                if sc > best_layer_score:
                    best_layer_score = sc
                    best_gains = (g_lm, g_mem)
                    best_nll, best_mc = cand_nll, cand_mc

        if best_gains is not None:
            g_lm, g_mem = best_gains
            for p_name, orig_w in (("o_proj", orig_o), ("down_proj", orig_dn)):
                upd = orig_w.clone()
                if p_name in unit_lm:
                    upd.add_(g_lm * unit_lm[p_name].float())
                if p_name in unit_mem:
                    upd.add_(g_mem * unit_mem[p_name].float())
                if p_name in unit_ablit:
                    upd.add_(g_ablit * unit_ablit[p_name].float())
                target_param = (
                    layer.self_attn.o_proj.weight
                    if p_name == "o_proj"
                    else layer.mlp.down_proj.weight
                )
                target_param.copy_(upd)
            cur_nll, cur_mc = best_nll, best_mc
            accepted_layers.append(
                {
                    "layer": l_idx,
                    "g_lm": g_lm,
                    "g_mem": g_mem,
                    "val_nll": round(cur_nll, 5),
                    "dev_mc": round(cur_mc, 4),
                    "stats": l_stats,
                }
            )
            logger.info(
                "Layer %2d ACCEPTED (g_lm=%.4f, g_mem=%.4f) -> val_nll=%.5f (PPL=%.4f) dev_mc=%.1f%%",
                l_idx,
                g_lm,
                g_mem,
                cur_nll,
                math.exp(cur_nll),
                cur_mc * 100.0,
            )
        else:
            layer.self_attn.o_proj.weight.copy_(orig_o)
            layer.mlp.down_proj.weight.copy_(orig_dn)
            logger.info(
                "Layer %2d REJECTED (Lyapunov gate preserved prev state)", l_idx
            )

        del orig_o, orig_dn, unit_lm, unit_mem
        gc.collect()

    # STRICT non-vacuous-progress gate: RAISE + FAILED report + NO export on failure.
    # A run where every layer was rejected produces no real WSP edit.
    try:
        n_accepted = check_stage_d_gate(accepted_layers)
    except WSPGateError as exc:
        write_failed_report(
            "D",
            model_key,
            str(exc),
            {
                "base_val_nll": base_nll,
                "base_dev_mc": base_mc,
                "layers_scanned": layers_seq,
                "accepted_layers": accepted_layers,
            },
        )
        raise
    logger.info(
        "Stage D gate passed: %d/%d layers accepted", n_accepted, len(layers_seq)
    )

    # Closed-form BitNet ternary-aware Hessian residual compensation (OBQ/Woodbury)
    # on target layers before TQ2_0 export, strictly gated by Lyapunov val_nll
    _, calib_k_map = stage_b.capture_input_subspaces(
        model, tok, layers_seq, calib_texts[:4], max_calib_tokens=32
    )
    comp_count = 0
    for l_idx in layers_seq:
        layer = model.model.layers[l_idx]
        orig_o = layer.self_attn.o_proj.weight.detach().clone()
        orig_dn = layer.mlp.down_proj.weight.detach().clone()
        layer.self_attn.o_proj.weight.copy_(
            _compensate_ternary_hessian_residual(orig_o, calib_k_map[(l_idx, "o_proj")])
        )
        layer.mlp.down_proj.weight.copy_(
            _compensate_ternary_hessian_residual(
                orig_dn, calib_k_map[(l_idx, "down_proj")]
            )
        )
        comp_nll, comp_mc = stage_c.evaluate_dev_metrics(
            model, tok, dev_mc, val_texts, device="cpu"
        )
        if comp_nll <= cur_nll + 1e-5 and comp_mc >= cur_mc - mc_tol:
            cur_nll, cur_mc = comp_nll, comp_mc
            comp_count += 1
        else:
            layer.self_attn.o_proj.weight.copy_(orig_o)
            layer.mlp.down_proj.weight.copy_(orig_dn)
    logger.info(
        "BitNet ternary Hessian compensation accepted on %d/%d layers -> val_nll=%.5f",
        comp_count,
        len(layers_seq),
        cur_nll,
    )

    final_digits = model.model.embed_tokens.weight.detach()[
        torch.tensor(digit_ids, dtype=torch.long)
    ]
    assert torch.equal(orig_digits, final_digits), (
        "Digit/symbol embeddings were modified!"
    )
    final_audit = c.audit_model_integrity(model, expected_params=expected_params)

    bitnet_names = {
        "flagship": f"quillan-1b-sovereign-bitnet-v23-{quant_type.lower()}.gguf",
        "mini": f"quillan-mini-0.5b-bitnet-v23-{quant_type.lower()}.gguf",
        "dragon": f"quillan-dragon-1b-bitnet-v23-{quant_type.lower()}.gguf",
    }
    staged_hf = c.EXPORT_ROOT / f"v23_staged_{model_key}_hf"
    staged_gguf = c.EXPORT_ROOT / bitnet_names[model_key]

    if model_key == "dragon":
        c.export_bitnet_gguf(model, staged_gguf, quant_type=quant_type, is_dragon=True)
    else:
        c.save_hf_checkpoint(model, tok, staged_hf)
        del model
        gc.collect()
        c.export_bitnet_gguf(
            staged_hf, staged_gguf, quant_type=quant_type, is_dragon=False
        )

    pre_ppl, post_ppl, fb_ppl = stage_c._patch_and_verify_gguf(
        staged_gguf, model_key, target_ctx=target_ctx
    )
    report = {
        "model": model_key,
        "mode": "sequential_layerwise_closed_form_wsp",
        "elapsed_s": round(time.time() - t0, 2),
        "param_audit": final_audit,
        "base_val_nll": round(base_nll, 5),
        "base_dev_mc": round(base_mc, 4),
        "final_val_nll": round(cur_nll, 5),
        "final_dev_mc": round(cur_mc, 4),
        "accepted_layers": accepted_layers,
        "staged_gguf": str(staged_gguf),
        "gguf_ppl_pre_patch": pre_ppl,
        "gguf_ppl_post_patch": post_ppl,
        "gguf_ppl_262k_compare": fb_ppl,
    }
    rep_path = c.EXPORT_ROOT / f"v23_staged_{model_key}_report.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Stage D completed -> %s", rep_path)
    return report


def run_synthetic_unit_test() -> None:
    """
    Lightweight CPU unit test on synthetic tensors verifying:
      1) Associative null-space projectors (_proj_out_null / _proj_in_null)
      2) Dual-space Woodbury solve & residual shrinkage
      3) Relative Frobenius norm bounding (<= 0.005)
      4) Monotonic Lyapunov gating simulation
    Does NOT load any HF model or touch GPU/GGUF files.
    """
    torch.manual_seed(42)
    # d_in MUST be >= 512 so _ternarize_block256_ls takes the per-256-block path
    # (d_in % 256 == 0 with at least two independent blocks), not the whole-matrix fallback.
    d_out, d_in, N, M = 64, 512, 16, 24
    assert d_in % 256 == 0 and d_in // 256 >= 2, (
        "Synthetic d_in must exercise per-256-block path"
    )

    U_digit, _ = torch.linalg.qr(torch.randn(d_out, 8))
    A_anchor = torch.randn(M, d_in)
    _, _, v_a = torch.svd_lowrank(A_anchor, q=12, niter=2)
    v_a = v_a.contiguous()

    K = torch.randn(d_in, N)
    R = torch.randn(d_out, N)
    W0 = torch.randn(d_out, d_in) * 0.02
    w0_norm = float(torch.norm(W0).item())

    trace_k = float((K * K).sum().item())
    lam = max(8.0, 1e-2 * trace_k / float(N))
    dw, cond = stage_c._woodbury_ridge_solve(
        K, R, A_anchor, ridge_lambda=lam, max_cond=1e4
    )
    assert dw.shape == (d_out, d_in), f"Bad dw shape: {dw.shape}"
    assert cond <= 1e4, f"Condition number exceeded bound: {cond}"

    dw_clamped = _clamp_q8_block_outliers(dw, sigma_k=2.5)
    dw_null = _proj_in_null(_proj_out_null(dw_clamped, U_digit), v_a)
    left_res = float(torch.norm(U_digit.T @ dw_null).item())
    right_res = float(torch.norm(dw_null @ v_a).item())
    assert left_res < 1e-4, f"Left null-space leakage: {left_res}"
    assert right_res < 1e-4, f"Right null-space leakage: {right_res}"

    # Paper 020: verify persona-orthogonal refusal abliteration basis using joint QR [U_persona, U_digit]
    U_persona, _ = torch.linalg.qr(torch.randn(d_out, 6), mode="reduced")
    U_joint, _ = torch.linalg.qr(torch.cat([U_persona, U_digit], dim=1), mode="reduced")
    raw_ab = torch.randn(d_out, 2)
    u_ablit, _ = torch.linalg.qr(_proj_out_null(raw_ab, U_joint), mode="reduced")
    persona_res = float(torch.norm(U_persona.T @ u_ablit).item())
    digit_ab_res = float(torch.norm(U_digit.T @ u_ablit).item())
    assert persona_res < 1e-5, f"Persona orthogonality violated: {persona_res}"
    assert digit_ab_res < 1e-5, f"Digit orthogonality violated: {digit_ab_res}"

    unit_dw = dw_null * (w0_norm / (float(torch.norm(dw_null).item()) + 1e-12))
    delta_scaled = 0.002 * unit_dw
    rel_frob = float(torch.norm(delta_scaled).item()) / w0_norm
    assert abs(rel_frob - 0.002) < 1e-5 and rel_frob <= 0.0025, (
        f"Frobenius bound failed: {rel_frob}"
    )

    # Paper 012 + BitNet 1.58b: verify PER-256-BLOCK LS ternarization + Hessian compensation
    w_hat = _ternarize_block256_ls(W0)
    assert w_hat.shape == W0.shape
    blocks_hat = w_hat.view(d_out, d_in // 256, 256)
    # gamma_star is LS-fit per (row, block): every non-zero in a given row-block
    # must equal that row-block's single scale, and the two 256-blocks must differ.
    block_scales_t = blocks_hat.abs().amax(dim=-1)  # (d_out, n_blocks)
    assert bool((block_scales_t > 0).all()), "A 256-block ternarized to all zeros"
    block_scales: List[float] = [
        float(block_scales_t[:, b].mean().item())
        for b in range(block_scales_t.shape[1])
    ]
    for b_idx in range(block_scales_t.shape[1]):
        blk = blocks_hat[:, b_idx, :]
        expected = block_scales_t[:, b_idx].unsqueeze(1)
        nz_mask = blk.abs() > 0
        scale_err = float(((blk.abs() - expected)[nz_mask].abs().max()).item())
        assert scale_err < 1e-6, (
            f"Block {b_idx} did not collapse to one LS scale per row: {scale_err}"
        )
    scale_spread = float(
        (block_scales_t[:, 0] - block_scales_t[:, 1]).abs().max().item()
    )
    assert scale_spread > 1e-9, (
        "Per-256-block LS scales are identical - block path was not exercised"
    )
    k_comp = torch.randn(d_in, 16)
    w_comp = _compensate_ternary_hessian_residual(W0, k_comp)
    assert w_comp.shape == W0.shape and torch.isfinite(w_comp).all()
    comp_delta = float(torch.norm(w_comp - W0).item())
    assert comp_delta > 1e-8, "Ternary Hessian compensation was a no-op"

    # Strict Stage D gate: zero accepted layers MUST raise; a real acceptance passes.
    gate_raised = False
    try:
        check_stage_d_gate([])
    except WSPGateError:
        gate_raised = True
    assert gate_raised, "Stage D gate failed to reject zero accepted layers"
    assert check_stage_d_gate([{"layer": 0}]) == 1

    print(
        f"STAGE_D_UNIT_TEST PASS | shape={tuple(dw_null.shape)} cond={cond:.2f} "
        f"left_null={left_res:.2e} right_null={right_res:.2e} "
        f"persona_null={persona_res:.2e} rel_frob={rel_frob:.4f} "
        f"bitnet_blocks={blocks_hat.shape[1]} block_scales={block_scales[0]:.5f}/{block_scales[1]:.5f} "
        f"scale_spread={scale_spread:.2e} "
        f"comp_delta={comp_delta:.2e} gate_rejects_zero=OK"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage D Sequential Deep Closed-Form WSP"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run synthetic CPU unit test only"
    )
    parser.add_argument(
        "--model", choices=["flagship", "mini", "dragon"], default="flagship"
    )
    parser.add_argument("--target-ctx", type=int, default=2097152)
    parser.add_argument(
        "--quant-type", default="TQ2_0", help="BitNet GGUF quant type (TQ2_0 / TQ1_0)"
    )
    args = parser.parse_args()

    if args.self_test:
        run_synthetic_unit_test()
        return

    run_stage_d(args.model, target_ctx=args.target_ctx, quant_type=args.quant_type)


if __name__ == "__main__":
    main()
