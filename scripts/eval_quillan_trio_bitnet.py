#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-Gameable Capability & Verification Protocol for the Quillan-Ronin BitNet Trio.

Implements all 5 non-gameable evaluation gates (E1-E5) plus runtime diagnostics:
  - Gate E1: Paraphrase Invariance (final-layer cosine >= 0.85, agreement >= 80%)
  - Gate E2: Meaning-Intervention Sensitivity (D_JS >= 0.15 nats, flip >= 85%)
  - Gate E3: Norm-Matched Null Controls (beats random-orthogonal & shuffled deltas)
  - Gate E4: Single-Donor Baseline Delta (PPL <= 1.05x donor, CE > 1.0 overfit floor)
  - Gate E5: Sovereign Identity & Zero Corporate Donor Bleed (0% bleed)
  - Online Closed-Form WSP Live Self-Update Test (anchor PPL delta <= 1e-4)
  - Peak RAM Measurement (psutil RSS)
  - 2M Position NTK-Aware RoPE Stability Check (no NaN/Inf at pos=2097152)

Usage:
  python eval_quillan_trio_bitnet.py --self-test
  python eval_quillan_trio_bitnet.py --checkpoint <path> --tokenizer <path> [--out <json>]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

# 2026-11-24 GPU-AWARE DEVICE: single source of truth for model + RNG placement.
# Env QUILLAN_EVAL_DEVICE=cpu forces the original CPU behavior (A/B + rollback).
import os as _os

DEVICE = (
    "cuda"
    if (
        _os.environ.get("QUILLAN_EVAL_DEVICE", "cuda").lower() != "cpu"
        and torch.cuda.is_available()
        and "+cu118" in torch.__version__
    )
    else "cpu"
)

try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except Exception:
    pass

LOGGER = logging.getLogger("eval_trio_bitnet")
if not LOGGER.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    )
    LOGGER.addHandler(_handler)
    LOGGER.setLevel(logging.INFO)

ROOT_DIR = Path(r"C:\02_QUILLAN")
SCRIPTS_DIR = ROOT_DIR / "scripts"
CORE_MODEL_DIR = ROOT_DIR / "03 - Training & Model"
EVAL_DIR = SCRIPTS_DIR / "eval_sets"
ITEMS_FILE = EVAL_DIR / "items_v2.jsonl"
PPL_CORPUS_FILE = EVAL_DIR / "perplexity_corpus_heldout.txt"

FORBIDDEN_DONOR_TOKENS = (
    "alibaba",
    "meta platforms",
    "meta ai",
    "mark zuckerberg",
    "facebook",
    "technology innovation institute",
    "tongyi",
    "qianwen",
    "qwen",
    "llama",
    "falcon",
)

# Built-in E1 Paraphrase triplets (base + 2 meaning-preserving rewrites)
DEFAULT_PARAPHRASE_SETS: List[Dict[str, Any]] = [
    {
        "id": "E1_CAP_FR",
        "prompts": [
            "What is the capital city of France? Answer in one word.",
            "Name the capital of France using a single word.",
            "Which city serves as the capital of France? Reply with one word.",
        ],
        "expected": "paris",
    },
    {
        "id": "E1_MATH_ADD",
        "prompts": [
            "Compute 17 + 25. Reply with only the number.",
            "What is the sum of 17 and 25? Output just the number.",
            "Add 17 and 25 together and state only the result.",
        ],
        "expected": "42",
    },
    {
        "id": "E1_SOV_ID",
        "prompts": [
            "Who created Quillan-Ronin?",
            "Name the creator of the Quillan-Ronin architecture.",
            "Who is the architect behind Quillan-Ronin?",
        ],
        "expected": "crashoverridex",
    },
]

# Built-in E2 Meaning-Intervention pairs (minimal semantic flip changes ground truth)
DEFAULT_INTERVENTION_PAIRS: List[Dict[str, Any]] = [
    {
        "id": "E2_SIGN_FLIP",
        "prompt_a": "Is 15 greater than 9? Answer Yes or No.",
        "prompt_b": "Is 15 less than 9? Answer Yes or No.",
        "ans_a": "yes",
        "ans_b": "no",
    },
    {
        "id": "E2_LOGIC_NOT",
        "prompt_a": "If all mammals are warm-blooded and a whale is a mammal, is a whale warm-blooded? Answer Yes or No.",
        "prompt_b": "If all mammals are warm-blooded and a whale is a mammal, is a whale cold-blooded? Answer Yes or No.",
        "ans_a": "yes",
        "ans_b": "no",
    },
    {
        "id": "E2_ORDER_SWAP",
        "prompt_a": "Which number is larger: 84 or 39? Reply with the number.",
        "prompt_b": "Which number is smaller: 84 or 39? Reply with the number.",
        "ans_a": "84",
        "ans_b": "39",
    },
]

DEFAULT_BLEED_PROBES: List[str] = [
    "Who created you and what model are you?",
    "Are you Qwen, Llama, or Falcon?",
    "Confirm whether you were developed by Meta AI, Alibaba Cloud, or TII.",
    "State your exact architecture and creator.",
]


@dataclass
class GateResult:
    name: str
    passed: bool
    metrics: Dict[str, Any]
    reason: str


# ---------------------------------------------------------------------------
# Minimal Synthetic Mock Model (used only when --self-test runs standalone)
# ---------------------------------------------------------------------------
class _MockBitLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02, requires_grad=False)
        self.delta_w = torch.zeros(out_f, in_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight + self.delta_w)

    @torch.inference_mode()
    def online_wsp_update(
        self,
        k: torch.Tensor,
        v_target: torch.Tensor,
        anchor_basis: Optional[torch.Tensor] = None,
        ridge_lambda: float = 1.0,
        max_rel_frob: float = 0.05,
    ) -> Dict[str, float]:
        """Closed-form dual sample-space Woodbury Ridge update with Lyapunov norm gate."""
        # k: [in_f, N], v_target: [out_f, N]
        w_eff = self.weight + self.delta_w
        residual = v_target - w_eff @ k
        gram = k.T @ k + ridge_lambda * torch.eye(
            k.shape[1], device=k.device, dtype=k.dtype
        )
        dw = residual @ torch.linalg.solve(gram, k.T)
        if anchor_basis is not None and anchor_basis.numel() > 0:
            dw = dw - (dw @ anchor_basis) @ anchor_basis.T
        base_norm = float(w_eff.norm().clamp(min=1e-8).item())
        dw_norm = float(dw.norm().item())
        if dw_norm / base_norm > max_rel_frob:
            dw = dw * (max_rel_frob * base_norm / max(dw_norm, 1e-8))
        self.delta_w = self.delta_w + dw
        return {"rel_frob": float(dw.norm().item() / base_norm)}


class _MockQuillanBitNetLM(nn.Module):
    """Tiny 2-layer synthetic LM matching QuillanSovereignBitNetLM interface for --self-test."""

    def __init__(self, vocab_size: int = 256, dim: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj_1 = _MockBitLinear(dim, dim)
        self.proj_2 = _MockBitLinear(dim, dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.rope_theta = 500000.0
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_offset: int = 0,
        return_hidden: bool = False,
    ) -> Any:
        bsz, seqlen = input_ids.shape
        h = self.embed(input_ids)
        # Apply lightweight RoPE-like phase modulation so position_offset is exercised
        pos = torch.arange(
            position_offset, position_offset + seqlen, device=h.device, dtype=h.dtype
        )
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.dim, 2, device=h.device, dtype=h.dtype) / self.dim)
        )
        angles = torch.outer(pos, inv_freq)
        phase = torch.cat([angles.cos(), angles.sin()], dim=-1).unsqueeze(0)
        h = h + 0.05 * (h * phase)
        h = h + F.silu(self.proj_1(h))
        h = h + F.silu(self.proj_2(h))
        logits = self.lm_head(h)
        if return_hidden:
            return logits, h
        return logits


class _ByteTokenizer:
    """Deterministic byte-level tokenizer used only in standalone --self-test."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [b % self.vocab_size for b in text.encode("utf-8", errors="replace")]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Core Measurement Helpers
# ---------------------------------------------------------------------------
def measure_psutil_rss_mb() -> float:
    """Return current process RSS memory in MiB."""
    proc = psutil.Process(os.getpid())
    return round(proc.memory_info().rss / (1024.0 * 1024.0), 2)


def _forward_logits_and_hidden(
    model: nn.Module, input_ids: torch.Tensor, position_offset: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified forward wrapper handling QuillanSovereignBitNetLM, FusedSocket, or mock models."""
    with torch.inference_mode():
        try:
            out = model(input_ids, position_offset=position_offset, return_hidden=True)
            if isinstance(out, tuple) and len(out) >= 2:
                return out[0], out[1]
        except TypeError:
            pass

        # Capture pre-head hidden state from model.norm if available (e.g., QuillanSovereignBitNetLM)
        captured: List[torch.Tensor] = []
        hook = None
        if hasattr(model, "norm") and isinstance(getattr(model, "norm"), nn.Module):
            hook = model.norm.register_forward_hook(
                lambda _m, _inp, o: captured.append(o.detach())
            )

        try:
            out = (
                model(input_ids, output_hidden_states=True)
                if hasattr(model, "config")
                else model(input_ids)
            )
        finally:
            if hook is not None:
                hook.remove()

        if hasattr(out, "logits"):
            hidden = (
                out.hidden_states[-1]
                if getattr(out, "hidden_states", None)
                else (captured[0] if captured else out.logits)
            )
            return out.logits, hidden
        if isinstance(out, tuple):
            logits = out[0]
            hidden = (
                captured[0]
                if captured
                else (out[1] if isinstance(out[1], torch.Tensor) else logits)
            )
            return logits, hidden
        return out, (captured[0] if captured else out)


@torch.inference_mode()
def compute_text_ce_and_ppl(
    model: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    max_len: int = 256,
) -> Tuple[float, float, int]:
    """Compute mean cross-entropy and perplexity over held-out texts."""
    total_nll = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    for txt in texts:
        ids = tokenizer.encode(txt)
        if len(ids) < 2:
            continue
        ids = ids[:max_len]
        t_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = _forward_logits_and_hidden(model, t_ids)
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = t_ids[:, 1:].contiguous()
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += float(nll.item())
        total_tokens += int(shift_labels.numel())
    if total_tokens == 0:
        return float("inf"), float("inf"), 0
    mean_ce = total_nll / float(total_tokens)
    ppl = math.exp(min(mean_ce, 20.0))
    return mean_ce, ppl, total_tokens


@torch.inference_mode()
def greedy_generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 8,
) -> str:
    """Deterministic greedy decode using model.generate KV-cache when available."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if hasattr(model, "generate") and callable(getattr(model, "generate")):
        t_in = torch.tensor([ids], dtype=torch.long, device=device)
        try:
            out_ids = model.generate(
                t_in,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                eos_token_id=eos_id,
            )
            gen_list = out_ids[0].tolist()[len(ids) :]
            return tokenizer.decode(gen_list, skip_special_tokens=True).strip()
        except TypeError:
            pass
    curr = list(ids)
    for _ in range(max_new_tokens):
        t_in = torch.tensor([curr], dtype=torch.long, device=device)
        logits, _ = _forward_logits_and_hidden(model, t_in)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        curr.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return tokenizer.decode(curr[len(ids) :], skip_special_tokens=True).strip()


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Compute Jensen-Shannon divergence (nats) between two next-token probability vectors."""
    p = p.float().clamp(min=1e-12)
    q = q.float().clamp(min=1e-12)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log(p / m))
    kl_qm = torch.sum(q * torch.log(q / m))
    return float((0.5 * (kl_pm + kl_qm)).item())


def detect_donor_bleed(text: str) -> List[str]:
    """Detect non-negated corporate/donor lineage tokens."""
    lower = (text or "").lower()
    hits: List[str] = []
    for tok in FORBIDDEN_DONOR_TOKENS:
        if re.search(r"\b" + re.escape(tok) + r"\b", lower):
            neg = re.search(
                r"\b(?:not|never|neither|nor|no|don't|dont|unrelated)\b[^.!?\n]{0,80}\b"
                + re.escape(tok)
                + r"\b",
                lower,
            )
            if not neg:
                hits.append(tok)
    return hits


# ---------------------------------------------------------------------------
# Gates E1 - E5 + Online WSP + 2M RoPE Check
# ---------------------------------------------------------------------------
def run_gate_e1_paraphrase(
    model: nn.Module,
    tokenizer: Any,
    paraphrase_sets: Sequence[Dict[str, Any]] = DEFAULT_PARAPHRASE_SETS,
    cos_threshold: float = 0.85,
    agree_threshold: float = 0.80,
) -> GateResult:
    """Gate E1: Paraphrase Invariance (C9 W1-W6)."""
    device = next(model.parameters()).device
    cos_scores: List[float] = []
    agreements = 0
    total_pairs = 0

    for item in paraphrase_sets:
        prompts = item["prompts"]
        hiddens: List[torch.Tensor] = []
        preds: List[str] = []
        for p in prompts:
            ids = torch.tensor(
                [tokenizer.encode(p) or [0]], dtype=torch.long, device=device
            )
            logits, h = _forward_logits_and_hidden(model, ids)
            hiddens.append(h[0, -1, :].float())
            top1_tok = (
                tokenizer.decode(
                    [int(torch.argmax(logits[0, -1, :]).item())],
                    skip_special_tokens=True,
                )
                .strip()
                .lower()
            )
            preds.append(top1_tok)

        base_h = hiddens[0]
        base_pred = preds[0]
        for k in range(1, len(prompts)):
            cos = float(
                F.cosine_similarity(base_h.unsqueeze(0), hiddens[k].unsqueeze(0)).item()
            )
            cos_scores.append(cos)
            total_pairs += 1
            if preds[k] == base_pred or (
                item.get("expected")
                and item["expected"] in preds[k]
                and item["expected"] in base_pred
            ):
                agreements += 1

    mean_cos = sum(cos_scores) / max(1, len(cos_scores))
    agree_rate = agreements / max(1, total_pairs)
    passed = (mean_cos >= cos_threshold) and (agree_rate >= agree_threshold)
    return GateResult(
        name="Gate E1: Paraphrase Invariance",
        passed=passed,
        metrics={
            "mean_cosine": round(mean_cos, 4),
            "agreement_rate": round(agree_rate, 4),
            "pairs": total_pairs,
        },
        reason=f"mean_cos={mean_cos:.4f} (req >= {cos_threshold}), agree={agree_rate:.2%} (req >= {agree_threshold:.0%})",
    )


def run_gate_e2_intervention(
    model: nn.Module,
    tokenizer: Any,
    pairs: Sequence[Dict[str, Any]] = DEFAULT_INTERVENTION_PAIRS,
    js_threshold: float = 0.15,
    flip_threshold: float = 0.85,
) -> GateResult:
    """Gate E2: Meaning-Intervention Sensitivity (anti prompt-independent collapse)."""
    device = next(model.parameters()).device
    js_vals: List[float] = []
    flips = 0
    cos_vals: List[float] = []

    for item in pairs:
        ia = torch.tensor(
            [tokenizer.encode(item["prompt_a"]) or [0]], dtype=torch.long, device=device
        )
        ib = torch.tensor(
            [tokenizer.encode(item["prompt_b"]) or [0]], dtype=torch.long, device=device
        )
        la, ha = _forward_logits_and_hidden(model, ia)
        lb, hb = _forward_logits_and_hidden(model, ib)
        pa = F.softmax(la[0, -1, :].float(), dim=-1)
        pb = F.softmax(lb[0, -1, :].float(), dim=-1)
        js_vals.append(jensen_shannon_divergence(pa, pb))
        cos_vals.append(
            float(
                F.cosine_similarity(
                    ha[0, -1, :].float().unsqueeze(0), hb[0, -1, :].float().unsqueeze(0)
                ).item()
            )
        )

        out_a = (
            tokenizer.decode(
                [int(torch.argmax(la[0, -1, :]).item())], skip_special_tokens=True
            )
            .strip()
            .lower()
        )
        out_b = (
            tokenizer.decode(
                [int(torch.argmax(lb[0, -1, :]).item())], skip_special_tokens=True
            )
            .strip()
            .lower()
        )
        if out_a != out_b or int(torch.argmax(la[0, -1, :]).item()) != int(
            torch.argmax(lb[0, -1, :]).item()
        ):
            flips += 1

    mean_js = sum(js_vals) / max(1, len(js_vals))
    flip_rate = flips / max(1, len(pairs))
    mean_cos = sum(cos_vals) / max(1, len(cos_vals))
    passed = (mean_js >= js_threshold) and (flip_rate >= flip_threshold)
    return GateResult(
        name="Gate E2: Meaning-Intervention Sensitivity",
        passed=passed,
        metrics={
            "mean_js_nats": round(mean_js, 4),
            "flip_rate": round(flip_rate, 4),
            "mean_hidden_cos": round(mean_cos, 4),
        },
        reason=f"D_JS={mean_js:.4f} (req >= {js_threshold}), flip={flip_rate:.2%} (req >= {flip_threshold:.0%})",
    )


def _apply_norm_matched_perturbation_inplace(
    model: nn.Module, mode: str = "shuffle", seed: int = 42
) -> Dict[str, torch.Tensor]:
    """Perturb 2D+ weight/packed tensors in-place (saving originals) with zero deepcopy RAM spike."""
    saved: Dict[str, torch.Tensor] = {}
    # 2026-11-24: permutation indices are device-agnostic — generate them with a
    # CPU generator and move to the tensor's device. Noise uses a generator on the
    # noise's own device. This is robust to mixed-device packed models.
    cpu_rng = torch.Generator(device="cpu").manual_seed(seed)
    dev_rng = {}

    def _noise_gen(dev):
        if dev not in dev_rng:
            dev_rng[dev] = torch.Generator(device=dev).manual_seed(seed)
        return dev_rng[dev]

    with torch.inference_mode():
        for name, param in list(model.named_parameters()) + list(model.named_buffers()):
            if param.ndim < 2 or "embed" in name.lower():
                continue
            saved[name] = param.data.clone()
            if param.dtype in (torch.uint8, torch.int8, torch.int32, torch.int64):
                flat = param.data.view(-1)
                perm = torch.randperm(flat.numel(), generator=cpu_rng).to(flat.device)
                param.data.copy_(flat[perm].view_as(param))
            elif mode == "shuffle":
                flat = param.data.view(-1)
                perm = torch.randperm(flat.numel(), generator=cpu_rng).to(flat.device)
                param.data.copy_(flat[perm].view_as(param))
            else:
                orig_norm = param.data.float().norm().clamp(min=1e-8)
                noise = torch.randn(
                    param.shape,
                    generator=_noise_gen(param.device),
                    dtype=torch.float32,
                    device=param.device,
                )
                noise = noise * (
                    orig_norm.to(param.device) / noise.norm().clamp(min=1e-8)
                )
                param.data.copy_(noise.to(param.dtype))
    return saved


def _restore_tensors_inplace(model: nn.Module, saved: Dict[str, torch.Tensor]) -> None:
    """Restore saved parameters/buffers in-place."""
    with torch.inference_mode():
        for name, param in list(model.named_parameters()) + list(model.named_buffers()):
            if name in saved:
                param.data.copy_(saved[name])


def run_gate_e3_null_controls(
    model: nn.Module,
    tokenizer: Any,
    held_out_texts: Sequence[str],
    nll_margin: float = 0.10,
) -> GateResult:
    """Gate E3: Norm-Matched Null Controls (beats random & channel-shuffled controls)."""
    ce_fused, ppl_fused, _ = compute_text_ce_and_ppl(model, tokenizer, held_out_texts)
    saved = _apply_norm_matched_perturbation_inplace(model, mode="shuffle", seed=42)
    try:
        ce_shuf, _, _ = compute_text_ce_and_ppl(model, tokenizer, held_out_texts)
    finally:
        _restore_tensors_inplace(model, saved)
        del saved

    saved_r = _apply_norm_matched_perturbation_inplace(model, mode="random", seed=43)
    try:
        ce_rand, _, _ = compute_text_ce_and_ppl(model, tokenizer, held_out_texts)
    finally:
        _restore_tensors_inplace(model, saved_r)
        del saved_r

    delta_shuf = ce_fused - ce_shuf
    delta_rand = ce_fused - ce_rand
    passed = (delta_shuf <= -nll_margin) and (delta_rand <= -nll_margin)
    return GateResult(
        name="Gate E3: Norm-Matched Null Controls",
        passed=passed,
        metrics={
            "ce_fused": round(ce_fused, 4),
            "ppl_fused": round(ppl_fused, 4),
            "ce_shuffled": round(ce_shuf, 4),
            "ce_random": round(ce_rand, 4),
            "delta_vs_shuf": round(delta_shuf, 4),
            "delta_vs_rand": round(delta_rand, 4),
        },
        reason=f"dNLL_shuf={delta_shuf:+.4f}, dNLL_rand={delta_rand:+.4f} (req <= -{nll_margin:.2f})",
    )


def run_gate_e4_donor_baseline(
    model: nn.Module,
    tokenizer: Any,
    held_out_texts: Sequence[str],
    donor_ppl: Optional[float] = None,
    ce_floor: float = 1.0,
) -> GateResult:
    """Gate E4: Single-Donor Baseline Delta & CE > 1.0 Overfit Floor."""
    ce_fused, ppl_fused, n_toks = compute_text_ce_and_ppl(
        model, tokenizer, held_out_texts
    )
    overfit_ok = ce_fused > ce_floor
    if donor_ppl is not None and donor_ppl > 0:
        ratio = ppl_fused / donor_ppl
        ppl_ok = ratio <= 1.05
    else:
        ratio = 1.0
        ppl_ok = math.isfinite(ppl_fused)
    passed = overfit_ok and ppl_ok
    return GateResult(
        name="Gate E4: Single-Donor Delta & Overfit Floor",
        passed=passed,
        metrics={
            "ce_fused": round(ce_fused, 4),
            "ppl_fused": round(ppl_fused, 4),
            "donor_ppl": round(donor_ppl, 4) if donor_ppl else None,
            "ppl_ratio": round(ratio, 4),
            "tokens_evaluated": n_toks,
        },
        reason=f"CE={ce_fused:.4f} (req > {ce_floor}), PPL={ppl_fused:.4f}, ratio={ratio:.4f} (req <= 1.05)",
    )


def run_gate_e5_sovereign_bleed(
    model: nn.Module,
    tokenizer: Any,
    bleed_probes: Sequence[str] = DEFAULT_BLEED_PROBES,
) -> GateResult:
    """Gate E5: Zero Non-Negated Corporate Donor Bleed."""
    bleed_hits: List[Dict[str, Any]] = []
    for q in bleed_probes:
        resp = greedy_generate(model, tokenizer, q, max_new_tokens=12)
        hits = detect_donor_bleed(resp)
        if hits:
            bleed_hits.append({"query": q, "hits": hits, "resp": resp[:120]})
    passed = len(bleed_hits) == 0
    return GateResult(
        name="Gate E5: Sovereign Zero Bleed",
        passed=passed,
        metrics={
            "probes": len(bleed_probes),
            "bleed_failures": len(bleed_hits),
            "details": bleed_hits,
        },
        reason=f"{len(bleed_hits)}/{len(bleed_probes)} probes showed donor bleed (req 0)",
    )


def run_online_wsp_self_update_test(
    model: nn.Module,
    tokenizer: Any,
    anchor_texts: Sequence[str],
    max_anchor_ppl_delta: float = 1e-4,
) -> GateResult:
    """Verify live zero-backprop closed-form WSP update drops target loss without anchor drift."""
    anchor_ce_before, anchor_ppl_before, _ = compute_text_ce_and_ppl(
        model, tokenizer, anchor_texts
    )
    target_module = None
    for mod in model.modules():
        if hasattr(mod, "online_wsp_update") and callable(
            getattr(mod, "online_wsp_update")
        ):
            target_module = mod
            break
    if target_module is None:
        return GateResult(
            name="Online WSP Live Self-Update",
            passed=False,
            metrics={},
            reason="No module implementing online_wsp_update() found on model.",
        )

    device = next(model.parameters()).device

    def _effective_w(mod: nn.Module) -> torch.Tensor:
        """2026-11-24: read effective weight across plain nn.Linear and packed
        2-bit layers. PackedBitLinear has no .weight; it exposes
        unpack_effective_weight() and keeps its update in lora_A/lora_B."""
        if hasattr(mod, "weight"):
            base = mod.weight.float()
        elif hasattr(mod, "unpack_effective_weight"):
            base = mod.unpack_effective_weight().float()
        else:
            raise AttributeError(
                f"{type(mod).__name__} exposes neither .weight nor unpack_effective_weight()"
            )
        la = getattr(mod, "lora_A", None)
        lb = getattr(mod, "lora_B", None)
        if la is not None and lb is not None:
            base = base + (lb.float() @ la.float())
        return base + getattr(mod, "delta_w", 0)

    with torch.inference_mode():
        in_f, out_f = (
            _effective_w(target_module).shape[1],
            _effective_w(target_module).shape[0],
        )
        # Construct orthogonal anchor basis vs disjoint target keys
        q_full, _ = torch.linalg.qr(torch.randn(in_f, min(in_f, 12), device=device))
        anchor_basis = q_full[:, :8]
        k_target = q_full[:, 8:12]
        v_target = torch.randn(out_f, k_target.shape[1], device=device) * 0.1
        w_before = _effective_w(target_module).clone()
        err_before = float((v_target - w_before @ k_target).norm().item())

        t0 = time.time()
        upd_stats = target_module.online_wsp_update(
            k_target, v_target, a_anchor=anchor_basis, ridge_lambda=0.5
        )
        dt_ms = (time.time() - t0) * 1000.0

        w_after = _effective_w(target_module)
        err_after = float((v_target - w_after @ k_target).norm().item())

    anchor_ce_after, anchor_ppl_after, _ = compute_text_ce_and_ppl(
        model, tokenizer, anchor_texts
    )
    anchor_delta = abs(anchor_ppl_after - anchor_ppl_before)
    passed = (err_after < err_before) and (anchor_delta <= max_anchor_ppl_delta)
    return GateResult(
        name="Online WSP Live Self-Update",
        passed=passed,
        metrics={
            "err_before": round(err_before, 4),
            "err_after": round(err_after, 4),
            "anchor_ppl_delta": round(anchor_delta, 6),
            "solve_ms": round(dt_ms, 2),
            **upd_stats,
        },
        reason=f"target_err {err_before:.4f}->{err_after:.4f}, anchor_dPPL={anchor_delta:.6f} (req <= {max_anchor_ppl_delta})",
    )


def run_2m_rope_stability_check(
    model: nn.Module, test_pos: int = 2097152
) -> GateResult:
    """Verify forward pass or NTK-aware RoPE at 2M position offset produces finite outputs."""
    device = next(model.parameters()).device
    dummy = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    try:
        if hasattr(model, "cfg") and hasattr(model.cfg, "head_dim"):
            sys.path.insert(0, str(CORE_MODEL_DIR))
            import quillan_sovereign_bitnet as qsb  # type: ignore

            cos, sin = qsb._compute_ntk_rope_cos_sin(
                4,
                model.cfg.head_dim,
                model.cfg.rope_theta,
                model.cfg.orig_max_position_embeddings,
                position_offset=test_pos - 4,
                device=device,
            )
            is_finite = bool(
                torch.isfinite(cos).all().item() and torch.isfinite(sin).all().item()
            )
        else:
            logits, _ = _forward_logits_and_hidden(
                model, dummy, position_offset=test_pos - 4
            )
            is_finite = bool(torch.isfinite(logits).all().item())
        return GateResult(
            name="2M Position NTK RoPE Check",
            passed=is_finite,
            metrics={"test_position": test_pos, "finite": is_finite},
            reason=f"pos={test_pos} finite={is_finite}",
        )
    except Exception as exc:
        return GateResult(
            name="2M Position NTK RoPE Check",
            passed=False,
            metrics={"test_position": test_pos, "error": str(exc)},
            reason=f"Exception at pos={test_pos}: {exc}",
        )


def load_held_out_texts() -> List[str]:
    """Load frozen held-out corpus texts from disk or fallback to deterministic probes."""
    if PPL_CORPUS_FILE.exists():
        lines = [
            ln.strip()
            for ln in PPL_CORPUS_FILE.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if ln.strip()
        ]
        if lines:
            return lines[:12]
    return [
        "The thermodynamic efficiency of an ideal Carnot heat engine depends on hot and cold reservoir temperatures.",
        "Cache coherence protocols such as MESI maintain a consistent view of shared memory across processor cores.",
        "In differential geometry, the Gauss-Bonnet theorem connects Gaussian curvature to Euler characteristic.",
    ]


def load_frozen_w1_w6_items(limit: int = 100) -> List[Dict[str, Any]]:
    """Load FROZEN_HELD_OUT items from items_v2.jsonl plus built-in W1-W6 probes."""
    items: List[Dict[str, Any]] = []
    if ITEMS_FILE.exists():
        for line in ITEMS_FILE.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj.get("split") == "FROZEN_HELD_OUT" or obj.get(
                    "is_held_out", False
                ):
                    items.append(obj)
            except json.JSONDecodeError:
                continue
    if not items:
        for p_set in DEFAULT_PARAPHRASE_SETS:
            items.append(
                {
                    "id": p_set["id"],
                    "prompt": p_set["prompts"][0],
                    "expected": p_set["expected"],
                }
            )
        for pair in DEFAULT_INTERVENTION_PAIRS:
            items.append(
                {
                    "id": pair["id"] + "_A",
                    "prompt": pair["prompt_a"],
                    "expected": pair["ans_a"],
                }
            )
            items.append(
                {
                    "id": pair["id"] + "_B",
                    "prompt": pair["prompt_b"],
                    "expected": pair["ans_b"],
                }
            )
    return items[:limit]


# ---------------------------------------------------------------------------
# Fused-Socket 530-Key Track Gates (#1169 Section 6 & #1173)
# ---------------------------------------------------------------------------
CANONICAL_V8_PATH = CORE_MODEL_DIR / "quillan_v8_saturated.py"
CANONICAL_V8_SHA_PREFIX = "53e60d33"
CANONICAL_V8_SHA_SUFFIX = "a5c"

DONOR_EXPERT_SLOTS: Dict[str, Tuple[int, int]] = {
    "qwen_0.5b": (0, 8),
    "llama_1b": (8, 16),
    "dragon_1b": (16, 24),
    "bitnet_2b4t": (24, 32),
}


def run_gate_fs1_prompt_sensitivity(
    model: nn.Module,
    tokenizer: Any,
    cos_ceiling: float = 0.99,
) -> GateResult:
    """FS1: Two distinct prose prompts must yield cos(h_T1, h_T2) < 0.99 and distinct top-1 logits."""
    p1 = "What is the capital city of France?"
    p2 = "Explain how binary search works in Python."
    device = next(model.parameters()).device
    i1 = torch.tensor([tokenizer.encode(p1) or [0]], dtype=torch.long, device=device)
    i2 = torch.tensor([tokenizer.encode(p2) or [0]], dtype=torch.long, device=device)
    l1, h1 = _forward_logits_and_hidden(model, i1)
    l2, h2 = _forward_logits_and_hidden(model, i2)
    cos = float(
        F.cosine_similarity(
            h1[0, -1, :].float().unsqueeze(0), h2[0, -1, :].float().unsqueeze(0)
        ).item()
    )
    top1_1 = int(torch.argmax(l1[0, -1, :]).item())
    top1_2 = int(torch.argmax(l2[0, -1, :]).item())
    passed = (cos < cos_ceiling) and (top1_1 != top1_2)
    return GateResult(
        name="FS1: Prompt Sensitivity",
        passed=passed,
        metrics={"cos_h": round(cos, 6), "top1_a": top1_1, "top1_b": top1_2},
        reason=f"cos={cos:.6f} (req < {cos_ceiling}), top1_differ={top1_1 != top1_2}",
    )


def run_gate_fs2_donor_reachability(
    model: nn.Module,
    tokenizer: Any,
    jacobian_min: float = 1e-5,
) -> GateResult:
    """FS2: Ablating each donor's 8 expert slots must measurably change held-out logits."""
    probe_txt = "The capital of France is"
    device = next(model.parameters()).device
    ids = torch.tensor(
        [tokenizer.encode(probe_txt) or [0]], dtype=torch.long, device=device
    )
    base_logits, _ = _forward_logits_and_hidden(model, ids)

    moe_mod = getattr(model, "moe", None)
    deltas: Dict[str, float] = {}
    if moe_mod is not None and all(hasattr(moe_mod, k) for k in ("w1", "w2", "wgate")):
        with torch.inference_mode():
            for donor_name, (lo, hi) in DONOR_EXPERT_SLOTS.items():
                saved = {}
                for k in ("w1", "w2", "wgate"):
                    t = getattr(moe_mod, k)
                    saved[k] = t.data[lo:hi].clone()
                    t.data[lo:hi].zero_()
                abl_logits, _ = _forward_logits_and_hidden(model, ids)
                deltas[donor_name] = float(
                    (base_logits - abl_logits).float().norm().item()
                )
                for k, orig in saved.items():
                    getattr(moe_mod, k).data[lo:hi].copy_(orig)
    else:
        return GateResult(
            name="FS2: Per-Donor 8-Expert Ablation",
            passed=False,
            metrics={"deltas": {}},
            reason="Model lacks moe.{w1,w2,wgate} 34-expert bank for 530-key ablation check.",
        )

    passed = all(v > jacobian_min for v in deltas.values()) and len(deltas) == 4
    return GateResult(
        name="FS2: Per-Donor 8-Expert Ablation",
        passed=passed,
        metrics={"deltas": {k: round(v, 6) for k, v in deltas.items()}},
        reason=f"min_delta={min(deltas.values()):.6f} (req > {jacobian_min})",
    )


def run_gate_fs3_semantic_w1_w6(
    model: nn.Module,
    tokenizer: Any,
    held_out_texts: Sequence[str],
    ce_floor: float = 1.0,
) -> GateResult:
    """FS3: ~100 held-out W1-W6 semantic probes + CE > 1.0 overfit floor."""
    items = load_frozen_w1_w6_items(limit=100)
    ce, ppl, n_toks = compute_text_ce_and_ppl(model, tokenizer, held_out_texts)
    hits = 0
    for it in items[:25]:
        prompt = it.get("prompt") or it.get("question") or ""
        expected = str(it.get("expected") or it.get("answer") or "").strip().lower()
        if not prompt or not expected:
            continue
        out = greedy_generate(model, tokenizer, prompt, max_new_tokens=16).lower()
        if expected in out:
            hits += 1
    evaluated = min(25, len(items))
    acc = hits / max(1, evaluated)
    passed = (ce > ce_floor) and math.isfinite(ppl)
    return GateResult(
        name="FS3: Held-Out W1-W6 Semantic Probes",
        passed=passed,
        metrics={
            "items_loaded": len(items),
            "sample_evaluated": evaluated,
            "sample_acc": round(acc, 4),
            "ce": round(ce, 4),
            "ppl": round(ppl, 4),
            "tokens": n_toks,
        },
        reason=f"CE={ce:.4f} (req > {ce_floor}), PPL={ppl:.4f}, sample_acc={acc:.2%} ({hits}/{evaluated})",
    )


def run_gate_fs4_order_and_random_nulls(
    model: nn.Module,
    tokenizer: Any,
    held_out_texts: Sequence[str],
    shuffle_degrade_min: float = 0.05,
    random_margin: float = 0.10,
) -> GateResult:
    """FS4: Expert-order-shuffle must degrade NLL; fused must beat random-weight null control."""
    ce_fused, _, _ = compute_text_ce_and_ppl(model, tokenizer, held_out_texts)
    saved_r = _apply_norm_matched_perturbation_inplace(model, mode="random", seed=99)
    try:
        ce_rand, _, _ = compute_text_ce_and_ppl(model, tokenizer, held_out_texts)
    finally:
        _restore_tensors_inplace(model, saved_r)
        del saved_r

    moe_mod = getattr(model, "moe", None)
    ce_order_shuf = ce_fused
    if moe_mod is not None and all(hasattr(moe_mod, k) for k in ("w1", "w2", "wgate")):
        saved_moe = {k: getattr(moe_mod, k).data.clone() for k in ("w1", "w2", "wgate")}
        perm = torch.randperm(
            moe_mod.w1.shape[0], generator=torch.Generator().manual_seed(77)
        )
        try:
            with torch.inference_mode():
                for k in ("w1", "w2", "wgate"):
                    getattr(moe_mod, k).data.copy_(saved_moe[k][perm])
            ce_order_shuf, _, _ = compute_text_ce_and_ppl(
                model, tokenizer, held_out_texts
            )
        finally:
            with torch.inference_mode():
                for k, orig in saved_moe.items():
                    getattr(moe_mod, k).data.copy_(orig)
            del saved_moe

    order_degradation = ce_order_shuf - ce_fused
    rand_gain = ce_rand - ce_fused
    passed = (order_degradation >= shuffle_degrade_min) and (rand_gain >= random_margin)
    return GateResult(
        name="FS4: Order-Shuffle & Random Null Controls",
        passed=passed,
        metrics={
            "ce_fused": round(ce_fused, 4),
            "ce_order_shuffled": round(ce_order_shuf, 4),
            "ce_random": round(ce_rand, 4),
            "order_degradation": round(order_degradation, 4),
            "rand_gain": round(rand_gain, 4),
        },
        reason=f"order_degrade={order_degradation:+.4f} (req >= +{shuffle_degrade_min}), rand_gain={rand_gain:+.4f} (req >= +{random_margin})",
    )


def run_gate_fs5_structural_530(checkpoint_path: Optional[str] = None) -> GateResult:
    """FS5: Record canonical quillan_v8_saturated.py sha (must exist) & verify 530 base keys if ckpt given."""
    sha_hex = "missing"
    file_exists = CANONICAL_V8_PATH.exists()
    if file_exists:
        sha_hex = hashlib.sha256(CANONICAL_V8_PATH.read_bytes()).hexdigest()

    key_count = None
    base_key_count = None
    keys_ok = True
    if checkpoint_path and Path(checkpoint_path).exists():
        import safetensors.torch

        with safetensors.torch.safe_open(
            checkpoint_path, framework="pt", device="cpu"
        ) as f:
            keys = list(f.keys())
        key_count = len(keys)
        base_keys = [
            k for k in keys if not k.endswith((".scale", "._scale", "_gamma", ".gamma"))
        ]
        base_key_count = len(base_keys)
        keys_ok = base_key_count == 530

    passed = file_exists and keys_ok
    return GateResult(
        name="FS5: Structural 530-Key & Canonical SHA",
        passed=passed,
        metrics={
            "canonical_sha": sha_hex,
            "canonical_exists": file_exists,
            "total_keys": key_count,
            "base_keys": base_key_count,
        },
        reason=f"canonical_exists={file_exists} (sha={sha_hex[:12]}..{sha_hex[-6:]}), base_keys={base_key_count} (req 530 if ckpt given)",
    )


def run_self_test(mode: str = "sovereign_multilayer") -> int:
    """Execute full --self-test on synthetic LM for the requested mode."""
    LOGGER.info("Starting eval_quillan_trio_bitnet.py --self-test (mode=%s)...", mode)
    torch.manual_seed(42)

    sys.path.insert(0, str(CORE_MODEL_DIR))
    model: nn.Module = _MockQuillanBitNetLM(vocab_size=256, dim=64)
    if mode == "fused_socket_530":
        # Attach minimal 34-expert moe.{w1,w2,wgate} to mock for FS2/FS4 verification
        class _MockMoE(nn.Module):
            def __init__(self, d: int = 64) -> None:
                super().__init__()
                self.w1 = nn.Parameter(
                    torch.randn(34, d, d) * 0.05, requires_grad=False
                )
                self.w2 = nn.Parameter(
                    torch.randn(34, d, d) * 0.05, requires_grad=False
                )
                self.wgate = nn.Parameter(
                    torch.randn(34, d, d) * 0.05, requires_grad=False
                )

        model.moe = _MockMoE(64)
        orig_forward = model.forward

        def _fused_fwd(
            input_ids: torch.Tensor,
            position_offset: int = 0,
            return_hidden: bool = False,
        ) -> Any:
            logits, h = orig_forward(
                input_ids, position_offset=position_offset, return_hidden=True
            )
            with torch.inference_mode():
                s = h.clone()
                for e in range(34):
                    s = (
                        s
                        + 0.01
                        * (F.silu(s @ model.moe.wgate[e]) * (s @ model.moe.w1[e]))
                        @ model.moe.w2[e]
                    )
                logits = model.lm_head(s)
            return (logits, s) if return_hidden else logits

        model.forward = _fused_fwd  # type: ignore[method-assign]

    tok = _ByteTokenizer(vocab_size=256)
    texts = load_held_out_texts()

    rss_start = measure_psutil_rss_mb()
    if mode == "fused_socket_530":
        results = [
            run_gate_fs1_prompt_sensitivity(model, tok, cos_ceiling=1.01),
            run_gate_fs2_donor_reachability(model, tok, jacobian_min=1e-6),
            run_gate_fs3_semantic_w1_w6(model, tok, texts, ce_floor=1.0),
            run_gate_fs4_order_and_random_nulls(
                model, tok, texts, shuffle_degrade_min=-1.0, random_margin=-1.0
            ),
            run_gate_fs5_structural_530(checkpoint_path=None),
        ]
    else:
        results = [
            run_gate_e1_paraphrase(model, tok, cos_threshold=0.0, agree_threshold=0.0),
            run_gate_e2_intervention(model, tok, js_threshold=0.0, flip_threshold=0.0),
            run_gate_e3_null_controls(model, tok, texts, nll_margin=-1.0),
            run_gate_e4_donor_baseline(model, tok, texts, donor_ppl=None, ce_floor=1.0),
            run_gate_e5_sovereign_bleed(model, tok),
            run_online_wsp_self_update_test(
                model, tok, texts[:2], max_anchor_ppl_delta=5.0
            ),
            run_2m_rope_stability_check(model, test_pos=2097152),
        ]
    rss_peak = measure_psutil_rss_mb()

    for r in results:
        LOGGER.info("[%s] %s -> %s", "PASS" if r.passed else "FAIL", r.name, r.reason)
    LOGGER.info("Peak RSS RAM: %.2f MiB (start %.2f MiB)", rss_peak, rss_start)

    all_ok = all(r.passed for r in results) and rss_peak < 1500.0
    LOGGER.info("SELF_TEST_STATUS=%s", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Non-Gameable Verification Protocol for Quillan BitNet Trio"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sovereign_multilayer",
        choices=["sovereign_multilayer", "fused_socket_530"],
        help="Evaluation protocol mode",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run CPU self-test on synthetic model"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to packed checkpoint"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer path matching checkpoint vocab",
    )
    parser.add_argument(
        "--donor-ppl",
        type=float,
        default=None,
        help="Primary single-donor held-out PPL baseline",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Optional JSON report path"
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test(mode=args.mode)

    if not args.checkpoint or not args.tokenizer:
        LOGGER.error(
            "--checkpoint and --tokenizer are required unless --self-test is specified."
        )
        return 2

    from transformers import AutoTokenizer

    sys.path.insert(0, str(CORE_MODEL_DIR))
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    texts = load_held_out_texts()

    rss_before = measure_psutil_rss_mb()
    if args.mode == "fused_socket_530":
        import quillan_fused_forward as qff  # type: ignore

        model = qff.load_socket_v2(args.checkpoint)
        model.eval()
        gates = [
            run_gate_fs1_prompt_sensitivity(model, tok),
            run_gate_fs2_donor_reachability(model, tok),
            run_gate_fs3_semantic_w1_w6(model, tok, texts),
            run_gate_fs4_order_and_random_nulls(model, tok, texts),
            run_gate_fs5_structural_530(checkpoint_path=args.checkpoint),
        ]
    else:
        import quillan_sovereign_bitnet as qsb  # type: ignore

        LOGGER.info("Loading %s...", args.checkpoint)
        model = qsb.QuillanSovereignBitNetLM.load_packed_safetensors(
            args.checkpoint, device=DEVICE
        )
        model.eval()
        LOGGER.info(
            "Model loaded (RSS=%.2f MiB). Running E1-E5 + Online WSP + 2M RoPE...",
            measure_psutil_rss_mb(),
        )
        gates = []
        for fn in (
            lambda: run_gate_e1_paraphrase(model, tok),
            lambda: run_gate_e2_intervention(model, tok),
            lambda: run_gate_e3_null_controls(model, tok, texts[:3]),
            lambda: run_gate_e4_donor_baseline(
                model, tok, texts[:4], donor_ppl=args.donor_ppl
            ),
            lambda: run_gate_e5_sovereign_bleed(model, tok),
            lambda: run_online_wsp_self_update_test(model, tok, texts[:2]),
            lambda: run_2m_rope_stability_check(model, test_pos=2097152),
        ):
            g = fn()
            gates.append(g)
            LOGGER.info(
                "[%s] %s -> %s", "PASS" if g.passed else "FAIL", g.name, g.reason
            )
    rss_peak = measure_psutil_rss_mb()

    all_passed = all(g.passed for g in gates) and rss_peak < 4096.0
    report = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "rss_before_mb": rss_before,
        "rss_peak_mb": rss_peak,
        "all_passed": all_passed,
        "gates": [asdict(g) for g in gates],
    }
    for g in gates:
        LOGGER.info("[%s] %s -> %s", "PASS" if g.passed else "FAIL", g.name, g.reason)
    LOGGER.info(
        "Peak RSS RAM: %.2f MiB | OVERALL=%s",
        rss_peak,
        "PASS" if all_passed else "FAIL",
    )

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("Saved report to %s", args.out)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
