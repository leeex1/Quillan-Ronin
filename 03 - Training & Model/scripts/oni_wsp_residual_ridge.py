#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closed-form WSP residual-ridge stage for Quillan's native ONI transfer checkpoints.

Scope (owner-locked 2026-11-26):
  * Writable leaf tensors ONLY: wte.weight (tied to lm_head.weight) and, per layer,
    h.{i}.attn.c_attn.weight, h.{i}.attn.c_proj.weight, h.{i}.moe.c_fc.weight,
    h.{i}.moe.c_proj.weight.  Every other state_dict key is protected by digest.
  * No backprop, no optimizer, no training.  Everything runs under torch.no_grad().
  * Target = ONE donor projected into ONI geometry with the saved alignment maps
    (P/Q, lowrank_resize, project_role are imported from oni_four_donor_transfer;
    this file does not duplicate that logic).  Residuals are measured on
    pre-hook activations captured with compute_aux=False, deliberation=False.
  * Outputs are written ONLY under exports/oni_transfer/wsp/.

Subcommands:
  apply   compute pre-registered closed-form deltas, save a CANDIDATE checkpoint
  verify  paired A/B with fresh reloads + sham / permuted-key controls + gates

No GGUF WSP scores are reused.  No improvement is claimed unless the frozen dev
set shows it AND the sham / permuted controls are beaten.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(r"F:\02_QUILLAN")
SCRIPTS = ROOT / "03 - Training & Model" / "scripts"
EXPORTS = ROOT / "exports"
OUT = EXPORTS / "oni_transfer"
WSP_OUT = OUT / "wsp"

for _p in (str(SCRIPTS), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from oni_v2_config import owner_config  # noqa: E402
from quillan_v5_4_oni import QuillanRoninOni  # noqa: E402
from oni_four_donor_transfer import (  # noqa: E402
    DONOR_FILES,
    EMB_KEY,
    GEOMETRY,
    N_REACHABLE,
    OWNER_VOCAB,
    ROLE_KEYS,
    donor_layer_count,
    donor_layer_index,
    load_index,
    log,
    lowrank_resize,  # noqa: F401  (reused inside project_role)
    project_role,
    read_rows,
    tensor_digest,  # noqa: F401
)
from safetensors import safe_open  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# PRE-REGISTERED PARAMETERS  (printed + saved before any run)
# ---------------------------------------------------------------------------

PREREG: Dict[str, Any] = {
    "stage": "oni_transfer_closed_form_wsp_residual_ridge",
    "date": "2026-11-26",
    "alpha": 0.05,
    "ridge_lambda": 1e-2,
    "rank_cap": 64,
    "frobenius_cap_ratio": 0.02,
    "seed": 20261126,
    "seq_len": 16,
    "n_calibration": 32,
    "n_dev": 16,
    "donor": {"mini": "qwen", "main": "qwen"},
    "clean_flags": {
        "compute_aux": False,
        "deliberation": False,
        "return_aux": False,
        "use_cache": False,
    },
    "target_rule": (
        "one donor projected into ONI geometry via saved alignment P/Q "
        "(embeddings: donor_rows @ P; layers: project_role); residual = "
        "target_output - current_student_output on captured pre-hook inputs"
    ),
    "ridge_rule": (
        "X = pre-hook inputs as [N, in]; R = residual as [out, N]; "
        "dW = R @ solve(X @ X.T + lambda*I, X) ; rank-cap via SVD of R ; "
        "Frobenius cap vs ||W|| ; W_new = W + alpha*dW"
    ),
    "controls": [
        "sham: Gaussian R matched to real R Frobenius norm, same seed",
        "permuted-key: shuffle R columns against fixed X, same seed",
    ],
    "gates": [
        "protected state_dict keys byte-identical (sha256)",
        "all dev forwards finite",
        "dev CE(wsp) < dev CE(base) and dev top1(wsp) >= dev top1(base)",
        "dev CE(wsp) < min(dev CE(sham), dev CE(permuted))",
    ],
    "no_claim_without_held_out_evidence": True,
}

# Fixed, self-contained prompt lists (no external datasets).  Each base sentence
# is repeated deterministically so every encoding is longer than seq_len + 1.
_CALIB_BASES: Tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog.",
    "A rising tide lifts every boat in the harbor.",
    "She recorded the measurement in a careful notebook.",
    "The engineer checked the bridge before the train arrived.",
    "Rain fell softly on the roof all afternoon long.",
    "He planted a young tree beside the garden gate.",
    "Every morning the baker opens the shop at six.",
    "The student solved the difficult problem by herself.",
    "A small candle lit the room during the storm.",
    "They walked together along the river path homeward.",
    "The old map showed a route across the mountains.",
    "Our compiler reports no errors after the latest fix.",
    "Winter mornings here are cold and very quiet.",
    "The committee reviewed the proposal with great care.",
    "Birds gather near the window when the sun appears.",
    "She kept a promise to return before the evening.",
    "A clear answer is better than a clever silence.",
    "The ledger balances after the final entry today.",
    "He sharpened the blade and began the long work.",
    "Children laughed in the courtyard after the lesson.",
    "The train arrived exactly on time at the station.",
    "A quiet mind can hear the smallest signal clearly.",
    "We checked every assumption before making the call.",
    "The garden grows better with patient daily watering.",
    "Night settled over the valley and the lights dimmed.",
    "The recipe calls for flour water salt and patience.",
    "He wrote the summary in one careful sitting.",
    "Truth survives the test when the proof is honest.",
    "The market opens early and closes at sunset.",
    "A steady hand makes the delicate cut possible.",
    "They celebrated the small victory with quiet pride.",
    "The library keeps its oldest books behind glass.",
)

_DEV_BASES: Tuple[str, ...] = (
    "A traveler asked the way to the northern village.",
    "The telescope revealed a faint distant galaxy.",
    "Rainwater collected in the wooden bucket overnight.",
    "She labeled each sample with a careful handwritten note.",
    "The mechanic tightened the last loose bolt today.",
    "Morning light crossed the empty kitchen floor.",
    "A single bell rang across the quiet town square.",
    "The archivist preserved the letters in acid free paper.",
    "He chose the longer path because it was safer.",
    "The kettle boiled while we finished the conversation.",
    "Small steps repeated daily produce remarkable results.",
    "The navigator marked the reef on the chart carefully.",
    "Autumn leaves covered the steps leading to the door.",
    "A good question is worth more than a quick answer.",
    "The verify step caught the error before release.",
    "Snow melted slowly from the roof into the gutters.",
)

CALIB_PROMPTS: Tuple[str, ...] = tuple((s + " ") * 4 for s in _CALIB_BASES)
DEV_PROMPTS: Tuple[str, ...] = tuple((s + " ") * 4 for s in _DEV_BASES)

SEQ_LEN = int(PREREG["seq_len"])


def prereg_path() -> Path:
    return WSP_OUT / "preregistered_params.json"


def write_prereg() -> None:
    WSP_OUT.mkdir(parents=True, exist_ok=True)
    obj = dict(PREREG)
    obj["calibration_bases"] = list(_CALIB_BASES)
    obj["dev_bases"] = list(_DEV_BASES)
    with open(prereg_path(), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    log(f"pre-registered parameters saved -> {prereg_path()}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def owner_ckpt_path(model_key: str) -> Path:
    return OUT / f"quillan_{model_key}_oni_transfer.pt"


def candidate_ckpt_path(model_key: str) -> Path:
    return WSP_OUT / f"quillan_{model_key}_oni_wsp_candidate.pt"


def alignment_path() -> Path:
    return OUT / "oni_donor_alignment_rebuilt.pt"


def writable_keys(n_layer: int) -> set:
    keys = {"wte.weight", "lm_head.weight"}  # lm_head aliases wte; never edited twice
    for i in range(n_layer):
        keys.add(f"h.{i}.attn.c_attn.weight")
        keys.add(f"h.{i}.attn.c_proj.weight")
        keys.add(f"h.{i}.moe.c_fc.weight")
        keys.add(f"h.{i}.moe.c_proj.weight")
    return keys


def edit_keys(n_layer: int) -> List[str]:
    keys = ["wte.weight"]
    for i in range(n_layer):
        keys.append(f"h.{i}.attn.c_attn.weight")
        keys.append(f"h.{i}.attn.c_proj.weight")
        keys.append(f"h.{i}.moe.c_fc.weight")
        keys.append(f"h.{i}.moe.c_proj.weight")
    return keys


def get_param(model: torch.nn.Module, key: str) -> torch.nn.Parameter:
    return model.get_parameter(key)


def fix_meta_buffers(model: torch.nn.Module, cfg: Any) -> None:
    """Materialize non-persistent RoPE buffers left on meta by meta-init."""
    base = 500000.0  # RotaryEmbedding default in quillan_v5_4_oni.py
    head_dim = int(cfg.head_dim)
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    for blk in model.h:
        rope = blk.attn.rope
        rope.inv_freq = inv.clone()
        rope._cos = torch.zeros(1)
        rope._sin = torch.zeros(1)
        rope._built_len = 0
    model.eval()


def fresh_model(
    model_key: str, ckpt_path: Path
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Meta-init + assign=True: no 2.6 GB weight duplication; checkpoint mmap'd."""
    cfg = owner_config(model_key)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True, mmap=True)
    with torch.device("meta"):
        model = QuillanRoninOni(cfg)
    res = model.load_state_dict(ckpt["state_dict"], strict=False, assign=True)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(
            f"state_dict mismatch: missing={len(res.missing_keys)} "
            f"unexpected={len(res.unexpected_keys)}"
        )
    # Re-tie: load_state_dict(assign=True) can split the tied wte/lm_head pair.
    model.lm_head.weight = model.wte.weight
    fix_meta_buffers(model, cfg)
    return model, ckpt


def clone_writable(model: torch.nn.Module, model_key: str) -> Dict[str, torch.Tensor]:
    cfg = owner_config(model_key)
    return {
        k: get_param(model, k).detach().clone() for k in edit_keys(int(cfg.n_layer))
    }


def install_writable(model: torch.nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    """Install a fresh set of writable weights; keeps the wte/lm_head tie."""
    for key, tensor in weights.items():
        cur = get_param(model, key)
        if cur.shape != tensor.shape:
            raise RuntimeError(
                f"shape mismatch for {key}: {tuple(cur.shape)} vs {tuple(tensor.shape)}"
            )
        new_p = torch.nn.Parameter(tensor.contiguous())
        module_path = key[: -len(".weight")]
        parent = model.get_submodule(module_path)
        setattr(parent, "weight", new_p)
    model.lm_head.weight = model.wte.weight  # single Parameter object


def protected_digest(model: torch.nn.Module, model_key: str) -> str:
    cfg = owner_config(model_key)
    sd = model.state_dict()
    allow = writable_keys(int(cfg.n_layer))
    h = hashlib.sha256()
    for key in sorted(sd):
        if key in allow:
            continue
        t = sd[key]
        h.update(key.encode("utf-8"))
        h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def tokenize(prompts: Tuple[str, ...]) -> List[List[int]]:
    tok = Tokenizer.from_file(
        str(EXPORTS / "quillan_tokenizer_v54_fourdonor" / "tokenizer.json")
    )
    if len(tok.get_vocab()) != OWNER_VOCAB:
        raise RuntimeError("tokenizer vocab mismatch")
    out: List[List[int]] = []
    for text in prompts:
        ids = [int(i) for i in tok.encode(text).ids if 0 < i < N_REACHABLE]
        if len(ids) < SEQ_LEN + 1:
            raise RuntimeError(f"prompt too short after filtering: {len(ids)} tokens")
        out.append(ids[: SEQ_LEN + 1])
    return out


def forward_inputs(token_lists: List[List[int]]) -> List[torch.Tensor]:
    return [torch.tensor([ids[:SEQ_LEN]], dtype=torch.long) for ids in token_lists]


def run_clean(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(
            ids,
            deliberation=False,
            compute_aux=False,
            return_aux=False,
            use_cache=False,
        )


# ---------------------------------------------------------------------------
# residual targets from ONE donor (one donor at a time, safetensors slices)
# ---------------------------------------------------------------------------


def donor_embedding_target(
    donor: str,
    idx: Dict[str, object],
    align: Dict[str, Any],
    model_key: str,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """[N, H] ONI-space target embeddings for the calibration token ids."""
    slots: torch.Tensor = idx["slots"]  # type: ignore[assignment]
    q2d: Dict[str, torch.Tensor] = idx["q2d"]  # type: ignore[assignment]
    did = q2d[donor][slots[token_ids]]
    ok = (did >= 0).nonzero(as_tuple=True)[0]
    H = int(GEOMETRY[model_key]["hidden"])
    target = align[f"target_embedding_{model_key}"][token_ids].clone()  # owner fallback
    if ok.numel():
        rows = read_rows(DONOR_FILES[donor], EMB_KEY, did[ok].tolist())
        P = align[f"maps_{model_key}"][donor]["P"].to(torch.float32)
        proj = rows.to(torch.float32) @ P
        target[ok] = proj[:, :H]
        del rows, proj
    return target.to(torch.float32)


def donor_layer_targets(
    donor: str,
    align: Dict[str, Any],
    model_key: str,
) -> Dict[str, torch.Tensor]:
    """Packed ONI-shape donor target weights for the four layer leaf families."""
    geom = GEOMETRY[model_key]
    H, F, L = int(geom["hidden"]), int(geom["ffn"]), int(geom["layers"])
    maps = align[f"maps_{model_key}"][donor]
    D = int(maps["donor_dim"])
    Q, P = maps["Q"].to(torch.float32), maps["P"].to(torch.float32)
    path = DONOR_FILES[donor]
    n_donor = donor_layer_count(path)
    out: Dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        for i in range(L):
            dl = donor_layer_index(i, L, n_donor)
            prefix = f"model.layers.{dl}."
            proj: Dict[str, torch.Tensor] = {}
            for role in ROLE_KEYS:
                key = prefix + ROLE_KEYS[role]
                if key not in fh.keys():
                    raise RuntimeError(f"missing donor key {key}")
                W = torch.as_tensor(fh.get_tensor(key), dtype=torch.float32)
                proj[role] = project_role(W, Q, P, D, H, F, role).to(torch.float32)
                del W
            out[f"h.{i}.attn.c_attn.weight"] = torch.cat(
                (proj["q"], proj["k"], proj["v"]), dim=0
            ).contiguous()
            out[f"h.{i}.attn.c_proj.weight"] = proj["o"].contiguous()
            out[f"h.{i}.moe.c_fc.weight"] = torch.cat(
                (proj["gate"], proj["up"]), dim=0
            ).contiguous()
            out[f"h.{i}.moe.c_proj.weight"] = proj["down"].contiguous()
            del proj
            gc.collect()
    return out


# ---------------------------------------------------------------------------
# closed-form ridge
# ---------------------------------------------------------------------------


def ridge_delta(
    R: torch.Tensor,
    X: torch.Tensor,
    lam: float,
    rank_cap: int,
    frob_cap: float,
    w_norm: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """dW = R @ solve(X @ X.T + lam*I, X); R [out,N], X [N,in]."""
    n = X.size(0)
    gram = X @ X.T
    # Scale-relative regularization: absolute lam underflows relative to the
    # gram diagonal when calibration prompts repeat token patterns, producing
    # a singular solve. Floor keeps it finite on small activations.
    scale = float(gram.diag().mean().item()) if n else 0.0
    lam_eff = max(lam * max(scale, 1e-8), 1e-8)
    gram.add_(torch.eye(n, dtype=gram.dtype) * lam_eff)
    try:
        V = torch.linalg.solve(gram, X)  # [N, in]
    except torch.linalg.LinAlgError:
        V = torch.linalg.lstsq(gram, X).solution
    dW = R @ V  # [out, in]
    pre_rank = float(dW.norm().item())
    if rank_cap > 0 and rank_cap < min(R.shape):
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
        k = min(rank_cap, int(S.numel()))
        dW = (U[:, :k] * S[:k]) @ (Vh[:k, :] @ V)
        del U, S, Vh
    post_rank = float(dW.norm().item())
    cap = frob_cap * w_norm
    scale = 1.0
    if post_rank > cap > 0:
        scale = cap / (post_rank + 1e-12)
        dW.mul_(scale)
    return dW, {
        "dW_norm_prerank": pre_rank,
        "dW_norm_postrank": post_rank,
        "frob_cap": cap,
        "cap_scale": scale,
        "ridge_lambda_effective": lam_eff,
        "dW_norm_final": float(dW.norm().item()),
    }


def one_hot_tokens(token_ids: torch.Tensor) -> torch.Tensor:
    X = torch.zeros((token_ids.numel(), OWNER_VOCAB), dtype=torch.float32)
    X.scatter_(1, token_ids.view(-1, 1), 1.0)
    return X


# ---------------------------------------------------------------------------
# capture pre-hook inputs
# ---------------------------------------------------------------------------


def capture_inputs(
    model: torch.nn.Module,
    inputs: List[torch.Tensor],
    model_key: str,
) -> Dict[str, torch.Tensor]:
    keys = edit_keys(int(owner_config(model_key).n_layer))
    buckets: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}

    def make_hook(key: str):
        def hook(_module: torch.nn.Module, inp: Tuple[torch.Tensor, ...]) -> None:
            buckets[key].append(
                inp[0].detach().float().reshape(-1, inp[0].shape[-1]).clone()
            )

        return hook

    handles = []
    for key in keys:
        module_path = key[: -len(".weight")]
        handles.append(
            model.get_submodule(module_path).register_forward_pre_hook(make_hook(key))
        )
    try:
        for ids in inputs:
            run_clean(model, ids)
    finally:
        for h in handles:
            h.remove()
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        if not buckets[key]:
            raise RuntimeError(f"no pre-hook input captured for {key}")
        out[key] = torch.cat(buckets[key], dim=0).contiguous()
    del buckets
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# APPLY
# ---------------------------------------------------------------------------


def apply_stage(model_key: str, donor: str) -> None:
    if model_key not in ("mini", "main"):
        raise SystemExit("model must be mini or main")
    WSP_OUT.mkdir(parents=True, exist_ok=True)
    write_prereg()
    t0 = time.time()
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.manual_seed(int(PREREG["seed"]))

    ckpt_path = owner_ckpt_path(model_key)
    if not ckpt_path.exists():
        raise SystemExit(f"missing baseline checkpoint {ckpt_path}")
    cfg = owner_config(model_key)
    n_layer = int(cfg.n_layer)
    geom = GEOMETRY[model_key]
    H, F = int(geom["hidden"]), int(geom["ffn"])
    log(f"=== WSP apply {model_key}: donor={donor} baseline={ckpt_path.name} ===")

    idx = load_index()
    align = torch.load(
        str(alignment_path()), map_location="cpu", weights_only=True, mmap=True
    )
    calib = tokenize(CALIB_PROMPTS)
    all_token_ids = torch.tensor(
        [ids[:SEQ_LEN] for ids in calib], dtype=torch.long
    ).reshape(-1)

    model, _ckpt = fresh_model(model_key, ckpt_path)
    base_weights = clone_writable(model, model_key)
    digest_before = protected_digest(model, model_key)
    log(f"protected keys digest (before): {digest_before[:24]}")

    captured = capture_inputs(model, forward_inputs(calib), model_key)
    layer_n = captured[edit_keys(n_layer)[1]].shape[0]
    log(
        f"captured pre-hook inputs for {len(captured)} writable modules; N(layer)={layer_n}"
    )

    # ---- embedding residual (one-hot keys) ----
    emb_target = donor_embedding_target(donor, idx, align, model_key, all_token_ids)
    emb_student = base_weights["wte.weight"][all_token_ids].to(torch.float32)
    R_emb = (emb_target - emb_student).T.contiguous()  # [H, N]
    X_emb = one_hot_tokens(all_token_ids)  # [N, vocab]
    dW_emb, emb_stats = ridge_delta(
        R_emb,
        X_emb,
        float(PREREG["ridge_lambda"]),
        int(PREREG["rank_cap"]),
        float(PREREG["frobenius_cap_ratio"]),
        float(base_weights["wte.weight"].norm().item()),
    )
    log(
        f"wte.weight residual norm={float(R_emb.norm().item()):.6f} "
        f"dW norm={emb_stats['dW_norm_final']:.6f} cap_scale={emb_stats['cap_scale']:.4f}"
    )
    new_weights: Dict[str, torch.Tensor] = {}
    new_weights["wte.weight"] = (
        base_weights["wte.weight"] + float(PREREG["alpha"]) * dW_emb.T
    )
    del dW_emb, X_emb, R_emb, emb_target, emb_student
    gc.collect()

    # ---- layer residuals (one donor at a time) ----
    targets = donor_layer_targets(donor, align, model_key)
    stats: Dict[str, Dict[str, float]] = {"wte.weight": emb_stats}
    for key in edit_keys(n_layer)[1:]:
        W = base_weights[key].to(torch.float32)
        Wt = targets[key]
        X = captured[key]  # [N, in]
        b = model.get_submodule(key[: -len(".weight")]).bias
        student_out = X @ W.T
        if b is not None:
            student_out = student_out + b.detach().float()
        R = (X @ Wt.T - student_out).T.contiguous()  # [out, N]
        rnorm = float(R.norm().item())
        if rnorm < 1e-12:
            new_weights[key] = base_weights[key].clone()
            stats[key] = {
                "residual_norm": 0.0,
                "dW_norm_final": 0.0,
                "zero_residual": 1.0,
            }
            del R, W, Wt
            continue
        dW, st = ridge_delta(
            R,
            X,
            float(PREREG["ridge_lambda"]),
            int(PREREG["rank_cap"]),
            float(PREREG["frobenius_cap_ratio"]),
            float(base_weights[key].norm().item()),
        )
        st["residual_norm"] = rnorm
        stats[key] = st
        new_weights[key] = base_weights[key] + float(PREREG["alpha"]) * dW
        log(
            f"{key}: residual={rnorm:.6f} dW={st['dW_norm_final']:.6f} "
            f"cap_scale={st['cap_scale']:.4f}"
        )
        del dW, R, W, Wt, student_out
        gc.collect()
    del targets, captured
    gc.collect()

    # finite forward + protected-key check on the edited model
    install_writable(model, new_weights)
    digest_after = protected_digest(model, model_key)
    protected_ok = digest_after == digest_before
    probe = forward_inputs(calib)[0]
    logits = run_clean(model, probe)
    finite = bool(torch.isfinite(logits).all())
    log(
        f"edited forward: shape={tuple(logits.shape)} finite={finite} "
        f"maxabs={float(logits.abs().max()):.4f}"
    )
    if not finite:
        raise RuntimeError("non-finite logits after WSP edit")
    if not protected_ok:
        raise RuntimeError(
            f"protected tensors changed: {digest_before[:24]} -> {digest_after[:24]}"
        )
    log(f"protected keys digest (after):  {digest_after[:24]}  equal={protected_ok}")

    metadata = {
        "build": "closed_form_wsp_residual_ridge",
        "model": model_key,
        "donor": donor,
        "baseline": str(ckpt_path),
        "preregistered": PREREG,
        "promoted": False,
        "promotion_note": (
            "candidate only; verify subcommand decides promotion on frozen dev evidence"
        ),
        "protected_keys_digest_before": digest_before,
        "protected_keys_digest_after": digest_after,
        "protected_keys_equal": protected_ok,
        "finite_forward": finite,
        "n_protected_keys": len(model.state_dict()) - len(writable_keys(n_layer)),
        "N_calibration_tokens": int(all_token_ids.numel()),
        "per_key_stats": stats,
        "H": H,
        "F": F,
        "n_layer": n_layer,
        "limitations": [
            "single projected donor target, not a ground-truth label",
            "no claim of coherence or benchmark improvement",
            "no backprop / optimizer used; closed-form ridge only",
            "dev set is a small fixed prompt list, not an external benchmark",
        ],
    }
    save_path = candidate_ckpt_path(model_key)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, str(save_path))
    log(f"saved CANDIDATE -> {save_path} ({save_path.stat().st_size / 1e9:.2f} GB)")
    log(f"apply finished in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# VERIFY
# ---------------------------------------------------------------------------


def evaluate(model: torch.nn.Module, token_lists: List[List[int]]) -> Dict[str, float]:
    ce_sum, correct = 0.0, 0
    for ids in token_lists:
        inp = torch.tensor([ids[:SEQ_LEN]], dtype=torch.long)
        target = torch.tensor([ids[SEQ_LEN]], dtype=torch.long)
        logits = run_clean(model, inp)[0, -1, :]
        ce_sum += float(F.cross_entropy(logits.unsqueeze(0), target).item())
        correct += int(logits.argmax(dim=-1).item() == int(target.item()))
    n = len(token_lists)
    return {"ce": ce_sum / n, "top1": correct / n, "n": float(n)}


def variant_weights(
    model_key: str,
    base: Dict[str, torch.Tensor],
    kind: str,
    calib_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build wsp / sham / permuted writable weight sets from the candidate stats."""
    cand = torch.load(
        str(candidate_ckpt_path(model_key)),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    meta = cand["metadata"]
    stats: Dict[str, Dict[str, float]] = meta["per_key_stats"]
    out: Dict[str, torch.Tensor] = {}
    if kind == "base":
        return {k: v.clone() for k, v in base.items()}
    if kind == "wsp":
        return {k: cand["state_dict"][k].detach().clone() for k in base}

    # recompute the ridge deltas with corrupted R (sham / permuted)
    torch.manual_seed(int(PREREG["seed"]))
    n_layer = int(owner_config(model_key).n_layer)
    align = torch.load(
        str(alignment_path()), map_location="cpu", weights_only=True, mmap=True
    )
    idx = load_index()
    calib = tokenize(CALIB_PROMPTS)
    m, _ = fresh_model(model_key, owner_ckpt_path(model_key))
    install_writable(m, base)
    X_by_key = capture_inputs(m, forward_inputs(calib), model_key)
    biases = {
        k: (
            m.get_submodule(k[: -len(".weight")]).bias.detach().float()
            if m.get_submodule(k[: -len(".weight")]).bias is not None
            else None
        )
        for k in base
        if k != "wte.weight"
    }
    del m
    gc.collect()

    emb_target = donor_embedding_target(meta["donor"], idx, align, model_key, calib_ids)
    emb_student = base["wte.weight"][calib_ids].to(torch.float32)
    R_emb = (emb_target - emb_student).T.contiguous()
    X_emb = one_hot_tokens(calib_ids)

    def corrupt(R: torch.Tensor) -> torch.Tensor:
        if kind == "sham":
            g = torch.randn(
                R.shape, generator=torch.Generator().manual_seed(int(PREREG["seed"]))
            )
            return g * (float(R.norm().item()) / (float(g.norm().item()) + 1e-12))
        perm = torch.randperm(
            R.shape[1], generator=torch.Generator().manual_seed(int(PREREG["seed"]))
        )
        return R[:, perm].contiguous()

    dW_emb, _ = ridge_delta(
        corrupt(R_emb),
        X_emb,
        float(PREREG["ridge_lambda"]),
        int(PREREG["rank_cap"]),
        float(PREREG["frobenius_cap_ratio"]),
        float(base["wte.weight"].norm().item()),
    )
    out["wte.weight"] = base["wte.weight"] + float(PREREG["alpha"]) * dW_emb.T
    del dW_emb, X_emb, R_emb, emb_target, emb_student
    gc.collect()

    targets = donor_layer_targets(meta["donor"], align, model_key)
    for key in edit_keys(n_layer)[1:]:
        st = stats[key]
        if float(st.get("zero_residual", 0.0)) > 0.5:
            out[key] = base[key].clone()
            continue
        W = base[key].to(torch.float32)
        X = X_by_key[key]
        student_out = X @ W.T
        if biases[key] is not None:
            student_out = student_out + biases[key]
        R = (X @ targets[key].T - student_out).T.contiguous()
        dW, _ = ridge_delta(
            corrupt(R),
            X,
            float(PREREG["ridge_lambda"]),
            int(PREREG["rank_cap"]),
            float(PREREG["frobenius_cap_ratio"]),
            float(base[key].norm().item()),
        )
        out[key] = base[key] + float(PREREG["alpha"]) * dW
        del dW, R, W, student_out
        gc.collect()
    del targets, X_by_key, cand
    gc.collect()
    return out


def verify_stage(model_key: str) -> None:
    WSP_OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    t0 = time.time()
    calib = tokenize(CALIB_PROMPTS)
    dev = tokenize(DEV_PROMPTS)
    calib_ids = torch.tensor(
        [ids[:SEQ_LEN] for ids in calib], dtype=torch.long
    ).reshape(-1)
    log(f"=== WSP verify {model_key}: calib={len(calib)} dev={len(dev)} ===")

    # baseline: fresh reload, protected digest, A/B metrics
    base_model, _ = fresh_model(model_key, owner_ckpt_path(model_key))
    base_weights = clone_writable(base_model, model_key)
    digest_base = protected_digest(base_model, model_key)
    m_base = evaluate(base_model, calib)
    d_base = evaluate(base_model, dev)
    log(
        f"BASE  calib ce={m_base['ce']:.5f} top1={m_base['top1']:.3f} | "
        f"dev ce={d_base['ce']:.5f} top1={d_base['top1']:.3f}"
    )
    del base_model
    gc.collect()

    results: Dict[str, Dict[str, Dict[str, float]]] = {
        "base": {"calib": m_base, "dev": d_base}
    }
    digests: Dict[str, str] = {"base": digest_base}

    # candidate + sham + permuted, each on its own fresh reload
    for kind in ("wsp", "sham", "permuted"):
        torch.manual_seed(int(PREREG["seed"]))
        weights = variant_weights(model_key, base_weights, kind, calib_ids)
        model, _ = fresh_model(model_key, owner_ckpt_path(model_key))
        install_writable(model, weights)
        dg = protected_digest(model, model_key)
        finite = all(
            bool(torch.isfinite(run_clean(model, inp)).all())
            for inp in forward_inputs(dev)
        )
        mc = evaluate(model, calib)
        md = evaluate(model, dev)
        log(
            f"{kind.upper():8s} calib ce={mc['ce']:.5f} top1={mc['top1']:.3f} | "
            f"dev ce={md['ce']:.5f} top1={md['top1']:.3f} | "
            f"finite={finite} protected_ok={dg == digest_base}"
        )
        results[kind] = {"calib": mc, "dev": md}
        digests[kind] = dg
        del model, weights
        gc.collect()

    # ---- gates ----
    ck = torch.load(
        str(candidate_ckpt_path(model_key)),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    meta = ck["metadata"]
    finite_meta = bool(meta["finite_forward"])
    d_wsp, d_basev = results["wsp"]["dev"], results["base"]["dev"]
    d_sham, d_perm = results["sham"]["dev"], results["permuted"]["dev"]
    gate_protected = all(v == digest_base for v in digests.values()) and bool(
        meta["protected_keys_equal"]
    )
    gate_finite = finite_meta
    gate_heldout = (d_wsp["ce"] < d_basev["ce"]) and (d_wsp["top1"] >= d_basev["top1"])
    gate_controls = d_wsp["ce"] < min(d_sham["ce"], d_perm["ce"])
    promoted = bool(gate_protected and gate_finite and gate_heldout and gate_controls)

    report = {
        "model": model_key,
        "donor": meta["donor"],
        "preregistered": PREREG,
        "calibration": {"n": len(calib), "ce": results["base"]["calib"]["ce"]},
        "dev": {k: v["dev"] for k, v in results.items()},
        "protected_digests": digests,
        "gates": {
            "protected_keys_identical": gate_protected,
            "dev_forwards_finite": gate_finite,
            "heldout_improvement": gate_heldout,
            "beats_sham_and_permuted": gate_controls,
        },
        "promoted": promoted,
        "decision": (
            "PROMOTE candidate as improved"
            if promoted
            else "DO NOT PROMOTE: held-out evidence insufficient or controls not beaten"
        ),
        "limitations": [
            "small fixed prompt bank; no external benchmark",
            "last-token CE/top1 only; one forward per prompt",
            "single projected donor target, not ground-truth supervision",
            "sham and permuted controls are diagnostics, not proof of generalization",
        ],
        "elapsed_s": time.time() - t0,
    }
    rp = WSP_OUT / f"wsp_gate_report_{model_key}.json"
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    if promoted:
        meta["promoted"] = True
        meta["promotion_report"] = str(rp)
        torch.save(
            {"state_dict": ck["state_dict"], "metadata": meta},
            str(candidate_ckpt_path(model_key)),
        )
    log(
        f"GATES protected={gate_protected} finite={gate_finite} "
        f"heldout={gate_heldout} controls={gate_controls} -> "
        f"{'PROMOTED' if promoted else 'NOT PROMOTED'}"
    )
    log(f"gate report -> {rp}")
    log(f"verify finished in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--model", choices=("mini", "main"), default="mini")
    p_apply.add_argument("--donor", choices=tuple(DONOR_FILES), default=None)
    p_ver = sub.add_parser("verify")
    p_ver.add_argument("--model", choices=("mini", "main"), default="mini")
    args = ap.parse_args()
    donor = getattr(args, "donor", None) or PREREG["donor"][str(args.model)]
    WSP_OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "apply":
        apply_stage(str(args.model), donor)
    else:
        verify_stage(str(args.model))


if __name__ == "__main__":
    main()
