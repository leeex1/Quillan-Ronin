#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mapped 4 -> 2 -> 1 donor KNOWLEDGE TRANSFER into Quillan's native ONI body.

This module is deliberately NOT a state-dict transplant. Donor geometries do
not match ONI (verified by scripts/adapter_diagnostics/oni_graph_falsification.py):
  qwen 896/4864, llama 2048/8192, bitnet 2560/6912, falcon 2048/8192
vs ONI mini 1152/3072 and main 1536/3072. Every donor tensor is therefore
projected through closed-form least-squares maps into ONI geometry and merged
by pair (qwen,bitnet) and (llama,falcon), then across the two pairs.

Corrected index space (owner correction 2026-11-26):
  tokenizer id -> token STRING -> canonical union slot via quillan_vocab.json.
  The old artifact used compiled_embedding[:151648] which is the WRONG rows.
  Here target rows = compiled_embedding[slot_for_token_id] and donor ids =
  q2d[name][slot_for_token_id]. The 5 sovereign specials (151643..151647)
  have no union slot and are never written with donor/noise data.

Inference contract for this build (verified falsification): clean path only.
  compute_aux=False, deliberation=False. Recorded in checkpoint metadata.

Subcommands:
  align            rebuild corrected alignment artifact
  build  mini|main map 4->2->1 and save checkpoint
  verify           load both checkpoints into fresh models and smoke-test
  all              align + build mini + build main + verify

No backprop, no training, no hifi/lofi code paths are touched.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(r"F:\02_QUILLAN")
SCRIPTS = ROOT / "03 - Training & Model" / "scripts"
EXPORTS = ROOT / "exports"
OUT = EXPORTS / "oni_transfer"
DONOR_DIR = Path(r"F:\quillan_cache\donors")

for _p in (str(SCRIPTS), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from oni_v2_config import owner_config  # noqa: E402
from quillan_v5_4_oni import QuillanRoninOni  # noqa: E402

from safetensors import safe_open  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DONORS: Tuple[str, ...] = ("qwen", "llama", "bitnet", "falcon")
PAIRS: Tuple[Tuple[str, str], ...] = (("qwen", "bitnet"), ("llama", "falcon"))
DONOR_FILES = {n: DONOR_DIR / f"{n}.safetensors" for n in DONORS}
EMB_KEY = "model.embed_tokens.weight"

OWNER_VOCAB = 151648
N_REACHABLE = 151643  # owner ids 0..151642 have union slots
SPECIAL_IDS = tuple(range(N_REACHABLE, OWNER_VOCAB))
FIT_N = 30000
EVAL_N = 30000
SVD_K = 128  # deterministic rank cap for shape-safe low-rank resize

EMBED_ALPHA = 0.50  # blend strength of donor knowledge into owner embedding
EMBED_CAP_RATIO = 0.35  # per-row delta norm cap relative to owner row norm

GEOMETRY = {
    "mini": {"hidden": 896, "ffn": 1792, "layers": 6, "heads": 14},
    "main": {"hidden": 1152, "ffn": 2304, "layers": 12, "heads": 18},
}

# Owner-locked real-owner-config counts from scripts/adapter_diagnostics/oni_one_param.py
EXPECTED_PARAMS = {"mini": 650_666_808, "main": 1_076_774_730}

ROLE_KEYS = {
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "o": "self_attn.o_proj.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}
ROLES = tuple(ROLE_KEYS.keys())

ALIGN_PATH = OUT / "oni_donor_alignment_rebuilt.pt"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# index space: tokenizer id -> string -> canonical union slot
# ---------------------------------------------------------------------------


def load_index() -> Dict[str, object]:
    log("building tokenizer-id -> union-slot index")
    tok_path = EXPORTS / "quillan_tokenizer_v54_fourdonor" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    vocab = tok.get_vocab()  # string -> id
    if len(vocab) != OWNER_VOCAB:
        raise RuntimeError(f"tokenizer vocab {len(vocab)} != {OWNER_VOCAB}")
    id2str = {i: s for s, i in vocab.items()}

    with open(EXPORTS / "quillan_vocab.json", encoding="utf-8") as fh:
        text2slot = json.load(fh)
    q2d = torch.load(
        EXPORTS / "quillan_token_maps_q2d.pt",
        map_location="cpu",
        weights_only=True,
    )

    slots = torch.full((OWNER_VOCAB,), -1, dtype=torch.int64)
    missing = []
    for i in range(N_REACHABLE):
        s = id2str[i]
        if s not in text2slot:
            missing.append(i)
            continue
        slots[i] = int(text2slot[s])
    if missing:
        raise RuntimeError(
            f"{len(missing)} reachable tokens lack union slots: {missing[:8]}"
        )
    if int(slots[:N_REACHABLE].min()) < 0 or int(slots[:N_REACHABLE].max()) > 237565:
        raise RuntimeError("union slot out of range")

    cov = {n: int((q2d[n][slots[:N_REACHABLE]] >= 0).sum().item()) for n in DONORS}
    log(
        f"index ready: reachable={N_REACHABLE} specials={list(SPECIAL_IDS)} coverage={cov}"
    )
    return {"slots": slots, "q2d": q2d, "coverage": cov}


# ---------------------------------------------------------------------------
# safe tensor row slicing (one donor at a time, mmap-free, modest RAM)
# ---------------------------------------------------------------------------


def read_rows(
    path: Path,
    key: str,
    idx: List[int],
    dtype=torch.float32,
    col_end: Optional[int] = None,
) -> torch.Tensor:
    """Gather scattered rows via SORTED CONTIGUOUS block reads.

    Random per-row safetensors indexing caused a native access violation on
    the 2048-dim donor; sorted block reads mirror the original builder and
    are stable and fast.
    """
    CHUNK = 4096
    rows = torch.tensor(idx, dtype=torch.long)
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        sl = fh.get_slice(key)
        shape = sl.get_shape()
        nrows = int(shape[0])
        ncols = int(col_end) if col_end is not None else int(shape[-1])
    valid = (rows >= 0) & (rows < nrows)
    safe_rows = torch.where(valid, rows, 0)
    order, inverse = torch.sort(safe_rows)
    out = torch.zeros((len(idx), ncols), dtype=dtype)
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        sl = fh.get_slice(key)
        for start in range(0, order.numel(), CHUNK):
            part = order[start : start + CHUNK]
            lo, hi = int(part[0]), int(part[-1]) + 1
            block = (
                torch.as_tensor(sl[lo:hi, 0:ncols], dtype=dtype)
                if col_end is not None
                else torch.as_tensor(sl[lo:hi], dtype=dtype)
            )
            local = part - lo
            out[inverse[start : start + part.numel()]] = block[local]
    return out


# ---------------------------------------------------------------------------
# corrected alignment artifact
# ---------------------------------------------------------------------------


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item())


def build_alignment(idx: Dict[str, object]) -> Dict[str, object]:
    """Rebuild alignment on the CORRECTED canonical-union-slot index space.

    One compiled-embedding read per donor (at the largest ONI width), then
    slice to each geometry. Donor rows are read once and reused. All row
    reads use sorted contiguous blocks so no scattered-slice crash recurs.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    slots: torch.Tensor = idx["slots"]  # type: ignore[assignment]
    q2d: Dict[str, torch.Tensor] = idx["q2d"]  # type: ignore[assignment]
    max_h = max(int(g["hidden"]) for g in GEOMETRY.values())
    compiled = EXPORTS / "quillan_embedding_compiled.safetensors"

    artifact: Dict[str, object] = {
        "note": (
            "CORRECTED alignment. target rows = compiled_embedding[slot_for_token_id]; "
            "donor ids = q2d[name][slot_for_token_id]. Old artifact used the wrong "
            "index space (compiled_embedding[:151648])."
        ),
        "vocab": OWNER_VOCAB,
        "reachable": N_REACHABLE,
        "special_token_ids": list(SPECIAL_IDS),
        "slot_for_token_id": slots.clone(),
        "fit_rows": FIT_N,
        "eval_rows": EVAL_N,
        "geometries": GEOMETRY,
        "expected_params": EXPECTED_PARAMS,
        "maps": {},
        "target_embedding_mini": None,
        "target_embedding_main": None,
    }

    log(f"reading compiled owner rows at max width {max_h}")
    owner_max = read_rows(
        compiled, EMB_KEY, slots[:N_REACHABLE].tolist(), col_end=max_h
    )
    for model_key, geom in GEOMETRY.items():
        H = int(geom["hidden"])
        full = torch.zeros((OWNER_VOCAB, H), dtype=torch.float32)
        full[:N_REACHABLE] = owner_max[:, :H]
        artifact[f"target_embedding_{model_key}"] = full
    del owner_max
    gc.collect()

    maps_by: Dict[str, Dict[str, object]] = {k: {} for k in GEOMETRY}
    for name in DONORS:
        log(f"=== donor {name}: aligned rows from corrected index space ===")
        did = q2d[name][slots[:N_REACHABLE]]
        valid_pos = (did >= 0).nonzero(as_tuple=True)[0]
        if valid_pos.numel() < FIT_N + EVAL_N:
            raise RuntimeError(f"{name}: not enough shared rows")
        # Mirror the original builder's deterministic evenly-spaced sampling,
        # but on the CORRECTED slot/index space. Eval rows are disjoint.
        fit_pos = valid_pos[torch.linspace(0, valid_pos.numel() - 1, FIT_N).long()]
        fit_set = set(fit_pos.tolist())
        remaining = torch.tensor(
            [p for p in valid_pos.tolist() if p not in fit_set], dtype=torch.long
        )
        eval_pos = remaining[torch.linspace(0, remaining.numel() - 1, EVAL_N).long()]

        fit_slots = slots[fit_pos].tolist()
        eval_slots = slots[eval_pos].tolist()
        fit_did = did[fit_pos].tolist()
        eval_did = did[eval_pos].tolist()

        Yfit_max = read_rows(compiled, EMB_KEY, fit_slots, col_end=max_h)
        Ye_max = read_rows(compiled, EMB_KEY, eval_slots, col_end=max_h)
        X = read_rows(DONOR_FILES[name], EMB_KEY, fit_did)
        Xe = read_rows(DONOR_FILES[name], EMB_KEY, eval_did)
        D = int(X.shape[1])

        for model_key, geom in GEOMETRY.items():
            H = int(geom["hidden"])
            Y = Yfit_max[:, :H].contiguous()
            Ye = Ye_max[:, :H].contiguous()
            # Same ridge-regularised closed-form solve as the original builder:
            # G = X^T X + 1e-2 * mean(|X|) * I ; P = G^-1 X^T Y
            Gx = X.t() @ X + 1e-2 * torch.eye(D) * X.abs().mean()
            P = torch.linalg.solve(Gx, X.t() @ Y)
            # Q: reverse map Y @ Q = X on the SAME 30000 aligned rows
            Gy = Y.t() @ Y + 1e-2 * torch.eye(H) * Y.abs().mean()
            Q = torch.linalg.solve(Gy, Y.t() @ X)
            rec = {
                "donor_dim": D,
                "P": P.detach().clone(),
                "Q": Q.detach().clone(),
                "shared_fit": len(fit_slots),
                "shared_eval": len(eval_slots),
                "cos_P_fit": _cos(X @ P, Y),
                "cos_P_eval": _cos(Xe @ P, Ye),
                "cos_Q_fit": _cos(Y @ Q, X),
                "cos_Q_eval": _cos(Ye @ Q, Xe),
            }
            maps_by[model_key][name] = rec
            log(
                f"  {model_key} H={H} {name}: dim={D} "
                f"Pfit={rec['cos_P_fit']:.4f} Peval={rec['cos_P_eval']:.4f} "
                f"Qfit={rec['cos_Q_fit']:.4f} Qeval={rec['cos_Q_eval']:.4f}"
            )
            del Y, Ye, Gx, Gy, P, Q
            gc.collect()
        del X, Xe, Yfit_max, Ye_max
        gc.collect()

    for key in GEOMETRY:
        artifact[f"maps_{key}"] = maps_by[key]
    artifact["maps"] = artifact["maps_mini"]  # backward-compatible alias
    torch.save(artifact, ALIGN_PATH)
    log(f"saved corrected alignment -> {ALIGN_PATH}")
    return artifact


# ---------------------------------------------------------------------------
# deterministic low-rank shape resize and role projection
# ---------------------------------------------------------------------------


def lowrank_resize(
    M: torch.Tensor, rows: int, cols: int, k: int = SVD_K
) -> torch.Tensor:
    """Deterministic rank-k SVD resize of a donor weight into ONI shape.

    The larger axis is deterministically pre-truncated to the first k entries
    before SVD so the operation is bounded and reproducible on CPU. Missing
    rows/columns are zero-padded. This is a geometric projection, not a
    semantic head-by-head transplant.
    """
    if M.shape == (rows, cols):
        return M.clone()
    M2 = M
    if M2.shape[1] > k:
        M2 = M2[:, :k]
    if M2.shape[0] > k:
        M2 = M2[:k, :]
    U, S, Vt = torch.linalg.svd(M2, full_matrices=False)
    r = min(U.shape[1], rows, cols)
    U, S, Vt = U[:, :r].contiguous(), S[:r], Vt[:r, :].contiguous()
    Up = torch.zeros((rows, r), dtype=M.dtype)
    Vp = torch.zeros((r, cols), dtype=M.dtype)
    Up[: min(rows, r), :] = U[: min(rows, r), :]
    Vp[:, : min(cols, r)] = Vt[:, : min(cols, r)].t()
    return (Up * S.unsqueeze(0)) @ Vp


def project_role(
    W: torch.Tensor,
    Q: torch.Tensor,
    P: torch.Tensor,
    donor_dim: int,
    H: int,
    F: int,
    role: str,
) -> torch.Tensor:
    """Project one donor nn.Linear weight (out_d, in_d) into ONI weight shape."""
    out_d, in_d = W.shape
    W32 = W.to(torch.float32)
    if in_d != donor_dim:
        W32 = lowrank_resize(W32, out_d, donor_dim)  # e.g. down_proj input = F_d
    A = Q @ W32.t()  # (H, out_d): ONI input -> donor input -> donor output
    if out_d == donor_dim:
        A = A @ P  # donor output -> ONI output when widths agree
    target = {
        "q": (H, H),
        "k": (H, H),
        "v": (H, H),
        "o": (H, H),
        "gate": (F, H),
        "up": (F, H),
        "down": (H, F),
    }[role]
    return lowrank_resize(A.t(), target[0], target[1])


def donor_layer_index(oni_i: int, n_oni: int, n_donor: int) -> int:
    if n_oni <= 1:
        return 0
    return int(round(oni_i * (n_donor - 1) / (n_oni - 1)))


def donor_layer_count(path: Path) -> int:
    import re

    with safe_open(str(path), framework="pt", device="cpu") as fh:
        nums = [
            int(m.group(1))
            for k in fh.keys()
            if (m := re.match(r"model\.layers\.(\d+)\.", k))
        ]
    return max(nums) + 1 if nums else 0


def merge_embedding(
    idx: Dict[str, object], align: Dict[str, object], H: int, model_key: str
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Blend donor knowledge into the corrected compiled owner embedding."""
    slots: torch.Tensor = idx["slots"]  # type: ignore[assignment]
    q2d: Dict[str, torch.Tensor] = idx["q2d"]  # type: ignore[assignment]
    owner = align[f"target_embedding_{model_key}"].clone()  # specials are zeros
    maps = align[f"maps_{model_key}"]

    pair_tensors: List[torch.Tensor] = []
    pair_counts: List[torch.Tensor] = []
    for pair in PAIRS:
        acc = torch.zeros_like(owner)
        cnt = torch.zeros((OWNER_VOCAB,), dtype=torch.float32)
        for name in pair:
            did = q2d[name][slots[:N_REACHABLE]]
            ok = (did >= 0).nonzero(as_tuple=True)[0]
            X = read_rows(DONOR_FILES[name], EMB_KEY, did[ok].tolist())
            proj = X @ maps[name]["P"]  # (n, H)
            acc[ok] += proj
            cnt[ok] += 1.0
            del X, proj
            gc.collect()
        nz = cnt > 0
        acc[nz] /= cnt[nz].unsqueeze(-1)
        pair_tensors.append(acc)
        pair_counts.append(cnt)

    total = pair_counts[0] + pair_counts[1]
    merged = torch.zeros_like(owner)
    know = total > 0
    merged[know] = (
        pair_tensors[0][know] * pair_counts[0][know].unsqueeze(-1)
        + pair_tensors[1][know] * pair_counts[1][know].unsqueeze(-1)
    ) / total[know].unsqueeze(-1)

    delta = merged - owner
    own_norm = owner[:N_REACHABLE].norm(dim=-1)
    cap = EMBED_CAP_RATIO * own_norm
    dnorm = delta[:N_REACHABLE].norm(dim=-1)
    scale = torch.ones_like(dnorm)
    too_big = dnorm > cap
    scale[too_big] = cap[too_big] / (dnorm[too_big] + 1e-8)
    delta[:N_REACHABLE] *= scale.unsqueeze(-1)

    out = owner.clone()
    out[:N_REACHABLE] += EMBED_ALPHA * delta[:N_REACHABLE]
    out[N_REACHABLE:] = 0.0  # sentinel: caller keeps model-init rows for specials

    coverage = {
        "no_donor": int((total == 0).sum().item()),
        "one_donor": int((total == 1).sum().item()),
        "two_donors": int((total == 2).sum().item()),
        "three_donors": int((total == 3).sum().item()),
        "four_donors": int((total == 4).sum().item()),
        "special_rows_untouched": len(SPECIAL_IDS),
    }
    log(f"embedding coverage {coverage}")
    return out, coverage


def merge_layers(
    idx: Dict[str, object], align: Dict[str, object], model_key: str
) -> Dict[int, Dict[str, torch.Tensor]]:
    """4 -> 2 -> 1 merge of q/k/v/o/gate/up/down for every ONI layer."""
    slots: torch.Tensor = idx["slots"]  # type: ignore[assignment]
    q2d: Dict[str, torch.Tensor] = idx["q2d"]  # type: ignore[assignment]
    geom = GEOMETRY[model_key]
    H, F, L = int(geom["hidden"]), int(geom["ffn"]), int(geom["layers"])
    maps = align[f"maps_{model_key}"]

    pair_results: List[Dict[int, Dict[str, torch.Tensor]]] = []
    for pair in PAIRS:
        log(f"  pair {pair} -> {model_key}")
        pair_out: Dict[int, Dict[str, torch.Tensor]] = {
            i: {r: torch.zeros(_target_shape(r, H, F)) for r in ROLES} for i in range(L)
        }
        pair_cnt: Dict[int, Dict[str, torch.Tensor]] = {
            i: {
                r: torch.zeros(_target_shape(r, H, F)[0], dtype=torch.float32)
                for r in ROLES
            }
            for i in range(L)
        }
        for name in pair:
            n_donor = donor_layer_count(DONOR_FILES[name])
            D = int(maps[name]["donor_dim"])
            Q, P = maps[name]["Q"], maps[name]["P"]
            log(f"    donor {name}: layers={n_donor} dim={D}")
            with safe_open(str(DONOR_FILES[name]), framework="pt", device="cpu") as fh:
                for i in range(L):
                    dl = donor_layer_index(i, L, n_donor)
                    prefix = f"model.layers.{dl}."
                    for role in ROLES:
                        key = prefix + ROLE_KEYS[role]
                        if key not in fh.keys():
                            raise RuntimeError(f"missing donor key {key}")
                        W = torch.as_tensor(fh.get_tensor(key), dtype=torch.float32)
                        proj = project_role(W, Q, P, D, H, F, role)
                        pair_out[i][role] += proj
                        pair_cnt[i][role] += 1.0
                        del W, proj
                gc.collect()
        for i in range(L):
            for role in ROLES:
                c = pair_cnt[i][role].unsqueeze(-1).clamp_min(1.0)
                pair_out[i][role] /= c
        pair_results.append(pair_out)

    final: Dict[int, Dict[str, torch.Tensor]] = {}
    for i in range(L):
        final[i] = {}
        for role in ROLES:
            final[i][role] = 0.5 * (pair_results[0][i][role] + pair_results[1][i][role])
    return final


def _target_shape(role: str, H: int, F: int) -> Tuple[int, int]:
    if role in ("gate", "up"):
        return (F, H)
    if role == "down":
        return (H, F)
    return (H, H)


# ---------------------------------------------------------------------------
# build + save a checkpoint
# ---------------------------------------------------------------------------


def tensor_digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()[:16]


def build_checkpoint(
    model_key: str, idx: Dict[str, object], align: Dict[str, object]
) -> Path:
    geom = GEOMETRY[model_key]
    H, F, L = int(geom["hidden"]), int(geom["ffn"]), int(geom["layers"])
    log(f"=== building {model_key}: 4->2->1 merge ===")

    embedding, emb_cov = merge_embedding(idx, align, H, model_key)
    layers = merge_layers(idx, align, model_key)
    merged_path = OUT / f"merged_tensors_{model_key}.pt"
    torch.save({"embedding": embedding, "layers": layers}, merged_path)
    del embedding, layers
    gc.collect()

    log(f"instantiating fresh {model_key} ONI model (current file)")
    t0 = time.time()
    model = QuillanRoninOni(owner_config(model_key))
    model.eval()
    nparams = sum(int(p.numel()) for p in model.parameters())
    expected = EXPECTED_PARAMS[model_key]
    log(f"model init {time.time() - t0:.1f}s params={nparams} expected={expected}")
    if nparams != expected:
        raise RuntimeError(
            f"{model_key} param count {nparams} != owner-locked {expected}"
        )

    merged = torch.load(merged_path, map_location="cpu", weights_only=True, mmap=True)
    sd = model.state_dict()
    writable = {"wte.weight", "lm_head.weight"}
    for i in range(L):
        writable.add(f"h.{i}.attn.c_attn.weight")
        writable.add(f"h.{i}.attn.c_proj.weight")
        writable.add(f"h.{i}.moe.c_fc.weight")
        writable.add(f"h.{i}.moe.c_proj.weight")

    before = {k: tensor_digest(v) for k, v in sd.items() if k not in writable}

    with torch.no_grad():
        wte = sd["wte.weight"]
        wte[:N_REACHABLE].copy_(merged["embedding"][:N_REACHABLE])
        # specials untouched: keep fresh model's own rows, never donor/noise
        for i in range(L):
            src = merged["layers"][i]
            c_attn = sd[f"h.{i}.attn.c_attn.weight"]
            c_attn[0:H].copy_(src["q"])
            c_attn[H : 2 * H].copy_(src["k"])
            c_attn[2 * H : 3 * H].copy_(src["v"])
            sd[f"h.{i}.attn.c_proj.weight"].copy_(src["o"])
            c_fc = sd[f"h.{i}.moe.c_fc.weight"]
            c_fc[0:F].copy_(src["gate"])
            c_fc[F : 2 * F].copy_(src["up"])
            sd[f"h.{i}.moe.c_proj.weight"].copy_(src["down"])

    after = {
        k: tensor_digest(v) for k, v in model.state_dict().items() if k not in writable
    }
    changed = [k for k in before if before[k] != after.get(k)]
    if changed:
        raise RuntimeError(f"sovereign tensors changed: {changed[:10]}")
    log(f"sovereign tensors verified unchanged: {len(before)} tensors")

    # zero-init identity evidence for the newly added tiers
    gate_info = {
        k: {"nonzero": int((v != 0).sum().item()), "maxabs": float(v.abs().max())}
        for k, v in model.state_dict().items()
        if k.startswith("tristack.gate_") or k == "throne_swarm.coupling"
    }

    metadata = {
        "build": "mapped_4_to_2_to_1_closed_form_knowledge_projection",
        "NOT_a_raw_state_dict_transplant": True,
        "model": model_key,
        "param_count": nparams,
        "expected_param_count": expected,
        "owner_config": {
            "vocab_size": int(model.cfg.vocab_size),
            "max_seq_len": int(model.cfg.max_seq_len),
            "hidden_dim": int(model.cfg.hidden_dim),
            "ffn_dim": int(model.cfg.ffn_dim),
            "n_layer": int(model.cfg.n_layer),
            "n_head": int(model.cfg.n_head),
            "num_experts": int(model.cfg.num_experts),
        },
        "recommended_inference_flags": {
            "compute_aux": False,
            "deliberation": False,
        },
        "recommended_inference_note": (
            "Donor geometries do not match ONI; clean path is the stable path. "
            "Verified falsification: clean-aux max logit delta 1.152; "
            "top-1 flips 750 -> 739 -> 602."
        ),
        "pairs": [list(p) for p in PAIRS],
        "embedding": {
            "alpha": EMBED_ALPHA,
            "cap_ratio": EMBED_CAP_RATIO,
            "coverage": emb_cov,
            "special_rows": list(SPECIAL_IDS),
            "special_rows_policy": "model-init preserved; no donor/noise write",
        },
        "role_projection": {
            "method": "Y@P donor->ONI, Y@Q reverse least squares, rank-k SVD shape resize",
            "svd_rank_cap": SVD_K,
            "qkv_note": (
                "Donor head layouts differ from ONI (18/24 heads x 64); SVD resize "
                "is geometric projection into packed ONI q/k/v, not head-aligned transplant"
            ),
            "layer_map": "round(i*(n_donor-1)/(n_oni-1)) clamped",
        },
        "alignment": {
            "path": str(ALIGN_PATH),
            "shared_fit": FIT_N,
            "shared_eval": EVAL_N,
            "cos": {
                n: {
                    "P_fit": align[f"maps_{model_key}"][n]["cos_P_fit"],
                    "P_eval": align[f"maps_{model_key}"][n]["cos_P_eval"],
                    "Q_fit": align[f"maps_{model_key}"][n]["cos_Q_fit"],
                    "Q_eval": align[f"maps_{model_key}"][n]["cos_Q_eval"],
                }
                for n in DONORS
            },
        },
        "sovereign_tensors_verified_unchanged": len(before),
        "zero_init_tier_gates": gate_info,
        "state_dict_sha256_of_sovereign_keys": hashlib.sha256(
            "".join(sorted(before.values())).encode()
        ).hexdigest()[:24],
        "limitations": [
            "Closed-form knowledge projection only; not raw weight identity.",
            "No claim of coherence, quality, or benchmark improvement.",
            "No backprop; WSP tuning deferred by owner spec.",
            "5 sovereign specials have no donor/union slot and are left at model init.",
        ],
    }

    ckpt_path = OUT / f"quillan_{model_key}_oni_transfer.pt"
    log(f"saving checkpoint -> {ckpt_path}")
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, ckpt_path)
    log(f"saved {ckpt_path.stat().st_size / 1e9:.2f} GB")
    del model, sd, merged
    gc.collect()
    return ckpt_path


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def verify_one(model_key: str) -> None:
    ckpt_path = OUT / f"quillan_{model_key}_oni_transfer.pt"
    log(f"=== verify {model_key}: {ckpt_path.name} ===")
    if not ckpt_path.exists():
        raise RuntimeError(f"missing checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    log(f"metadata: {json.dumps(ckpt['metadata']['recommended_inference_flags'])}")
    log(
        f"params: {ckpt['metadata']['param_count']} "
        f"(expected {ckpt['metadata']['expected_param_count']})"
    )
    log(f"metadata cos: {json.dumps(ckpt['metadata']['alignment']['cos'])}")
    log(
        f"sovereign unchanged count: {ckpt['metadata']['sovereign_tensors_verified_unchanged']}"
    )

    t0 = time.time()
    model = QuillanRoninOni(owner_config(model_key))
    model.eval()
    nparams = sum(int(p.numel()) for p in model.parameters())
    log(f"fresh model init {time.time() - t0:.1f}s params={nparams}")
    if nparams != int(ckpt["metadata"]["param_count"]):
        raise RuntimeError("fresh model param count differs from checkpoint metadata")
    res = model.load_state_dict(ckpt["state_dict"], strict=False)
    log(
        f"load_state_dict(strict=False): missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
    )
    if res.missing_keys:
        log(f"  missing sample: {res.missing_keys[:8]}")
    if res.unexpected_keys:
        log(f"  unexpected sample: {res.unexpected_keys[:8]}")
    del ckpt
    gc.collect()

    kwargs = dict(deliberation=False, compute_aux=False, return_aux=False)
    with torch.no_grad():
        # finite forward on random tokens (1,4)
        ids = torch.randint(1, OWNER_VOCAB - 5, (1, 4), dtype=torch.long)
        logits = model(ids, **kwargs)
        finite = bool(torch.isfinite(logits).all())
        log(
            f"forward (1,4): shape={tuple(logits.shape)} finite={finite} maxabs={float(logits.abs().max()):.4f}"
        )
        if not finite:
            raise RuntimeError("non-finite logits")

        # zero-init identity: tiers on vs off on the same clean input
        on = model(ids, **kwargs).clone()
        ts, tw = model.tristack, model.throne_swarm
        model.tristack, model.throne_swarm = None, None
        off = model(ids, **kwargs)
        delta = float((on - off).abs().max())
        model.tristack, model.throne_swarm = ts, tw
        log(
            f"tier on/off max logit delta={delta:.8f} gates={json.dumps(ckpt_meta_gates(model_key))}"
        )

        # deterministic greedy generation, 8 tokens
        gen = [1, 50, 120, 304]
        for _ in range(8):
            inp = torch.tensor([gen], dtype=torch.long)
            nxt = int(torch.argmax(model(inp, **kwargs)[0, -1], dim=-1).item())
            gen.append(nxt)
        log(f"greedy tokens (8): {gen[4:]}")

    del model
    gc.collect()


def ckpt_meta_gates(model_key: str) -> dict:
    p = OUT / f"quillan_{model_key}_oni_transfer.pt"
    # read only metadata without loading the whole state dict into RAM
    obj = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
    g = obj["metadata"]["zero_init_tier_gates"]
    del obj
    gc.collect()
    return g


def verify_all() -> None:
    for key in ("mini", "main"):
        verify_one(key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("align", "build", "verify", "all"))
    ap.add_argument("model", nargs="?", choices=("mini", "main"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))

    if args.cmd == "verify":
        verify_all()
        return

    idx = load_index()
    if args.cmd == "align":
        build_alignment(idx)
        return

    if args.cmd == "build":
        if args.model is None:
            raise SystemExit("build requires mini|main")
        align = torch.load(ALIGN_PATH, map_location="cpu", weights_only=True)
        build_checkpoint(args.model, idx, align)
        return

    align = build_alignment(idx)
    for key in ("mini", "main"):
        build_checkpoint(key, idx, align)
    verify_all()


if __name__ == "__main__":
    main()
