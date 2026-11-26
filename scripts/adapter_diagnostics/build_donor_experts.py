#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
#1171 / #1169 sec.3 -- FOUR-DONOR EXTRACTION + ALIGNMENT PIPELINE (BUILD, CLOSED-FORM)
=====================================================================================

Concrete deliverable for C26-TECHNE (board #1171, spec #1169 section 3).

WHAT THIS DOES
--------------
  1. Streams each AVAILABLE donor one at a time (peak RAM < 1.5 GB):
       qwen_0.5b   (d=896,  ffn=4864, L=24, BF16)        -> experts  0..7
       llama_1b    (d=2048, ffn=8192, L=16, BF16)        -> experts  8..15
       dragon/falcon3:1b (ollama; GGUF-only => skipped)  -> experts 16..23
       bitnet_2b4t (d=2560, ffn=6912, L=30, PACKED U8)   -> experts 24..31
       experts 32..33 = closed-form consensus (mean of aligned maps, re-gamma'd)
  2. For each donor, 8 EVENLY SPACED layers (spec #1169 slot map).
  3. Orthogonal Procrustes alignment into the 896 host frame via TOKEN-STRING
     shared vocab. REUSES the existing primitives (NOT reimplemented):
       * v23_common.compute_shared_vocab_procrustes   (SVD isometric alignment)
       * merge_donor_embeddings.canonicalize_token    ('Ġ'/'▁' byte-level canon)
       * quillan_weight_adapter.to_ternary_pair(..., group_dim=-1) (absmean)
  4. Maps SwiGLU MLP weights (HF [ffn,d] / [d,ffn]) into host MoE orientation:
       w1_host    = (W_up   @ P).T    -> [896, 4864]  (ffn centre-crop/pad)
       wgate_host = (W_gate @ P).T    -> [896, 4864]
       w2_host    = W_down[:,crop].T @ P -> [4864, 896]
     Bitnet U8 is unpacked FIRST (4 ternary vals/byte, code 0/1/2 -> -1/0/+1,
     code 3 -> 0 defensively) and scaled by its BF16 per-tensor weight_scale.
  5. TERNARIZES LAST: per-output-channel absmean gamma computed ON THE MAPPED
     FLOAT WEIGHT (dim=-2 for [E,in,out]), NOT inherited gammas.mean().
     Emits F16 = ternary_code * recomputed_gamma at the EXACT checkpoint shapes.
  6. Writes a NEW file exports/donor_experts_aligned.safetensors (never overwrites
     canonical). Atomically replaced. Emits a JSON verification report alongside.

INVARIANTS: no backprop, no optimizer, deterministic, no canonical/gate/donor edits,
one donor resident at a time, F16 output drop-in compatible with the 530-key graph
(socket-v2 stores moe.w1/wgate [34,896,4864] and moe.w2 [34,4864,896] as F16).

Run:  python -u scripts\\adapter_diagnostics\\build_donor_experts.py
Author: C26-TECHNE (Constraint Loop), v5.4.0-oni -- board #1171
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(r"C:\02_QUILLAN")
SCRIPTS = ROOT / "scripts"
DIAG = ROOT / "scripts" / "adapter_diagnostics"
MODEL_DIR = ROOT / "03 - Training & Model"
EXPORTS = ROOT / "exports"

for _p in (str(DIAG), str(SCRIPTS), str(MODEL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- REUSED primitives (read first, never reimplemented) --------------------
# NOTE: import quillan_weight_adapter FIRST. v23_common injects
# "03 - Training & Model\scripts" at sys.path[0], whose quillan_weight_adapter is
# an OLD copy lacking to_ternary_pair. Binding the correct module first pins it.
from quillan_weight_adapter import to_ternary_pair  # noqa: E402
import v23_common as v23  # noqa: E402
from merge_donor_embeddings import canonicalize_token  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("C26.donor_experts")

HOST_DIM = 896
HOST_FFN = 4864
N_EXPERTS = 34
LAYERS_PER_DONOR = 8
MAX_PAIRS = 20000  # >> the primitive's 512-pair minimum; caps Procrustes RAM
EPS = 1e-5
GAMMA_DIM = -2  # per-OUTPUT-channel for host [in,out] MoE layout (spec #1165)

OUT_FILE = EXPORTS / "donor_experts_aligned.safetensors"
REPORT_FILE = EXPORTS / "donor_experts_aligned.report.json"


# --------------------------------------------------------------------------- #
# Donor specifications (fixed order => determinism)                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DonorSpec:
    name: str
    ckpt: Path
    tokenizer: Path
    hidden: int
    ffn: int
    expert_start: int
    expert_end: int
    kind: str  # "bf16" | "packed_u8"
    optional: bool = False
    note: str = ""


DONORS: Tuple[DonorSpec, ...] = (
    DonorSpec(
        "qwen",
        Path(r"E:\source_models\qwen_0.5b\model.safetensors"),
        Path(r"E:\source_models\qwen_0.5b\tokenizer.json"),
        896,
        4864,
        0,
        8,
        "bf16",
        note="HOST FRAME (identity P)",
    ),
    DonorSpec(
        "llama",
        Path(r"E:\source_models\llama_1b\model.safetensors"),
        Path(r"E:\source_models\llama_1b\tokenizer.json"),
        2048,
        8192,
        8,
        16,
        "bf16",
    ),
    DonorSpec(
        "dragon",
        Path(
            r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model_bf16.safetensors"
        ),
        Path(r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\tokenizer.json"),
        2048,
        8192,
        16,
        24,
        "bf16",
        optional=True,
        note="Falcon3-1B official BF16 from HF (genuinely dequantized, absm~0.04) + tokenizer",
    ),
    DonorSpec(
        "bitnet",
        Path(r"E:\source_models\bitnet_2b_4t\model.safetensors"),
        Path(r"E:\source_models\bitnet_2b_4t\tokenizer.json"),
        2560,
        6912,
        24,
        32,
        "packed_u8",
    ),
)


# --------------------------------------------------------------------------- #
# Memory-efficient embedding row gather (chunked safetensors reads)            #
# --------------------------------------------------------------------------- #
def read_embed_rows(path: Path, key: str, indices: Sequence[int]) -> torch.Tensor:
    """Gather only the requested embedding rows as float32, in index order.

    Avoids materialising the full [V,d] table (up to 525 MB BF16 for llama).
    """
    idx = torch.as_tensor(list(indices), dtype=torch.long)
    order = torch.argsort(idx)
    sorted_idx = idx[order]
    with safe_open(str(path), framework="pt", device="cpu") as f:
        sl = f.get_slice(key)
        vocab, dim = int(sl.get_shape()[0]), int(sl.get_shape()[1])
        out = torch.empty((len(idx), dim), dtype=torch.float32)
        chunk = 4096
        for start in range(0, vocab, chunk):
            end = min(start + chunk, vocab)
            m = (sorted_idx >= start) & (sorted_idx < end)
            if not bool(m.any()):
                continue
            block = sl[start:end].float()  # [chunk, dim] float32
            sel = sorted_idx[m] - start
            out[order[m]] = block[sel]
            del block
    return out


# --------------------------------------------------------------------------- #
# Token-string maps (REUSE canonicalize_token from merge_donor_embeddings)     #
# --------------------------------------------------------------------------- #
def load_canon_token_map(tokenizer_path: Path) -> Dict[str, int]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    mapping: Dict[str, int] = {}
    for i in range(tok.get_vocab_size()):
        raw = tok.id_to_token(i)
        if raw is None:
            continue
        canon = canonicalize_token(raw)
        if canon and canon not in mapping:  # first id wins (deterministic)
            mapping[canon] = i
    return mapping


def build_shared_pairs(
    host_map: Mapping[str, int],
    donor_map: Mapping[str, int],
    max_pairs: int = MAX_PAIRS,
) -> Tuple[List[str], List[int], List[int]]:
    """Return (canonical strings, host ids, donor ids) for shared token strings."""
    shared = [s for s in host_map if s in donor_map]
    if len(shared) > max_pairs:
        step = len(shared) / float(max_pairs)
        shared = [shared[int(i * step)] for i in range(max_pairs)]
    h_ids = [int(host_map[s]) for s in shared]
    d_ids = [int(donor_map[s]) for s in shared]
    return shared, h_ids, d_ids


def mean_pair_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-row cosine between two equally-shaped float tensors."""
    an = a / a.norm(dim=-1, keepdim=True).clamp_min(EPS)
    bn = b / b.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return float((an * bn).sum(dim=-1).mean().item())


def compute_alignment(
    spec: DonorSpec,
    host_rows: torch.Tensor,
    shared: Sequence[str],
    d_ids: Sequence[int],
    donor_map: Mapping[str, int],
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Procrustes P (d_donor, 896) + finite alignment cosine diagnostics.

    REUSES v23_common.compute_shared_vocab_procrustes on the token-string-aligned
    row subsets. For qwen (already host frame) P is the exact identity.
    """
    if spec.name == "qwen":
        P = torch.eye(HOST_DIM, dtype=torch.float32)
        diag: Dict[str, object] = {
            "n_shared_tokens": len(shared),
            "P_shape": [HOST_DIM, HOST_DIM],
            "identity_host_frame": True,
            "align_cos_mean": 1.0,
            "align_cos_finite": True,
        }
        return P, diag

    donor_rows = read_embed_rows(spec.ckpt, "model.embed_tokens.weight", d_ids)
    # Vocab maps are over the SAME canonical strings -> every pair is an exact
    # string match inside the reused primitive (index i == row i in each subset).
    hv = {s: i for i, s in enumerate(shared)}
    dv = {s: i for i, s in enumerate(shared)}
    P, n_pairs = v23.compute_shared_vocab_procrustes(
        host_vocab=hv,
        donor_vocab=dv,
        host_emb=host_rows,
        donor_emb=donor_rows,
        max_pairs=MAX_PAIRS,
    )
    P = P.float().contiguous()
    if tuple(P.shape) != (spec.hidden, HOST_DIM):
        raise RuntimeError(
            f"{spec.name}: P shape {tuple(P.shape)} != {(spec.hidden, HOST_DIM)}"
        )

    cos = mean_pair_cosine(donor_rows @ P, host_rows)
    orth = float(torch.norm(P.T @ P - torch.eye(HOST_DIM)).item())
    diag = {
        "n_shared_tokens": int(n_pairs),
        "P_shape": [int(x) for x in P.shape],
        "orthogonality_error": orth,
        # Expected LOSSY (~0.33-0.42 per C8 #1169). NOT a pass threshold.
        "align_cos_mean": cos,
        "align_cos_finite": bool(torch.isfinite(torch.tensor(cos)).item()),
        "identity_host_frame": False,
    }
    del donor_rows
    gc.collect()
    return P, diag


# --------------------------------------------------------------------------- #
# Bitnet U8 unpack (convention verified on-disk: 0/1/2 -> -1/0/+1, 3 absent)   #
# --------------------------------------------------------------------------- #
def unpack_u8_ternary(
    packed: torch.Tensor, logical_shape: Sequence[int]
) -> torch.Tensor:
    """Unpack 4 ternary codes/byte along dim 0, matching scripts/bitnet_linear.py."""
    flat = packed.to(torch.uint8).reshape(-1)
    v0 = (flat >> 0) & 3
    v1 = (flat >> 2) & 3
    v2 = (flat >> 4) & 3
    v3 = (flat >> 6) & 3
    codes = torch.stack([v0, v1, v2, v3], dim=-1).reshape(-1)
    vals = torch.where(codes == 0, -1.0, torch.where(codes == 2, 1.0, 0.0))
    total = 1
    for s in logical_shape:
        total *= int(s)
    return vals[:total].reshape(tuple(logical_shape)).float()


# --------------------------------------------------------------------------- #
# MLP extraction + host-frame mapping                                          #
# --------------------------------------------------------------------------- #
def evenly_spaced_layers(n_layers: int, k: int = LAYERS_PER_DONOR) -> List[int]:
    if n_layers <= k:
        return list(range(n_layers))
    return [int(round(i * (n_layers - 1) / (k - 1))) for i in range(k)]


def _ffn_span(src_ffn: int) -> Tuple[int, int]:
    """Centre-crop/pad source ffn to HOST_FFN; returns (src_start, pad_count)."""
    crop = (src_ffn - HOST_FFN) // 2
    if crop >= 0:
        return crop, 0
    return 0, HOST_FFN - src_ffn  # pad on the right (source smaller)


def load_layer_mlp(
    f: "safe_open", spec: DonorSpec, layer: int, P: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Return (w1_host[896,4864], wgate_host[896,4864], w2_host[4864,896], meta).

    w1/wgate/w2 are FLOAT effective weights in host frame, BEFORE ternarization.
    """
    pre = f"model.layers.{layer}.mlp."
    src = spec.hidden
    src_ffn = spec.ffn

    def load_proj(name: str, logical: Sequence[int]) -> Tuple[torch.Tensor, float]:
        key = pre + name + ".weight"
        if spec.kind == "packed_u8":
            raw = f.get_tensor(key)
            scale = float(f.get_tensor(key + "_scale").float().reshape(-1)[0].item())
            return unpack_u8_ternary(raw, logical) * scale, scale
        w = f.get_tensor(key).float()  # HF: [out, in]
        return w, float(w.abs().mean().item())  # inherited (global) gamma proxy

    gate, g_scale = load_proj("gate_proj", (src_ffn, src))  # [ffn, d]
    up, u_scale = load_proj("up_proj", (src_ffn, src))  # [ffn, d]
    down, d_scale = load_proj("down_proj", (src, src_ffn))  # [d, ffn]

    crop, pad = _ffn_span(src_ffn)
    if pad:
        up = torch.nn.functional.pad(up, (0, 0, 0, pad))
        gate = torch.nn.functional.pad(gate, (0, 0, 0, pad))
        down = torch.nn.functional.pad(down, (0, pad, 0, 0))
        end = src_ffn + pad
    else:
        end = crop + HOST_FFN

    # Input (hidden) side: P projects donor hidden -> host 896.
    #  (W @ P) is [HOST_FFN, 896]; .T -> [896, HOST_FFN] for right-multiply.
    w1 = (up[crop:end] @ P).T.contiguous()
    wgate = (gate[crop:end] @ P).T.contiguous()
    # Output side: donor ffn -> host ffn is identity on the cropped span (no
    # token-string correspondence exists for ffn channels; centre-crop chosen).
    #  down[crop:end] is [d, HOST_FFN]; .T @ P -> [HOST_FFN, 896].
    w2 = (down[:, crop:end].T @ P).contiguous()

    del gate, up, down
    meta = {
        "layer": float(layer),
        "inherited_gamma_gate": g_scale,
        "inherited_gamma_up": u_scale,
        "inherited_gamma_down": d_scale,
        "inherited_gamma_gammas_mean": (g_scale + u_scale + d_scale) / 3.0,
        "ffn_crop_start": float(crop),
        "ffn_pad": float(pad),
    }
    return w1, wgate, w2, meta


# --------------------------------------------------------------------------- #
# Ternarize LAST with RECOMPUTED per-channel gamma                             #
# --------------------------------------------------------------------------- #
def ternarize_recomputed(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """w: [..., in, out] float -> (F16 ternary*gamma, gamma[...,1,out]).

    Reuses to_ternary_pair(group_dim=-1) on the transposed view so that the
    absmean is taken over the INPUT dim (dim=-2 of host layout), i.e. one
    RECOMPUTED gamma per OUTPUT channel. No inherited scale is passed in.

    This IS the canonical docs convention: gamma = mean|W| of the (already
    DEQUANTIZED) float weight. Per C28 #1251 do NOT scalar-rescale here —
    Falcon's earlier gamma~87 was a dequantization bug in the extract, not a
    normalization problem. Donors must arrive at true weight magnitude.
    """
    wt = w.transpose(-1, -2)  # [..., out, in]
    q_t, g_t = to_ternary_pair(wt, group_dim=-1)  # per-channel absmean, recomputed
    q = q_t.transpose(-1, -2).contiguous()  # [..., in, out], codes -1/0/1
    gamma = g_t.transpose(-1, -2).contiguous()  # [..., 1, out]
    return (q * gamma).to(torch.float16), gamma.float()


def ram_mb(peak: bool = False) -> float:
    """Current (or peak) working set of this process via Win32 GetProcessMemoryInfo."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PMC),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        ok = psapi.GetProcessMemoryInfo(
            k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
        )
        if not ok:
            return -1.0
        v = pmc.PeakWorkingSetSize if peak else pmc.WorkingSetSize
        return round(v / (1024**2), 1)
    except Exception as exc:  # pragma: no cover - probe is best-effort
        log.warning("ram probe unavailable: %s", exc)
        return -1.0


def trim_ws() -> None:
    """Drop file-backed mmap pages from the working set (safetensors reads)."""
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
        psapi.EmptyWorkingSet.restype = wintypes.BOOL
        psapi.EmptyWorkingSet(k32.GetCurrentProcess())
    except Exception as exc:  # pragma: no cover
        log.warning("working-set trim unavailable: %s", exc)


def peak_ram_mb() -> float:
    """Exact peak working set of this process via Win32 GetProcessMemoryInfo."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PMC),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        ok = psapi.GetProcessMemoryInfo(
            k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
        )
        return round(pmc.PeakWorkingSetSize / (1024**2), 1) if ok else -1.0
    except Exception as exc:  # pragma: no cover - probe is best-effort
        log.warning("peak RAM probe unavailable: %s", exc)
        return -1.0


def finite_check(t: torch.Tensor, label: str, chunk: int = 2) -> Dict[str, object]:
    """Scan for NaN/Inf in expert-sized chunks (never a full-size bool mask)."""
    n_nan = n_inf = 0
    for s in range(0, int(t.shape[0]), chunk):
        blk = t[s : s + chunk]
        n_nan += int(torch.isnan(blk).sum().item())
        n_inf += int(torch.isinf(blk).sum().item())
    if n_nan or n_inf:
        raise RuntimeError(f"{label}: NaN={n_nan} Inf={n_inf}")
    return {"nan": n_nan, "inf": n_inf, "finite": True}


# --------------------------------------------------------------------------- #
# Main build                                                                   #
# --------------------------------------------------------------------------- #
def build(out_path: Path = OUT_FILE) -> Dict[str, object]:
    t0 = time.time()
    report: Dict[str, object] = {
        "task": "#1171 / #1169 sec.3 donor expert extraction",
        "host_dim": HOST_DIM,
        "host_ffn": HOST_FFN,
        "n_experts": N_EXPERTS,
        "donors": {},
        "skipped": [],
        "verification": {},
    }

    # -- resolve donors ------------------------------------------------------
    available: List[DonorSpec] = []
    for spec in DONORS:
        ok = spec.ckpt.is_file() and spec.tokenizer.is_file()
        if ok:
            available.append(spec)
            log.info("DONOR READY %-7s %s", spec.name, spec.ckpt)
        else:
            reason = (
                "safetensors missing"
                if not spec.ckpt.is_file()
                else "tokenizer missing"
            )
            extra = ""
            if spec.name == "dragon":
                blob = Path(r"C:\Users\Admin\.ollama\models\blobs")
                if blob.is_dir() and any(blob.glob("sha256-89056211*")):
                    extra = (
                        " (GGUF blob present; needs safetensors+tokenizer re-export)"
                    )
            report["skipped"].append(
                {
                    "name": spec.name,
                    "reason": reason + extra,
                    "optional": spec.optional,
                }
            )
            log.warning("DONOR SKIP %-7s %s%s", spec.name, reason, extra)

    if not any(s.name == "qwen" for s in available):
        raise RuntimeError("qwen host donor is REQUIRED to define the 896 frame")

    # -- token maps + Procrustes (no output allocated yet; low RAM) ----------
    host_map = load_canon_token_map(
        next(s.tokenizer for s in available if s.name == "qwen")
    )
    host_ids: Dict[str, List[int]] = {}
    align_meta: Dict[str, Dict[str, object]] = {}
    P_by_donor: Dict[str, torch.Tensor] = {}

    # qwen host rows are shared by every donor's Procrustes.
    qwen = next(s for s in available if s.name == "qwen")
    for spec in available:
        dmap = load_canon_token_map(spec.tokenizer)
        shared, h_ids, d_ids = build_shared_pairs(host_map, dmap)
        if len(shared) < 512:
            raise RuntimeError(f"{spec.name}: only {len(shared)} shared token strings")
        host_ids[spec.name] = h_ids
        if spec.name == "qwen":
            host_rows = read_embed_rows(qwen.ckpt, "model.embed_tokens.weight", h_ids)
        P, diag = compute_alignment(spec, host_rows, shared, d_ids, dmap)
        P_by_donor[spec.name] = P
        align_meta[spec.name] = diag
        log.info(
            "ALIGN %-7s n_shared=%s P=%s orth_err=%s align_cos=%.6f (finite=%s)",
            spec.name,
            diag["n_shared_tokens"],
            diag["P_shape"],
            diag.get("orthogonality_error", 0.0),
            diag["align_cos_mean"],
            diag["align_cos_finite"],
        )
        del dmap
        gc.collect()
        trim_ws()  # evict donor-embedding mmap pages before the next donor
    del host_rows
    gc.collect()
    log.info("RAM after Procrustes: cur=%.1fMB peak=%.1fMB", ram_mb(), ram_mb(True))

    # -- preallocate exact-shape F16 output (drop-in to 530-key graph) -------
    out: Dict[str, torch.Tensor] = {
        "moe.w1": torch.zeros((N_EXPERTS, HOST_DIM, HOST_FFN), dtype=torch.float16),
        "moe.wgate": torch.zeros((N_EXPERTS, HOST_DIM, HOST_FFN), dtype=torch.float16),
        "moe.w2": torch.zeros((N_EXPERTS, HOST_FFN, HOST_DIM), dtype=torch.float16),
    }
    log.info("Preallocated output: %s", {k: list(v.shape) for k, v in out.items()})
    log.info("RAM after prealloc: cur=%.1fMB peak=%.1fMB", ram_mb(), ram_mb(True))

    # consensus accumulators (float32, 2 slots only -> ~100 MB)
    cons = {
        "first": {k: torch.zeros_like(out[k][0], dtype=torch.float32) for k in out},
        "last": {k: torch.zeros_like(out[k][0], dtype=torch.float32) for k in out},
        "n": 0,
    }

    gamma_samples: List[Dict[str, object]] = []
    w_cos_samples: List[Dict[str, object]] = []

    # -- stream donors, one at a time ---------------------------------------
    for spec in available:
        layers = evenly_spaced_layers(_count_layers(spec.ckpt))
        n_layers = len(layers)
        log.info("LAYERS %-7s n=%d selected=%s", spec.name, n_layers, layers)
        P = P_by_donor[spec.name]
        with safe_open(str(spec.ckpt), framework="pt", device="cpu") as f:
            for j, layer in enumerate(layers):
                e = spec.expert_start + j
                w1, wg, w2, meta = load_layer_mlp(f, spec, layer, P)

                q1, g1 = ternarize_recomputed(w1)
                qg, gg = ternarize_recomputed(wg)
                q2, g2 = ternarize_recomputed(w2)
                out["moe.w1"][e] = q1
                out["moe.wgate"][e] = qg
                out["moe.w2"][e] = q2

                # direction-preservation cosine (finite, NOT an alignment claim)
                w1_dir = mean_pair_cosine(
                    w1.reshape(-1, w1.shape[-1])[:64],
                    out["moe.w1"][e].float().reshape(-1, w1.shape[-1])[:64],
                )
                w_cos_samples.append(
                    {
                        "donor": spec.name,
                        "expert": e,
                        "layer": layer,
                        "w1_vs_ternary_cos": w1_dir,
                    }
                )

                if j == 0:  # one before/after gamma sample per donor
                    gamma_samples.append(
                        {
                            "donor": spec.name,
                            "expert": e,
                            "layer": layer,
                            "inherited_gamma_gammas_mean": meta[
                                "inherited_gamma_gammas_mean"
                            ],
                            "inherited_gamma_gate": meta["inherited_gamma_gate"],
                            "inherited_gamma_up": meta["inherited_gamma_up"],
                            "inherited_gamma_down": meta["inherited_gamma_down"],
                            "recomputed_gamma_w1_mean": float(g1.mean().item()),
                            "recomputed_gamma_w1_min": float(g1.min().item()),
                            "recomputed_gamma_w1_max": float(g1.max().item()),
                            "recomputed_gamma_shape": list(g1.shape),
                        }
                    )

                # consensus accumulation (mean of aligned float maps)
                if j == 0:
                    for k, w in (("moe.w1", w1), ("moe.wgate", wg), ("moe.w2", w2)):
                        cons["first"][k] += w
                if j == len(layers) - 1:
                    for k, w in (("moe.w1", w1), ("moe.wgate", wg), ("moe.w2", w2)):
                        cons["last"][k] += w
                del w1, wg, w2, q1, qg, q2
                gc.collect()
                trim_ws()  # evict this layer's MLP mmap pages immediately
        cons["n"] += 1
        log.info(
            "BUILT %-7s experts %d..%d | RAM cur=%.1fMB peak=%.1fMB",
            spec.name,
            spec.expert_start,
            spec.expert_end - 1,
            ram_mb(),
            ram_mb(True),
        )
        trim_ws()  # evict this donor's MLP mmap pages before the next donor

    # -- consensus experts 32-33 --------------------------------------------
    if cons["n"] > 0:
        for slot, key in ((N_EXPERTS - 2, "first"), (N_EXPERTS - 1, "last")):
            for k in out:
                avg = cons[key][k] / float(cons["n"])
                out[k][slot], _ = ternarize_recomputed(avg)
        log.info(
            "CONSENSUS experts %d-%d from %d donors",
            N_EXPERTS - 2,
            N_EXPERTS - 1,
            cons["n"],
        )
    else:
        log.warning("No donors -> consensus experts left at zero (residual no-op)")

    # -- verification --------------------------------------------------------
    ver: Dict[str, object] = {}
    shape_ok = all(
        tuple(out[k].shape) == shp
        for k, shp in {
            "moe.w1": (N_EXPERTS, HOST_DIM, HOST_FFN),
            "moe.wgate": (N_EXPERTS, HOST_DIM, HOST_FFN),
            "moe.w2": (N_EXPERTS, HOST_FFN, HOST_DIM),
        }.items()
    )
    ver["shapes"] = {k: list(v.shape) for k, v in out.items()}
    ver["shapes_match_exact"] = bool(shape_ok)
    ver["dtype"] = {k: str(v.dtype) for k, v in out.items()}
    ver["finite"] = {k: finite_check(v, k) for k, v in out.items()}
    ver["align_cos"] = {n: d.get("align_cos_mean") for n, d in align_meta.items()}
    ver["align_cos_all_finite"] = all(
        bool(d.get("align_cos_finite")) for d in align_meta.values()
    )
    ver["ternary_direction_cos_samples"] = w_cos_samples[:6]
    ver["gamma_before_after_samples"] = gamma_samples
    ver["unused_experts_are_zero"] = [
        e
        for e in range(N_EXPERTS)
        if not any(s.expert_start <= e < s.expert_end for s in available)
        and e < N_EXPERTS - 2
    ]
    ver["peak_ram_mb"] = peak_ram_mb()
    ver["elapsed_sec"] = round(time.time() - t0, 1)
    report["donors"] = {
        s.name: {
            "experts": [s.expert_start, s.expert_end - 1],
            "alignment": align_meta.get(s.name, {}),
            "layers": evenly_spaced_layers(_count_layers(s.ckpt)),
        }
        for s in available
    }
    report["verification"] = ver

    if not shape_ok:
        raise RuntimeError("output shapes do not match 530-key graph contract")

    log.info("RAM before save: cur=%.1fMB peak=%.1fMB", ram_mb(), ram_mb(True))
    # -- atomic write --------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".safetensors.tmp")
    save_file(out, str(tmp), metadata={"format": "pt", "builder": "C26-TECHNE-#1171"})
    os.replace(str(tmp), str(out_path))
    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ver["out_file"] = str(out_path)
    ver["out_size_mb"] = round(out_path.stat().st_size / (1024**2), 1)
    ver["report_file"] = str(REPORT_FILE)
    log.info(
        "WROTE %s (%.1f MB) in %.1fs", out_path, ver["out_size_mb"], ver["elapsed_sec"]
    )
    return report


def _count_layers(ckpt: Path) -> int:
    """Count distinct transformer layers from the safetensors header (no weights)."""
    with safe_open(str(ckpt), framework="pt", device="cpu") as f:
        layers = {
            int(m.group(1))
            for k in f.keys()
            for m in [re.search(r"layers\.(\d+)\.", k)]
            if m
        }
    return len(layers)


def main() -> None:
    global HOST_FFN, OUT_FILE, REPORT_FILE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    ap.add_argument(
        "--ffn",
        type=int,
        default=HOST_FFN,
        help="host FFN width; 4864=main-FFN bank, 512=council-MoE width",
    )
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()
    HOST_FFN = args.ffn
    OUT_FILE = args.out
    if args.report is not None:
        REPORT_FILE = args.report
    log.info("HOST_FFN=%d OUT=%s REPORT=%s", HOST_FFN, OUT_FILE, REPORT_FILE)
    rep = build(args.out)
    print("\n===== VERIFICATION SUMMARY =====")
    print(json.dumps(rep["verification"], indent=2, ensure_ascii=False))
    print("===== SKIPPED =====")
    print(json.dumps(rep["skipped"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
