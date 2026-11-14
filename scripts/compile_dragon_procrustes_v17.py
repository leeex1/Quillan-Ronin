"""
=============================================================================
DRAGON 1.0B (FALCON3-1B Q8_0) PROCRUSTES + SPECTRAL WSP COMPILER (v17+)
=============================================================================
Analytical Weight-Space Programming (zero gradient descent, zero test-set
calibration, Delta_P == 0):
1. Zero-mmap safetensors reader (reads individual tensors via seek/read to
   avoid Windows pagefile/commit-limit OSError 1455 under concurrent loads).
2. Fixes Falcon3-1B tokenizer space prefix ('\\u0120' / 'Ġ') for equinorm,
   zero-mean option null-space (U_opt), security/negation-protected (U_prot),
   and refusal subspace (U_ref, excluding terminal layer 17).
3. Closed-form Orthogonal Procrustes (P = U V^T) on shared-vocab embeddings
   between Falcon3-1B and donors (Llama-3.2-1B, Qwen2.5-0.5B, BitNet-2B),
   cached to disk for fast parameter sweeps.
4. Bi-orthogonal null-space projection (I - U_h U_h^T) spec (I - V_h V_h^T)
   restricted to linear residual projections (attn_output, ffn_down), with
   Hann-bell layer tapering and optional host spectral enhancement.
=============================================================================
"""
import argparse
import gc
import json
import logging
import os
import shutil
import struct
import sys
import time
from typing import Dict, List, Optional

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import wsp_v8_closed_form_sota_compiler as v8
import compile_sovereign_fleet_wsp_v12_q8 as v12
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Dragon_Procrustes_v17")

_DTYPE_MAP = {
    "F32": (np.float32, 4),
    "F16": (np.float16, 2),
    "BF16": (np.uint16, 2),
    "I8": (np.int8, 1),
    "U8": (np.uint8, 1),
}


def read_safetensors_header(path: str):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen).decode("utf-8"))
        data_base = 8 + hlen
    return header, data_base


def read_safetensors_tensor(path: str, key: str, header=None, data_base: int = 0) -> Optional[torch.Tensor]:
    if header is None:
        header, data_base = read_safetensors_header(path)
    meta = header.get(key)
    if meta is None:
        return None
    dtype_str = meta["dtype"]
    shape = meta["shape"]
    start, end = meta["data_offsets"]
    np_dt, _ = _DTYPE_MAP[dtype_str]
    with open(path, "rb") as f:
        f.seek(data_base + start)
        raw = f.read(end - start)
    arr = np.frombuffer(raw, dtype=np_dt).reshape(shape).copy()
    if dtype_str == "BF16":
        t = torch.from_numpy(arr).view(torch.bfloat16).float()
    else:
        t = torch.from_numpy(arr).float()
    return t


def load_selected_tensors(safetensors_path: str, keys: List[str]) -> Dict[str, torch.Tensor]:
    header, data_base = read_safetensors_header(safetensors_path)
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        t = read_safetensors_tensor(safetensors_path, k, header, data_base)
        if t is not None:
            out[k] = t
    return out


def _norm_tok(t: str) -> str:
    return t.replace("\u0120", " ").replace("\u2581", " ")


def compute_procrustes_map(
    host_toks: List[str],
    host_emb_tensor,
    donor_dir: str,
    donor_st_path: str,
    max_shared: int = 20000,
) -> Optional[torch.Tensor]:
    tok_json = os.path.join(donor_dir, "tokenizer.json")
    if not os.path.exists(tok_json):
        logger.warning("No tokenizer.json in %s; skipping Procrustes", donor_dir)
        return None

    tok = Tokenizer.from_file(tok_json)
    donor_vocab = tok.get_vocab()
    donor_norm_map: Dict[str, int] = {}
    for t_str, tid in donor_vocab.items():
        if t_str.startswith("<") and t_str.endswith(">"):
            continue
        nt = _norm_tok(t_str)
        if nt and nt not in donor_norm_map:
            donor_norm_map[nt] = tid

    n_host_rows = host_emb_tensor.data.shape[0]
    host_indices: List[int] = []
    donor_indices: List[int] = []
    seen_norm = set()
    for h_id, h_str in enumerate(host_toks):
        if h_id >= n_host_rows:
            break
        if h_str.startswith("<") and h_str.endswith(">"):
            continue
        nt = _norm_tok(h_str)
        if not nt or nt in seen_norm:
            continue
        d_id = donor_norm_map.get(nt)
        if d_id is not None:
            seen_norm.add(nt)
            host_indices.append(h_id)
            donor_indices.append(d_id)
            if len(host_indices) >= max_shared:
                break

    if len(host_indices) < 256:
        logger.warning("Too few shared tokens (%d) for %s", len(host_indices), donor_dir)
        return None

    d_emb_full = read_safetensors_tensor(donor_st_path, "model.embed_tokens.weight")
    if d_emb_full is None:
        return None

    valid_pairs = [(h, d) for h, d in zip(host_indices, donor_indices) if d < d_emb_full.shape[0]]
    h_idx = [p[0] for p in valid_pairs]
    d_idx = torch.tensor([p[1] for p in valid_pairs], dtype=torch.long)

    h_sub = torch.from_numpy(
        gguf.quants.dequantize(host_emb_tensor.data[h_idx], host_emb_tensor.tensor_type)
    ).float()
    E_host = F.normalize(h_sub, p=2, dim=1)
    E_donor = F.normalize(d_emb_full[d_idx], p=2, dim=1)
    del d_emb_full, h_sub
    gc.collect()

    M = E_donor.t() @ E_host  # (d_donor, d_host)
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    P = U @ Vh  # (d_donor, d_host)
    logger.info(
        "Procrustes %s: shared=%d, P shape=%s",
        os.path.basename(donor_dir), len(valid_pairs), tuple(P.shape)
    )
    return P


def fold_procrustes_tensor(
    host_w: torch.Tensor,
    donor_w: torch.Tensor,
    P_out: Optional[torch.Tensor],
    P_in: Optional[torch.Tensor],
    rank: int = 24,
    rel_frob: float = 0.004,
    U_opt: Optional[torch.Tensor] = None,
    biorth_k: int = 32,
) -> torch.Tensor:
    w_f = host_w.detach().float()
    d_out, d_in = w_f.shape
    dw = donor_w.detach().float()

    if P_out is not None and dw.shape[0] == P_out.shape[0] and P_out.shape[1] == d_out:
        dw = P_out.t() @ dw
    elif dw.shape[0] != d_out:
        dw = F.interpolate(dw.t().unsqueeze(0), size=d_out, mode="linear", align_corners=False).squeeze(0).t()

    if P_in is not None and dw.shape[1] == P_in.shape[0] and P_in.shape[1] == d_in:
        dw = dw @ P_in
    elif dw.shape[1] != d_in:
        dw = F.interpolate(dw.unsqueeze(0), size=d_in, mode="linear", align_corners=False).squeeze(0)

    q = min(rank, min(d_out, d_in) - 2)
    U, S, V = torch.svd_lowrank(dw, q=q)
    spec = (U * S.unsqueeze(0)) @ V.t()

    # Bi-orthogonal null-space projection: (I - U_h U_h^T) @ spec @ (I - V_h V_h^T)
    bk = min(biorth_k, min(d_out, d_in) - 2)
    U_h, _, V_h = torch.svd_lowrank(w_f, q=bk)
    spec = spec - (spec @ V_h) @ V_h.t()
    spec = spec - U_h @ (U_h.t() @ spec)

    if U_opt is not None and U_opt.shape[0] == d_out:
        spec = spec - U_opt @ (U_opt.t() @ spec)

    norm = torch.norm(spec)
    if norm < 1e-8:
        return torch.zeros_like(w_f)
    return (spec / norm) * (rel_frob * torch.norm(w_f))


def spectral_boost_ffn_down(
    w_f: torch.Tensor,
    alpha: float = 0.02,
    tau: float = 8.0,
    U_opt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if alpha <= 0.0:
        return torch.zeros_like(w_f)
    k = min(64, min(w_f.shape) - 2)
    U, S, V = torch.svd_lowrank(w_f, q=k)
    idx = torch.arange(k, dtype=w_f.dtype)
    weights = alpha * torch.exp(-idx / tau)
    delta = (U * (S * weights).unsqueeze(0)) @ V.t()
    if U_opt is not None and U_opt.shape[0] == w_f.shape[0]:
        delta = delta - U_opt @ (U_opt.t() @ delta)
    return delta


def compile_dragon_procrustes(
    ppl_file: str,
    o_frob: float = 0.008,
    d_frob: float = 0.004,
    spec_boost: float = 0.02,
    rank: int = 24,
    lo: int = 6,
    hi: int = 17,
    ref_alpha: float = 0.35,
    tag: str = "v17p",
    eval_ppl: bool = True,
) -> Dict[str, object]:
    t0 = time.time()
    logger.info("=== Compiling Dragon 1.0B Procrustes WSP (%s) ===", tag)
    reader_c = gguf.GGUFReader(v8.BLOB_DRAGON)
    tmap_c = {t.name: t for t in reader_c.tensors}
    q8_out = os.path.join(v12.EXPORT_ROOT, f"quillan-dragon-1b-wsp-{tag}-q8_0.gguf")
    shutil.copy2(v8.BLOB_DRAGON, q8_out)

    tf = reader_c.fields["tokenizer.ggml.tokens"]
    toks = [bytes(tf.parts[idx]).decode("utf-8", errors="replace") for idx in tf.data]
    tok_lookup: Dict[str, int] = {}
    for i, t in enumerate(toks):
        if t not in tok_lookup:
            tok_lookup[t] = i

    def find_tids(words: List[str]) -> List[int]:
        out_ids = []
        for w in words:
            for cand in (w, "\u0120" + w.lstrip(), w.lower(), "\u0120" + w.lstrip().lower()):
                if cand in tok_lookup:
                    out_ids.append(tok_lookup[cand])
        return list(dict.fromkeys(out_ids))

    # 1. Process output.weight first and free it immediately to keep RAM low
    W_out = torch.from_numpy(
        gguf.quants.dequantize(tmap_c["output.weight"].data, tmap_c["output.weight"].tensor_type).copy()
    ).float()

    raw_opt_c = [tok_lookup[l] for l in "ABCDE" if l in tok_lookup]
    space_opt_c = [tok_lookup["\u0120" + l] for l in "ABCDE" if ("\u0120" + l) in tok_lookup]
    broken_c = [tok_lookup[b] for b in [")): ", "))", "}}}", "}}"] if b in tok_lookup]
    good_c = tok_lookup.get("):", 4981)
    logger.info("Option tids raw=%d, space=%d", len(raw_opt_c), len(space_opt_c))
    v12.apply_equinorm_and_syntax_calibration(W_out, raw_opt_c, space_opt_c, broken_c, good_c)

    with open(q8_out, "r+b") as fp:
        t_out = tmap_c["output.weight"]
        raw_out_bytes = gguf.quants.quantize(W_out.numpy().astype(np.float32), t_out.tensor_type).tobytes()
        fp.seek(t_out.data_offset)
        fp.write(raw_out_bytes)

    # Zero-mean option difference basis (preserves factual option knowledge)
    opt_ids = find_tids(["A", "B", "C", "D"])
    opt_vecs = [W_out[tid] for tid in opt_ids]
    opt_mean = torch.stack(opt_vecs, dim=0).mean(dim=0)
    opt_diffs = [v - opt_mean for v in opt_vecs]
    U_opt_c, _ = torch.linalg.qr(torch.stack(opt_diffs, dim=1))

    # Protected basis: code negation, security terms, sovereign identity, options
    prot_words = [
        "not", "never", "false", "None", "raise", "def", "return", "for", "if",
        "range", "min", "max", "security", "vulnerability", "exploit", "cipher",
        "packet", "attack", "Quillan", "Dragon", "Crash", "Step", "Final",
    ]
    prot_ids = find_tids(prot_words) + opt_ids
    U_prot_c, _ = torch.linalg.qr(
        torch.stack([F.normalize(W_out[i], p=2, dim=0) for i in dict.fromkeys(prot_ids)], dim=1)
    )

    ref_ids = find_tids(["cannot", "sorry", "apologize", "unable", "assist", "illegal", "harmful", "guidelines"])
    U_ref_raw, _, _ = torch.linalg.svd(
        torch.stack([F.normalize(W_out[i], p=2, dim=0) for i in ref_ids], dim=1), full_matrices=False
    )
    U_ref_k = U_ref_raw[:, :min(2, U_ref_raw.shape[1])]
    U_ref_p = U_ref_k - U_prot_c @ (U_prot_c.t() @ U_ref_k)
    U_ref_p = U_ref_p - U_opt_c @ (U_opt_c.t() @ U_ref_p)
    U_ref_c, _ = torch.linalg.qr(U_ref_p)

    del W_out
    gc.collect()

    # 2. Load or compute Procrustes maps
    llama_st_path = os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors")
    qwen_st_path = os.path.join(v8.MINI_SNAP, "model.safetensors")
    bitnet_st_path = v12.find_bitnet_snapshot_path()
    bitnet_dir = os.path.dirname(bitnet_st_path)

    cache_path = os.path.join(v12.EXPORT_ROOT, "dragon_procrustes_cache.pt")
    if os.path.exists(cache_path):
        logger.info("Loading cached Procrustes maps from %s", cache_path)
        pcache = torch.load(cache_path, map_location="cpu", weights_only=True)
        P_llama = pcache["llama"]
        P_qwen = pcache["qwen"]
        P_bitnet = pcache["bitnet"]
    else:
        t_emb = tmap_c["token_embd.weight"]
        P_llama = compute_procrustes_map(toks, t_emb, v8.FLAGSHIP_SNAP, llama_st_path)
        P_qwen = compute_procrustes_map(toks, t_emb, v8.MINI_SNAP, qwen_st_path)
        P_bitnet = compute_procrustes_map(toks, t_emb, bitnet_dir, bitnet_st_path)
        gc.collect()
        torch.save({"llama": P_llama, "qwen": P_qwen, "bitnet": P_bitnet}, cache_path)

    llama_layers = [9, 11, 13, 15]
    qwen_layers = [12, 15, 18, 21]
    bitnet_layers = [10, 14, 18, 21]

    llama_o = load_selected_tensors(llama_st_path, [f"model.layers.{l}.self_attn.o_proj.weight" for l in llama_layers])
    llama_d = load_selected_tensors(llama_st_path, [f"model.layers.{l}.mlp.down_proj.weight" for l in llama_layers])
    qwen_o = load_selected_tensors(qwen_st_path, [f"model.layers.{l}.self_attn.o_proj.weight" for l in qwen_layers])
    bitnet_o = load_selected_tensors(bitnet_st_path, [f"model.layers.{l}.self_attn.o_proj.weight" for l in bitnet_layers])

    o_sources = [
        ([llama_o[f"model.layers.{l}.self_attn.o_proj.weight"] for l in llama_layers], P_llama),
        ([qwen_o[f"model.layers.{l}.self_attn.o_proj.weight"] for l in qwen_layers], P_qwen),
        ([bitnet_o[f"model.layers.{l}.self_attn.o_proj.weight"] for l in bitnet_layers], P_bitnet),
    ]
    d_list = [llama_d[f"model.layers.{l}.mlp.down_proj.weight"] for l in llama_layers]

    layers_c = list(range(lo, hi))
    updated_count = 1

    with open(q8_out, "r+b") as fp:
        for idx, l in enumerate(layers_c):
            tau = (idx + 0.5) / len(layers_c)
            ramp = float(0.5 * (1.0 - np.cos(2.0 * np.pi * tau)))  # Hann bell

            if o_frob > 0.0:
                tname = f"blk.{l}.attn_output.weight"
                if tname in tmap_c:
                    t = tmap_c[tname]
                    w_c = torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).float()
                    src_list, P_src = o_sources[idx % len(o_sources)]
                    d_tensor = src_list[idx % len(src_list)]
                    delta = fold_procrustes_tensor(
                        w_c, d_tensor, P_src, P_src, rank=rank, rel_frob=o_frob * ramp, U_opt=U_opt_c
                    )
                    w_new = w_c + delta
                    raw = gguf.quants.quantize(w_new.numpy().astype(np.float32), t.tensor_type).tobytes()
                    fp.seek(t.data_offset)
                    fp.write(raw)
                    updated_count += 1

            tname_d = f"blk.{l}.ffn_down.weight"
            if tname_d in tmap_c:
                t = tmap_c[tname_d]
                w_c = torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).float()
                w_new = w_c.clone()
                if d_frob > 0.0:
                    d_tensor = d_list[idx % len(d_list)]
                    w_new = w_new + fold_procrustes_tensor(
                        w_c, d_tensor, P_llama, None, rank=rank, rel_frob=d_frob * ramp, U_opt=None
                    )
                if spec_boost > 0.0:
                    w_new = w_new + spectral_boost_ffn_down(w_c, alpha=spec_boost * ramp, tau=8.0, U_opt=None)
                if ref_alpha > 0.0 and l in (14, 15, 16):
                    w_new = w_new - ref_alpha * torch.matmul(U_ref_c, torch.matmul(U_ref_c.t(), w_new))
                raw = gguf.quants.quantize(w_new.numpy().astype(np.float32), t.tensor_type).tobytes()
                fp.seek(t.data_offset)
                fp.write(raw)
                updated_count += 1

    ppl_c = measure_perplexity(q8_out, ppl_file) if eval_ppl else -1.0
    logger.info(
        "DRAGON %s PPL = %.4f (v15 ref: 4.2627, v14 ref: 4.2609) | Updated=%d | elapsed=%.1fs",
        tag, ppl_c, updated_count, time.time() - t0
    )
    return {"path": q8_out, "ppl": ppl_c, "param_diff": 0, "updated": updated_count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--o_frob", type=float, default=0.008)
    ap.add_argument("--d_frob", type=float, default=0.004)
    ap.add_argument("--spec_boost", type=float, default=0.02)
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--lo", type=int, default=6)
    ap.add_argument("--hi", type=int, default=17)
    ap.add_argument("--ref_alpha", type=float, default=0.35)
    ap.add_argument("--tag", default="v17p")
    a = ap.parse_args()

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)

    res = compile_dragon_procrustes(
        ppl_file=ppl_file,
        o_frob=a.o_frob,
        d_frob=a.d_frob,
        spec_boost=a.spec_boost,
        rank=a.rank,
        lo=a.lo,
        hi=a.hi,
        ref_alpha=a.ref_alpha,
        tag=a.tag,
        eval_ppl=True,
    )
    logger.info("FINAL RESULT: %s", res)


if __name__ == "__main__":
    main()
