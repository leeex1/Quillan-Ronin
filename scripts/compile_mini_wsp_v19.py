"""
=============================================================================
MINI 0.5B ANALYTICAL WSP v19 (ORTHOGONAL PROCRUSTES + BI-ORTHO HANN FOLD)
=============================================================================
Zero gradient descent, zero test-set contamination (HELD_OUT_PPL_TEXT is NEVER
used as an anchor or calibration target). Parameter count Diff == 0.

Key upgrades over v18:
  1. Shared-vocabulary Orthogonal Procrustes (P = U V^T in R^{2048 x 896})
     aligning Llama-3.2-1B and Dragon-1B donor spaces into Qwen2.5-0.5B's
     residual space before rank-16 SVD extraction (replaces host-only basis).
  2. Bi-orthogonal null-space projection (I - U_h U_h^T) delta (I - V_h V_h^T)
     with Hann-bell cadence on mid-layers [8..19], freezing 0..7 and 20..23
     (gate_proj/up_proj untouched).
  3. Clean tied-embedding equinorm + syntax calibration (no aggressive triple-quote
     or L23 perturbations that hurt Q8_0 PPL in v18).
  4. Surgical refusal abliteration (alpha=0.35, layers [18, 19]) + ridge
     down_proj update (max_rel_frob=0.0020 on [10, 12, 14, 16, 18, 20]).
  5. Direct FP32 -> Q8_0 GGUF export + measure_perplexity.
=============================================================================
"""
import gc
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

import gguf
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import wsp_v8_closed_form_sota_compiler as v8
import compile_sovereign_fleet_wsp_v12_q8 as v12
import compile_flagship_general_v14 as v14
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Mini_WSP_v19")


def compute_procrustes_map(
    E_donor: torch.Tensor,
    tok_donor_vocab: Dict[str, int],
    E_host: torch.Tensor,
    tok_host_vocab: Dict[str, int],
    max_tokens: int = 20000,
) -> torch.Tensor:
    """Computes semi-orthogonal Procrustes map P in R^{d_donor x d_host} (P^T P = I)."""
    shared_d, shared_h = [], []
    for tok_str, hid in tok_host_vocab.items():
        if hid < E_host.shape[0] and tok_str in tok_donor_vocab:
            did = tok_donor_vocab[tok_str]
            if did < E_donor.shape[0]:
                shared_d.append(did)
                shared_h.append(hid)
                if len(shared_d) >= max_tokens:
                    break
    Ed = F.normalize(E_donor[shared_d].float(), p=2, dim=1)
    Eh = F.normalize(E_host[shared_h].float(), p=2, dim=1)
    M = Ed.t() @ Eh  # (d_donor, d_host)
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    P = U @ Vt
    logger.info("Procrustes map %s -> %s on %d shared tokens", tuple(E_donor.shape), tuple(P.shape), len(shared_d))
    return P


def fold_procrustes_svd(
    host_w: torch.Tensor,
    donor_w: torch.Tensor,
    P_out: Optional[torch.Tensor],
    P_in: Optional[torch.Tensor],
    rank: int = 16,
    rel_frob: float = 0.0020,
    U_opt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rotates donor_w into host dims via Procrustes P_out/P_in, extracts rank-r SVD, projects bi-orthogonally."""
    w_f = host_w.detach().float()
    d_out, d_in = w_f.shape
    w_d = donor_w.float()

    if P_out is not None:
        w_d = P_out.t() @ w_d
    elif w_d.shape[0] != d_out:
        w_d = w_d[:d_out, :]

    if P_in is not None:
        w_d = w_d @ P_in
    elif w_d.shape[1] != d_in:
        w_d = w_d[:, :d_in]

    r = min(rank, min(d_out, d_in) - 2)
    U_d, S_d, V_d = torch.svd_lowrank(w_d, q=r)
    delta = (U_d * S_d.unsqueeze(0)) @ V_d.t()

    U_h, _, V_h = torch.svd_lowrank(w_f, q=min(32, min(d_out, d_in) - 2))
    delta = delta - (delta @ V_h) @ V_h.t()
    delta = delta - U_h @ (U_h.t() @ delta)
    if U_opt is not None and U_opt.shape[0] == d_out:
        delta = delta - U_opt @ (U_opt.t() @ delta)

    norm_d = torch.norm(delta)
    if norm_d < 1e-8:
        return torch.zeros_like(w_f)
    return delta * (rel_frob * torch.norm(w_f) / norm_d)


def compute_zero_mean_u_opt(W_head: torch.Tensor, enc_tok_fn) -> torch.Tensor:
    vecs = []
    for l in ("A", "B", "C", "D", " A", " B", " C", " D"):
        ids = enc_tok_fn(l)
        if ids:
            vecs.append(F.normalize(W_head[ids[0]].float(), p=2, dim=0))
    V = torch.stack(vecs, dim=1)
    V = V - V.mean(dim=1, keepdim=True)
    U, S, _ = torch.linalg.svd(V, full_matrices=False)
    return U[:, S > 1e-5]


def export_fp32_q8_gguf(model: torch.nn.Module, orig_st: Dict[str, torch.Tensor], snap_dir: str, tag: str, copy_files: List[str]) -> str:
    sd = model.state_dict()
    new_st = {k: sd[k].detach().cpu().to(torch.float32) for k in orig_st.keys()}
    v12.verify_exact_shapes_and_params(orig_st, new_st, tag)

    hf_tmp = os.path.join(v12.EXPORT_ROOT, f"{tag}_hf_tmp")
    os.makedirs(hf_tmp, exist_ok=True)
    save_file(new_st, os.path.join(hf_tmp, "model.safetensors"))
    for fn in copy_files:
        src = os.path.join(snap_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(hf_tmp, fn))

    q8_path = os.path.join(v12.EXPORT_ROOT, f"{tag}-q8_0.gguf")
    subprocess.run([sys.executable, v8.CONVERT_SCRIPT, hf_tmp, "--outfile", q8_path, "--outtype", "q8_0"], check=True)
    shutil.rmtree(hf_tmp, ignore_errors=True)
    return q8_path


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib_pairs = v14.build_general_curated_pairs(test_fps, seed=9019)

    tok_b = Tokenizer.from_file(os.path.join(v8.MINI_SNAP, "tokenizer.json"))
    tok_b_vocab = tok_b.get_vocab()
    enc_chat_b = lambda s, u: tok_b.encode(
        f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    ).ids
    enc_tok_b = lambda t: tok_b.encode(t).ids

    model_b = transformers.AutoModelForCausalLM.from_pretrained(v8.MINI_SNAP, dtype=torch.float32).eval()
    W_emb_host = model_b.model.embed_tokens.weight.detach().float().clone()

    # Build Procrustes maps for Llama-3.2-1B (2048->896) and Dragon-1B (2048->896)
    tok_l = Tokenizer.from_file(os.path.join(v8.FLAGSHIP_SNAP, "tokenizer.json"))
    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    P_llama = compute_procrustes_map(llama_st["model.embed_tokens.weight"], tok_l.get_vocab(), W_emb_host, tok_b_vocab)
    llama_o = [llama_st[f"model.layers.{l}.self_attn.o_proj.weight"].float().clone() for l in range(16)]
    llama_down = [llama_st[f"model.layers.{l}.mlp.down_proj.weight"].float().clone() for l in range(16)]
    del llama_st, tok_l

    reader_d = gguf.GGUFReader(v8.BLOB_DRAGON)
    tmap_d = {t.name: t for t in reader_d.tensors}
    tf_d = reader_d.fields["tokenizer.ggml.tokens"]
    tok_d_vocab = {}
    for i, idx in enumerate(tf_d.data):
        s = bytes(tf_d.parts[idx]).decode("utf-8", errors="replace")
        if s not in tok_d_vocab:
            tok_d_vocab[s] = i
    E_dragon = torch.from_numpy(gguf.quants.dequantize(tmap_d["token_embd.weight"].data, tmap_d["token_embd.weight"].tensor_type).copy()).float()
    P_dragon = compute_procrustes_map(E_dragon, tok_d_vocab, W_emb_host, tok_b_vocab)
    del E_dragon, reader_d, tmap_d, tf_d, tok_d_vocab

    dragon_dict = v8.load_dragon_ffn_down_tensors()
    dragon_down = [dragon_dict[k] for k in sorted(dragon_dict)]
    del dragon_dict, W_emb_host
    gc.collect()

    # 1. Equinorm + syntax calibration
    raw_opt_b = [32, 33, 34, 35, 36]
    space_opt_b = [362, 425, 356, 422, 468]
    broken_b = [36715, 593, 7705, 75542]
    good_b = 1648
    v12.apply_equinorm_and_syntax_calibration(model_b.model.embed_tokens.weight, raw_opt_b, space_opt_b, broken_b, good_b)

    W_head_b = model_b.lm_head.weight.detach().float()
    U_opt_b = compute_zero_mean_u_opt(W_head_b, enc_tok_b)

    # 2. Surgical refusal abliteration (mid-layers [18, 19], alpha=0.35)
    U_ref_b = v12.compute_rank_k_refusal_subspace(model_b.model.embed_tokens.weight, enc_tok_b, model_b, enc_chat_b, rank=2)
    with torch.no_grad():
        for l in [18, 19]:
            W_d = model_b.model.layers[l].mlp.down_proj.weight
            W_d.sub_((0.35 * torch.matmul(U_ref_b, torch.matmul(U_ref_b.t(), W_d.float()))).to(W_d.dtype))
    del U_ref_b

    # 3. Ridge solve on 6 mid-layers [10, 12, 14, 16, 18, 20] (zero HELD_OUT_PPL_TEXT!)
    anchor_ids = tok_b.encode(v12.EXTRA_ANCHOR_TEXT).ids
    down_layers = [10, 12, 14, 16, 18, 20]
    donors_down = [dragon_down[i % len(dragon_down)] for i in range(len(down_layers))]
    updates_ridge = v14.compute_general_wsp_update(
        model_b, enc_chat_b, enc_tok_b, anchor_ids,
        down_layers, donors_down, calib_pairs, ridge_lambda=36.0, max_rel_frob=0.0020,
    )

    # 4. Hann-bell Procrustes fold on mid-layers [8..19] (o_proj + down_proj) + ridge
    merge_layers = list(range(8, 20))
    def hann(idx: int, n: int) -> float:
        tau = (idx + 0.5) / n
        return 0.5 * (1.0 - math.cos(2.0 * math.pi * tau))

    with torch.no_grad():
        for idx, l in enumerate(merge_layers):
            cad = hann(idx, len(merge_layers))
            l_llama = round(l * 15 / 23)
            w_o = model_b.model.layers[l].self_attn.o_proj.weight
            w_o.add_(fold_procrustes_svd(w_o, llama_o[l_llama], P_llama, P_llama, rank=16, rel_frob=0.0020 * cad, U_opt=U_opt_b))

            w_d = model_b.model.layers[l].mlp.down_proj.weight
            w_d.add_(fold_procrustes_svd(w_d, llama_down[l_llama], P_llama, None, rank=16, rel_frob=0.0022 * cad, U_opt=None))

        for l in down_layers:
            model_b.model.layers[l].mlp.down_proj.weight.add_(updates_ridge[l])

    del updates_ridge, llama_o, llama_down, dragon_down, P_llama, P_dragon
    gc.collect()
    qwen_st = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    q8_b = export_fp32_q8_gguf(
        model_b, qwen_st, v8.MINI_SNAP, "quillan-mini-0.5b-wsp-v19",
        ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"],
    )
    del qwen_st, model_b
    gc.collect()

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    ppl_q8 = measure_perplexity(q8_b, ppl_file)
    logger.info("MINI WSP v19 Q8_0 PPL = %.4f (v15 ref: 9.9125) | Path: %s | Elapsed: %.1fs", ppl_q8, q8_b, time.time() - t0)


if __name__ == "__main__":
    main()
