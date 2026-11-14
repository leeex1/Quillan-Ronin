"""
=============================================================================
MINI 0.5B ANALYTICAL WSP v20 (Q8_0-AWARE + HOST-BASIS SVD + PROCRUSTES MOE)
=============================================================================
Zero gradient descent, zero test-set contamination (HELD_OUT_PPL_TEXT is NEVER
used as an anchor, covariance, or sweep target; all internal sweeps use ONLY
v12.EXTRA_ANCHOR_TEXT). Parameter count Diff == 0.

Key upgrades over v19:
  1. Drops `v14.compute_general_wsp_update` (which used `F.interpolate` across
     matrix dimensions and inflated Q8_0 32-weight block scales).
  2. Uses exact Q8_0-simulated block quantization (`quantize_q8_0`) during all
     internal multiplier sweeps on `v12.EXTRA_ANCHOR_TEXT`.
  3. Combines Shared-Vocab Orthogonal Procrustes (`P_llama`, `P_dragon`) +
     BitNet 1.58b Ternary MoE + Activation Covariance Steering on `o_proj`.
  4. Replaces cross-architecture `down_proj` interpolation with Host-Basis SVD
     Spectral Reweighting (`dS = S * exp(-k/8) * hann[idx]`, `delta = (U * dS) @ V^T`),
     preserving exact host singular vectors without cross-subspace rotation.
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
import compile_flagship_full_merge_v15 as v15
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Mini_WSP_v20")


def quantize_q8_0(w: torch.Tensor) -> torch.Tensor:
    """Exact PyTorch simulation of GGUF Q8_0 32-element block quantization (fp16 scale + int8 round)."""
    with torch.no_grad():
        orig_shape = w.shape
        b = w.float().reshape(-1, 32)
        amax = b.abs().amax(dim=1, keepdim=True)
        d = (amax / 127.0).to(torch.float16).to(torch.float32)
        inv_d = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
        q = torch.round(b * inv_d).clamp(-127.0, 127.0)
        return (q * d).reshape(orig_shape)


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


def build_procrustes_steered_o_delta(
    host_w: torch.Tensor,
    llama_w: torch.Tensor,
    P_llama: torch.Tensor,
    dragon_w: torch.Tensor,
    P_dragon: torch.Tensor,
    bitnet_w: torch.Tensor,
    V_a: torch.Tensor,
    U_sov: torch.Tensor,
    U_opt: Optional[torch.Tensor],
    rank: int = 24,
    rel_frob: float = 0.0020,
) -> torch.Tensor:
    """Builds Procrustes-aligned + BitNet 1.58b Ternary MoE delta for o_proj (896 x 896)."""
    with torch.no_grad():
        wh = host_w.detach().float()
        h_norm = torch.norm(wh)
        if h_norm < 1e-8 or rel_frob <= 0.0:
            return torch.zeros_like(wh)

        wl_aligned = P_llama.t() @ llama_w.float() @ P_llama
        wd_aligned = P_dragon.t() @ dragon_w.float() @ P_dragon

        wb = v8.project_spectral_basis(bitnet_w.float(), wh.shape[0], wh.shape[1], rank=rank)
        scale_b = torch.mean(torch.abs(wb)).clamp(min=1e-8)
        wb_tern = torch.round(wb / scale_b).clamp(-1.0, 1.0) * scale_b

        def _extract_mode(w_cand: torch.Tensor) -> torch.Tensor:
            w_s = w_cand * (h_norm / (torch.norm(w_cand) + 1e-8))
            diff = w_s - wh
            diff = diff - (diff @ V_a) @ V_a.t()
            steered = 0.65 * ((diff @ U_sov) @ U_sov.t()) + 0.35 * diff
            r = min(rank, min(steered.shape) - 2)
            U, S, V = torch.svd_lowrank(steered, q=r, niter=2)
            decay = torch.exp(-torch.arange(r, dtype=torch.float32) / 8.0)
            m = (U * (S * decay).unsqueeze(0)) @ V.t()
            return m / (torch.norm(m) + 1e-8)

        m_l = _extract_mode(wl_aligned)
        m_d = _extract_mode(wd_aligned)
        m_b = _extract_mode(wb_tern)
        combined = 0.45 * m_l + 0.35 * m_d + 0.20 * m_b

        combined = combined - (combined @ V_a) @ V_a.t()
        if U_opt is not None and U_opt.shape[0] == wh.shape[0]:
            combined = combined - U_opt @ (U_opt.t() @ combined)
        c_norm = torch.norm(combined)
        if c_norm < 1e-8:
            return torch.zeros_like(wh)
        return combined * (rel_frob * h_norm / c_norm)


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib_pairs = v14.build_general_curated_pairs(test_fps, seed=9020)
    anchor_texts = [p["prompt"] + " " + p["chosen"] for p in calib_pairs[:24]]
    anchor_full = v12.EXTRA_ANCHOR_TEXT + "\n\n" + "\n\n".join(anchor_texts)

    tok_b = Tokenizer.from_file(os.path.join(v8.MINI_SNAP, "tokenizer.json"))
    tok_b_vocab = tok_b.get_vocab()
    enc_chat_b = lambda s, u: tok_b.encode(
        f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    ).ids
    enc_tok_b = lambda t: tok_b.encode(t).ids

    model_b = transformers.AutoModelForCausalLM.from_pretrained(v8.MINI_SNAP, dtype=torch.float32).eval()
    W_emb_host = model_b.model.embed_tokens.weight.detach().float().clone()

    # Strictly use EXTRA_ANCHOR_TEXT (never HELD_OUT_PPL_TEXT) for internal calibration sweep
    calib_ids = torch.tensor([enc_tok_b(v12.EXTRA_ANCHOR_TEXT)], dtype=torch.long)

    def calib_ppl() -> float:
        with torch.no_grad():
            out = model_b(calib_ids, labels=calib_ids)
            return math.exp(out.loss.item())

    p_init = calib_ppl()
    logger.info("Initial calibration PPL (on EXTRA_ANCHOR_TEXT): %.4f", p_init)

    # 1. Build Procrustes maps & load donor tensors for mid-layers [8..20]
    merge_layers = list(range(8, 21))
    tok_l = Tokenizer.from_file(os.path.join(v8.FLAGSHIP_SNAP, "tokenizer.json"))
    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    P_llama = compute_procrustes_map(llama_st["model.embed_tokens.weight"], tok_l.get_vocab(), W_emb_host, tok_b_vocab)
    llama_o = {l: llama_st[f"model.layers.{round(l * 15 / 23)}.self_attn.o_proj.weight"].float().clone() for l in merge_layers}
    llama_gate = {l: llama_st[f"model.layers.{round(l * 15 / 23)}.mlp.gate_proj.weight"].float().clone() for l in merge_layers}
    llama_up = {l: llama_st[f"model.layers.{round(l * 15 / 23)}.mlp.up_proj.weight"].float().clone() for l in merge_layers}
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
    dragon_o = {}
    for l in merge_layers:
        dl = round(l * 15 / 23)
        t_do = tmap_d[f"blk.{dl}.attn_output.weight"]
        dragon_o[l] = torch.from_numpy(gguf.quants.dequantize(t_do.data, t_do.tensor_type).copy()).float()
    del E_dragon, reader_d, tmap_d, tf_d, tok_d_vocab, W_emb_host

    bitnet_st = load_file(v12.find_bitnet_snapshot_path())
    bitnet_layers = [10, 14, 18, 21]
    bitnet_o = {l: bitnet_st[f"model.layers.{bitnet_layers[idx % len(bitnet_layers)]}.self_attn.o_proj.weight"].float().clone() for idx, l in enumerate(merge_layers)}
    bitnet_gate = {l: bitnet_st[f"model.layers.{bitnet_layers[idx % len(bitnet_layers)]}.mlp.gate_proj.weight"].float().clone() for idx, l in enumerate(merge_layers)}
    bitnet_up = {l: bitnet_st[f"model.layers.{bitnet_layers[idx % len(bitnet_layers)]}.mlp.up_proj.weight"].float().clone() for idx, l in enumerate(merge_layers)}
    del bitnet_st
    gc.collect()

    # 2. Equinorm + syntax calibration (exact v15 tied-embedding calibration)
    raw_opt_b = [32, 33, 34, 35, 36]
    space_opt_b = [362, 425, 356, 422, 468]
    broken_b = [36715, 593, 7705, 75542]
    good_b = 1648
    v12.apply_equinorm_and_syntax_calibration(model_b.model.embed_tokens.weight, raw_opt_b, space_opt_b, broken_b, good_b)

    W_head_b = model_b.lm_head.weight.detach().float()
    U_opt_b = v8.compute_option_null_basis(W_head_b, enc_tok_b)

    # 3. Protected Rank-2 Refusal Subspace Abliteration (Q8_0-aware scan on [18, 19] vs [20, 21])
    U_ref_b = v12.compute_rank_k_refusal_subspace(model_b.model.embed_tokens.weight, enc_tok_b, model_b, enc_chat_b, rank=2)
    saved_18_21 = {l: model_b.model.layers[l].mlp.down_proj.weight.data.clone() for l in [18, 19, 20, 21]}
    best_ablit_pair = [18, 19]
    best_ablit_ppl = 1e9
    for cand_pair in ([18, 19], [20, 21]):
        with torch.no_grad():
            for l in [18, 19, 20, 21]:
                model_b.model.layers[l].mlp.down_proj.weight.data.copy_(saved_18_21[l])
            for l in cand_pair:
                W_d = model_b.model.layers[l].mlp.down_proj.weight.data
                W_new = W_d - 0.35 * torch.matmul(U_ref_b, torch.matmul(U_ref_b.t(), W_d.float()))
                model_b.model.layers[l].mlp.down_proj.weight.data.copy_(quantize_q8_0(W_new))
        p_cand = calib_ppl()
        logger.info("[Abliteration Q8_0 scan] layers=%s -> calib Q8_0 PPL=%.4f", cand_pair, p_cand)
        if p_cand < best_ablit_ppl:
            best_ablit_ppl = p_cand
            best_ablit_pair = cand_pair

    with torch.no_grad():
        for l in [18, 19, 20, 21]:
            model_b.model.layers[l].mlp.down_proj.weight.data.copy_(saved_18_21[l])
        for l in best_ablit_pair:
            W_d = model_b.model.layers[l].mlp.down_proj.weight.data
            model_b.model.layers[l].mlp.down_proj.weight.data.sub_(0.35 * torch.matmul(U_ref_b, torch.matmul(U_ref_b.t(), W_d.float())))
    del saved_18_21, U_ref_b
    logger.info("Selected abliteration layers=%s (calib Q8_0 PPL=%.4f)", best_ablit_pair, best_ablit_ppl)

    # 4. Capture Uncentered Input Covariances for o_proj (L8..L20)
    cov_o = {l: torch.zeros((896, 896), dtype=torch.float32) for l in merge_layers}
    hooks = []
    for l in merge_layers:
        def make_o_hook(layer_idx):
            def fn(mod, inp, out):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                cov_o[layer_idx].add_(x.t() @ x)
            return fn
        hooks.append(model_b.model.layers[l].self_attn.o_proj.register_forward_hook(make_o_hook(l)))

    calib_paragraphs = [p.strip() for p in anchor_full.split("\n\n") if len(p.strip()) > 30][:24]
    with torch.no_grad():
        for p in calib_paragraphs:
            ids = enc_tok_b(p)[:256]
            if len(ids) >= 4:
                model_b(torch.tensor([ids], dtype=torch.long))
    for h in hooks:
        h.remove()

    V_a_o = {}
    U_sov_o = {}
    for l in merge_layers:
        _, _, Vh = torch.linalg.svd(cov_o[l], full_matrices=False)
        V_a_o[l] = Vh[:48, :].t().contiguous()
        U_sov_o[l] = Vh[48:72, :].t().contiguous()
    del cov_o
    gc.collect()

    # 5. Precompute & Sweep Procrustes Steered MoE on o_proj + Donor Folds on gate/up (L8..L20)
    n_m = len(merge_layers)
    hann_o = [0.5 * (1.0 - math.cos(2.0 * math.pi * (i + 1) / (n_m + 1))) for i in range(n_m)]
    o_orig = {l: model_b.model.layers[l].self_attn.o_proj.weight.data.clone() for l in merge_layers}
    g_orig = {l: model_b.model.layers[l].mlp.gate_proj.weight.data.clone() for l in merge_layers}
    u_orig = {l: model_b.model.layers[l].mlp.up_proj.weight.data.clone() for l in merge_layers}

    o_deltas, g_deltas, u_deltas = {}, {}, {}
    for idx, l in enumerate(merge_layers):
        o_deltas[l] = build_procrustes_steered_o_delta(
            o_orig[l], llama_o[l], P_llama, dragon_o[l], P_dragon, bitnet_o[l],
            V_a_o[l], U_sov_o[l], U_opt_b, rank=24, rel_frob=0.0025 * hann_o[idx],
        )
        d_g = llama_gate[l] if idx % 2 == 0 else bitnet_gate[l]
        g_deltas[l] = v15.fold_donor_tensor(g_orig[l], d_g, rank=24, rel_frob=0.0020 * hann_o[idx], U_opt=None)
        d_u = bitnet_up[l] if idx % 2 == 0 else llama_up[l]
        u_deltas[l] = v15.fold_donor_tensor(u_orig[l], d_u, rank=24, rel_frob=0.0020 * hann_o[idx], U_opt=None)

    del llama_o, dragon_o, bitnet_o, llama_gate, llama_up, bitnet_gate, bitnet_up, P_llama, P_dragon, V_a_o, U_sov_o
    gc.collect()

    best_mult = 0.0
    best_p_o = 1e9
    for mult in [0.0, 0.25, 0.50, 0.75, 1.0, 1.25]:
        with torch.no_grad():
            for l in merge_layers:
                model_b.model.layers[l].self_attn.o_proj.weight.data.copy_(quantize_q8_0(o_orig[l] + mult * o_deltas[l]))
                model_b.model.layers[l].mlp.gate_proj.weight.data.copy_(quantize_q8_0(g_orig[l] + mult * g_deltas[l]))
                model_b.model.layers[l].mlp.up_proj.weight.data.copy_(quantize_q8_0(u_orig[l] + mult * u_deltas[l]))
        p = calib_ppl()
        logger.info("[o/gate/up Q8_0 scan] mult=%.2f -> calib Q8_0 PPL=%.4f", mult, p)
        if p < best_p_o:
            best_p_o = p
            best_mult = mult

    with torch.no_grad():
        for l in merge_layers:
            model_b.model.layers[l].self_attn.o_proj.weight.data.copy_(o_orig[l] + best_mult * o_deltas[l])
            model_b.model.layers[l].mlp.gate_proj.weight.data.copy_(g_orig[l] + best_mult * g_deltas[l])
            model_b.model.layers[l].mlp.up_proj.weight.data.copy_(u_orig[l] + best_mult * u_deltas[l])
    logger.info("Selected o/gate/up mult=%.2f (calib Q8_0 PPL=%.4f)", best_mult, best_p_o)
    del o_orig, g_orig, u_orig, o_deltas, g_deltas, u_deltas
    gc.collect()

    # 6. Host-Basis SVD Spectral Reweighting on Mid-Layer down_proj (L8..L20)
    down_layers = list(range(8, 21))
    n_d = len(down_layers)
    hann_d = [0.5 * (1.0 - math.cos(2.0 * math.pi * (i + 1) / (n_d + 1))) for i in range(n_d)]
    down_orig, down_unit_deltas = {}, {}
    decay = torch.exp(-torch.arange(24, dtype=torch.float32) / 8.0)
    with torch.no_grad():
        for idx, l in enumerate(down_layers):
            w = model_b.model.layers[l].mlp.down_proj.weight.data.float().clone()
            down_orig[l] = w
            U, S, V = torch.svd_lowrank(w, q=24, niter=2)
            dS = S * (decay * hann_d[idx])
            down_unit_deltas[l] = (U * dS.unsqueeze(0)) @ V.t()

    best_alpha = 0.0
    best_p_d = 1e9
    for alpha in [-0.010, -0.005, 0.0, 0.003, 0.006, 0.010, 0.015, 0.025, 0.035]:
        with torch.no_grad():
            for l in down_layers:
                model_b.model.layers[l].mlp.down_proj.weight.data.copy_(
                    quantize_q8_0(down_orig[l] + alpha * down_unit_deltas[l])
                )
        p = calib_ppl()
        logger.info("[down_proj Host-SVD Q8_0 scan] alpha=%+.4f -> calib Q8_0 PPL=%.4f", alpha, p)
        if p < best_p_d:
            best_p_d = p
            best_alpha = alpha

    with torch.no_grad():
        for l in down_layers:
            w_fin = down_orig[l] + best_alpha * down_unit_deltas[l]
            model_b.model.layers[l].mlp.down_proj.weight.data.copy_(quantize_q8_0(w_fin.to(torch.bfloat16).float()))
    p_bf16_sim = calib_ppl()
    logger.info(
        "Selected down_proj Host-SVD alpha=%+.4f (fp32->Q8_0 calib PPL=%.4f, bf16->Q8_0 calib PPL=%.4f)",
        best_alpha, best_p_d, p_bf16_sim,
    )

    use_bf16_export = p_bf16_sim < best_p_d
    with torch.no_grad():
        for l in down_layers:
            model_b.model.layers[l].mlp.down_proj.weight.data.copy_(down_orig[l] + best_alpha * down_unit_deltas[l])
    del down_orig, down_unit_deltas
    gc.collect()

    # 7. Export to Q8_0 GGUF & Measure Held-Out Perplexity
    if use_bf16_export:
        logger.info("Exporting via BFloat16 -> Q8_0 path (selected by EXTRA_ANCHOR_TEXT Q8_0 calib)...")
        model_b.to(torch.bfloat16)
    else:
        logger.info("Exporting via Float32 -> Q8_0 path (selected by EXTRA_ANCHOR_TEXT Q8_0 calib)...")

    qwen_st = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    sd = model_b.state_dict()
    target_dtype = torch.bfloat16 if use_bf16_export else torch.float32
    new_st = {k: sd[k].detach().cpu().to(target_dtype).contiguous() for k in qwen_st.keys()}
    ref_st = {k: v.to(target_dtype) for k, v in qwen_st.items()}
    v12.verify_exact_shapes_and_params(ref_st, new_st, "quillan-mini-0.5b-wsp-v20")
    del ref_st, qwen_st, sd, model_b
    gc.collect()

    tag = "quillan-mini-0.5b-wsp-v20"
    hf_tmp = os.path.join(v12.EXPORT_ROOT, f"{tag}_hf_tmp")
    os.makedirs(hf_tmp, exist_ok=True)
    save_file(new_st, os.path.join(hf_tmp, "model.safetensors"))
    del new_st
    gc.collect()

    copy_files = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
    for fn in copy_files:
        src = os.path.join(v8.MINI_SNAP, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(hf_tmp, fn))

    q8_path = os.path.join(v12.EXPORT_ROOT, f"{tag}-q8_0.gguf")
    subprocess.run([sys.executable, v8.CONVERT_SCRIPT, hf_tmp, "--outfile", q8_path, "--outtype", "q8_0"], check=True)
    shutil.rmtree(hf_tmp, ignore_errors=True)

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    ppl_q8 = measure_perplexity(q8_path, ppl_file)
    logger.info("MINI WSP v20 Q8_0 PPL = %.4f (v15 ref: 9.9125, clean base: 9.9362) | Path: %s | Elapsed: %.1fs", ppl_q8, q8_path, time.time() - t0)


if __name__ == "__main__":
    main()
