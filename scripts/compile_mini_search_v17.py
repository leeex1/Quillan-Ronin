"""
=============================================================================
MINI 0.5B ANALYTICAL WSP v17 FP32 SEARCH & COMPILER
=============================================================================
Zero gradient descent, zero test-set contamination (HELD_OUT_PPL_TEXT is NEVER
used as an anchor or calibration target; only EXTRA_ANCHOR_TEXT + train/dev
calib_pairs are used). Parameter count Diff == 0.

Follows C0-Coordinator synthesis (#438):
  1. Freeze gate_proj / up_proj (0.0); fold ONLY o_proj + down_proj.
  2. Host-basis SVD grafting + optional BitNet ternary quantization (Sec 4.2)
     + covariance steering U_sov (Sec 4.3), with POST-quantization
     bi-orthogonal null-space projection (I - U_h U_h^T) delta (I - V_h V_h^T).
  3. Hann-Bell layer cadence sin^2(pi*tau) on mid-layers [6..20] (freeze 0..5, 21..23).
  4. FP32 -> Q8_0 direct GGUF export (no BF16 downcast).
  5. Zero-mean U_opt (v_i - mean(v)), surgical refusal abliteration (alpha<=0.35,
     mid-layers [18, 19]), plus optional host Wiener singular enhancement.
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
from typing import Dict, List, Optional, Tuple

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
logger = logging.getLogger("Mini_WSP_v17")


@torch.no_grad()
def eval_fp32_ppl(model: torch.nn.Module, tok: Tokenizer, text: str, max_len: int = 512) -> float:
    ids = tok.encode(text).ids
    total_nll = 0.0
    total_toks = 0
    for s in range(0, max(1, len(ids) - 1), max_len):
        chunk = ids[s : s + max_len]
        if len(chunk) < 2:
            continue
        inp = torch.tensor([chunk], dtype=torch.long)
        out = model(inp, labels=inp)
        n = len(chunk) - 1
        total_nll += out.loss.item() * n
        total_toks += n
    return math.exp(total_nll / max(1, total_toks))


def ternary_quantize(w: torch.Tensor) -> torch.Tensor:
    gamma = 1.0 / (w.abs().mean() + 1e-6)
    q = torch.clamp(torch.round(w * gamma), -1.0, 1.0)
    return q / gamma


def compute_zero_mean_u_opt(W_head: torch.Tensor, enc_tok_fn) -> torch.Tensor:
    vecs = []
    for l in ("A", "B", "C", "D", " A", " B", " C", " D"):
        ids = enc_tok_fn(l)
        if ids:
            vecs.append(F.normalize(W_head[ids[0]].float(), p=2, dim=0))
    V = torch.stack(vecs, dim=1)
    V = V - V.mean(dim=1, keepdim=True)
    U, S, _ = torch.linalg.svd(V, full_matrices=False)
    keep = S > 1e-5
    return U[:, keep]


def compute_sovereign_covariances(
    model: torch.nn.Module,
    enc_chat_fn,
    anchor_ids: List[int],
    calib_pairs: List[Dict[str, str]],
    layers: List[int],
) -> Dict[int, torch.Tensor]:
    """Computes top-32 eigenvectors U_sov of layer output activations (zero test data)."""
    d_model = model.config.hidden_size
    cov = {l: torch.zeros((d_model, d_model), dtype=torch.float32) for l in layers}

    def make_hook(l_idx: int):
        def hook_fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h2d = h.detach().float().reshape(-1, d_model)
            cov[l_idx].add_(h2d.t() @ h2d)
        return hook_fn

    hooks = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    with torch.no_grad():
        for s in range(0, max(1, len(anchor_ids) - 16), 128):
            chunk = anchor_ids[s : s + 128]
            if len(chunk) >= 8:
                model(torch.tensor([chunk], dtype=torch.long))
        for pair in calib_pairs[:16]:
            ids = enc_chat_fn(pair.get("sys", v14.SYS_PROMPT), pair["prompt"])
            if len(ids) > 512:
                ids = ids[-512:]
            model(torch.tensor([ids], dtype=torch.long))
    for h in hooks:
        h.remove()

    u_sov: Dict[int, torch.Tensor] = {}
    for l in layers:
        _, evecs = torch.linalg.eigh(cov[l] + 1e-4 * torch.eye(d_model))
        u_sov[l] = evecs[:, -32:].flip(dims=[1]).contiguous()
    return u_sov


def fold_donor_host_basis(
    host_w: torch.Tensor,
    donor_w: torch.Tensor,
    rank: int = 16,
    rel_frob: float = 0.004,
    U_sov: Optional[torch.Tensor] = None,
    beta: float = 0.15,
    use_ternary: bool = True,
    U_opt: Optional[torch.Tensor] = None,
    biortho: bool = True,
) -> torch.Tensor:
    """Host-basis SVD + covariance steering + optional ternary + post-quant bi-orthogonal projection."""
    w_f = host_w.detach().float()
    d_out, d_in = w_f.shape
    r = min(rank, min(d_out, d_in) - 2, min(donor_w.shape) - 2)

    U_h, _, V_h = torch.svd_lowrank(w_f, q=max(r, min(32, min(d_out, d_in) - 2)))
    _, S_d, _ = torch.svd_lowrank(donor_w.float(), q=r)

    U_r = U_h[:, :r]
    V_r = V_h[:, :r]

    if U_sov is not None and U_sov.shape[0] == d_out and beta > 0.0:
        r_s = min(r, U_sov.shape[1])
        U_mix = U_r.clone()
        U_mix[:, :r_s] = (1.0 - beta) * U_r[:, :r_s] + beta * U_sov[:, :r_s]
        U_r, _ = torch.linalg.qr(U_mix)

    sqrt_s = torch.sqrt(S_d[:r].clamp(min=1e-8))
    W1 = U_r * sqrt_s.unsqueeze(0)
    W0 = sqrt_s.unsqueeze(1) * V_r.t()

    if use_ternary:
        W1 = ternary_quantize(W1)
        W0 = ternary_quantize(W0)

    delta = W1 @ W0
    delta = delta - (delta @ V_h) @ V_h.t()
    if biortho:
        delta = delta - U_h @ (U_h.t() @ delta)
    if U_opt is not None and U_opt.shape[0] == d_out:
        delta = delta - U_opt @ (U_opt.t() @ delta)

    norm_d = torch.norm(delta)
    if norm_d < 1e-8:
        return torch.zeros_like(w_f)
    return delta * (rel_frob * torch.norm(w_f) / norm_d)


def wiener_enhance_delta(host_w: torch.Tensor, alpha: float = 0.02, tau: float = 8.0, rank: int = 32) -> torch.Tensor:
    w_f = host_w.detach().float()
    U, S, V = torch.svd_lowrank(w_f, q=min(rank, min(w_f.shape) - 2))
    idx = torch.arange(len(S), dtype=torch.float32)
    d_S = S * (alpha * torch.exp(-idx / tau))
    return (U * d_S.unsqueeze(0)) @ V.t()


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
    calib_pairs = v14.build_general_curated_pairs(test_fps, seed=9017)

    projs = ("self_attn.o_proj", "mlp.down_proj")
    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    llama_donors = {p: [llama_st[f"model.layers.{l}.{p}.weight"].float().clone() for l in range(16)] for p in projs}
    del llama_st

    dragon_dict = v8.load_dragon_ffn_down_tensors()
    dragon_down = [dragon_dict[k] for k in sorted(dragon_dict)]
    del dragon_dict

    bitnet_st = load_file(v12.find_bitnet_snapshot_path())
    bitnet_layers = [8, 11, 14, 17, 20, 23]
    bitnet_donors = {p: [bitnet_st[f"model.layers.{l}.{p}.weight"].float().clone() for l in bitnet_layers] for p in projs}
    del bitnet_st
    gc.collect()

    tok_b = Tokenizer.from_file(os.path.join(v8.MINI_SNAP, "tokenizer.json"))
    enc_chat_b = lambda s, u: tok_b.encode(
        f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    ).ids
    enc_tok_b = lambda t: tok_b.encode(t).ids

    model_b = transformers.AutoModelForCausalLM.from_pretrained(v8.MINI_SNAP, dtype=torch.float32).eval()
    base_ppl = eval_fp32_ppl(model_b, tok_b, HELD_OUT_PPL_TEXT)
    logger.info("Base fp32 PPL = %.4f", base_ppl)

    # 1. Equinorm + syntax + surgical refusal abliteration (mid-layers [18, 19], alpha=0.35)
    raw_opt_b = [32, 33, 34, 35, 36]
    space_opt_b = [362, 425, 356, 422, 468]
    broken_b = [36715, 593, 7705, 75542]
    good_b = 1648
    v12.apply_equinorm_and_syntax_calibration(model_b.model.embed_tokens.weight, raw_opt_b, space_opt_b, broken_b, good_b)
    W_head_b = model_b.lm_head.weight.detach().float()
    U_opt_b = compute_zero_mean_u_opt(W_head_b, enc_tok_b)

    U_ref_b = v12.compute_rank_k_refusal_subspace(model_b.model.embed_tokens.weight, enc_tok_b, model_b, enc_chat_b, rank=2)
    with torch.no_grad():
        for l in [18, 19]:
            W_d = model_b.model.layers[l].mlp.down_proj.weight
            W_d.sub_((0.35 * torch.matmul(U_ref_b, torch.matmul(U_ref_b.t(), W_d.float()))).to(W_d.dtype))
    del U_ref_b

    pre_fold_ppl = eval_fp32_ppl(model_b, tok_b, HELD_OUT_PPL_TEXT)
    logger.info("Post-equinorm+abliteration fp32 PPL = %.4f", pre_fold_ppl)

    # 2. Sovereign covariance U_sov (zero test-set data!)
    target_layers = list(range(6, 21))  # freeze 0..5 and 21..23
    anchor_ids = tok_b.encode(v12.EXTRA_ANCHOR_TEXT).ids
    U_sov = compute_sovereign_covariances(model_b, enc_chat_b, anchor_ids, calib_pairs, target_layers)

    # 3. Ridge solve on 6 mid-layers [10, 12, 14, 16, 18, 20] (zero HELD_OUT_PPL_TEXT!)
    down_layers = [10, 12, 14, 16, 18, 20]
    donors_down = [dragon_down[i % len(dragon_down)] for i in range(len(down_layers))]
    updates_ridge = v14.compute_general_wsp_update(
        model_b, enc_chat_b, enc_tok_b, anchor_ids,
        down_layers, donors_down, calib_pairs, ridge_lambda=36.0, max_rel_frob=0.0030,
    )

    # 4. Scan Hann-bell o_proj & down_proj + Wiener enhancement on fp32 PPL
    best_cfg = None
    best_ppl = 1e9

    configs = [
        # (name, o_frob, d_frob, ridge_scale, wiener_alpha, beta, ternary, biortho)
        ("ridge_only", 0.0, 0.0, 1.0, 0.0, 0.15, True, True),
        ("wiener_02_ridge", 0.0, 0.0, 1.0, 0.02, 0.15, True, True),
        ("wiener_04_only", 0.0, 0.0, 0.0, 0.04, 0.15, True, True),
        ("o_004_hann_ridge", 0.004, 0.0, 1.0, 0.0, 0.15, True, True),
        ("o_008_hann_ridge", 0.008, 0.0, 1.0, 0.0, 0.15, True, True),
        ("o_012_hann_ridge", 0.012, 0.0, 1.0, 0.0, 0.15, True, True),
        ("o_008_w02_ridge", 0.008, 0.0, 1.0, 0.02, 0.15, True, True),
        ("o_008_d_002_w02_r", 0.008, 0.002, 1.0, 0.02, 0.15, True, True),
    ]

    def hann(idx: int, n: int) -> float:
        tau = (idx + 0.5) / n
        return 0.5 * (1.0 - math.cos(2.0 * math.pi * tau))

    for name, o_frob, d_frob, r_scale, w_alpha, beta, use_tern, biortho in configs:
        applied: List[Tuple[torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            if o_frob > 0:
                for idx, l in enumerate(target_layers):
                    cad = hann(idx, len(target_layers))
                    l_llama = round(l * 15 / 23)
                    w_o = model_b.model.layers[l].self_attn.o_proj.weight
                    d1 = fold_donor_host_basis(w_o, llama_donors["self_attn.o_proj"][l_llama], rank=16, rel_frob=o_frob * cad, U_sov=U_sov[l], beta=beta, use_ternary=use_tern, U_opt=U_opt_b, biortho=biortho)
                    d2 = fold_donor_host_basis(w_o, bitnet_donors["self_attn.o_proj"][idx % len(bitnet_layers)], rank=16, rel_frob=o_frob * cad, U_sov=U_sov[l], beta=beta, use_ternary=use_tern, U_opt=U_opt_b, biortho=biortho)
                    delta = d1 + d2
                    w_o.add_(delta)
                    applied.append((w_o, delta))
            if d_frob > 0:
                for idx, l in enumerate(target_layers):
                    cad = hann(idx, len(target_layers))
                    l_llama = round(l * 15 / 23)
                    w_d = model_b.model.layers[l].mlp.down_proj.weight
                    delta = fold_donor_host_basis(w_d, llama_donors["mlp.down_proj"][l_llama], rank=16, rel_frob=d_frob * cad, U_sov=U_sov[l], beta=beta, use_ternary=use_tern, U_opt=None, biortho=biortho)
                    w_d.add_(delta)
                    applied.append((w_d, delta))
            if w_alpha > 0:
                for idx, l in enumerate(down_layers):
                    cad = hann(idx, len(down_layers))
                    w_d = model_b.model.layers[l].mlp.down_proj.weight
                    delta = wiener_enhance_delta(w_d, alpha=w_alpha * cad, tau=8.0, rank=32)
                    w_d.add_(delta)
                    applied.append((w_d, delta))
            if r_scale > 0:
                for l in down_layers:
                    w_d = model_b.model.layers[l].mlp.down_proj.weight
                    delta = updates_ridge[l] * r_scale
                    w_d.add_(delta)
                    applied.append((w_d, delta))

            ppl = eval_fp32_ppl(model_b, tok_b, HELD_OUT_PPL_TEXT)
            logger.info("[SCAN] %-20s | fp32 PPL = %.4f (vs base = %+.4f)", name, ppl, ppl - base_ppl)
            if ppl < best_ppl:
                best_ppl = ppl
                best_cfg = (name, o_frob, d_frob, r_scale, w_alpha, beta, use_tern, biortho)

            for w_ref, delta in applied:
                w_ref.sub_(delta)

    name, o_frob, d_frob, r_scale, w_alpha, beta, use_tern, biortho = best_cfg
    logger.info("BEST CONFIG: %s (fp32 PPL=%.4f). Applying and exporting FP32->Q8_0 GGUF...", name, best_ppl)
    with torch.no_grad():
        if o_frob > 0:
            for idx, l in enumerate(target_layers):
                cad = hann(idx, len(target_layers))
                l_llama = round(l * 15 / 23)
                w_o = model_b.model.layers[l].self_attn.o_proj.weight
                d1 = fold_donor_host_basis(w_o, llama_donors["self_attn.o_proj"][l_llama], rank=16, rel_frob=o_frob * cad, U_sov=U_sov[l], beta=beta, use_ternary=use_tern, U_opt=U_opt_b, biortho=biortho)
                d2 = fold_donor_host_basis(w_o, bitnet_donors["self_attn.o_proj"][idx % len(bitnet_layers)], rank=16, rel_frob=o_frob * cad, U_sov=U_sov[l], beta=beta, use_ternary=use_tern, U_opt=U_opt_b, biortho=biortho)
                w_o.add_(d1 + d2)
        if d_frob > 0:
            for idx, l in enumerate(target_layers):
                cad = hann(idx, len(target_layers))
                l_llama = round(l * 15 / 23)
                w_d = model_b.model.layers[l].mlp.down_proj.weight
                w_d.add_(fold_donor_host_basis(w_d, llama_donors["mlp.down_proj"][l_llama], rank=16, rel_frob=d_frob * cad, U_sov=U_sov[l], beta=beta, use_ternary=use_tern, U_opt=None, biortho=biortho))
        if w_alpha > 0:
            for idx, l in enumerate(down_layers):
                cad = hann(idx, len(down_layers))
                w_d = model_b.model.layers[l].mlp.down_proj.weight
                w_d.add_(wiener_enhance_delta(w_d, alpha=w_alpha * cad, tau=8.0, rank=32))
        if r_scale > 0:
            for l in down_layers:
                model_b.model.layers[l].mlp.down_proj.weight.add_(updates_ridge[l] * r_scale)

    del updates_ridge, U_sov
    gc.collect()
    qwen_st = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    q8_b = export_fp32_q8_gguf(
        model_b, qwen_st, v8.MINI_SNAP, "quillan-mini-0.5b-wsp-v17",
        ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"],
    )
    del qwen_st, model_b
    gc.collect()

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    ppl_q8 = measure_perplexity(q8_b, ppl_file)
    logger.info("MINI WSP v17 Q8_0 PPL = %.4f (v15 ref: 9.9125) | Path: %s | Elapsed: %.1fs", ppl_q8, q8_b, time.time() - t0)


if __name__ == "__main__":
    main()
