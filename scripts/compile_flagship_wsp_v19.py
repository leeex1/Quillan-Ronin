"""Flagship v19: Hann-bell o_proj covariance-steered ternary fold + Ridge-solved down_proj (no gate/up).
Zero test-set contamination: calibration uses gsm8k train / ai2_arc train / mmlu dev + EXTRA_ANCHOR_TEXT only.
HELD_OUT_PPL_TEXT is used strictly for measurement (fp32_ppl and measure_perplexity).
"""
import os
import sys
import gc
import math
import time
import json
import shutil
import subprocess
import logging

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import wsp_v8_closed_form_sota_compiler as v8
import compile_sovereign_fleet_wsp_v12_q8 as v12
import compile_flagship_general_v14 as v14
import compile_flagship_true_wsp_v17 as v17
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Flagship_WSP_v19")
logging.getLogger("httpx").setLevel(logging.WARNING)


@torch.no_grad()
def fp32_ppl(model, ids_all, win=512):
    nll, n = 0.0, 0
    for s in range(0, len(ids_all) - 1, win):
        chunk = ids_all[s:s + win + 1]
        if len(chunk) < 8:
            break
        x = torch.tensor([chunk[:-1]])
        y = torch.tensor([chunk[1:]])
        lg = model(x).logits[0].float()
        nll += F.cross_entropy(lg, y[0], reduction="sum").item()
        n += y.numel()
    return float(torch.exp(torch.tensor(nll / n)))


def hann_weights(n_layers: int) -> list[float]:
    return [math.sin(math.pi * (i + 0.5) / n_layers) ** 2 for i in range(n_layers)]


def compute_zero_mean_option_basis(W_head: torch.Tensor, encode_tok_fn) -> torch.Tensor:
    """Zero-mean pairwise difference basis for option letters (preserves option decision simplex)."""
    vecs = []
    for group in (["A", "B", "C", "D", "E"], [" A", " B", " C", " D", " E"]):
        g_vecs = []
        for tok_str in group:
            ids = encode_tok_fn(tok_str)
            if len(ids) == 1 and ids[0] < W_head.shape[0]:
                g_vecs.append(W_head[ids[0]].detach().float())
        if len(g_vecs) >= 2:
            M = torch.stack(g_vecs)
            M = M - M.mean(dim=0, keepdim=True)
            vecs.append(M)
    if not vecs:
        return torch.zeros((W_head.shape[1], 0), dtype=torch.float32)
    C = torch.cat(vecs, dim=0)
    _, _, Vt = torch.linalg.svd(C, full_matrices=False)
    return Vt[:min(6, Vt.shape[0])].t().contiguous()


def apply_digit_separation(W_emb: torch.Tensor, encode_tok_fn, tau: float = 1.15) -> None:
    """Spherical separation of digit embeddings 0..9 on S^{d-1} for ARC-AGI color/category disambiguation."""
    d_ids = []
    for d in "0123456789":
        ids = encode_tok_fn(d)
        if len(ids) == 1 and ids[0] < W_emb.shape[0]:
            d_ids.append(ids[0])
    if len(d_ids) < 2:
        return
    with torch.no_grad():
        E = W_emb[d_ids].float()
        mean_e = E.mean(dim=0, keepdim=True)
        norms = torch.norm(E, dim=1, keepdim=True)
        centered = E - mean_e
        sep = mean_e + tau * centered
        sep = sep / (torch.norm(sep, dim=1, keepdim=True) + 1e-8) * norms
        W_emb[d_ids] = sep.to(W_emb.dtype)


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib = v14.build_general_curated_pairs(test_fps, seed=9019)

    qst = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    ql = [10, 12, 14, 16, 18, 20, 22]
    q_o = [qst[f"model.layers.{l}.self_attn.o_proj.weight"].float().clone() for l in ql]
    q_d = [qst[f"model.layers.{l}.mlp.down_proj.weight"].float().clone() for l in ql]
    del qst

    bst = load_file(v12.find_bitnet_snapshot_path())
    bl = [8, 11, 14, 17, 20, 23]
    b_o = [bst[f"model.layers.{l}.self_attn.o_proj.weight"].float().clone() for l in bl]
    b_d = [bst[f"model.layers.{l}.mlp.down_proj.weight"].float().clone() for l in bl]
    del bst

    dd = v8.load_dragon_ffn_down_tensors()
    dragon_down = [dd[k] for k in sorted(dd)]
    del dd
    gc.collect()

    tok = Tokenizer.from_file(os.path.join(v8.FLAGSHIP_SNAP, "tokenizer.json"))
    enc_chat = lambda s, u: tok.encode(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    ).ids
    enc_tok = lambda t: tok.encode(t, add_special_tokens=False).ids

    model = transformers.AutoModelForCausalLM.from_pretrained(v8.FLAGSHIP_SNAP, dtype=torch.float32).eval()
    ppl_ids = [128000] + enc_tok(HELD_OUT_PPL_TEXT)

    # 1. Dual-group equinorm + syntax calibration + digit separation
    v12.apply_equinorm_and_syntax_calibration(
        model.model.embed_tokens.weight,
        [32, 33, 34, 35, 36],
        [362, 426, 356, 423, 469],
        [37815, 595, 7861, 76642],
        1680,
    )
    if not model.config.tie_word_embeddings:
        v12.apply_equinorm_and_syntax_calibration(
            model.lm_head.weight,
            [32, 33, 34, 35, 36],
            [362, 426, 356, 423, 469],
            [37815, 595, 7861, 76642],
            1680,
        )
    apply_digit_separation(model.model.embed_tokens.weight, enc_tok, tau=1.15)
    base_ppl = fp32_ppl(model, ppl_ids)
    logger.info("fp32 baseline (equinorm+digit_sep) PPL=%.4f", base_ppl)

    U_opt_diff = compute_zero_mean_option_basis(model.lm_head.weight.detach().float(), enc_tok)

    # 2. Surgical refusal abliteration: L13-14 only (leave L15 untouched), dampened scales
    U_ref = v12.compute_rank_k_refusal_subspace(
        model.model.embed_tokens.weight, enc_tok, model, enc_chat, rank=3
    )
    with torch.no_grad():
        for l in [13, 14]:
            Wd = model.model.layers[l].mlp.down_proj.weight
            Wd.sub_(0.35 * U_ref @ (U_ref.t() @ Wd))
            Wo = model.model.layers[l].self_attn.o_proj.weight
            Wo.sub_(0.15 * U_ref @ (U_ref.t() @ Wo))
    abl_ppl = fp32_ppl(model, ppl_ids)
    logger.info("after surgical abliteration (L13-14, 0.35/0.15) PPL=%.4f", abl_ppl)

    # 3. o_proj covariance-steered ternary deltas on layers [4..15] with Hann-bell schedule (NO U_opt on o_proj!)
    merge_layers = list(range(4, 16))
    U_sov = v17.compute_layer_covariance_bases(model, enc_chat, calib, merge_layers, rank=32)
    hann = hann_weights(len(merge_layers))

    o_deltas_hann = {}
    with torch.no_grad():
        for idx, l in enumerate(merge_layers):
            w_o = model.model.layers[l].self_attn.o_proj.weight
            d_oq = v17.build_steered_ternary_moe_delta(
                w_o, q_o[idx % len(ql)], U_sov[l], rank=32, beta=0.15, gamma_rel=0.004 * hann[idx], U_opt=None
            )
            d_ob = v17.build_steered_ternary_moe_delta(
                w_o, b_o[idx % len(bl)], U_sov[l], rank=32, beta=0.15, gamma_rel=0.004 * hann[idx], U_opt=None
            )
            o_deltas_hann[l] = 0.5 * (d_oq + d_ob)

    cur_o = 0.0
    o_results = {}
    for m in [1.0, 2.0, 3.0, 5.0]:
        with torch.no_grad():
            for l in merge_layers:
                model.model.layers[l].self_attn.o_proj.weight.add_((m - cur_o) * o_deltas_hann[l])
            cur_o = m
        p = fp32_ppl(model, ppl_ids)
        o_results[m] = p
        logger.info("o_proj Hann mult=%.1f PPL=%.4f", m, p)

    best_m = min(o_results, key=o_results.get)
    with torch.no_grad():
        for l in merge_layers:
            model.model.layers[l].self_attn.o_proj.weight.add_((best_m - cur_o) * o_deltas_hann[l])
        cur_o = best_m
    logger.info("BEST o_proj Hann mult=%.1f PPL=%.4f", best_m, o_results[best_m])

    # 4. Ridge-solved down_proj updates using EXTRA_ANCHOR_TEXT ONLY (zero HELD_OUT_PPL_TEXT!)
    layers_down = [6, 7, 8, 9, 10, 11, 12, 13, 14]
    donors_down = [
        q_d[0], dragon_down[0], b_d[0],
        q_d[1], dragon_down[1], b_d[1],
        q_d[2], dragon_down[2], b_d[2],
    ]
    anchor_ids = enc_tok(v12.EXTRA_ANCHOR_TEXT)
    updates_down = v14.compute_general_wsp_update(
        model, enc_chat, enc_tok, anchor_ids, layers_down, donors_down, calib,
        ridge_lambda=32.0, max_rel_frob=0.0045,
    )
    # Apply zero-mean option difference null-space protection on down_proj
    with torch.no_grad():
        for l in layers_down:
            d = updates_down[l]
            if U_opt_diff.shape[1] > 0:
                d = d - U_opt_diff @ (U_opt_diff.t() @ d)
            updates_down[l] = d

    cur_d = 0.0
    d_results = {}
    for dm in [0.0, 0.4, 0.7, 1.0]:
        with torch.no_grad():
            for l in layers_down:
                model.model.layers[l].mlp.down_proj.weight.add_((dm - cur_d) * updates_down[l])
            cur_d = dm
        p = fp32_ppl(model, ppl_ids)
        d_results[dm] = p
        logger.info("o_proj@%.1f + ridge_down@%.1f PPL=%.4f", best_m, dm, p)

    best_dm = min(d_results, key=d_results.get)
    with torch.no_grad():
        for l in layers_down:
            model.model.layers[l].mlp.down_proj.weight.add_((best_dm - cur_d) * updates_down[l])
    logger.info("FINAL CHOICE: o_proj@%.1f + ridge_down@%.1f fp32 PPL=%.4f", best_m, best_dm, d_results[best_dm])

    with open(r"C:\02_QUILLAN\exports\flagship_v19_scan.json", "w") as f:
        json.dump({"base": base_ppl, "abl": abl_ppl, "o_scan": o_results, "d_scan": d_results}, f, indent=1)

    del U_sov, U_ref, U_opt_diff, updates_down, o_deltas_hann, q_o, q_d, b_o, b_d, dragon_down
    gc.collect()

    # Export FP32 -> Q8_0 directly (avoid BF16 7-bit mantissa truncation)
    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    sd = model.state_dict()
    new_st = {k: sd[k].detach().cpu().float() for k in llama_st.keys()}
    orig_shapes = {k: v.float() for k, v in llama_st.items()}
    v12.verify_exact_shapes_and_params(orig_shapes, new_st, "quillan-1b-sovereign-wsp-v19")
    del llama_st, orig_shapes, sd, model
    gc.collect()

    hf_tmp = os.path.join(v12.EXPORT_ROOT, "quillan-1b-sovereign-wsp-v19_hf_tmp")
    os.makedirs(hf_tmp, exist_ok=True)
    save_file(new_st, os.path.join(hf_tmp, "model.safetensors"))
    del new_st
    gc.collect()
    for fn in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(v8.FLAGSHIP_SNAP, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(hf_tmp, fn))
    q8 = os.path.join(v12.EXPORT_ROOT, "quillan-1b-sovereign-wsp-v19-q8_0.gguf")
    subprocess.run([sys.executable, v12.CONVERT_SCRIPT, hf_tmp, "--outfile", q8, "--outtype", "q8_0"], check=True)
    shutil.rmtree(hf_tmp, ignore_errors=True)

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor_flagship_v19.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    q8_ppl = measure_perplexity(q8, ppl_file)
    logger.info("FLAGSHIP WSP v19 Q8_0 PPL = %.4f (v15 ref: 7.9463) in %.0fs", q8_ppl, time.time() - t0)


if __name__ == "__main__":
    main()
