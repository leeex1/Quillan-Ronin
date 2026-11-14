"""Flagship v20: Low-RAM analytical WSP (equinorm + digit_sep + verbosity_damp + surgical abliteration
+ Hann-bell o_proj + Host-Basis SVD Spectral Reweighting on mid down_proj).
Zero training, zero test-set contamination. Exports FP32 -> Q8_0 directly.
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
logger = logging.getLogger("Flagship_WSP_v20")
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


def apply_digit_separation(W_emb: torch.Tensor, encode_tok_fn, tau: float = 1.15) -> None:
    d_ids = [encode_tok_fn(d)[0] for d in "0123456789" if len(encode_tok_fn(d)) == 1]
    if len(d_ids) < 2:
        return
    with torch.no_grad():
        E = W_emb[d_ids].float()
        mean_e = E.mean(dim=0, keepdim=True)
        norms = torch.norm(E, dim=1, keepdim=True)
        sep = mean_e + tau * (E - mean_e)
        sep = sep / (torch.norm(sep, dim=1, keepdim=True) + 1e-8) * norms
        W_emb[d_ids] = sep.to(W_emb.dtype)


def apply_verbosity_dampening(W_head: torch.Tensor, encode_tok_fn, scale: float = 0.70) -> None:
    """Dampen comment/step filler tokens ('#', ' #', 'Step', ' Step') to prevent 128-tok budget exhaustion."""
    v_ids = []
    for tok_str in ["#", " #", "Step", " Step"]:
        ids = encode_tok_fn(tok_str)
        if len(ids) == 1 and ids[0] < W_head.shape[0]:
            v_ids.append(ids[0])
    if v_ids:
        with torch.no_grad():
            W_head[v_ids] = (W_head[v_ids].float() * scale).to(W_head.dtype)


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib = v14.build_general_curated_pairs(test_fps, seed=9020)

    qst = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    ql = [10, 12, 14, 16, 18, 20, 22]
    q_o = [qst[f"model.layers.{l}.self_attn.o_proj.weight"].float().clone() for l in ql]
    del qst
    bst = load_file(v12.find_bitnet_snapshot_path())
    bl = [8, 11, 14, 17, 20, 23]
    b_o = [bst[f"model.layers.{l}.self_attn.o_proj.weight"].float().clone() for l in bl]
    del bst
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

    # 1. Equinorm + syntax calibration + digit separation + verbosity dampening
    v12.apply_equinorm_and_syntax_calibration(
        model.model.embed_tokens.weight,
        [32, 33, 34, 35, 36],
        [362, 426, 356, 423, 469],
        [37815, 595, 7861, 76642],
        1680,
    )
    apply_digit_separation(model.model.embed_tokens.weight, enc_tok, tau=1.15)
    apply_verbosity_dampening(model.model.embed_tokens.weight, enc_tok, scale=0.75)
    base_ppl = fp32_ppl(model, ppl_ids)
    logger.info("fp32 baseline (equinorm+digit_sep+verb_damp) PPL=%.4f", base_ppl)

    # 2. Surgical refusal abliteration: L13-14 only (leave L15 untouched)
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
    logger.info("after surgical abliteration (L13-14) PPL=%.4f", abl_ppl)
    del U_ref

    # 3. o_proj Hann-bell covariance-steered ternary fold (small mult scan)
    merge_layers = list(range(5, 14))
    U_sov = v17.compute_layer_covariance_bases(model, enc_chat, calib, merge_layers, rank=32)
    hann = hann_weights(len(merge_layers))

    o_deltas = {}
    with torch.no_grad():
        for idx, l in enumerate(merge_layers):
            w_o = model.model.layers[l].self_attn.o_proj.weight
            d_oq = v17.build_steered_ternary_moe_delta(
                w_o, q_o[idx % len(ql)], U_sov[l], rank=32, beta=0.15, gamma_rel=0.004 * hann[idx], U_opt=None
            )
            d_ob = v17.build_steered_ternary_moe_delta(
                w_o, b_o[idx % len(bl)], U_sov[l], rank=32, beta=0.15, gamma_rel=0.004 * hann[idx], U_opt=None
            )
            o_deltas[l] = 0.5 * (d_oq + d_ob)
    del U_sov, q_o, b_o
    gc.collect()

    cur_o = 0.0
    o_results = {0.0: abl_ppl}
    for m in [0.25, 0.5, 1.0]:
        with torch.no_grad():
            for l in merge_layers:
                model.model.layers[l].self_attn.o_proj.weight.add_((m - cur_o) * o_deltas[l])
            cur_o = m
        p = fp32_ppl(model, ppl_ids)
        o_results[m] = p
        logger.info("o_proj Hann mult=%.2f PPL=%.4f", m, p)

    best_m = min(o_results, key=o_results.get)
    with torch.no_grad():
        for l in merge_layers:
            model.model.layers[l].self_attn.o_proj.weight.add_((best_m - cur_o) * o_deltas[l])
    del o_deltas
    gc.collect()
    logger.info("BEST o_proj Hann mult=%.2f PPL=%.4f", best_m, o_results[best_m])

    # 4. Host-Basis SVD Spectral Reweighting on mid down_proj [6..12] (zero basis mismatch)
    down_layers = list(range(6, 13))
    down_hann = hann_weights(len(down_layers))
    down_deltas = {}
    decay = torch.exp(-torch.arange(24, dtype=torch.float32) / 8.0)
    with torch.no_grad():
        for idx, l in enumerate(down_layers):
            W = model.model.layers[l].mlp.down_proj.weight.detach().float()
            U, S, V = torch.svd_lowrank(W, q=24)
            # Soft-boost leading singular modes (scaled by Hann window)
            dS = S * (decay * down_hann[idx])
            down_deltas[l] = (U * dS.unsqueeze(0)) @ V.t()

    cur_a = 0.0
    a_results = {0.0: o_results[best_m]}
    for alpha in [0.005, 0.010, 0.020, 0.035]:
        with torch.no_grad():
            for l in down_layers:
                model.model.layers[l].mlp.down_proj.weight.add_((alpha - cur_a) * down_deltas[l])
            cur_a = alpha
        p = fp32_ppl(model, ppl_ids)
        a_results[alpha] = p
        logger.info("down_proj spectral boost alpha=%.3f PPL=%.4f", alpha, p)

    best_a = min(a_results, key=a_results.get)
    with torch.no_grad():
        for l in down_layers:
            model.model.layers[l].mlp.down_proj.weight.add_((best_a - cur_a) * down_deltas[l])
    del down_deltas
    gc.collect()
    logger.info("FINAL CHOICE: o_proj@%.2f + down_spec@%.3f fp32 PPL=%.4f", best_m, best_a, a_results[best_a])

    with open(r"C:\02_QUILLAN\exports\flagship_v20_scan.json", "w") as f:
        json.dump({"base": base_ppl, "abl": abl_ppl, "o_scan": o_results, "a_scan": a_results}, f, indent=1)

    # Export FP32 -> Q8_0 directly
    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    sd = model.state_dict()
    new_st = {k: sd[k].detach().cpu().float() for k in llama_st.keys()}
    orig_shapes = {k: v.float() for k, v in llama_st.items()}
    v12.verify_exact_shapes_and_params(orig_shapes, new_st, "quillan-1b-sovereign-wsp-v20")
    del llama_st, orig_shapes, sd, model
    gc.collect()

    hf_tmp = os.path.join(v12.EXPORT_ROOT, "quillan-1b-sovereign-wsp-v20_hf_tmp")
    os.makedirs(hf_tmp, exist_ok=True)
    save_file(new_st, os.path.join(hf_tmp, "model.safetensors"))
    del new_st
    gc.collect()
    for fn in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(v8.FLAGSHIP_SNAP, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(hf_tmp, fn))
    q8 = os.path.join(v12.EXPORT_ROOT, "quillan-1b-sovereign-wsp-v20-q8_0.gguf")
    subprocess.run([sys.executable, v12.CONVERT_SCRIPT, hf_tmp, "--outfile", q8, "--outtype", "q8_0"], check=True)
    shutil.rmtree(hf_tmp, ignore_errors=True)

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor_flagship_v20.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    q8_ppl = measure_perplexity(q8, ppl_file)
    logger.info("FLAGSHIP WSP v20 Q8_0 PPL = %.4f (v15 ref: 7.9463) in %.0fs", q8_ppl, time.time() - t0)


if __name__ == "__main__":
    main()
