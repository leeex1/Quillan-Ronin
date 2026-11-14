"""
=============================================================================
FLAGSHIP 1.2B SURGICAL PER-LAYER WSP COMPILER (v22)
=============================================================================
Built directly from `diagnose_layer_attribution.py` findings on Llama-3.2-1B:
  - L12 mlp.down_proj: worst [math] margin (-2.2920) & #2 [code] (-1.2698)
  - L11 & L14 mlp.down_proj: worst [logic_mc] margin (-3.5244, -3.1530)
  - L15 mlp.down_proj & o_proj: worst [code] (-1.3658) & [logic_mc] (-1.1141)
Key guarantees:
  1. Zero gradient descent, zero test-set calibration, Diff == 0 params.
  2. Sweep selection uses ONLY `v12.EXTRA_ANCHOR_TEXT` (HELD_OUT_PPL_TEXT is
     strictly only in final `measure_perplexity`).
  3. Preserves v20's o_proj Hann-bell covariance fold + Host-Basis SVD down_proj.
  4. Applies surgical single-layer null-space-projected edits ONLY to L11, L12,
     L14, L15 to remove preamble/docstring/step degradation.
=============================================================================
"""
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from typing import List

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
import wsp_v8_closed_form_sota_compiler as v8
import compile_sovereign_fleet_wsp_v12_q8 as v12
import compile_flagship_general_v14 as v14
import compile_flagship_true_wsp_v17 as v17
import compile_flagship_wsp_v20 as f20
import compile_mini_surgical_v21 as ms21
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Flagship_Surgical_v22")
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib = v14.build_general_curated_pairs(test_fps, seed=9022)

    qst = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    ql = [10, 12, 14, 16, 18, 20, 22]
    q_o = [qst[f"model.layers.{l}.self_attn.o_proj.weight"].float().clone() for l in ql]
    del qst
    bst = load_file(v12.find_bitnet_snapshot_path())
    bl = [8, 11, 14, 17, 20, 23]
    b_o = [bst[f"model.layers.{l}.self_attn.o_proj.weight"].float().clone() for l in bl]
    del bst
    gc.collect()

    tok_hf = transformers.AutoTokenizer.from_pretrained(v8.FLAGSHIP_SNAP, trust_remote_code=True)
    tok = Tokenizer.from_file(os.path.join(v8.FLAGSHIP_SNAP, "tokenizer.json"))
    enc_chat = lambda s, u: tok.encode(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    ).ids
    enc_tok = lambda t: tok.encode(t, add_special_tokens=False).ids

    model = transformers.AutoModelForCausalLM.from_pretrained(v8.FLAGSHIP_SNAP, dtype=torch.float32).eval()
    # Strictly use EXTRA_ANCHOR_TEXT for internal fp32 sweep (never HELD_OUT_PPL_TEXT)
    calib_ids = [128000] + enc_tok(v12.EXTRA_ANCHOR_TEXT)

    # 1. Equinorm + syntax + digit separation + non-prose verbosity dampening
    v12.apply_equinorm_and_syntax_calibration(
        model.model.embed_tokens.weight,
        [32, 33, 34, 35, 36],
        [362, 426, 356, 423, 469],
        [37815, 595, 7861, 76642],
        1680,
    )
    f20.apply_digit_separation(model.model.embed_tokens.weight, enc_tok, tau=1.15)
    f20.apply_verbosity_dampening(model.model.embed_tokens.weight, enc_tok, scale=0.75)
    with torch.no_grad():
        for s in ['"""', '    """', "'''", "##", " ##"]:
            ids = enc_tok(s)
            if len(ids) == 1:
                model.model.embed_tokens.weight.data[ids[0]] *= 0.78
    logger.info("fp32 baseline (EXTRA_ANCHOR_TEXT) PPL=%.4f", f20.fp32_ppl(model, calib_ids))

    # 2. Surgical refusal abliteration: L13-14 only (leave L15 untouched)
    U_ref = v12.compute_rank_k_refusal_subspace(model.model.embed_tokens.weight, enc_tok, model, enc_chat, rank=3)
    with torch.no_grad():
        for l in [13, 14]:
            Wd = model.model.layers[l].mlp.down_proj.weight
            Wd.sub_(0.35 * U_ref @ (U_ref.t() @ Wd))
            Wo = model.model.layers[l].self_attn.o_proj.weight
            Wo.sub_(0.15 * U_ref @ (U_ref.t() @ Wo))
    del U_ref

    # 3. o_proj Hann-bell covariance-steered ternary fold (L5..L13)
    merge_layers = list(range(5, 14))
    U_sov = v17.compute_layer_covariance_bases(model, enc_chat, calib, merge_layers, rank=32)
    hann = f20.hann_weights(len(merge_layers))
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
    o_results = {0.0: f20.fp32_ppl(model, calib_ids)}
    for m in [0.25, 0.5, 1.0]:
        with torch.no_grad():
            for l in merge_layers:
                model.model.layers[l].self_attn.o_proj.weight.add_((m - cur_o) * o_deltas[l])
            cur_o = m
        o_results[m] = f20.fp32_ppl(model, calib_ids)
    best_m = min(o_results, key=o_results.get)
    with torch.no_grad():
        for l in merge_layers:
            model.model.layers[l].self_attn.o_proj.weight.add_((best_m - cur_o) * o_deltas[l])
    del o_deltas
    logger.info("BEST o_proj Hann mult=%.2f calib PPL=%.4f", best_m, o_results[best_m])

    # 4. Host-Basis SVD Spectral Reweighting on mid down_proj [6..12]
    down_layers = list(range(6, 13))
    down_hann = f20.hann_weights(len(down_layers))
    down_deltas = {}
    decay = torch.exp(-torch.arange(24, dtype=torch.float32) / 8.0)
    with torch.no_grad():
        for idx, l in enumerate(down_layers):
            W = model.model.layers[l].mlp.down_proj.weight.detach().float()
            U, S, V = torch.svd_lowrank(W, q=24)
            dS = S * (decay * down_hann[idx])
            down_deltas[l] = (U * dS.unsqueeze(0)) @ V.t()

    cur_a = 0.0
    a_results = {0.0: o_results[best_m]}
    for alpha in [0.005, 0.010, 0.020, 0.035]:
        with torch.no_grad():
            for l in down_layers:
                model.model.layers[l].mlp.down_proj.weight.add_((alpha - cur_a) * down_deltas[l])
            cur_a = alpha
        a_results[alpha] = f20.fp32_ppl(model, calib_ids)
    best_a = min(a_results, key=a_results.get)
    with torch.no_grad():
        for l in down_layers:
            model.model.layers[l].mlp.down_proj.weight.add_((best_a - cur_a) * down_deltas[l])
    del down_deltas
    logger.info("BEST down_proj Host-SVD alpha=%.3f calib PPL=%.4f", best_a, a_results[best_a])

    # 5. Capture prose null-space V_a for surgical layers [11, 12, 14, 15]
    anchor_full = v12.EXTRA_ANCHOR_TEXT + "\n\n" + "\n\n".join(p["prompt"] + " " + p["chosen"] for p in calib[:20])
    act_d = {l: [] for l in (11, 12, 14, 15)}
    act_o = {15: []}
    hooks = []
    for l in (11, 12, 14, 15):
        def _mk_d(li):
            def _fn(mod, inp, out):
                act_d[li].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]))
            return _fn
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(_mk_d(l)))
    hooks.append(model.model.layers[15].self_attn.o_proj.register_forward_hook(
        lambda m, i, o: act_o[15].append(i[0].detach().float().reshape(-1, i[0].shape[-1]))
    ))
    with torch.no_grad():
        for p in [x.strip() for x in anchor_full.split("\n\n") if len(x.strip()) > 30][:20]:
            ids = enc_tok(p)[:128]
            if len(ids) >= 4:
                model(torch.tensor([ids], dtype=torch.long))
    for h in hooks:
        h.remove()

    V_a_d = {l: torch.svd_lowrank(torch.cat(act_d[l], dim=0), q=48, niter=2)[2].contiguous() for l in (11, 12, 14, 15)}
    V_a_o15 = torch.svd_lowrank(torch.cat(act_o[15], dim=0), q=48, niter=2)[2].contiguous()
    del act_d, act_o
    gc.collect()

    # 6. SURGICAL SINGLE-LAYER EDITS (L12 math/code, L11 & L14 logic_mc, L15 code/logic)
    E = model.model.embed_tokens.weight.detach().float()
    def _rows(strs):
        return torch.stack([E[enc_tok(s)[0]] for s in strs if len(enc_tok(s)) > 0], dim=0)

    math_prompts = [
        "A baker makes 12 loaves/hr for 3 hrs, sells 10. How many left?\n",
        "What is 15 * 4 + 12?\n",
    ]
    mc_prompts = [
        "Which option is prime?\nA) 9\nB) 11\nC) 15\nD) 21\nReply with 'Answer: <letter>'.\n",
        "All bloops are razzies. All razzies are lazzies. All bloops are:\nA) fast\nB) lazzies\nC) blue\nD) small\nReply with 'Answer: <letter>'.\n",
    ]
    code_prompts = [
        "Complete Python function body only:\ndef add_two(a: int, b: int) -> int:\n",
        "Complete Python function body only:\ndef is_even(n: int) -> bool:\n",
    ]

    # L12 down_proj: Math (-2.292) & Code (-1.270)
    X12 = ms21.capture_module_inputs(model, tok_hf, math_prompts + code_prompts, 12, "down_proj")
    d12 = ms21.surgical_single_layer_steering(
        model.model.layers[12].mlp.down_proj.weight.data,
        X12, V_a_d[12], _rows(["26", "72", "    return"]), _rows(["To", "Step", "##", '"""']),
        alpha_damp=0.18, rel_frob_cap=0.0018,
    )
    with torch.no_grad():
        model.model.layers[12].mlp.down_proj.weight.data.add_(d12)

    # L11 & L14 down_proj: Logic/MC (-3.524, -3.153)
    for l_mc, rf in ((11, 0.0016), (14, 0.0016)):
        X_mc = ms21.capture_module_inputs(model, tok_hf, mc_prompts, l_mc, "down_proj")
        d_mc = ms21.surgical_single_layer_steering(
            model.model.layers[l_mc].mlp.down_proj.weight.data,
            X_mc, V_a_d[l_mc], _rows(["Answer"]), _rows(["To", "Here", "Let"]),
            alpha_damp=0.15, rel_frob_cap=rf,
        )
        with torch.no_grad():
            model.model.layers[l_mc].mlp.down_proj.weight.data.add_(d_mc)

    # L15 down_proj & o_proj: Code (-1.366) & Logic/MC (-1.114)
    X15_d = ms21.capture_module_inputs(model, tok_hf, code_prompts + mc_prompts, 15, "down_proj")
    d15_d = ms21.surgical_single_layer_steering(
        model.model.layers[15].mlp.down_proj.weight.data,
        X15_d, V_a_d[15], _rows(["    return", "Answer"]), _rows(["Here", "To", '"""', "##"]),
        alpha_damp=0.16, rel_frob_cap=0.0015,
    )
    X15_o = ms21.capture_module_inputs(model, tok_hf, mc_prompts, 15, "o_proj")
    d15_o = ms21.surgical_single_layer_steering(
        model.model.layers[15].self_attn.o_proj.weight.data,
        X15_o, V_a_o15, _rows(["Answer"]), _rows(["To", "Here"]),
        alpha_damp=0.12, rel_frob_cap=0.0012,
    )
    with torch.no_grad():
        model.model.layers[15].mlp.down_proj.weight.data.add_(d15_d)
        model.model.layers[15].self_attn.o_proj.weight.data.add_(d15_o)

    logger.info("After surgical L11/L12/L14/L15 edits calib PPL=%.4f", f20.fp32_ppl(model, calib_ids))

    # Export FP32 -> Q8_0 directly
    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    sd = model.state_dict()
    new_st = {k: sd[k].detach().cpu().float() for k in llama_st.keys()}
    orig_shapes = {k: v.float() for k, v in llama_st.items()}
    v12.verify_exact_shapes_and_params(orig_shapes, new_st, "quillan-1b-sovereign-wsp-v22")
    del llama_st, orig_shapes, sd, model
    gc.collect()

    hf_tmp = os.path.join(v12.EXPORT_ROOT, "quillan-1b-sovereign-wsp-v22_hf_tmp")
    os.makedirs(hf_tmp, exist_ok=True)
    save_file(new_st, os.path.join(hf_tmp, "model.safetensors"))
    del new_st
    gc.collect()
    for fn in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(v8.FLAGSHIP_SNAP, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(hf_tmp, fn))
    q8 = os.path.join(v12.EXPORT_ROOT, "quillan-1b-sovereign-wsp-v22-q8_0.gguf")
    subprocess.run([sys.executable, v12.CONVERT_SCRIPT, hf_tmp, "--outfile", q8, "--outtype", "q8_0"], check=True)
    shutil.rmtree(hf_tmp, ignore_errors=True)

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor_flagship_v22.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    q8_ppl = measure_perplexity(q8, ppl_file)
    logger.info("FLAGSHIP SURGICAL v22 Q8_0 PPL = %.4f (v20 ref: 7.9088, v15 ref: 7.9463) in %.0fs", q8_ppl, time.time() - t0)


if __name__ == "__main__":
    main()
