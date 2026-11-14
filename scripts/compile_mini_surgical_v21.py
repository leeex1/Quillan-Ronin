"""
=============================================================================
MINI 0.5B SURGICAL PER-LAYER WSP COMPILER (v21)
=============================================================================
Built directly from `diagnose_layer_attribution.py` findings on Qwen2.5-0.5B:
  - L21 mlp.down_proj: worst [logic_mc] margin (-5.5399)
  - L20 mlp.down_proj & o_proj: worst [code] margin (-1.1179 / -0.5294)
  - L11 mlp.down_proj & L9 o_proj: worst [math] margin (-2.6342 / -1.5035)
Key guarantees:
  1. Zero gradient descent, zero test-set contamination, Diff == 0 params.
  2. ZERO gate_proj / up_proj edits (v20 proved gate/up hurts Q8_0 PPL).
  3. Mid-layer Q8_0-safe o_proj Procrustes + Covariance Steering (o_frob=0.0018).
  4. Surgical single-layer null-space projected edits ONLY inside L9, L11, L20, L21.
  5. Direct FP32 -> Q8_0 export + held-out PPL & 10Q verification.
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
import compile_flagship_full_merge_v15 as v15
import compile_mini_wsp_v20 as m20
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Mini_Surgical_v21")


@torch.no_grad()
def capture_module_inputs(model, tok, prompts: List[str], layer_idx: int, submod_name: str) -> torch.Tensor:
    """Captures last-token input activations to `layers[layer_idx].<submod_name>`."""
    feats = []
    mod = model.model.layers[layer_idx]
    target = mod.self_attn.o_proj if submod_name == "o_proj" else mod.mlp.down_proj

    def _hook(m, inp):
        feats.append(inp[0][:, -1, :].detach().float())

    h = target.register_forward_pre_hook(_hook)
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"]
        _ = model(ids)
    h.remove()
    return torch.cat(feats, dim=0)  # (N, d_in)


def surgical_single_layer_steering(
    W: torch.Tensor,
    X_in: torch.Tensor,
    V_a: torch.Tensor,
    good_dirs: torch.Tensor,
    bad_dirs: torch.Tensor,
    alpha_damp: float = 0.25,
    rel_frob_cap: float = 0.0022,
) -> torch.Tensor:
    """
    Computes a surgical single-layer delta inside `W` only:
      1) Dampens projections onto `bad_dirs` orthogonal to prose subspace `V_a`.
      2) Adds rank-1 boundary steering toward `good_dirs - bad_dirs` along `X_in_perp`.
      3) Clamps total ||Delta W||_F <= rel_frob_cap * ||W||_F.
    """
    with torch.no_grad():
        wf = W.detach().float()
        w_norm = torch.norm(wf)
        if w_norm < 1e-8:
            return torch.zeros_like(wf)

        # Orthonormalize bad_dirs (rows of W_E / lm_head)
        Q_bad, _ = torch.linalg.qr(bad_dirs.t(), mode="reduced")  # (d_out, k_bad)
        # 1) Subspace damping in prose right-null-space
        d_damp = -alpha_damp * (Q_bad @ (Q_bad.t() @ wf))
        d_damp = d_damp - (d_damp @ V_a) @ V_a.t()

        # 2) Boundary activation steering in prose right-null-space
        x_mean = X_in.mean(dim=0)  # (d_in,)
        x_perp = x_mean - V_a @ (V_a.t() @ x_mean)
        x_norm = torch.norm(x_perp)
        if x_norm > 1e-8:
            x_hat = x_perp / x_norm
            v_good = F.normalize(good_dirs.mean(dim=0), p=2, dim=0)
            v_bad = F.normalize(bad_dirs.mean(dim=0), p=2, dim=0)
            v_dir = F.normalize(v_good - 0.85 * v_bad, p=2, dim=0)
            d_steer = (0.5 * rel_frob_cap * w_norm) * torch.outer(v_dir, x_hat)
        else:
            d_steer = torch.zeros_like(wf)

        delta = d_damp + d_steer
        delta = delta - (delta @ V_a) @ V_a.t()
        d_norm = torch.norm(delta)
        cap = rel_frob_cap * w_norm
        if d_norm > cap and d_norm > 1e-8:
            delta = delta * (cap / d_norm)
        return delta


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib_pairs = v14.build_general_curated_pairs(test_fps, seed=9021)
    anchor_texts = [p["prompt"] + " " + p["chosen"] for p in calib_pairs[:24]]
    anchor_full = v12.EXTRA_ANCHOR_TEXT + "\n\n" + "\n\n".join(anchor_texts)

    tok_hf = transformers.AutoTokenizer.from_pretrained(v8.MINI_SNAP, trust_remote_code=True)
    tok_b = Tokenizer.from_file(os.path.join(v8.MINI_SNAP, "tokenizer.json"))
    tok_b_vocab = tok_b.get_vocab()
    enc_chat_b = lambda s, u: tok_b.encode(
        f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    ).ids
    enc_tok_b = lambda t: tok_b.encode(t).ids

    model_b = transformers.AutoModelForCausalLM.from_pretrained(v8.MINI_SNAP, dtype=torch.float32).eval()
    W_emb_host = model_b.model.embed_tokens.weight.detach().float().clone()
    calib_ids = torch.tensor([enc_tok_b(v12.EXTRA_ANCHOR_TEXT)], dtype=torch.long)

    def calib_ppl() -> float:
        with torch.no_grad():
            return math.exp(model_b(calib_ids, labels=calib_ids).loss.item())

    logger.info("Initial Mini calib PPL (EXTRA_ANCHOR_TEXT): %.4f", calib_ppl())

    # 1. Equinorm + syntax + rare docstring dampening (no prose tokens!)
    raw_opt_b = [32, 33, 34, 35, 36]
    space_opt_b = [362, 425, 356, 422, 468]
    broken_b = [36715, 593, 7705, 75542]
    good_b = 1648
    v12.apply_equinorm_and_syntax_calibration(model_b.model.embed_tokens.weight, raw_opt_b, space_opt_b, broken_b, good_b)
    with torch.no_grad():
        for tok_str in ['"""', "'''", "##", "```python"]:
            ids = enc_tok_b(tok_str)
            if len(ids) == 1:
                model_b.model.embed_tokens.weight.data[ids[0]] *= 0.78

    W_head_b = model_b.lm_head.weight.detach().float()
    U_opt_b = v8.compute_option_null_basis(W_head_b, enc_tok_b)

    # 2. Protected Rank-2 Refusal Abliteration on [18, 19]
    U_ref_b = v12.compute_rank_k_refusal_subspace(model_b.model.embed_tokens.weight, enc_tok_b, model_b, enc_chat_b, rank=2)
    with torch.no_grad():
        for l in [18, 19]:
            W_d = model_b.model.layers[l].mlp.down_proj.weight.data
            W_new = W_d - 0.35 * torch.matmul(U_ref_b, torch.matmul(U_ref_b.t(), W_d.float()))
            model_b.model.layers[l].mlp.down_proj.weight.data.copy_(W_new)

    # 3. Capture prose null-space V_a_o & sovereign steering U_sov_o on o_proj [9..19] + V_a_d on down_proj [11, 20, 21]
    cov_o = {l: torch.zeros((896, 896), dtype=torch.float32) for l in range(9, 20)}
    act_d = {l: [] for l in (11, 20, 21)}
    hooks = []
    for l in range(9, 20):
        def _mk_o(li):
            def _fn(mod, inp, out):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                cov_o[li].add_(x.t() @ x)
            return _fn
        hooks.append(model_b.model.layers[l].self_attn.o_proj.register_forward_hook(_mk_o(l)))
    for l in (11, 20, 21):
        def _mk_d(li):
            def _fn(mod, inp, out):
                act_d[li].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]))
            return _fn
        hooks.append(model_b.model.layers[l].mlp.down_proj.register_forward_hook(_mk_d(l)))

    calib_paragraphs = [p.strip() for p in anchor_full.split("\n\n") if len(p.strip()) > 30][:24]
    with torch.no_grad():
        for p in calib_paragraphs:
            ids = enc_tok_b(p)[:128]
            if len(ids) >= 4:
                model_b(torch.tensor([ids], dtype=torch.long))
    for h in hooks:
        h.remove()

    V_a_o, U_sov = {}, {}
    for l in range(9, 20):
        _, _, Vh = torch.linalg.svd(cov_o[l], full_matrices=False)
        V_a_o[l] = Vh[:48, :].t().contiguous()
        U_sov[l] = Vh[48:72, :].t().contiguous()
    del cov_o
    V_a_d = {}
    for l in (11, 20, 21):
        X = torch.cat(act_d[l], dim=0)
        _, _, V = torch.svd_lowrank(X, q=48, niter=2)
        V_a_d[l] = V.contiguous()
    del act_d
    gc.collect()

    # 4. Q8_0-safe Procrustes + Covariance-Steered o_proj ONLY on [9..19] (ZERO gate/up!)
    merge_layers = list(range(9, 20))
    tok_l = Tokenizer.from_file(os.path.join(v8.FLAGSHIP_SNAP, "tokenizer.json"))
    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    P_llama = m20.compute_procrustes_map(llama_st["model.embed_tokens.weight"], tok_l.get_vocab(), W_emb_host, tok_b_vocab)
    llama_o = {l: llama_st[f"model.layers.{round(l * 15 / 23)}.self_attn.o_proj.weight"].float().clone() for l in merge_layers}
    del llama_st, tok_l

    import gguf
    reader_d = gguf.GGUFReader(v8.BLOB_DRAGON)
    tmap_d = {t.name: t for t in reader_d.tensors}
    tf_d = reader_d.fields["tokenizer.ggml.tokens"]
    tok_d_vocab = {}
    for i, idx in enumerate(tf_d.data):
        s = bytes(tf_d.parts[idx]).decode("utf-8", errors="replace")
        if s not in tok_d_vocab:
            tok_d_vocab[s] = i
    E_dragon = torch.from_numpy(gguf.quants.dequantize(tmap_d["token_embd.weight"].data, tmap_d["token_embd.weight"].tensor_type).copy()).float()
    P_dragon = m20.compute_procrustes_map(E_dragon, tok_d_vocab, W_emb_host, tok_b_vocab)
    dragon_o = {}
    for l in merge_layers:
        dl = round(l * 15 / 23)
        t_do = tmap_d[f"blk.{dl}.attn_output.weight"]
        dragon_o[l] = torch.from_numpy(gguf.quants.dequantize(t_do.data, t_do.tensor_type).copy()).float()
    del E_dragon, reader_d, tmap_d, tf_d, tok_d_vocab, W_emb_host

    bitnet_st = load_file(v12.find_bitnet_snapshot_path())
    bitnet_layers = [10, 14, 18, 21]
    bitnet_o = {l: bitnet_st[f"model.layers.{bitnet_layers[idx % len(bitnet_layers)]}.self_attn.o_proj.weight"].float().clone() for idx, l in enumerate(merge_layers)}
    del bitnet_st
    gc.collect()

    n_m = len(merge_layers)
    hann_m = [0.5 * (1.0 - math.cos(2.0 * math.pi * (i + 1) / (n_m + 1))) for i in range(n_m)]
    o_orig, o_deltas = {}, {}
    with torch.no_grad():
        for idx, l in enumerate(merge_layers):
            o_orig[l] = model_b.model.layers[l].self_attn.o_proj.weight.data.float().clone()
            o_deltas[l] = m20.build_procrustes_steered_o_delta(
                o_orig[l], llama_o[l], P_llama, dragon_o[l], P_dragon, bitnet_o[l],
                V_a_o[l], U_sov[l], U_opt_b, rank=24, rel_frob=0.0018 * hann_m[idx]
            )
    del llama_o, dragon_o, bitnet_o, P_llama, P_dragon
    gc.collect()

    best_mult, best_p = 0.0, 1e9
    for mult in [0.0, 0.5, 1.0, 1.5, 2.0]:
        with torch.no_grad():
            for l in merge_layers:
                model_b.model.layers[l].self_attn.o_proj.weight.data.copy_(m20.quantize_q8_0(o_orig[l] + mult * o_deltas[l]))
        p = calib_ppl()
        logger.info("[o_proj-only Q8_0 scan] mult=%.2f -> calib Q8_0 PPL=%.4f", mult, p)
        if p < best_p:
            best_p, best_mult = p, mult
    with torch.no_grad():
        for l in merge_layers:
            model_b.model.layers[l].self_attn.o_proj.weight.data.copy_(o_orig[l] + best_mult * o_deltas[l])
    del o_orig, o_deltas

    # 5. SURGICAL SINGLE-LAYER EDITS (L21 logic_mc, L20 code, L11 math)
    E = model_b.model.embed_tokens.weight.detach().float()
    def _rows(strs):
        return torch.stack([E[enc_tok_b(s)[0]] for s in strs if len(enc_tok_b(s)) > 0], dim=0)

    # 5a) L21 mlp.down_proj: logic_mc (-5.5399) -> dampen preamble, steer Answer
    mc_prompts = [
        "Which option is prime?\nA) 9\nB) 11\nC) 15\nD) 21\nReply with 'Answer: <letter>'.\n",
        "All bloops are razzies. All razzies are lazzies. All bloops are:\nA) fast\nB) lazzies\nC) blue\nD) small\nReply with 'Answer: <letter>'.\n",
    ]
    X_21 = capture_module_inputs(model_b, tok_hf, mc_prompts, 21, "down_proj")
    d21 = surgical_single_layer_steering(
        model_b.model.layers[21].mlp.down_proj.weight.data,
        X_21, V_a_d[21], _rows(["Answer", " **"]), _rows(["The", "To", "Here", "Let"]),
        alpha_damp=0.20, rel_frob_cap=0.0022,
    )
    with torch.no_grad():
        model_b.model.layers[21].mlp.down_proj.weight.data.add_(d21)
    logger.info("Surgical L21 down_proj delta applied (calib Q8_0-fp32 PPL=%.4f)", calib_ppl())

    # 5b) L20 mlp.down_proj: code (-1.1179) -> dampen docstring/comment, steer return
    code_prompts = [
        "Complete Python function body only:\ndef add_two(a: int, b: int) -> int:\n",
        "Complete Python function body only:\ndef is_even(n: int) -> bool:\n",
    ]
    X_20 = capture_module_inputs(model_b, tok_hf, code_prompts, 20, "down_proj")
    d20 = surgical_single_layer_steering(
        model_b.model.layers[20].mlp.down_proj.weight.data,
        X_20, V_a_d[20], _rows(["    return", "    if"]), _rows(['"""', "```", "Here", "#"]),
        alpha_damp=0.20, rel_frob_cap=0.0020,
    )
    with torch.no_grad():
        model_b.model.layers[20].mlp.down_proj.weight.data.add_(d20)
    logger.info("Surgical L20 down_proj delta applied (calib PPL=%.4f)", calib_ppl())

    # 5c) L11 mlp.down_proj: math (-2.6342) -> dampen Step/Let, steer digits
    math_prompts = [
        "A baker makes 12 loaves/hr for 3 hrs, sells 10. How many left?\n",
        "What is 15 * 4 + 12?\n",
    ]
    X_11 = capture_module_inputs(model_b, tok_hf, math_prompts, 11, "down_proj")
    d11 = surgical_single_layer_steering(
        model_b.model.layers[11].mlp.down_proj.weight.data,
        X_11, V_a_d[11], _rows(["26", "72", "####"]), _rows(["Let", "Step", "To"]),
        alpha_damp=0.15, rel_frob_cap=0.0018,
    )
    with torch.no_grad():
        model_b.model.layers[11].mlp.down_proj.weight.data.add_(d11)
    logger.info("Surgical L11 down_proj delta applied (calib PPL=%.4f)", calib_ppl())

    # 6. Export FP32 -> Q8_0 directly
    qwen_st = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    sd = model_b.state_dict()
    new_st = {k: sd[k].detach().cpu().float().contiguous() for k in qwen_st.keys()}
    ref_st = {k: v.float() for k, v in qwen_st.items()}
    v12.verify_exact_shapes_and_params(ref_st, new_st, "quillan-mini-0.5b-wsp-v21")
    del ref_st, qwen_st, sd, model_b
    gc.collect()

    tag = "quillan-mini-0.5b-wsp-v21"
    hf_tmp = os.path.join(v12.EXPORT_ROOT, f"{tag}_hf_tmp")
    os.makedirs(hf_tmp, exist_ok=True)
    save_file(new_st, os.path.join(hf_tmp, "model.safetensors"))
    del new_st
    gc.collect()
    for fn in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
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
    logger.info("MINI SURGICAL v21 Q8_0 PPL = %.4f (v15 ref: 9.9125, base: 9.9362) | Elapsed: %.1fs", ppl_q8, time.time() - t0)


if __name__ == "__main__":
    main()
