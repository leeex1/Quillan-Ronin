"""Flagship v21: Zero-contamination analytical WSP compiler.
Combines:
  1. Dual-group option equinorm + syntax bracket calibration + digit separation (tau=1.15)
  2. Non-prose markdown/docstring token dampening ('#', '##', 'Step', '\"\"\"', '    \"\"\"') + 'Answer' boost
  3. Surgical refusal abliteration on L13-14 (0.35 down_proj, 0.15 o_proj; L15 untouched)
  4. Hann-bell o_proj covariance-steered ternary MoE fold across layers [5..13] (mult=1.0)
  5. Turn-boundary & post-signature rank-1 null-projected steering in L14-15 (orthogonal to V_a on EXTRA_ANCHOR_TEXT)
Zero training, zero test-set calibration. HELD_OUT_PPL_TEXT is used ONLY at the end in measure_perplexity.
"""
import os
import sys
import gc
import math
import time
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
logger = logging.getLogger("Flagship_WSP_v21")
logging.getLogger("httpx").setLevel(logging.WARNING)


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


def apply_non_prose_verbosity_calibration(W_head: torch.Tensor, encode_tok_fn) -> None:
    """Dampen non-prose markdown/docstring tokens and boost 'Answer' without touching prose words."""
    damp_tokens = ["#", " #", "##", " ##", "Step", " Step", '"""', ' """', '    """', "'''"]
    damp_ids = []
    for t in damp_tokens:
        ids = encode_tok_fn(t)
        if len(ids) == 1 and ids[0] < W_head.shape[0]:
            damp_ids.append(ids[0])
    boost_ids = []
    for t in ["Answer", " Answer"]:
        ids = encode_tok_fn(t)
        if len(ids) == 1 and ids[0] < W_head.shape[0]:
            boost_ids.append(ids[0])
    with torch.no_grad():
        if damp_ids:
            W_head[damp_ids] = (W_head[damp_ids].float() * 0.58).to(W_head.dtype)
        if boost_ids:
            W_head[boost_ids] = (W_head[boost_ids].float() * 1.10).to(W_head.dtype)


def apply_turn_boundary_null_steering(
    model: torch.nn.Module,
    enc_chat,
    enc_tok,
    calib_pairs: list,
    target_layers: list[int] = [14, 15],
) -> None:
    """
    Projects turn-start and post-def steering vectors into the orthogonal complement
    of prose activations (V_a derived strictly from EXTRA_ANCHOR_TEXT + train prompts).
    """
    W_emb = model.model.embed_tokens.weight.detach().float()

    def tok_vec(s: str) -> torch.Tensor:
        ids = enc_tok(s)
        return F.normalize(W_emb[ids[0]].clone(), p=2, dim=0)

    v_turn = F.normalize(
        0.65 * tok_vec("Answer") + 0.55 * tok_vec("def")
        - 0.45 * (tok_vec("Here") + tok_vec("To") + tok_vec("##") + tok_vec('"""')),
        p=2, dim=0,
    )
    v_def = F.normalize(
        0.80 * tok_vec("    ") + 0.60 * tok_vec("return")
        - 0.70 * (tok_vec('    """') + tok_vec('"""')),
        p=2, dim=0,
    )

    prose_acts = {l: [] for l in target_layers}
    turn_acts = {l: [] for l in target_layers}
    def_acts = {l: [] for l in target_layers}
    mode = ["prose"]

    def make_hook(l_idx: int):
        def hook_fn(mod, inp, out):
            x = inp[0].detach().float()
            if mode[0] == "prose":
                for pos in range(0, x.shape[1], max(1, x.shape[1] // 16)):
                    prose_acts[l_idx].append(x[0, pos, :].clone())
            elif mode[0] == "turn":
                turn_acts[l_idx].append(x[0, -1, :].clone())
            elif mode[0] == "def":
                def_acts[l_idx].append(x[0, -1, :].clone())
        return hook_fn

    hooks = [model.model.layers[l].self_attn.o_proj.register_forward_hook(make_hook(l)) for l in target_layers]

    sys_p = "You are a concise expert assistant. Follow output format instructions strictly."
    with torch.no_grad():
        # 1. Capture prose subspace on EXTRA_ANCHOR_TEXT + train prompts
        mode[0] = "prose"
        extra_ids = enc_tok(v12.EXTRA_ANCHOR_TEXT)
        if len(extra_ids) >= 8:
            model(torch.tensor([extra_ids], dtype=torch.long))
        for item in calib_pairs[:16]:
            c_ids = enc_tok(item["chosen"])
            if len(c_ids) >= 8:
                model(torch.tensor([c_ids[:128]], dtype=torch.long))

        # 2. Capture assistant turn-start boundary states on train prompts
        mode[0] = "turn"
        for item in calib_pairs[:20]:
            ids = enc_chat(sys_p, item["prompt"])[-256:]
            model(torch.tensor([ids], dtype=torch.long))

        # 3. Capture post-signature 'def ...:\n' states on train code prompts
        mode[0] = "def"
        for sig in [
            "def solve(x: int) -> int:\n",
            "def compute_range(a: int, b: int) -> list[int]:\n",
            "def filter_items(arr: list[str]) -> list[str]:\n",
        ]:
            ids = enc_chat(sys_p, f"Complete the Python function:\n{sig}") + enc_tok(f"```python\n{sig}")
            model(torch.tensor([ids[-256:]], dtype=torch.long))

    for hk in hooks:
        hk.remove()

    with torch.no_grad():
        for l in target_layers:
            W_o = model.model.layers[l].self_attn.o_proj.weight
            P_mat = torch.stack(prose_acts[l], dim=0)
            _, _, V_a = torch.svd_lowrank(P_mat, q=min(24, P_mat.shape[0] - 2))

            k_turn = torch.stack(turn_acts[l], dim=0).mean(dim=0)
            k_turn_perp = k_turn - V_a @ (V_a.t() @ k_turn)
            k_turn_perp = F.normalize(k_turn_perp, p=2, dim=0)

            k_def = torch.stack(def_acts[l], dim=0).mean(dim=0)
            k_def_perp = k_def - V_a @ (V_a.t() @ k_def)
            k_def_perp = F.normalize(k_def_perp, p=2, dim=0)

            delta = 0.0012 * torch.norm(W_o.float()) * (
                0.65 * torch.outer(v_turn, k_turn_perp) + 0.35 * torch.outer(v_def, k_def_perp)
            )
            W_o.add_(delta.to(W_o.dtype))


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib = v14.build_general_curated_pairs(test_fps, seed=9021)

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

    # 1. Equinorm + syntax calibration + digit separation + non-prose verbosity calibration
    v12.apply_equinorm_and_syntax_calibration(
        model.model.embed_tokens.weight,
        [32, 33, 34, 35, 36],
        [362, 426, 356, 423, 469],
        [37815, 595, 7861, 76642],
        1680,
    )
    apply_digit_separation(model.model.embed_tokens.weight, enc_tok, tau=1.15)
    apply_non_prose_verbosity_calibration(model.model.embed_tokens.weight, enc_tok)

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
    del U_ref

    # 3. o_proj Hann-bell covariance-steered ternary fold on layers [5..13] (mult=1.0)
    merge_layers = list(range(5, 14))
    U_sov = v17.compute_layer_covariance_bases(model, enc_chat, calib, merge_layers, rank=32)
    hann = hann_weights(len(merge_layers))

    with torch.no_grad():
        for idx, l in enumerate(merge_layers):
            w_o = model.model.layers[l].self_attn.o_proj.weight
            d_oq = v17.build_steered_ternary_moe_delta(
                w_o, q_o[idx % len(ql)], U_sov[l], rank=32, beta=0.15, gamma_rel=0.004 * hann[idx], U_opt=None
            )
            d_ob = v17.build_steered_ternary_moe_delta(
                w_o, b_o[idx % len(bl)], U_sov[l], rank=32, beta=0.15, gamma_rel=0.004 * hann[idx], U_opt=None
            )
            w_o.add_(0.5 * (d_oq + d_ob))
    del U_sov, q_o, b_o
    gc.collect()

    # 4. Turn-boundary & post-signature null-projected steering in L14-15
    apply_turn_boundary_null_steering(model, enc_chat, enc_tok, calib, target_layers=[14, 15])

    # 5. Export FP32 -> Q8_0 directly (freeing model before convert_hf_to_gguf.py)
    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    sd = model.state_dict()
    new_st = {k: sd[k].detach().cpu().float() for k in llama_st.keys()}
    orig_shapes = {k: v.float() for k, v in llama_st.items()}
    v12.verify_exact_shapes_and_params(orig_shapes, new_st, "quillan-1b-sovereign-wsp-v21")
    del llama_st, orig_shapes, sd, model
    gc.collect()

    hf_tmp = os.path.join(v12.EXPORT_ROOT, "quillan-1b-sovereign-wsp-v21_hf_tmp")
    os.makedirs(hf_tmp, exist_ok=True)
    save_file(new_st, os.path.join(hf_tmp, "model.safetensors"))
    del new_st
    gc.collect()
    for fn in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(v8.FLAGSHIP_SNAP, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(hf_tmp, fn))
    q8 = os.path.join(v12.EXPORT_ROOT, "quillan-1b-sovereign-wsp-v21-q8_0.gguf")
    subprocess.run([sys.executable, v12.CONVERT_SCRIPT, hf_tmp, "--outfile", q8, "--outtype", "q8_0"], check=True)
    shutil.rmtree(hf_tmp, ignore_errors=True)

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor_flagship_v21.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    q8_ppl = measure_perplexity(q8, ppl_file)
    logger.info("FLAGSHIP WSP v21 Q8_0 PPL = %.4f (v15 ref: 7.9463) in %.0fs", q8_ppl, time.time() - t0)


if __name__ == "__main__":
    main()
