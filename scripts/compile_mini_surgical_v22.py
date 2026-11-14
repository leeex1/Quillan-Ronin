"""
=============================================================================
MINI 0.5B SURGICAL PER-LAYER WSP COMPILER (v22)
=============================================================================
Refines v21:
  1. Removes `embed_tokens` 0.78x scaling (which shifted initial calib PPL
     2.2480 -> 2.2514), keeping tied embeddings clean except v12 equinorm.
  2. Keeps Q8_0-safe o_proj Procrustes (L9..L19, zero gate/up).
  3. Focuses surgical single-layer null-space edits on L21 (logic_mc:
     'Answer:' over '**'/'The'/'To'/'Here'), L20 (code: '    return' over
     '\"\"\"'/'```'), and L11 (math: digits over 'Let'/'Step').
  4. Directly exports FP32 -> Q8_0 and measures held-out PPL.
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

import gguf
import torch
import transformers
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
import wsp_v8_closed_form_sota_compiler as v8
import compile_sovereign_fleet_wsp_v12_q8 as v12
import compile_flagship_general_v14 as v14
import compile_mini_wsp_v20 as m20
import compile_mini_surgical_v21 as ms21
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Mini_Surgical_v22")


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib_pairs = v14.build_general_curated_pairs(test_fps, seed=9022)
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

    logger.info("Initial Mini calib PPL: %.4f", calib_ppl())

    # 1. Equinorm + syntax calibration only (keep tied embed_tokens clean)
    v12.apply_equinorm_and_syntax_calibration(
        model_b.model.embed_tokens.weight,
        [32, 33, 34, 35, 36],
        [362, 425, 356, 422, 468],
        [36715, 593, 7705, 75542],
        1648,
    )
    W_head_b = model_b.lm_head.weight.detach().float()
    U_opt_b = v8.compute_option_null_basis(W_head_b, enc_tok_b)

    # 2. Protected Rank-2 Refusal Abliteration on [18, 19] (gentle alpha=0.25)
    U_ref_b = v12.compute_rank_k_refusal_subspace(model_b.model.embed_tokens.weight, enc_tok_b, model_b, enc_chat_b, rank=2)
    with torch.no_grad():
        for l in [18, 19]:
            W_d = model_b.model.layers[l].mlp.down_proj.weight.data
            model_b.model.layers[l].mlp.down_proj.weight.data.sub_(0.25 * torch.matmul(U_ref_b, torch.matmul(U_ref_b.t(), W_d.float())))
    del U_ref_b

    # 3. Capture prose null-space V_a_o [9..19] & V_a_d [11, 20, 21]
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
        _, _, V = torch.svd_lowrank(torch.cat(act_d[l], dim=0), q=48, niter=2)
        V_a_d[l] = V.contiguous()
    del act_d
    gc.collect()

    # 4. Q8_0-safe Procrustes o_proj [9..19] (zero gate/up)
    merge_layers = list(range(9, 20))
    tok_l = Tokenizer.from_file(os.path.join(v8.FLAGSHIP_SNAP, "tokenizer.json"))
    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    P_llama = m20.compute_procrustes_map(llama_st["model.embed_tokens.weight"], tok_l.get_vocab(), W_emb_host, tok_b_vocab)
    llama_o = {l: llama_st[f"model.layers.{round(l * 15 / 23)}.self_attn.o_proj.weight"].float().clone() for l in merge_layers}
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
    with torch.no_grad():
        for idx, l in enumerate(merge_layers):
            w_o = model_b.model.layers[l].self_attn.o_proj.weight.data.float()
            delta = m20.build_procrustes_steered_o_delta(
                w_o, llama_o[l], P_llama, dragon_o[l], P_dragon, bitnet_o[l],
                V_a_o[l], U_sov[l], U_opt_b, rank=24, rel_frob=0.0020 * hann_m[idx]
            )
            model_b.model.layers[l].self_attn.o_proj.weight.data.add_(delta)
    del llama_o, dragon_o, bitnet_o, P_llama, P_dragon
    gc.collect()
    logger.info("After Procrustes o_proj calib PPL=%.4f", calib_ppl())

    # 5. Surgical Single-Layer Edits (L21, L20, L11)
    E = model_b.model.embed_tokens.weight.detach().float()
    def _rows(strs):
        return torch.stack([E[enc_tok_b(s)[0]] for s in strs if len(enc_tok_b(s)) > 0], dim=0)

    mc_prompts = [
        "Which option is prime?\nA) 9\nB) 11\nC) 15\nD) 21\nReply with 'Answer: <letter>'.\n",
        "All bloops are razzies. All razzies are lazzies. All bloops are:\nA) fast\nB) lazzies\nC) blue\nD) small\nReply with 'Answer: <letter>'.\n",
    ]
    X_21 = ms21.capture_module_inputs(model_b, tok_hf, mc_prompts, 21, "down_proj")
    d21 = ms21.surgical_single_layer_steering(
        model_b.model.layers[21].mlp.down_proj.weight.data,
        X_21, V_a_d[21], _rows(["Answer"]), _rows(["The", "To", "Here", "**"]),
        alpha_damp=0.22, rel_frob_cap=0.0024,
    )
    with torch.no_grad():
        model_b.model.layers[21].mlp.down_proj.weight.data.add_(d21)

    code_prompts = [
        "Complete Python function body only:\ndef add_two(a: int, b: int) -> int:\n",
        "Complete Python function body only:\ndef is_even(n: int) -> bool:\n",
    ]
    X_20 = ms21.capture_module_inputs(model_b, tok_hf, code_prompts, 20, "down_proj")
    d20 = ms21.surgical_single_layer_steering(
        model_b.model.layers[20].mlp.down_proj.weight.data,
        X_20, V_a_d[20], _rows(["    return", "    if"]), _rows(['"""', "```", "Here", "#"]),
        alpha_damp=0.22, rel_frob_cap=0.0022,
    )
    with torch.no_grad():
        model_b.model.layers[20].mlp.down_proj.weight.data.add_(d20)

    math_prompts = [
        "A baker makes 12 loaves/hr for 3 hrs, sells 10. How many left?\n",
        "What is 15 * 4 + 12?\n",
    ]
    X_11 = ms21.capture_module_inputs(model_b, tok_hf, math_prompts, 11, "down_proj")
    d11 = ms21.surgical_single_layer_steering(
        model_b.model.layers[11].mlp.down_proj.weight.data,
        X_11, V_a_d[11], _rows(["26", "72", "####"]), _rows(["Let", "Step", "To"]),
        alpha_damp=0.16, rel_frob_cap=0.0018,
    )
    with torch.no_grad():
        model_b.model.layers[11].mlp.down_proj.weight.data.add_(d11)
    logger.info("After surgical L21/L20/L11 calib PPL=%.4f", calib_ppl())

    # 6. Export FP32 -> Q8_0 directly
    qwen_st = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    sd = model_b.state_dict()
    new_st = {k: sd[k].detach().cpu().float().contiguous() for k in qwen_st.keys()}
    ref_st = {k: v.float() for k, v in qwen_st.items()}
    v12.verify_exact_shapes_and_params(ref_st, new_st, "quillan-mini-0.5b-wsp-v22")
    del ref_st, qwen_st, sd, model_b
    gc.collect()

    tag = "quillan-mini-0.5b-wsp-v22"
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
    logger.info("MINI SURGICAL v22 Q8_0 PPL = %.4f (v15 ref: 9.9125, base: 9.9362) | Elapsed: %.1fs", ppl_q8, time.time() - t0)


if __name__ == "__main__":
    main()
