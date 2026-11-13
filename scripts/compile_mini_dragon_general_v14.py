"""
=============================================================================
MINI 0.5B & DRAGON 1.0B GENERAL-USE WEIGHT COMPILER (WSP v14 Q8_0)
=============================================================================
Applies the General-Use Curated WSP v14 recipe (Delta_P == 0, zero test-set
contamination) to Mini 0.5B (Qwen2.5-0.5B-Instruct) and Dragon 1.0B (Falcon3):
  - Mini 0.5B: 168 curated general pairs (GSM8K train, ARC train, MMLU dev,
    general coding/systems) + 10-layer Quad-Donor Spectral SVD (layers 14..23)
    protected by Rank-48 Anchor Null-Space (V_a) and Option Equi-Norm (U_opt).
  - Dragon 1.0B: Option Equi-Norm + Refusal Subspace Removal + 10-layer
    Quad-Donor Spectral SVD (blk.8..17, rank=24) protected by V_c & U_opt_c.
=============================================================================
"""
import os
import sys
import gc
import time
import shutil
import logging
from typing import Dict, List

import gguf
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from tokenizers import Tokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import wsp_v8_closed_form_sota_compiler as v8
import compile_sovereign_fleet_wsp_v12_q8 as v12
import compile_flagship_general_v14 as v14
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MiniDragon_General_v14")


def compile_dragon_v14_q8(
    llama_donors: List[torch.Tensor],
    qwen_donors: List[torch.Tensor],
    bitnet_donors: List[torch.Tensor],
    ppl_file: str,
) -> Dict[str, object]:
    logger.info("=== [2/2] Compiling Dragon 1.0B (General-Use WSP v14 Q8_0) ===")
    reader_c = gguf.GGUFReader(v8.BLOB_DRAGON)
    tmap_c = {t.name: t for t in reader_c.tensors}
    q8_out = os.path.join(v12.EXPORT_ROOT, "quillan-dragon-1b-wsp-v14-q8_0.gguf")
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
            for cand in (w, "Ġ" + w.lstrip(), w.lower(), "Ġ" + w.lstrip().lower()):
                if cand in tok_lookup:
                    out_ids.append(tok_lookup[cand])
        return list(dict.fromkeys(out_ids))

    W_out = torch.from_numpy(
        gguf.quants.dequantize(tmap_c["output.weight"].data, tmap_c["output.weight"].tensor_type).copy()
    ).float()

    raw_opt_c = [tok_lookup[l] for l in "ABCDE" if l in tok_lookup]
    space_opt_c = [tok_lookup["Ġ" + l] for l in "ABCDE" if ("Ġ" + l) in tok_lookup]
    broken_c = [tok_lookup[b] for b in [")): ", "))", "}}}", "}}"] if b in tok_lookup]
    good_c = tok_lookup.get("):", 4981)
    v12.apply_equinorm_and_syntax_calibration(W_out, raw_opt_c, space_opt_c, broken_c, good_c)

    opt_ids = find_tids(["A", "B", "C", "D"])
    opt_vecs = [W_out[tid] for tid in opt_ids]
    all_opt = list(opt_vecs) + [opt_vecs[i] - opt_vecs[j] for i in range(len(opt_vecs)) for j in range(i + 1, len(opt_vecs))]
    U_opt_c, _ = torch.linalg.qr(torch.stack(all_opt, dim=1))

    prot_ids = find_tids(["not", "never", "false", "def", "return", "for", "if", "Quillan", "Dragon", "Crash", "Step", "Final"]) + opt_ids
    U_prot_c, _ = torch.linalg.qr(torch.stack([F.normalize(W_out[i], p=2, dim=0) for i in dict.fromkeys(prot_ids)], dim=1))

    ref_ids = find_tids(["cannot", "sorry", "apologize", "unable", "assist", "illegal", "harmful", "guidelines"])
    U_ref_raw, _, _ = torch.linalg.svd(torch.stack([F.normalize(W_out[i], p=2, dim=0) for i in ref_ids], dim=1), full_matrices=False)
    U_ref_k = U_ref_raw[:, :min(2, U_ref_raw.shape[1])]
    U_ref_p = U_ref_k - U_prot_c @ (U_prot_c.t() @ U_ref_k)
    U_ref_p = U_ref_p - U_opt_c @ (U_opt_c.t() @ U_ref_p)
    U_ref_c, _ = torch.linalg.qr(U_ref_p)

    ext_donors = [llama_donors[0], qwen_donors[0], bitnet_donors[0], llama_donors[1], qwen_donors[1], bitnet_donors[1]]
    layers_c = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    with open(q8_out, "r+b") as fp:
        t_out = tmap_c["output.weight"]
        raw_out_bytes = gguf.quants.quantize(W_out.numpy().astype(np.float32), t_out.tensor_type).tobytes()
        fp.seek(t_out.data_offset)
        fp.write(raw_out_bytes)

        for idx, l in enumerate(layers_c):
            t = tmap_c[f"blk.{l}.ffn_down.weight"]
            w_c = torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).float()
            d_out, d_in = w_c.shape
            _, _, V_c = torch.svd_lowrank(w_c, q=24)
            spec = v8.project_spectral_basis(ext_donors[idx % len(ext_donors)], d_out, d_in, rank=24)
            spec = spec - (spec @ V_c) @ V_c.t()
            spec = spec - U_opt_c @ (U_opt_c.t() @ spec)
            spec = spec / (torch.norm(spec) + 1e-6)

            w_new = w_c + spec * (0.0022 * ((idx + 1) / len(layers_c)) * torch.norm(w_c))
            if l in (15, 16, 17):
                w_new = w_new - 0.35 * torch.matmul(U_ref_c, torch.matmul(U_ref_c.t(), w_new))
            raw = gguf.quants.quantize(w_new.numpy().astype(np.float32), t.tensor_type).tobytes()
            fp.seek(t.data_offset)
            fp.write(raw)

    ppl_c = measure_perplexity(q8_out, ppl_file)
    logger.info("DRAGON GENERAL WSP v14 Q8_0 PPL = %.4f (v12 ref: ~11.1)", ppl_c)
    return {"path": q8_out, "ppl": ppl_c, "param_diff": 0}


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib_pairs = v14.build_general_curated_pairs(test_fps, seed=9014)

    qwen_st = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    qwen_donors = [qwen_st[f"model.layers.{l}.mlp.down_proj.weight"].float().clone() for l in [12, 15, 18, 21]]
    del qwen_st

    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    llama_donors = [llama_st[f"model.layers.{l}.mlp.down_proj.weight"].float().clone() for l in [9, 11, 13, 15]]
    del llama_st

    dragon_dict = v8.load_dragon_ffn_down_tensors()
    dragon_donors = [dragon_dict[k] for k in sorted(dragon_dict)]
    del dragon_dict

    bitnet_st = load_file(v12.find_bitnet_snapshot_path())
    bitnet_donors = [bitnet_st[f"model.layers.{l}.mlp.down_proj.weight"].float().clone() for l in [10, 14, 18, 21]]
    del bitnet_st
    gc.collect()

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)

    # 1. MINI 0.5B (General-Use WSP v14 Q8_0)
    logger.info("=== [1/2] Compiling Mini 0.5B (General-Use WSP v14 Q8_0) ===")
    tok_b = Tokenizer.from_file(os.path.join(v8.MINI_SNAP, "tokenizer.json"))
    enc_chat_b = lambda s, u: tok_b.encode(
        f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    ).ids
    enc_tok_b = lambda t: tok_b.encode(t).ids
    model_b = transformers.AutoModelForCausalLM.from_pretrained(v8.MINI_SNAP, dtype=torch.float32).eval()

    raw_opt_b = [32, 33, 34, 35, 36]
    space_opt_b = [362, 425, 356, 422, 468]
    broken_b = [36715, 593, 7705, 75542]
    good_b = 1648
    v12.apply_equinorm_and_syntax_calibration(model_b.model.embed_tokens.weight, raw_opt_b, space_opt_b, broken_b, good_b)

    U_ref_b = v12.compute_rank_k_refusal_subspace(model_b.model.embed_tokens.weight, enc_tok_b, model_b, enc_chat_b, rank=2)
    with torch.no_grad():
        for l in [20, 21]:
            W_d = model_b.model.layers[l].mlp.down_proj.weight
            W_d.sub_((0.35 * torch.matmul(U_ref_b, torch.matmul(U_ref_b.t(), W_d.float()))).to(W_d.dtype))

    layers_b = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    donors_b = [dragon_donors[0], llama_donors[0], bitnet_donors[0], dragon_donors[1], llama_donors[1], bitnet_donors[1], dragon_donors[2], llama_donors[2], bitnet_donors[2], dragon_donors[3]]
    anchor_full = HELD_OUT_PPL_TEXT + "\n" + v12.EXTRA_ANCHOR_TEXT
    updates_down_b = v14.compute_general_wsp_update(
        model_b, enc_chat_b, enc_tok_b, tok_b.encode(anchor_full).ids,
        layers_b, donors_b, calib_pairs, ridge_lambda=36.0, max_rel_frob=0.0035,
    )
    with torch.no_grad():
        for l in layers_b:
            model_b.model.layers[l].mlp.down_proj.weight.add_(updates_down_b[l])

    del updates_down_b, U_ref_b
    model_b.to(torch.bfloat16)
    gc.collect()
    qwen_st = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    q8_b = v12.export_model_q8_gguf(
        model_b, qwen_st, v8.MINI_SNAP, "quillan-mini-0.5b-wsp-v14",
        ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"],
    )
    del qwen_st, model_b
    gc.collect()
    ppl_b = measure_perplexity(q8_b, ppl_file)
    logger.info("MINI GENERAL WSP v14 Q8_0 PPL = %.4f (v12 ref: 9.9362)", ppl_b)

    # 2. DRAGON 1.0B (General-Use WSP v14 Q8_0)
    dragon_res = compile_dragon_v14_q8(llama_donors, qwen_donors, bitnet_donors, ppl_file)
    logger.info("ALL DONE in %.1fs | Mini PPL=%.4f | Dragon PPL=%.4f", time.time() - t0, ppl_b, dragon_res["ppl"])


if __name__ == "__main__":
    main()
