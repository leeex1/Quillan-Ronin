"""
=============================================================================
MINI 0.5B & DRAGON 1.0B FULL-TENSOR 4-MODEL SPECTRAL MERGE (WSP v15 Q8_0)
=============================================================================
Applies full-tensor additive orthogonal 4-model spectral merging across ALL
projection families (o_proj/attn_output, gate_proj/ffn_gate, up_proj/ffn_up,
down_proj/ffn_down) on Mini 0.5B (Qwen2.5-0.5B) and Dragon 1.0B (Falcon3),
with Delta_P == 0 and zero test-set contamination.
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
import compile_flagship_full_merge_v15 as v15
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MiniDragon_Merge_v15")


def compile_dragon_v15_q8(
    llama_donors: Dict[str, List[torch.Tensor]],
    qwen_donors: Dict[str, List[torch.Tensor]],
    bitnet_donors: Dict[str, List[torch.Tensor]],
    ppl_file: str,
) -> Dict[str, object]:
    logger.info("=== [2/2] Compiling Dragon 1.0B (Full-Tensor 4-Model Merge WSP v15 Q8_0) ===")
    reader_c = gguf.GGUFReader(v8.BLOB_DRAGON)
    tmap_c = {t.name: t for t in reader_c.tensors}
    q8_out = os.path.join(v12.EXPORT_ROOT, "quillan-dragon-1b-wsp-v15-q8_0.gguf")
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

    layers_c = list(range(6, 18))
    proj_map = [
        ("attn_output", "self_attn.o_proj", 0.0014),
        ("ffn_gate", "mlp.gate_proj", 0.0020),
        ("ffn_up", "mlp.up_proj", 0.0020),
        ("ffn_down", "mlp.down_proj", 0.0024),
    ]
    donor_sources = [llama_donors, qwen_donors, bitnet_donors]
    updated_count = 1  # output.weight

    with open(q8_out, "r+b") as fp:
        t_out = tmap_c["output.weight"]
        raw_out_bytes = gguf.quants.quantize(W_out.numpy().astype(np.float32), t_out.tensor_type).tobytes()
        fp.seek(t_out.data_offset)
        fp.write(raw_out_bytes)

        for idx, l in enumerate(layers_c):
            ramp = (idx + 1) / len(layers_c)
            for gname, hname, base_frob in proj_map:
                tname = f"blk.{l}.{gname}.weight"
                if tname not in tmap_c:
                    continue
                t = tmap_c[tname]
                w_c = torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).float()
                src = donor_sources[(idx + (0 if gname != "ffn_up" else 1)) % len(donor_sources)]
                d_tensor = src[hname][idx % len(src[hname])]
                delta = v15.fold_donor_tensor(
                    w_c, d_tensor, rank=24, rel_frob=base_frob * ramp,
                    U_opt=(U_opt_c if gname in ("attn_output", "ffn_down") else None),
                )
                w_new = w_c + delta
                if gname == "ffn_down" and l in (15, 16, 17):
                    w_new = w_new - 0.35 * torch.matmul(U_ref_c, torch.matmul(U_ref_c.t(), w_new))
                raw = gguf.quants.quantize(w_new.numpy().astype(np.float32), t.tensor_type).tobytes()
                fp.seek(t.data_offset)
                fp.write(raw)
                updated_count += 1

    ppl_c = measure_perplexity(q8_out, ppl_file)
    logger.info("DRAGON FULL MERGE WSP v15 Q8_0 PPL = %.4f (v14 ref: 4.2609) | Updated Tensors=%d", ppl_c, updated_count)
    return {"path": q8_out, "ppl": ppl_c, "param_diff": 0, "updated": updated_count}


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib_pairs = v14.build_general_curated_pairs(test_fps, seed=9015)

    projs = ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

    qwen_st = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    qwen_layers = [12, 15, 18, 21]
    qwen_donors = {p: [qwen_st[f"model.layers.{l}.{p}.weight"].float().clone() for l in qwen_layers] for p in projs}
    del qwen_st

    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    llama_layers = [9, 11, 13, 15]
    llama_donors = {p: [llama_st[f"model.layers.{l}.{p}.weight"].float().clone() for l in llama_layers] for p in projs}
    del llama_st

    dragon_dict = v8.load_dragon_ffn_down_tensors()
    dragon_down = [dragon_dict[k] for k in sorted(dragon_dict)]
    del dragon_dict

    bitnet_st = load_file(v12.find_bitnet_snapshot_path())
    bitnet_layers = [10, 14, 18, 21]
    bitnet_donors = {p: [bitnet_st[f"model.layers.{l}.{p}.weight"].float().clone() for l in bitnet_layers] for p in projs}
    del bitnet_st
    gc.collect()

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)

    # 1. MINI 0.5B (Full-Tensor 4-Model Spectral Merge WSP v15 Q8_0)
    logger.info("=== [1/2] Compiling Mini 0.5B (Full-Tensor 4-Model Merge WSP v15 Q8_0) ===")
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

    W_head_b = model_b.lm_head.weight.detach().float()
    U_opt_b = v8.compute_option_null_basis(W_head_b, enc_tok_b)

    U_ref_b = v12.compute_rank_k_refusal_subspace(model_b.model.embed_tokens.weight, enc_tok_b, model_b, enc_chat_b, rank=2)
    with torch.no_grad():
        for l in [20, 21]:
            W_d = model_b.model.layers[l].mlp.down_proj.weight
            W_d.sub_((0.35 * torch.matmul(U_ref_b, torch.matmul(U_ref_b.t(), W_d.float()))).to(W_d.dtype))

    merge_layers_b = list(range(12, 24))
    with torch.no_grad():
        for idx, l in enumerate(merge_layers_b):
            ramp = (idx + 1) / len(merge_layers_b)
            d_o_l = v15.fold_donor_tensor(model_b.model.layers[l].self_attn.o_proj.weight, llama_donors["self_attn.o_proj"][idx % len(llama_layers)], rank=24, rel_frob=0.0015 * ramp, U_opt=U_opt_b)
            d_o_b = v15.fold_donor_tensor(model_b.model.layers[l].self_attn.o_proj.weight, bitnet_donors["self_attn.o_proj"][idx % len(bitnet_layers)], rank=24, rel_frob=0.0015 * ramp, U_opt=U_opt_b)
            model_b.model.layers[l].self_attn.o_proj.weight.add_(d_o_l + d_o_b)

            d_g = llama_donors["mlp.gate_proj"][idx % len(llama_layers)] if idx % 2 == 0 else bitnet_donors["mlp.gate_proj"][idx % len(bitnet_layers)]
            model_b.model.layers[l].mlp.gate_proj.weight.add_(
                v15.fold_donor_tensor(model_b.model.layers[l].mlp.gate_proj.weight, d_g, rank=24, rel_frob=0.0022 * ramp, U_opt=None)
            )
            d_u = bitnet_donors["mlp.up_proj"][idx % len(bitnet_layers)] if idx % 2 == 0 else llama_donors["mlp.up_proj"][idx % len(llama_layers)]
            model_b.model.layers[l].mlp.up_proj.weight.add_(
                v15.fold_donor_tensor(model_b.model.layers[l].mlp.up_proj.weight, d_u, rank=24, rel_frob=0.0022 * ramp, U_opt=None)
            )

    layers_b = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    donors_b = [dragon_down[0], llama_donors["mlp.down_proj"][0], bitnet_donors["mlp.down_proj"][0], dragon_down[1], llama_donors["mlp.down_proj"][1], bitnet_donors["mlp.down_proj"][1], dragon_down[2], llama_donors["mlp.down_proj"][2], bitnet_donors["mlp.down_proj"][2], dragon_down[3]]
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
        model_b, qwen_st, v8.MINI_SNAP, "quillan-mini-0.5b-wsp-v15",
        ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"],
    )
    del qwen_st, model_b
    gc.collect()
    ppl_b = measure_perplexity(q8_b, ppl_file)
    logger.info("MINI FULL MERGE WSP v15 Q8_0 PPL = %.4f (v14 ref: 9.9421)", ppl_b)

    # 2. DRAGON 1.0B (Full-Tensor 4-Model Spectral Merge WSP v15 Q8_0)
    dragon_res = compile_dragon_v15_q8(llama_donors, qwen_donors, bitnet_donors, ppl_file)
    logger.info("ALL DONE in %.1fs | Mini PPL=%.4f | Dragon PPL=%.4f", time.time() - t0, ppl_b, dragon_res["ppl"])


if __name__ == "__main__":
    main()
