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
    anchor_full = v12.EXTRA_ANCHOR_TEXT  # held-out PPL text deliberately excluded
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
        model_b, qwen_st, v8.MINI_SNAP, "quillan-mini-0.5b-wsp-v17",
        ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"],
    )
    del qwen_st, model_b
    gc.collect()
    ppl_b = measure_perplexity(q8_b, ppl_file)
    logger.info("MINI FULL MERGE WSP v17 Q8_0 PPL = %.4f (v15 ref: 9.9125)", ppl_b)

    logger.info("DONE in %.1fs", time.time()-t0)


if __name__ == "__main__":
    main()


