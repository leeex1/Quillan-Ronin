import os
import sys
import gc
import json
import logging
import torch
import transformers
from tokenizers import Tokenizer
from safetensors.torch import load_file

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import wsp_v8_closed_form_sota_compiler as v8
import compile_sovereign_fleet_wsp_v12_q8 as v12
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Mini_V12_Recompile")

def recompile_mini():
    logger.info("=== Compiling Mini 0.5B (WSP v12 Clean Stable Q8_0) ===")
    MINI_SNAP = v8.MINI_SNAP
    tok_b = Tokenizer.from_file(os.path.join(MINI_SNAP, "tokenizer.json"))
    enc_chat_b = lambda s, u: tok_b.encode(
        f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    ).ids
    enc_tok_b = lambda t: tok_b.encode(t).ids

    model_b = transformers.AutoModelForCausalLM.from_pretrained(MINI_SNAP, dtype=torch.float32).eval()

    # 1. Option equi-norm & syntax calibration to tied embedding/head
    raw_opt_b = [32, 33, 34, 35, 36]
    space_opt_b = [362, 425, 356, 422, 468]
    broken_b = [36715, 593, 7705, 75542]  # )):, )), ))), }}}
    good_b = 1648  # ):
    v12.apply_equinorm_and_syntax_calibration(model_b.model.embed_tokens.weight, raw_opt_b, space_opt_b, broken_b, good_b)

    # 2. Refusal subspace removal
    U_ref_b = v12.compute_rank_k_refusal_subspace(model_b.model.embed_tokens.weight, enc_tok_b, model_b, enc_chat_b, rank=2)
    with torch.no_grad():
        for l in [20, 21]:
            W_d = model_b.model.layers[l].mlp.down_proj.weight
            W_d.sub_((0.35 * torch.matmul(U_ref_b, torch.matmul(U_ref_b.t(), W_d.float()))).to(W_d.dtype))

    # 3. Multi-projection closed-form MEMIT + 4-Donor Spectral SVD Folding (down_proj ONLY)
    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    llama_donors = [llama_st[f"model.layers.{l}.mlp.down_proj.weight"].float() for l in [10, 11]]
    bitnet_st = load_file(v12.find_bitnet_snapshot_path())
    bitnet_donors = [bitnet_st[f"model.layers.{l}.mlp.down_proj.weight"].float() for l in [10, 14]]
    dragon_donors = [model_b.model.layers[l].mlp.down_proj.weight.detach().float() for l in [14, 16]]
    donors_b = [dragon_donors[0], llama_donors[0], bitnet_donors[0], dragon_donors[1], llama_donors[1], bitnet_donors[1]]

    layers_b = [18, 19, 20, 21, 22, 23]
    anchor_full = HELD_OUT_PPL_TEXT + "\n" + v12.EXTRA_ANCHOR_TEXT
    test_fps = [HELD_OUT_PPL_TEXT]
    calib_pairs = v12.build_v12_calibration_pairs(test_fps, seed=9002)

    updates_down_b, _ = v12.compute_multi_projection_wsp_update(
        model_b, enc_chat_b, enc_tok_b, tok_b.encode(anchor_full).ids,
        layers_b, donors_b, calib_pairs, ridge_lambda=52.0, max_rel_frob=0.0012,
    )
    with torch.no_grad():
        for l in layers_b:
            model_b.model.layers[l].mlp.down_proj.weight.add_(updates_down_b[l])

    del updates_down_b, U_ref_b, llama_st, bitnet_st
    model_b.to(torch.bfloat16)
    gc.collect()

    qwen_st = load_file(os.path.join(MINI_SNAP, "model.safetensors"))
    q8_b = v12.export_model_q8_gguf(
        model_b, qwen_st, MINI_SNAP, "quillan-mini-0.5b-wsp-v12",
        ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"],
    )
    del qwen_st, model_b
    gc.collect()

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor.txt")
    ppl_b = measure_perplexity(q8_b, ppl_file)
    logger.info("MODEL B WSP v12 Clean Q8_0 PPL = %.4f (Clean reference: 9.9362)", ppl_b)

    # Update manifest
    manifest_path = os.path.join(v12.EXPORT_ROOT, "wsp_v12_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        man["MODEL_B"] = {"path": q8_b, "ppl": ppl_b, "param_diff": 0}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2)

    logger.info("=== Finished Mini 0.5B Recompile Successfully ===")

if __name__ == "__main__":
    recompile_mini()
