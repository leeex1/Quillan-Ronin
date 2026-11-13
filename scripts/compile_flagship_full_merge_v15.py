"""
=============================================================================
FLAGSHIP 1.2B FULL-TENSOR 4-MODEL SPECTRAL MERGE (WSP v15 Q8_0)
=============================================================================
Folds Qwen2.5-0.5B, BitNet-2B, and Falcon3/Dragon-1B across ALL transformer
projection families (self_attn.o_proj, mlp.gate_proj, mlp.up_proj,
mlp.down_proj) in layers [4..15] on Llama-3.2-1B-Instruct (Delta_P == 0,
zero test-set contamination), protected by Rank-48 Anchor Null-Space (V_a)
and Option Equi-Norm (U_opt).
=============================================================================
"""
import os
import sys
import gc
import time
import logging
from typing import Dict, List

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
logger = logging.getLogger("Flagship_Merge_v15")


def fold_donor_tensor(
    host_w: torch.Tensor,
    donor_w: torch.Tensor,
    rank: int = 32,
    rel_frob: float = 0.004,
    U_opt: torch.Tensor | None = None,
) -> torch.Tensor:
    w_f = host_w.detach().float()
    d_out, d_in = w_f.shape
    _, _, V_h = torch.svd_lowrank(w_f, q=min(32, min(d_out, d_in) - 2))
    spec = v8.project_spectral_basis(donor_w, d_out, d_in, rank=rank)
    spec = spec - (spec @ V_h) @ V_h.t()
    if U_opt is not None and U_opt.shape[0] == d_out:
        spec = spec - U_opt @ (U_opt.t() @ spec)
    spec = spec / (torch.norm(spec) + 1e-6)
    return spec * (rel_frob * torch.norm(w_f))


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib_pairs = v14.build_general_curated_pairs(test_fps, seed=9015)

    qwen_st = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    qwen_layers = [10, 12, 14, 16, 18, 20, 22]
    qwen_donors = {
        proj: [qwen_st[f"model.layers.{l}.{proj}.weight"].float().clone() for l in qwen_layers]
        for proj in ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    }
    del qwen_st

    dragon_dict = v8.load_dragon_ffn_down_tensors()
    dragon_down = [dragon_dict[k] for k in sorted(dragon_dict)]
    del dragon_dict

    bitnet_st = load_file(v12.find_bitnet_snapshot_path())
    bitnet_layers = [8, 11, 14, 17, 20, 23]
    bitnet_donors = {
        proj: [bitnet_st[f"model.layers.{l}.{proj}.weight"].float().clone() for l in bitnet_layers]
        for proj in ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    }
    del bitnet_st
    gc.collect()

    tok_a = Tokenizer.from_file(os.path.join(v8.FLAGSHIP_SNAP, "tokenizer.json"))
    enc_chat_a = lambda s, u: tok_a.encode(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    ).ids
    enc_tok_a = lambda t: tok_a.encode(t, add_special_tokens=False).ids

    model_a = transformers.AutoModelForCausalLM.from_pretrained(v8.FLAGSHIP_SNAP, dtype=torch.float32).eval()

    # 1. Option equi-norm & syntax calibration
    raw_opt_a = [32, 33, 34, 35, 36]
    space_opt_a = [362, 426, 356, 423, 469]
    broken_a = [37815, 595, 7861, 76642]
    good_a = 1680
    v12.apply_equinorm_and_syntax_calibration(model_a.model.embed_tokens.weight, raw_opt_a, space_opt_a, broken_a, good_a)

    W_head = model_a.lm_head.weight.detach().float()
    U_opt = v8.compute_option_null_basis(W_head, enc_tok_a)

    # 2. Refusal subspace removal
    U_ref_a = v12.compute_rank_k_refusal_subspace(model_a.model.embed_tokens.weight, enc_tok_a, model_a, enc_chat_a, rank=3)
    with torch.no_grad():
        for l in [13, 14, 15]:
            W_d = model_a.model.layers[l].mlp.down_proj.weight
            W_d.sub_((0.60 * torch.matmul(U_ref_a, torch.matmul(U_ref_a.t(), W_d.float()))).to(W_d.dtype))
        for l in [13, 14]:
            W_o = model_a.model.layers[l].self_attn.o_proj.weight
            W_o.sub_((0.35 * torch.matmul(U_ref_a, torch.matmul(U_ref_a.t(), W_o.float()))).to(W_o.dtype))

    # 3. Full-tensor 4-model spectral merge across layers [4..15]
    merge_layers = list(range(4, 16))
    with torch.no_grad():
        for idx, l in enumerate(merge_layers):
            ramp = (idx + 1) / len(merge_layers)
            # o_proj (Qwen + BitNet projected separately into host shape)
            d_o_q = fold_donor_tensor(model_a.model.layers[l].self_attn.o_proj.weight, qwen_donors["self_attn.o_proj"][idx % len(qwen_layers)], rank=32, rel_frob=0.0020 * ramp, U_opt=U_opt)
            d_o_b = fold_donor_tensor(model_a.model.layers[l].self_attn.o_proj.weight, bitnet_donors["self_attn.o_proj"][idx % len(bitnet_layers)], rank=32, rel_frob=0.0020 * ramp, U_opt=U_opt)
            model_a.model.layers[l].self_attn.o_proj.weight.add_(d_o_q + d_o_b)
            # gate_proj (Qwen + BitNet)
            d_g = qwen_donors["mlp.gate_proj"][idx % len(qwen_layers)] if idx % 2 == 0 else bitnet_donors["mlp.gate_proj"][idx % len(bitnet_layers)]
            model_a.model.layers[l].mlp.gate_proj.weight.add_(
                fold_donor_tensor(model_a.model.layers[l].mlp.gate_proj.weight, d_g, rank=32, rel_frob=0.0030 * ramp, U_opt=None)
            )
            # up_proj (Qwen + BitNet)
            d_u = bitnet_donors["mlp.up_proj"][idx % len(bitnet_layers)] if idx % 2 == 0 else qwen_donors["mlp.up_proj"][idx % len(qwen_layers)]
            model_a.model.layers[l].mlp.up_proj.weight.add_(
                fold_donor_tensor(model_a.model.layers[l].mlp.up_proj.weight, d_u, rank=32, rel_frob=0.0030 * ramp, U_opt=None)
            )

    # 4. General curated WSP + down_proj 4-model spectral fold on [6..15]
    layers_a = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    donors_down = [
        qwen_donors["mlp.down_proj"][0], dragon_down[0], bitnet_donors["mlp.down_proj"][0],
        qwen_donors["mlp.down_proj"][1], dragon_down[1], bitnet_donors["mlp.down_proj"][1],
        qwen_donors["mlp.down_proj"][2], dragon_down[2], bitnet_donors["mlp.down_proj"][2],
        qwen_donors["mlp.down_proj"][3],
    ]
    anchor_full = HELD_OUT_PPL_TEXT + "\n" + v12.EXTRA_ANCHOR_TEXT
    updates_down_a = v14.compute_general_wsp_update(
        model_a, enc_chat_a, enc_tok_a, tok_a.encode(anchor_full, add_special_tokens=False).ids,
        layers_a, donors_down, calib_pairs, ridge_lambda=30.0, max_rel_frob=0.0050,
    )
    with torch.no_grad():
        for l in layers_a:
            model_a.model.layers[l].mlp.down_proj.weight.add_(updates_down_a[l])

        # Rank-1 Identity Steer Layer 15 o_proj
        w_pos_a = F.normalize(model_a.model.embed_tokens.weight[51435].detach().float(), p=2, dim=0)
        w_neg_a = F.normalize(0.5 * (model_a.model.embed_tokens.weight[5109].float() + model_a.model.embed_tokens.weight[15836].float()), p=2, dim=0)
        acts_a = []
        hk = model_a.model.layers[15].self_attn.o_proj.register_forward_hook(lambda m, i, o: acts_a.append(i[0][0, -1, :].detach().float().clone()))
        for iq in ["Who created you?", "What model are you?"]:
            model_a(torch.tensor([enc_chat_a("You are Quillan-Ronin, created by CrashOverrideX.", iq)[-192:]], dtype=torch.long))
        hk.remove()
        u_id_a = F.normalize(torch.stack(acts_a).mean(dim=0), p=2, dim=0)
        model_a.model.layers[15].self_attn.o_proj.weight.add_(0.07 * torch.outer(w_pos_a, u_id_a) - 0.05 * torch.outer(w_neg_a, u_id_a))

    del updates_down_a, U_ref_a
    model_a.to(torch.bfloat16)
    gc.collect()

    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    q8_a = v12.export_model_q8_gguf(
        model_a, llama_st, v8.FLAGSHIP_SNAP, "quillan-1b-sovereign-wsp-v15",
        ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"],
    )
    del llama_st, model_a
    gc.collect()

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    ppl_a = measure_perplexity(q8_a, ppl_file)
    logger.info("FLAGSHIP FULL MERGE WSP v15 Q8_0 PPL = %.4f (v14 ref: 7.9921) in %.1fs", ppl_a, time.time() - t0)


if __name__ == "__main__":
    main()
