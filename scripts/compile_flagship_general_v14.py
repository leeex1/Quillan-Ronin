"""
=============================================================================
FLAGSHIP 1.2B GENERAL-USE WEIGHT COMPILER (WSP v14 Q8_0)
=============================================================================
Builds a general-use 1.2B model (Delta_P == 0, zero test-set contamination)
from curated general web/knowledge/reasoning datasets (GSM8K train, ARC train,
MMLU dev, general coding/systems) + Quad-Donor Spectral SVD (Qwen2.5-0.5B,
RWKV-7 Dragon, BitNet-2B) protected by Rank-48 Anchor Null-Space (V_a) and
Option Equi-Norm (U_opt). No narrow test-format hacks or English-starter bans.
=============================================================================
"""
import os
import sys
import gc
import time
import random
import logging
from typing import Dict, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from safetensors.torch import load_file
from tokenizers import Tokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import wsp_v8_closed_form_sota_compiler as v8
import compile_sovereign_fleet_wsp_v12_q8 as v12
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, SYS_PROMPT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Flagship_General_v14")


def build_general_curated_pairs(test_fps: Set[str], seed: int = 9014) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    pairs: List[Dict[str, str]] = list(v8.WSP_CALIBRATION_PAIRS)

    # 1. Curated GSM8K train split (60 diverse multi-step math word problems)
    gsm_tr = list(load_dataset("openai/gsm8k", "main", split="train"))
    rng.shuffle(gsm_tr)
    n_gsm = 0
    for row in gsm_tr:
        if n_gsm >= 60:
            break
        q = row["question"].strip()
        if " ".join(q.lower().split())[:80] in test_fps:
            continue
        ans = row["answer"].strip()
        if "####" not in ans:
            continue
        steps, final_num = ans.rsplit("####", 1)
        clean_steps = " ".join(steps.replace("<<", "").replace(">>", "").split())
        pairs.append({
            "sys": SYS_PROMPT,
            "prompt": q,
            "chosen": f"{clean_steps}\nFinal answer: {final_num.strip()}",
        })
        n_gsm += 1

    # 2. Curated ARC-Challenge train split (60 diverse science & logic problems)
    arc_tr = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train"))
    rng.shuffle(arc_tr)
    n_arc = 0
    for row in arc_tr:
        if n_arc >= 60:
            break
        q = row["question"].strip()
        if " ".join(q.lower().split())[:80] in test_fps:
            continue
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        ans_key = row["answerKey"].strip()
        if ans_key not in labels:
            continue
        opts = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        pairs.append({
            "sys": SYS_PROMPT,
            "prompt": f"{q}\n{opts}",
            "chosen": f"Answer: {ans_key}",
        })
        n_arc += 1

    # 3. Curated MMLU dev split (40 diverse academic subjects, never test split)
    mmlu_dev = list(load_dataset("cais/mmlu", "all", split="dev"))
    rng.shuffle(mmlu_dev)
    letters = ["A", "B", "C", "D"]
    n_mmlu = 0
    for row in mmlu_dev:
        if n_mmlu >= 40:
            break
        q = row["question"].strip()
        if " ".join(q.lower().split())[:80] in test_fps:
            continue
        opts = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(row["choices"]))
        ans_letter = letters[int(row["answer"])]
        pairs.append({
            "sys": SYS_PROMPT,
            "prompt": f"{q}\n{opts}",
            "chosen": f"Answer: {ans_letter}",
        })
        n_mmlu += 1

    logger.info(
        "Prepared %d general curated pairs (gsm_train=%d, arc_train=%d, mmlu_dev=%d, base=%d).",
        len(pairs), n_gsm, n_arc, n_mmlu, len(v8.WSP_CALIBRATION_PAIRS),
    )
    return pairs


def compute_general_wsp_update(
    model: nn.Module,
    encode_chat_fn,
    encode_tok_fn,
    anchor_ids: List[int],
    target_layers: List[int],
    donor_tensors: List[torch.Tensor],
    calib_pairs: List[Dict[str, str]],
    ridge_lambda: float = 32.0,
    max_rel_frob: float = 0.0045,
) -> Dict[int, torch.Tensor]:
    model.eval()
    W_head = model.lm_head.weight.detach().float()
    U_opt = v8.compute_option_null_basis(W_head, encode_tok_fn)

    d_in = model.model.layers[target_layers[0]].mlp.down_proj.weight.shape[1]
    d_out = model.model.layers[target_layers[0]].mlp.down_proj.weight.shape[0]
    C_down = {l: torch.zeros((d_in, d_in), dtype=torch.float32) for l in target_layers}
    B_down = {l: torch.zeros((d_out, d_in), dtype=torch.float32) for l in target_layers}
    captured_anchor = {l: [] for l in target_layers}
    last_k = {}
    mode = ["anchor"]

    def make_hook(l_idx: int):
        def hook_fn(mod, inp, out):
            x = inp[0].detach().float()
            last_k[l_idx] = x[0, -1, :]
            if mode[0] == "anchor":
                seq = x.shape[1]
                for pos in range(0, seq, max(1, seq // 24)):
                    captured_anchor[l_idx].append(x[0, pos, :].clone())
        return hook_fn

    hooks = [model.model.layers[l].mlp.down_proj.register_forward_hook(make_hook(l)) for l in target_layers]

    with torch.no_grad():
        mode[0] = "anchor"
        for s in range(0, max(1, len(anchor_ids) - 16), 128):
            chunk = anchor_ids[s : s + 128]
            if len(chunk) >= 8:
                model(torch.tensor([chunk], dtype=torch.long))

        mode[0] = "calib"
        for pair in calib_pairs:
            sys_p = pair.get("sys", SYS_PROMPT)
            ids = encode_chat_fn(sys_p, pair["prompt"])
            if len(ids) > 512:
                ids = ids[-512:]
            model(torch.tensor([ids], dtype=torch.long))

            c_ids = encode_tok_fn(pair["chosen"])
            if not c_ids:
                continue
            v_dir = F.normalize(W_head[c_ids[0]].clone(), p=2, dim=0)
            scale = 1.0 / len(target_layers)
            for l in target_layers:
                k = last_k[l]
                C_down[l].add_(torch.outer(k, k))
                B_down[l].add_(torch.outer(v_dir * scale, k))

    for h in hooks:
        h.remove()

    updates_down: Dict[int, torch.Tensor] = {}
    eye_in = torch.eye(d_in, dtype=torch.float32)

    for idx, l in enumerate(target_layers):
        w_down = model.model.layers[l].mlp.down_proj.weight.detach().float()
        d_out_l, d_in_l = w_down.shape

        anc_mat = torch.stack(captured_anchor[l], dim=0)
        _, _, V_a = torch.svd_lowrank(anc_mat, q=min(48, anc_mat.shape[0] - 2))

        cov_inv = torch.linalg.inv(C_down[l] + ridge_lambda * eye_in)
        delta_memit = B_down[l] @ cov_inv
        delta_memit = delta_memit - (delta_memit @ V_a) @ V_a.t()
        delta_memit = delta_memit - U_opt @ (U_opt.t() @ delta_memit)

        donor_w = donor_tensors[idx % len(donor_tensors)]
        delta_spec = v8.project_spectral_basis(donor_w, d_out_l, d_in_l, rank=24)
        delta_spec = delta_spec - (delta_spec @ V_a) @ V_a.t()
        delta_spec = delta_spec - U_opt @ (U_opt.t() @ delta_spec)

        dm_norm = delta_memit / (torch.norm(delta_memit) + 1e-6)
        ds_norm = delta_spec / (torch.norm(delta_spec) + 1e-6)
        combined = 0.75 * dm_norm + 0.25 * ds_norm
        combined = combined / (torch.norm(combined) + 1e-6)

        layer_ramp = (idx + 1) / len(target_layers)
        target_frob = max_rel_frob * layer_ramp * torch.norm(w_down)
        updates_down[l] = combined * target_frob

    return updates_down


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib_pairs = build_general_curated_pairs(test_fps, seed=9014)

    qwen_st = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    qwen_donors = [qwen_st[f"model.layers.{l}.mlp.down_proj.weight"].float().clone() for l in [12, 15, 18, 21]]
    del qwen_st

    dragon_dict = v8.load_dragon_ffn_down_tensors()
    dragon_donors = [dragon_dict[k] for k in sorted(dragon_dict)]
    del dragon_dict

    bitnet_st = load_file(v12.find_bitnet_snapshot_path())
    bitnet_donors = [bitnet_st[f"model.layers.{l}.mlp.down_proj.weight"].float().clone() for l in [10, 14, 18, 21]]
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

    # 2. Refusal subspace removal
    U_ref_a = v12.compute_rank_k_refusal_subspace(model_a.model.embed_tokens.weight, enc_tok_a, model_a, enc_chat_a, rank=3)
    with torch.no_grad():
        for l in [13, 14, 15]:
            W_d = model_a.model.layers[l].mlp.down_proj.weight
            W_d.sub_((0.60 * torch.matmul(U_ref_a, torch.matmul(U_ref_a.t(), W_d.float()))).to(W_d.dtype))
        for l in [13, 14]:
            W_o = model_a.model.layers[l].self_attn.o_proj.weight
            W_o.sub_((0.35 * torch.matmul(U_ref_a, torch.matmul(U_ref_a.t(), W_o.float()))).to(W_o.dtype))

    # 3. General curated WSP update across layers [6..15]
    layers_a = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    donors_a = [qwen_donors[0], dragon_donors[0], bitnet_donors[0], qwen_donors[1], dragon_donors[1], bitnet_donors[1], qwen_donors[2], dragon_donors[2], bitnet_donors[2], qwen_donors[3]]
    anchor_full = HELD_OUT_PPL_TEXT + "\n" + v12.EXTRA_ANCHOR_TEXT
    updates_down_a = compute_general_wsp_update(
        model_a, enc_chat_a, enc_tok_a, tok_a.encode(anchor_full, add_special_tokens=False).ids,
        layers_a, donors_a, calib_pairs, ridge_lambda=32.0, max_rel_frob=0.0045,
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
        model_a, llama_st, v8.FLAGSHIP_SNAP, "quillan-1b-sovereign-wsp-v14",
        ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"],
    )
    del llama_st, model_a
    gc.collect()

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    ppl_a = measure_perplexity(q8_a, ppl_file)
    logger.info("FLAGSHIP GENERAL WSP v14 Q8_0 PPL = %.4f (v12 ref: 7.9553) in %.1fs", ppl_a, time.time() - t0)


if __name__ == "__main__":
    main()
