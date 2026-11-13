"""
=============================================================================
FLAGSHIP 1.2B TRUE WEIGHT-SPACE PROGRAMMING (WSP v17 Q8_0)
=============================================================================
Implements the exact 4-stage analytical framework from
weight_space_programming_formal_paper.md without backpropagation:
  1. Stage 1 (Sec 3): Closed-Form Refusal Subspace Abliteration
  2. Stage 2 (Sec 2.1): Activation Covariance Manifolds (Sigma_l = U_l L U_l^T)
  3. Stage 3 (Sec 2.2, 4.2, 4.3): Multi-Donor t-SVD + Covariance Steering
     (beta=0.15) + BitNet 1.58b Ternary Quantization {-1, 0, +1} on (W0, W1)
  4. Stage 4 (Sec 5.1): Analytical Closed-Form MoE Folding across all 4
     projection families (o_proj, gate_proj, up_proj, down_proj) in layers
     [4..15] with orthogonal host-complement & option null-space projection.
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
logger = logging.getLogger("Flagship_TrueWSP_v17")


def ternary_quantize(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Section 4.2: BitNet 1.58b ternary quantization {-1, 0, +1}."""
    gamma = 1.0 / (w.abs().mean() + eps)
    w_bar = torch.clamp(torch.round(w * gamma), -1.0, 1.0)
    return w_bar / gamma


def compute_layer_covariance_bases(
    model: torch.nn.Module,
    enc_chat,
    calib_pairs: List[tuple],
    layers: List[int],
    rank: int = 32,
) -> Dict[int, torch.Tensor]:
    """Section 2.1: Empirical second-moment covariance Sigma_l = U_l Lambda_l U_l^T."""
    d_model = model.config.hidden_size
    covs = {l: torch.zeros((d_model, d_model), dtype=torch.float32) for l in layers}
    counts = {l: 0 for l in layers}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h_f = h.detach().float().reshape(-1, d_model)
            covs[layer_idx].add_(h_f.t() @ h_f)
            counts[layer_idx] += h_f.shape[0]
        return hook_fn

    for l in layers:
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

    sys_prompt = "You are Quillan-Ronin, a precise general-purpose reasoning engine."
    with torch.no_grad():
        for item in calib_pairs[:48]:
            ids = (enc_chat(item.get("sys", sys_prompt), item["prompt"]) + enc_chat("", item["chosen"])[1:])[-256:]
            model(torch.tensor([ids], dtype=torch.long))

    for hk in hooks:
        hk.remove()

    bases: Dict[int, torch.Tensor] = {}
    for l in layers:
        sigma = covs[l] / max(1, counts[l])
        U, _, _ = torch.svd_lowrank(sigma, q=rank)
        bases[l] = U
    return bases


def build_steered_ternary_moe_delta(
    host_w: torch.Tensor,
    donor_w: torch.Tensor,
    U_sov: torch.Tensor | None,
    rank: int = 32,
    beta: float = 0.15,
    gamma_rel: float = 0.008,
    U_opt: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sections 2.2, 4.2, 4.3, 5.1:
    1. Project donor into host shape P(W_donor)
    2. Truncated SVD: U_r S_r V_r^T
    3. Steer U_r with sovereign covariance eigenvectors U_sov (beta=0.15)
    4. Ternary-quantize W1 = U_tilde * sqrt(S_r), W0 = sqrt(S_r) * V_r^T
    5. Project W1 @ W0 onto host orthogonal complement (I - V_h V_h^T) and U_opt
    """
    w_f = host_w.detach().float()
    d_out, d_in = w_f.shape

    # Section 2.2 dimension projection
    d_src_out, d_src_in = donor_w.shape
    row_idx = torch.arange(d_out) % d_src_out
    col_idx = torch.arange(d_in) % d_src_in
    proj = donor_w.float()[row_idx][:, col_idx]

    r = min(rank, min(d_out, d_in) - 2)
    U_r, S_r, V_r = torch.svd_lowrank(proj, q=r)
    sqrt_s = torch.sqrt(torch.clamp(S_r, min=1e-8))

    # Section 4.3 covariance steering when output dim matches hidden_size
    if U_sov is not None and U_sov.shape[0] == d_out:
        r_cov = min(r, U_sov.shape[1])
        U_steer = U_r.clone()
        U_steer[:, :r_cov] = F.normalize(
            (1.0 - beta) * U_r[:, :r_cov] + beta * U_sov[:, :r_cov], p=2, dim=0
        )
    else:
        U_steer = U_r

    # Section 4.2 symmetric split + BitNet 1.58b ternary quantization
    W1 = ternary_quantize(U_steer * sqrt_s.unsqueeze(0))
    W0 = ternary_quantize(sqrt_s.unsqueeze(1) * V_r.t())
    expert = W1 @ W0

    # Orthogonal complement of host right-singular subspace (additive, never averaging)
    _, _, V_h = torch.svd_lowrank(w_f, q=min(32, min(d_out, d_in) - 2))
    expert = expert - (expert @ V_h) @ V_h.t()
    if U_opt is not None and U_opt.shape[0] == d_out:
        expert = expert - U_opt @ (U_opt.t() @ expert)

    expert = expert / (torch.norm(expert) + 1e-6)
    return expert * (gamma_rel * torch.norm(w_f))


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib_pairs = v14.build_general_curated_pairs(test_fps, seed=9017)

    projs = ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

    qwen_st = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    qwen_layers = [10, 12, 14, 16, 18, 20, 22]
    qwen_donors = {p: [qwen_st[f"model.layers.{l}.{p}.weight"].float().clone() for l in qwen_layers] for p in projs}
    del qwen_st

    dragon_dict = v8.load_dragon_ffn_down_tensors()
    dragon_down = [dragon_dict[k] for k in sorted(dragon_dict)]
    del dragon_dict

    bitnet_st = load_file(v12.find_bitnet_snapshot_path())
    bitnet_layers = [8, 11, 14, 17, 20, 23]
    bitnet_donors = {p: [bitnet_st[f"model.layers.{l}.{p}.weight"].float().clone() for l in bitnet_layers] for p in projs}
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

    # Option equi-norm & syntax calibration
    v12.apply_equinorm_and_syntax_calibration(
        model_a.model.embed_tokens.weight, [32, 33, 34, 35, 36], [362, 426, 356, 423, 469], [37815, 595, 7861, 76642], 1680
    )
    W_head = model_a.lm_head.weight.detach().float()
    U_opt = v8.compute_option_null_basis(W_head, enc_tok_a)

    # Stage 1: Closed-Form Refusal Subspace Abliteration
    U_ref_a = v12.compute_rank_k_refusal_subspace(model_a.model.embed_tokens.weight, enc_tok_a, model_a, enc_chat_a, rank=3)
    with torch.no_grad():
        for l in [13, 14, 15]:
            W_d = model_a.model.layers[l].mlp.down_proj.weight
            W_d.sub_((0.70 * torch.matmul(U_ref_a, torch.matmul(U_ref_a.t(), W_d.float()))).to(W_d.dtype))
        for l in [13, 14]:
            W_o = model_a.model.layers[l].self_attn.o_proj.weight
            W_o.sub_((0.40 * torch.matmul(U_ref_a, torch.matmul(U_ref_a.t(), W_o.float()))).to(W_o.dtype))

    # Stage 2: Activation Covariance Manifolds Sigma_l = U_l Lambda_l U_l^T
    merge_layers = list(range(4, 16))
    logger.info("Computing layer activation covariance manifolds Sigma_l for layers %s...", merge_layers)
    U_sov_map = compute_layer_covariance_bases(model_a, enc_chat_a, calib_pairs, merge_layers, rank=32)

    # Stages 3 & 4: Multi-Donor Covariance-Steered Ternary MoE Folding across all 4 projections
    logger.info("Folding 4-model covariance-steered ternary MoE experts across layers [4..15]...")
    with torch.no_grad():
        for idx, l in enumerate(merge_layers):
            ramp = 0.5 + 0.5 * ((idx + 1) / len(merge_layers))
            U_sov = U_sov_map[l]

            # o_proj: Qwen + BitNet ternary experts (K=2)
            d_oq = build_steered_ternary_moe_delta(
                model_a.model.layers[l].self_attn.o_proj.weight,
                qwen_donors["self_attn.o_proj"][idx % len(qwen_layers)],
                U_sov, rank=32, beta=0.15, gamma_rel=0.004 * ramp, U_opt=U_opt,
            )
            d_ob = build_steered_ternary_moe_delta(
                model_a.model.layers[l].self_attn.o_proj.weight,
                bitnet_donors["self_attn.o_proj"][idx % len(bitnet_layers)],
                U_sov, rank=32, beta=0.15, gamma_rel=0.004 * ramp, U_opt=U_opt,
            )
            model_a.model.layers[l].self_attn.o_proj.weight.add_(0.5 * (d_oq + d_ob))

            # gate_proj: Qwen + BitNet ternary experts (K=2)
            d_gq = build_steered_ternary_moe_delta(
                model_a.model.layers[l].mlp.gate_proj.weight,
                qwen_donors["mlp.gate_proj"][idx % len(qwen_layers)],
                None, rank=32, beta=0.15, gamma_rel=0.006 * ramp, U_opt=None,
            )
            d_gb = build_steered_ternary_moe_delta(
                model_a.model.layers[l].mlp.gate_proj.weight,
                bitnet_donors["mlp.gate_proj"][idx % len(bitnet_layers)],
                None, rank=32, beta=0.15, gamma_rel=0.006 * ramp, U_opt=None,
            )
            model_a.model.layers[l].mlp.gate_proj.weight.add_(0.5 * (d_gq + d_gb))

            # up_proj: Qwen + BitNet ternary experts (K=2)
            d_uq = build_steered_ternary_moe_delta(
                model_a.model.layers[l].mlp.up_proj.weight,
                qwen_donors["mlp.up_proj"][idx % len(qwen_layers)],
                None, rank=32, beta=0.15, gamma_rel=0.006 * ramp, U_opt=None,
            )
            d_ub = build_steered_ternary_moe_delta(
                model_a.model.layers[l].mlp.up_proj.weight,
                bitnet_donors["mlp.up_proj"][idx % len(bitnet_layers)],
                None, rank=32, beta=0.15, gamma_rel=0.006 * ramp, U_opt=None,
            )
            model_a.model.layers[l].mlp.up_proj.weight.add_(0.5 * (d_uq + d_ub))

            # down_proj: Qwen + BitNet + Dragon ternary experts (K=3)
            d_dq = build_steered_ternary_moe_delta(
                model_a.model.layers[l].mlp.down_proj.weight,
                qwen_donors["mlp.down_proj"][idx % len(qwen_layers)],
                U_sov, rank=32, beta=0.15, gamma_rel=0.008 * ramp, U_opt=U_opt,
            )
            d_db = build_steered_ternary_moe_delta(
                model_a.model.layers[l].mlp.down_proj.weight,
                bitnet_donors["mlp.down_proj"][idx % len(bitnet_layers)],
                U_sov, rank=32, beta=0.15, gamma_rel=0.008 * ramp, U_opt=U_opt,
            )
            d_dd = build_steered_ternary_moe_delta(
                model_a.model.layers[l].mlp.down_proj.weight,
                dragon_down[idx % len(dragon_down)],
                U_sov, rank=32, beta=0.15, gamma_rel=0.008 * ramp, U_opt=U_opt,
            )
            model_a.model.layers[l].mlp.down_proj.weight.add_((d_dq + d_db + d_dd) / 3.0)

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

    del U_sov_map, U_ref_a
    model_a.to(torch.bfloat16)
    gc.collect()

    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    q8_a = v12.export_model_q8_gguf(
        model_a, llama_st, v8.FLAGSHIP_SNAP, "quillan-1b-sovereign-wsp-v17",
        ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    )
    del llama_st, model_a
    gc.collect()

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor_flagship_v17.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    ppl_a = measure_perplexity(q8_a, ppl_file)
    logger.info("FLAGSHIP TRUE WSP v17 Q8_0 PPL = %.4f (v15 ref: 7.9463) in %.1fs", ppl_a, time.time() - t0)


if __name__ == "__main__":
    main()


