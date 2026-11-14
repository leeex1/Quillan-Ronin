"""Flagship v18: fp32 PPL scan of covariance-steered ternary folds (no training, closed-form only).
Calibration: gsm8k train / ai2_arc train / mmlu dev only. HELD_OUT_PPL_TEXT used for MEASUREMENT only.
Exports the best multiplier (if it beats the fp32 baseline) to quillan-1b-sovereign-wsp-v18-q8_0.gguf.
"""
import os, sys, gc, time, json, logging
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from tokenizers import Tokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import wsp_v8_closed_form_sota_compiler as v8
import compile_sovereign_fleet_wsp_v12_q8 as v12
import compile_flagship_general_v14 as v14
import compile_flagship_true_wsp_v17 as v17
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Flagship_Scan_v18")
logging.getLogger("httpx").setLevel(logging.WARNING)

MULTS = [0.0, 0.5, 1.0, 2.0, 3.0]


@torch.no_grad()
def fp32_ppl(model, ids_all, win=512):
    nll, n = 0.0, 0
    for s in range(0, len(ids_all) - 1, win):
        chunk = ids_all[s:s + win + 1]
        if len(chunk) < 8:
            break
        x = torch.tensor([chunk[:-1]]); y = torch.tensor([chunk[1:]])
        lg = model(x).logits[0].float()
        nll += F.cross_entropy(lg, y[0], reduction="sum").item(); n += y.numel()
    return float(torch.exp(torch.tensor(nll / n)))


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib = v14.build_general_curated_pairs(test_fps, seed=9018)
    projs = ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

    qst = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    ql = [10, 12, 14, 16, 18, 20, 22]
    qd = {p: [qst[f"model.layers.{l}.{p}.weight"].float().clone() for l in ql] for p in projs}
    del qst
    bst = load_file(v12.find_bitnet_snapshot_path())
    bl = [8, 11, 14, 17, 20, 23]
    bd = {p: [bst[f"model.layers.{l}.{p}.weight"].float().clone() for l in bl] for p in projs}
    del bst
    dd = v8.load_dragon_ffn_down_tensors()
    dragon_down = [dd[k] for k in sorted(dd)]
    del dd; gc.collect()

    tok = Tokenizer.from_file(os.path.join(v8.FLAGSHIP_SNAP, "tokenizer.json"))
    enc_chat = lambda s, u: tok.encode(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False).ids
    enc_tok = lambda t: tok.encode(t, add_special_tokens=False).ids
    model = transformers.AutoModelForCausalLM.from_pretrained(v8.FLAGSHIP_SNAP, dtype=torch.float32).eval()
    ppl_ids = [128000] + enc_tok(HELD_OUT_PPL_TEXT)

    v12.apply_equinorm_and_syntax_calibration(
        model.model.embed_tokens.weight, [32, 33, 34, 35, 36], [362, 426, 356, 423, 469], [37815, 595, 7861, 76642], 1680)
    base_ppl = fp32_ppl(model, ppl_ids)
    logger.info("fp32 baseline (equinorm only) PPL=%.4f", base_ppl)
    U_opt = v8.compute_option_null_basis(model.lm_head.weight.detach().float(), enc_tok)
    U_ref = v12.compute_rank_k_refusal_subspace(model.model.embed_tokens.weight, enc_tok, model, enc_chat, rank=3)
    with torch.no_grad():
        for l in [13, 14, 15]:
            W = model.model.layers[l].mlp.down_proj.weight; W.sub_(0.70 * U_ref @ (U_ref.t() @ W))
        for l in [13, 14]:
            W = model.model.layers[l].self_attn.o_proj.weight; W.sub_(0.40 * U_ref @ (U_ref.t() @ W))
    abl_ppl = fp32_ppl(model, ppl_ids)
    logger.info("after abliteration PPL=%.4f", abl_ppl)

    layers = list(range(4, 16))
    U_sov = v17.compute_layer_covariance_bases(model, enc_chat, calib, layers, rank=32)
    deltas = {}
    with torch.no_grad():
        for idx, l in enumerate(layers):
            ramp = 0.5 + 0.5 * ((idx + 1) / len(layers))
            L = model.model.layers[l]
            def B(w, donors, us, uo, g):
                return v17.build_steered_ternary_moe_delta(w, donors, us, rank=32, beta=0.15, gamma_rel=g * ramp, U_opt=uo)
            qi, bi = idx % len(ql), idx % len(bl)
            deltas[(l, "o")] = 0.5 * (B(L.self_attn.o_proj.weight, qd[projs[0]][qi], U_sov[l], U_opt, 0.004) + B(L.self_attn.o_proj.weight, bd[projs[0]][bi], U_sov[l], U_opt, 0.004))
            deltas[(l, "g")] = 0.5 * (B(L.mlp.gate_proj.weight, qd[projs[1]][qi], None, None, 0.006) + B(L.mlp.gate_proj.weight, bd[projs[1]][bi], None, None, 0.006))
            deltas[(l, "u")] = 0.5 * (B(L.mlp.up_proj.weight, qd[projs[2]][qi], None, None, 0.006) + B(L.mlp.up_proj.weight, bd[projs[2]][bi], None, None, 0.006))
            deltas[(l, "d")] = (B(L.mlp.down_proj.weight, qd[projs[3]][qi], U_sov[l], U_opt, 0.008) + B(L.mlp.down_proj.weight, bd[projs[3]][bi], U_sov[l], U_opt, 0.008) + B(L.mlp.down_proj.weight, dragon_down[idx % len(dragon_down)], U_sov[l], U_opt, 0.008)) / 3.0
    logger.info("deltas built in %.0fs", time.time() - t0)

    def getw(l, k):
        L = model.model.layers[l]
        return {"o": L.self_attn.o_proj, "g": L.mlp.gate_proj, "u": L.mlp.up_proj, "d": L.mlp.down_proj}[k].weight

    applied = {}
    def set_mult(kinds, m):
        with torch.no_grad():
            for (l, k), d in deltas.items():
                if k not in kinds:
                    continue
                cur = applied.get((l, k), 0.0)
                getw(l, k).add_((m - cur) * d); applied[(l, k)] = m

    results = {}
    for kinds in ("odgu", "d", "o", "gu"):
        for m in MULTS[1:]:
            set_mult("odgu", 0.0)
            set_mult(kinds, m)
            p = fp32_ppl(model, ppl_ids)
            results[f"{kinds}@{m}"] = p
            logger.info("scan kinds=%s mult=%.1f PPL=%.4f (abl-only %.4f)", kinds, m, p, abl_ppl)
    best = min(results, key=results.get)
    logger.info("BEST %s PPL=%.4f ; abl-only %.4f ; baseline %.4f", best, results[best], abl_ppl, base_ppl)
    with open(r"C:\02_QUILLAN\exports\flagship_v18_scan.json", "w") as f:
        json.dump({"base": base_ppl, "abl": abl_ppl, "scan": results}, f, indent=1)

    kinds, m = best.split("@")
    set_mult("odgu", 0.0)
    set_mult(kinds, float(m))
    model.to(torch.bfloat16)
    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    q8 = v12.export_model_q8_gguf(model, llama_st, v8.FLAGSHIP_SNAP, "quillan-1b-sovereign-wsp-v18",
                                  ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor_flagship_v18.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    logger.info("GGUF Q8_0 PPL (%s) = %.4f (v15 ref 7.9463) in %.0fs", best, measure_perplexity(q8, ppl_file), time.time() - t0)


if __name__ == "__main__":
    main()
