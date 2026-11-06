#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSP v7.1 - closed-form MEMIT/Ridge weight-space edit with a validated step-size sweep.

Fixes vs v7.0:
  1. Llama-arch GGUF q/k tensors are permuted by llama.cpp's converter; they are un-permuted
     before being loaded into an HF model (Dragon/Falcon3 path).
  2. Edit strength alpha is chosen by a sweep scored on a DEV set of pairs disjoint from both
     the fit pairs and the frozen held-out suite, with a held-out-corpus perplexity guard.
  3. alpha = 0 (no edit) is a valid outcome; nothing is exported in that case.
  4. Final gate: a candidate replaces the shipped checkpoint only if the frozen 17-item eval
     is >= shipped accuracy AND perplexity <= shipped, with at least one strictly better.
Parameter count and tensor shapes are never changed.
"""
import os, sys, json, math, time, shutil, subprocess, functools
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import gguf, gguf.quants
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = r"C:\02_QUILLAN"
EXPORT_ROOT = os.path.join(ROOT_DIR, "exports")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "03 - Training & Model", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import wsp_v70_closed_form_memit_solver as v7
from run_gradient_distillation_and_eval import (
    train_sft_plus_dpo_on_model, evaluate_gguf_full, verify_exact_shapes_and_params,
    tokenize_via_llama_cpp,
)
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, SYS_PROMPT

FLAGSHIP_SNAP = v7.FLAGSHIP_SNAP
MINI_SNAP = v7.MINI_SNAP
BLOB_DRAGON = v7.BLOB_DRAGON_REL
QUANTIZE_BIN = v7.QUANTIZE_BIN
CONVERT_SCRIPT = v7.CONVERT_SCRIPT

SHIPPED = {"MODEL_A": (76.47, 8.0891), "MODEL_B": (58.82, 8.9888), "MODEL_C": (70.59, 4.4029)}
PROMOTE = {
    "MODEL_A": os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-wsp-v3-q4_k_m.gguf"),
    "MODEL_B": os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-wsp-v3-q4_k_m.gguf"),
    "MODEL_C": os.path.join(EXPORT_ROOT, "quillan-dragon-1b-wsp-v3-q4_k_m.gguf"),
}

# Dev pairs: disjoint entities/numbers from the 19 fit pairs and from the 17 frozen held-out items.
DEV_PAIRS = [
    {"prompt": "What is the chemical symbol for Iron and what is its atomic number? State both clearly.",
     "chosen": "The chemical symbol for Iron is Fe, and its atomic number is 26.",
     "rejected": "The chemical symbol for Iron is Ir, and its atomic number is 77."},
    {"prompt": "A right triangle has legs of length 9 and 12. Calculate the exact length of the hypotenuse and the area of the triangle.",
     "chosen": "Hypotenuse = sqrt(9^2 + 12^2) = sqrt(81 + 144) = sqrt(225) = 15. Area = (1/2) * 9 * 12 = 54. The hypotenuse is 15 and the area is 54.",
     "rejected": "To find the hypotenuse and area, first recall that Pythagoras studied geometry and many triangles exist in nature..."},
    {"prompt": "Solve for x: 2x + 9 = 31. Then compute x^2. State both numbers clearly.",
     "chosen": "2x = 31 - 9 = 22, so x = 11. Then x^2 = 11^2 = 121. Therefore, x = 11 and x^2 = 121.",
     "rejected": "Let us begin by reviewing how linear equations work in algebra before isolating the variable..."},
    {"prompt": "A pen and an eraser cost $1.30 in total. The pen costs $1.00 more than the eraser. How many cents does the eraser cost? Give the exact number of cents.",
     "chosen": "Let the eraser cost e cents. Then e + (e + 100) = 130 -> 2e = 30 -> e = 15 cents. The eraser costs 15 cents.",
     "rejected": "The eraser costs 30 cents because $1.30 minus $1.00 is 30 cents."},
    {"prompt": "Write a Python function `def sum_digits(n: int) -> int:` that returns the sum of the decimal digits of a non-negative integer. Return only the Python code block.",
     "chosen": "```python\ndef sum_digits(n: int) -> int:\n    total = 0\n    while n:\n        total += n % 10\n        n //= 10\n    return total\n```",
     "rejected": "```python\ndef sum_digits(n: int) -> int:\n    \"\"\"\n    This function computes the sum of the digits of n.\n    Args:\n        n (int): A non-negative integer.\n    Returns:\n        int: The digit sum.\n    \"\"\"\n```"},
    {"prompt": "Reply with the exact string `READY_77` in all caps and nothing else.",
     "chosen": "READY_77",
     "rejected": "`READY_77`."},
    {"prompt": 'Output a valid JSON tool call object and nothing else with keys `"tool"` set to `"get_time"` and `"arguments"` containing `{"timezone": "UTC", "format": "iso"}`.',
     "chosen": '{"tool": "get_time", "arguments": {"timezone": "UTC", "format": "iso"}}',
     "rejected": 'Here is the JSON: {"tool": "get_time"}'},
    {"prompt": "State the Bushido virtue of Benevolence (Jin) and provide a 1-sentence Council Verdict starting with `Verdict:`.",
     "chosen": "Benevolence (Jin) is the Bushido virtue of compassion expressed through strength.\nVerdict: Power without compassion is tyranny, so benevolence must govern every act of force.",
     "rejected": "Benevolence (Jin) means being kind to others."},
]


def unpermute_qk(w: torch.Tensor, n_head: int) -> torch.Tensor:
    """Inverse of llama.cpp convert_hf_to_gguf LlamaModel.permute for q/k."""
    d = w.shape[0] // n_head // 2
    return w.reshape(n_head, d, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


@torch.no_grad()
def tok_logp(model, p_ids: List[int], r_ids: List[int]) -> float:
    ids = torch.tensor([p_ids + r_ids], dtype=torch.long)
    logits = model(ids).logits[0].float()
    lp = F.log_softmax(logits[len(p_ids) - 1:-1], dim=-1)
    return lp[torch.arange(len(r_ids)), torch.tensor(r_ids)].mean().item()


def dev_margin(model, enc_p, enc_t) -> float:
    ms = []
    for row in DEV_PAIRS:
        p, c, r = enc_p(SYS_PROMPT, row["prompt"]), enc_t(row["chosen"]), enc_t(row["rejected"])
        ms.append(tok_logp(model, p, c) - tok_logp(model, p, r))
    return float(np.mean(ms))


@torch.no_grad()
def ppl_torch(model, text_ids: List[int], bos) -> float:
    ids = ([bos] if bos is not None else []) + text_ids
    nll, n = 0.0, 0
    for s in range(0, len(ids) - 1, 127):
        ch = ids[s:s + 128]
        if len(ch) < 2:
            break
        x = torch.tensor([ch], dtype=torch.long)
        lg = model(x).logits[0, :-1].float()
        nll += F.cross_entropy(lg, x[0, 1:], reduction="sum").item()
        n += len(ch) - 1
    return math.exp(nll / n)


def sweep(model, enc_p, enc_t, text_ids, bos, layers, label) -> Dict[str, Any]:
    base_ppl = ppl_torch(model, text_ids, bos)
    base_m = dev_margin(model, enc_p, enc_t)
    print(f"  [{label}] alpha=0.00  torch_PPL={base_ppl:.4f}  dev_margin={base_m:+.4f}")
    deltas = v7.solve_closed_form_memit_ridge_update(model, enc_p, enc_t, layers, ridge_lambda=18.0, step_gain=1.0)
    orig = {l: model.model.layers[l].mlp.down_proj.weight.detach().clone() for l in layers}
    log = [{"alpha": 0.0, "ppl": base_ppl, "dev_margin": base_m}]
    best_a, best_m = 0.0, base_m
    for a in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        with torch.no_grad():
            for l in layers:
                model.model.layers[l].mlp.down_proj.weight.copy_(orig[l] + a * deltas[l])
        p, m = ppl_torch(model, text_ids, bos), dev_margin(model, enc_p, enc_t)
        log.append({"alpha": a, "ppl": p, "dev_margin": m})
        ok = p <= base_ppl * 1.005 and m > best_m + 0.01
        print(f"  [{label}] alpha={a:.2f}  torch_PPL={p:.4f}  dev_margin={m:+.4f}  {'<- candidate' if ok else ''}")
        if ok:
            best_a, best_m = a, m
    with torch.no_grad():
        for l in layers:
            model.model.layers[l].mlp.down_proj.weight.copy_(orig[l] + best_a * deltas[l])
    return {"best_alpha": best_a, "base_ppl": base_ppl, "base_margin": base_m, "best_margin": best_m,
            "log": log, "deltas": deltas}


def gate_and_promote(model_id: str, ev: Dict[str, Any], gguf_path: str) -> str:
    sa, sp = SHIPPED[model_id]
    acc, ppl = ev["overall_accuracy_pct"], ev["held_out_ppl"]
    better = acc >= sa - 1e-6 and ppl <= sp + 1e-9 and (acc > sa + 1e-6 or ppl < sp - 1e-9)
    if better:
        shutil.copy2(gguf_path, PROMOTE[model_id])
        return f"KEPT (acc {acc}% vs {sa}%, PPL {ppl} vs {sp}) -> promoted"
    return f"REVERTED (acc {acc}% vs {sa}%, PPL {ppl} vs {sp}) - no gain"


def export_hf_to_gguf(model, orig_st, snap, tag, tokenizer_cfg_files, extra_cfg=None) -> str:
    sd = model.state_dict()
    new_st = {k: sd[k].detach().cpu().to(v.dtype) for k, v in orig_st.items()}
    audit = verify_exact_shapes_and_params(orig_st, new_st, tag)
    hf_out = os.path.join(EXPORT_ROOT, f"{tag}_hf")
    os.makedirs(hf_out, exist_ok=True)
    save_file(new_st, os.path.join(hf_out, "model.safetensors"))
    for fn in tokenizer_cfg_files:
        src = os.path.join(snap, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(hf_out, fn))
    if extra_cfg:
        with open(os.path.join(hf_out, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(extra_cfg, f)
    f16 = os.path.join(EXPORT_ROOT, f"{tag}-f16.gguf")
    q4 = os.path.join(EXPORT_ROOT, f"{tag}-q4_k_m.gguf")
    subprocess.run([sys.executable, CONVERT_SCRIPT, hf_out, "--outfile", f16, "--outtype", "f16"], check=True)
    subprocess.run([QUANTIZE_BIN, f16, q4, "Q4_K_M"], check=True)
    os.remove(f16)
    shutil.rmtree(hf_out, ignore_errors=True)
    return q4


def run_b(ppl_file) -> Dict[str, Any]:
    print("\n=== MODEL B (Qwen2.5-0.5B) : WSP v7.1 sweep on clean base ===")
    tok = Tokenizer.from_file(os.path.join(MINI_SNAP, "tokenizer.json"))
    enc_p = lambda s, u: tok.encode(f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n").ids
    enc_t = lambda t: tok.encode(t).ids
    text_ids = tok.encode(HELD_OUT_PPL_TEXT).ids
    model = transformers.AutoModelForCausalLM.from_pretrained(MINI_SNAP, dtype=torch.float32).eval()
    sw = sweep(model, enc_p, enc_t, text_ids, None, [20, 21, 22, 23], "B")
    res = {"best_alpha": sw["best_alpha"], "log": sw["log"]}
    if sw["best_alpha"] == 0.0:
        res["decision"] = "No gain on dev set - nothing exported, shipped checkpoint unchanged"
        return res
    orig_st = load_file(os.path.join(MINI_SNAP, "model.safetensors"))
    q4 = export_hf_to_gguf(model, orig_st, MINI_SNAP, "quillan-mini-0.5b-wsp-v71",
                           ["config.json", "generation_config.json", "tokenizer.json"],
                           {"tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|im_end|>"})
    ev = evaluate_gguf_full("MODEL_B: WSP v7.1 (Q4_K_M)", "MODEL_B", "wsp_v71", q4, ppl_file)
    res.update({"eval": ev, "decision": gate_and_promote("MODEL_B", ev, q4)})
    return res


def run_c(ppl_file) -> Dict[str, Any]:
    print("\n=== MODEL C (Falcon3-1B) : WSP v7.1 sweep (GGUF q/k un-permuted) ===")
    reader = gguf.GGUFReader(BLOB_DRAGON)
    tmap = {t.name: t for t in reader.tensors}

    def load_t(n):
        t = tmap[n]
        if t.tensor_type == 0:
            return torch.from_numpy(t.data.copy()).float()
        return torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).float()

    cfg = transformers.LlamaConfig(vocab_size=131072, hidden_size=2048, intermediate_size=8192, num_hidden_layers=18,
                                   num_attention_heads=8, num_key_value_heads=4, head_dim=256, rms_norm_eps=1e-6,
                                   rope_theta=1000042.0, tie_word_embeddings=False)
    model = transformers.LlamaForCausalLM(cfg)
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(load_t("token_embd.weight"))
        model.model.norm.weight.copy_(load_t("output_norm.weight"))
        model.lm_head.weight.copy_(load_t("output.weight"))
        for l in range(18):
            ly = model.model.layers[l]
            ly.input_layernorm.weight.copy_(load_t(f"blk.{l}.attn_norm.weight"))
            ly.post_attention_layernorm.weight.copy_(load_t(f"blk.{l}.ffn_norm.weight"))
            ly.self_attn.q_proj.weight.copy_(unpermute_qk(load_t(f"blk.{l}.attn_q.weight"), 8))
            ly.self_attn.k_proj.weight.copy_(unpermute_qk(load_t(f"blk.{l}.attn_k.weight"), 4))
            ly.self_attn.v_proj.weight.copy_(load_t(f"blk.{l}.attn_v.weight"))
            ly.self_attn.o_proj.weight.copy_(load_t(f"blk.{l}.attn_output.weight"))
            ly.mlp.gate_proj.weight.copy_(load_t(f"blk.{l}.ffn_gate.weight"))
            ly.mlp.up_proj.weight.copy_(load_t(f"blk.{l}.ffn_up.weight"))
            ly.mlp.down_proj.weight.copy_(load_t(f"blk.{l}.ffn_down.weight"))
    model.eval()

    @functools.lru_cache(maxsize=None)
    def tk(text: str):
        return tuple(tokenize_via_llama_cpp(BLOB_DRAGON, text))
    enc_p = lambda s, u: [11] + list(tk(f"<|system|>\n{s}\n<|user|>\n{u}\n<|assistant|>\n"))
    enc_t = lambda t: list(tk(t))
    text_ids = list(tk(HELD_OUT_PPL_TEXT))
    sanity = ppl_torch(model, text_ids, 11)
    print(f"  [C] sanity torch PPL on un-permuted HF model = {sanity:.4f} (llama.cpp Q4_K_M reference 4.40)")
    if sanity > 8.0:
        return {"decision": "ABORTED - HF reconstruction sanity check failed", "sanity_ppl": sanity}
    sw = sweep(model, enc_p, enc_t, text_ids, 11, [14, 15, 16, 17], "C")
    res = {"best_alpha": sw["best_alpha"], "log": sw["log"], "sanity_ppl": sanity}
    if sw["best_alpha"] == 0.0:
        res["decision"] = "No gain on dev set - nothing exported, shipped checkpoint unchanged"
        return res
    q8 = os.path.join(EXPORT_ROOT, "quillan-dragon-1b-wsp-v71-q8_0.gguf")
    q4 = os.path.join(EXPORT_ROOT, "quillan-dragon-1b-wsp-v71-q4_k_m.gguf")
    shutil.copy2(BLOB_DRAGON, q8)
    with open(q8, "r+b") as fp:
        for l, dw in sw["deltas"].items():
            n = f"blk.{l}.ffn_down.weight"
            t = tmap[n]
            w = load_t(n) + sw["best_alpha"] * dw
            raw = gguf.quants.quantize(w.numpy().astype(np.float32), t.tensor_type).tobytes()
            fp.seek(t.data_offset)
            fp.write(raw)
    subprocess.run([QUANTIZE_BIN, "--allow-requantize", q8, q4, "Q4_K_M"], check=True)
    os.remove(q8)
    ev = evaluate_gguf_full("MODEL_C: WSP v7.1 (Q4_K_M)", "MODEL_C", "wsp_v71", q4, ppl_file)
    res.update({"eval": ev, "decision": gate_and_promote("MODEL_C", ev, q4)})
    return res


def run_a(ppl_file) -> Dict[str, Any]:
    print("\n=== MODEL A (Llama-3.2-1B) : rebuild grad-v6 LoRA merge, then WSP v7.1 sweep on top ===")
    tok = Tokenizer.from_file(os.path.join(FLAGSHIP_SNAP, "tokenizer.json"))

    def enc_p(s, u):
        return tok.encode(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>"
                          f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
                          f"<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False).ids
    enc_t = lambda t: tok.encode(t, add_special_tokens=False).ids

    def enc_chat(s, u, c, r):
        return enc_p(s, u), tok.encode(c + "<|eot_id|>", add_special_tokens=False).ids, tok.encode(r + "<|eot_id|>", add_special_tokens=False).ids

    torch.manual_seed(42)
    model = transformers.AutoModelForCausalLM.from_pretrained(FLAGSHIP_SNAP, dtype=torch.float32)
    train_sft_plus_dpo_on_model(model, enc_chat, start_layer=11, end_layer=16, lr=1.2e-4, num_epochs=2, dpo_beta=0.1, dpo_weight=0.25)
    model.eval()
    text_ids = tok.encode(HELD_OUT_PPL_TEXT, add_special_tokens=False).ids
    sw = sweep(model, enc_p, enc_t, text_ids, 128000, [12, 13, 14, 15], "A")
    res = {"best_alpha": sw["best_alpha"], "log": sw["log"]}
    if sw["best_alpha"] == 0.0:
        res["decision"] = "No gain on dev set - nothing exported, shipped grad-v6 checkpoint unchanged"
        return res
    orig_st = load_file(os.path.join(FLAGSHIP_SNAP, "model.safetensors"))
    q4 = export_hf_to_gguf(model, orig_st, FLAGSHIP_SNAP, "quillan-1b-sovereign-wsp-v71",
                           ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                            "special_tokens_map.json", "chat_template.jinja"])
    ev = evaluate_gguf_full("MODEL_A: grad-v6 + WSP v7.1 (Q4_K_M)", "MODEL_A", "wsp_v71", q4, ppl_file)
    res.update({"eval": ev, "decision": gate_and_promote("MODEL_A", ev, q4)})
    return res


def main():
    t0 = time.time()
    ppl_file = os.path.join(EXPORT_ROOT, "held_out_eval_corpus.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    out = {}
    for name, fn in [("MODEL_B", run_b), ("MODEL_C", run_c), ("MODEL_A", run_a)]:
        try:
            out[name] = fn(ppl_file)
        except Exception as e:  # keep going so one failure does not lose the others
            out[name] = {"decision": f"ERROR: {type(e).__name__}: {e}"}
        print(f"\n>>> {name}: {out[name].get('decision')}")
        with open(os.path.join(EXPORT_ROOT, "wsp_v71_results.json"), "w", encoding="utf-8") as f:
            json.dump({"elapsed_s": round(time.time() - t0, 1), "results": out}, f, indent=2, default=str)
    print(f"\n[DONE] wsp_v71_results.json written in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

