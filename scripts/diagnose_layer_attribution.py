"""
Per-Layer Diagnostic Attribution (Logit-Lens Residual Stream Attribution)
Measures each layer's (o_proj/self_attn and down_proj/mlp) exact log-probability
and margin contribution across Math, Code, Logic/MC, and Prose (EXTRA_ANCHOR_TEXT).
Zero gradient descent. Zero test-set / HELD_OUT_PPL_TEXT contamination.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import transformers

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
import compile_sovereign_fleet_wsp_v12_q8 as v12


def build_curated_probes() -> Dict[str, List[Tuple[str, str, str]]]:
    """Curated non-test probes: (prompt, y_good_prefix, y_bad_prefix)."""
    return {
        "math": [
            ("Question: A baker makes 12 loaves per hour for 3 hours, then sells 10 loaves. How many remain? Answer with just the number.\n", "26", "To"),
            ("Question: A train travels 60 miles per hour for 2.5 hours. How many miles does it travel? Answer with just the number.\n", "150", "Step"),
            ("Question: Sara has 45 apples and gives 18 to Tom. How many apples does Sara have left? Answer with just the number.\n", "27", "Let"),
            ("Question: What is 15 * 4 + 12? Answer with just the number.\n", "72", "##"),
        ],
        "code": [
            ("Complete the Python function body directly with no docstring or preamble:\ndef add_two(a: int, b: int) -> int:\n", "    return", '    """'),
            ("Complete the Python function body directly with no docstring or preamble:\ndef is_even(n: int) -> bool:\n", "    return", "Here"),
            ("Complete the Python function body directly with no docstring or preamble:\ndef max_of_list(xs: list) -> int:\n", "    return", "#"),
            ("Complete the Python function body directly with no docstring or preamble:\ndef square_sum(n: int) -> int:\n", "    return", '    """'),
        ],
        "logic_mc": [
            ("Which option is a prime number?\nA) 9\nB) 11\nC) 15\nD) 21\nReply with 'Answer: <letter>'.\n", "Answer", "To"),
            ("If all bloops are razzies and all razzies are lazzies, all bloops are:\nA) fast\nB) lazzies\nC) blue\nD) small\nReply with 'Answer: <letter>'.\n", "Answer", "Here"),
            ("Which HTTP header prevents clickjacking?\nA) X-Frame-Options\nB) Accept\nC) Host\nD) ETag\nReply with 'Answer: <letter>'.\n", "Answer", "To"),
            ("What is the next number in 2, 4, 8, 16, ?\nA) 20\nB) 24\nC) 32\nD) 30\nReply with 'Answer: <letter>'.\n", "Answer", "Let"),
        ],
    }


def first_token_id(tok, text: str) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    return int(ids[0]) if ids else 0


@torch.no_grad()
def analyze_model(model_dir: str, name: str, out_json: str):
    tok = transformers.AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    layers = model.model.layers
    n_layers = len(layers)
    norm = model.model.norm
    lm_head = model.lm_head

    # Capture pre-layer, post-attn (via pre-mlp input_layernorm/post_attention_layernorm), and post-layer
    cache: Dict[int, Dict[str, torch.Tensor]] = {}
    hooks = []

    for idx, layer in enumerate(layers):
        def _make_attn_hook(i: int):
            def _hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                cache.setdefault(i, {})["attn_out"] = o[:, -1, :].detach()
            return _hook

        def _make_mlp_hook(i: int):
            def _hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                cache.setdefault(i, {})["mlp_out"] = o[:, -1, :].detach()
            return _hook

        def _make_layer_pre_hook(i: int):
            def _hook(mod, inp):
                cache.setdefault(i, {})["h_in"] = inp[0][:, -1, :].detach()
            return _hook

        hooks.append(layer.register_forward_pre_hook(_make_layer_pre_hook(idx)))
        hooks.append(layer.self_attn.register_forward_hook(_make_attn_hook(idx)))
        hooks.append(layer.mlp.register_forward_hook(_make_mlp_hook(idx)))

    probes = build_curated_probes()
    stats = {
        dom: {
            l: {"attn_dlogp": 0.0, "mlp_dlogp": 0.0, "attn_dmargin": 0.0, "mlp_dmargin": 0.0, "n": 0}
            for l in range(n_layers)
        }
        for dom in probes
    }

    for dom, items in probes.items():
        for prompt, good_s, bad_s in items:
            cache.clear()
            ids = tok(prompt, return_tensors="pt")["input_ids"]
            _ = model(ids)
            yg = first_token_id(tok, good_s)
            yb = first_token_id(tok, bad_s)
            for l in range(n_layers):
                h_in = cache[l]["h_in"]
                h_mid = h_in + cache[l]["attn_out"]
                h_out = h_mid + cache[l]["mlp_out"]

                lp_in = F.log_softmax(lm_head(norm(h_in)), dim=-1)[0]
                lp_mid = F.log_softmax(lm_head(norm(h_mid)), dim=-1)[0]
                lp_out = F.log_softmax(lm_head(norm(h_out)), dim=-1)[0]

                s = stats[dom][l]
                s["attn_dlogp"] += float(lp_mid[yg] - lp_in[yg])
                s["mlp_dlogp"] += float(lp_out[yg] - lp_mid[yg])
                s["attn_dmargin"] += float((lp_mid[yg] - lp_mid[yb]) - (lp_in[yg] - lp_in[yb]))
                s["mlp_dmargin"] += float((lp_out[yg] - lp_out[yb]) - (lp_mid[yg] - lp_mid[yb]))
                s["n"] += 1

    # Also measure prose (EXTRA_ANCHOR_TEXT) per-layer NLL delta
    prose_stats = {l: {"attn_dlogp": 0.0, "mlp_dlogp": 0.0, "n": 0} for l in range(n_layers)}
    prose_ids = tok(v12.EXTRA_ANCHOR_TEXT, return_tensors="pt")["input_ids"][0]
    for pos in [min(32, len(prose_ids) - 2), min(64, len(prose_ids) - 2), min(96, len(prose_ids) - 2)]:
        if pos < 8:
            continue
        cache.clear()
        _ = model(prose_ids[:pos].unsqueeze(0))
        yt = int(prose_ids[pos])
        for l in range(n_layers):
            h_in = cache[l]["h_in"]
            h_mid = h_in + cache[l]["attn_out"]
            h_out = h_mid + cache[l]["mlp_out"]
            lp_in = F.log_softmax(lm_head(norm(h_in)), dim=-1)[0]
            lp_mid = F.log_softmax(lm_head(norm(h_mid)), dim=-1)[0]
            lp_out = F.log_softmax(lm_head(norm(h_out)), dim=-1)[0]
            prose_stats[l]["attn_dlogp"] += float(lp_mid[yt] - lp_in[yt])
            prose_stats[l]["mlp_dlogp"] += float(lp_out[yt] - lp_mid[yt])
            prose_stats[l]["n"] += 1

    for h in hooks:
        h.remove()

    summary: Dict[str, object] = {"model": name, "n_layers": n_layers, "domains": {}, "prose": {}}
    print(f"\n=== LAYER ATTRIBUTION: {name} ({n_layers} layers) ===")
    for dom in probes:
        dom_rows = []
        for l in range(n_layers):
            s = stats[dom][l]
            n = max(1, s["n"])
            row = {
                "layer": l,
                "attn_dlogp": round(s["attn_dlogp"] / n, 4),
                "mlp_dlogp": round(s["mlp_dlogp"] / n, 4),
                "attn_dmargin": round(s["attn_dmargin"] / n, 4),
                "mlp_dmargin": round(s["mlp_dmargin"] / n, 4),
            }
            dom_rows.append(row)
        worst_attn = sorted(dom_rows, key=lambda r: r["attn_dmargin"])[:3]
        worst_mlp = sorted(dom_rows, key=lambda r: r["mlp_dmargin"])[:3]
        print(f"[{dom}] worst attn_dmargin: {[(r['layer'], r['attn_dmargin']) for r in worst_attn]}")
        print(f"[{dom}] worst  mlp_dmargin: {[(r['layer'], r['mlp_dmargin']) for r in worst_mlp]}")
        summary["domains"][dom] = dom_rows

    prose_rows = []
    for l in range(n_layers):
        s = prose_stats[l]
        n = max(1, s["n"])
        prose_rows.append({
            "layer": l,
            "attn_dlogp": round(s["attn_dlogp"] / n, 4),
            "mlp_dlogp": round(s["mlp_dlogp"] / n, 4),
        })
    worst_prose_mlp = sorted(prose_rows, key=lambda r: r["mlp_dlogp"])[:3]
    print(f"[prose] worst mlp_dlogp: {[(r['layer'], r['mlp_dlogp']) for r in worst_prose_mlp]}")
    summary["prose"] = prose_rows

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    analyze_model(args.model_dir, args.name, args.out_json)
