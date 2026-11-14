"""
Per-Layer Logit-Lens Attribution directly on Dragon 1.0B GGUF (18-layer Llama arch).
Zero gradient descent. Zero test-set / HELD_OUT_PPL_TEXT contamination.
"""
import argparse
import json
import math
import os
import sys
from typing import Dict, List

import gguf
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
import compile_sovereign_fleet_wsp_v12_q8 as v12
import diagnose_layer_attribution as dla


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def apply_rope(x: torch.Tensor, pos: torch.Tensor, theta: float = 1000042.0) -> torch.Tensor:
    # x: (B, H, T, D)
    D = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    freqs = torch.outer(pos.float(), inv_freq)  # (T, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


@torch.no_grad()
def analyze_dragon_gguf(gguf_path: str, out_json: str):
    reader = gguf.GGUFReader(gguf_path)
    tmap = {t.name: t for t in reader.tensors}

    def dq(name: str) -> torch.Tensor:
        t = tmap[name]
        return torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).float()

    # Build byte-level/BPE vocab lookup using Falcon/Llama tokenizer if available, else GGUF tokens
    tf = reader.fields["tokenizer.ggml.tokens"]
    toks = [bytes(tf.parts[i]).decode("utf-8", errors="replace") for i in tf.data]
    tok2id = {s: i for i, s in enumerate(toks)}

    def encode_simple(text: str) -> List[int]:
        # Greedy longest-match on GGUF vocab (using \u0120 for space)
        s = text.replace(" ", "\u0120").replace("\n", "\u010a")
        ids, i = [], 0
        while i < len(s):
            matched = False
            for L in range(min(16, len(s) - i), 0, -1):
                sub = s[i : i + L]
                if sub in tok2id:
                    ids.append(tok2id[sub])
                    i += L
                    matched = True
                    break
            if not matched:
                ids.append(tok2id.get(s[i], 0))
                i += 1
        return ids

    def tok_id(prefix: str) -> int:
        p = prefix.replace(" ", "\u0120")
        if p in tok2id:
            return tok2id[p]
        if ("\u0120" + p) in tok2id:
            return tok2id["\u0120" + p]
        return encode_simple(prefix)[0]

    E = dq("token_embd.weight")
    out_w = dq("output.weight")
    out_norm_w = dq("output_norm.weight")

    n_layers = 18
    n_heads = 8
    n_kv = 4
    d_head = 256
    groups = n_heads // n_kv

    layers = []
    for l in range(n_layers):
        layers.append({
            "an": dq(f"blk.{l}.attn_norm.weight"),
            "wq": dq(f"blk.{l}.attn_q.weight"),
            "wk": dq(f"blk.{l}.attn_k.weight"),
            "wv": dq(f"blk.{l}.attn_v.weight"),
            "wo": dq(f"blk.{l}.attn_output.weight"),
            "fn": dq(f"blk.{l}.ffn_norm.weight"),
            "wg": dq(f"blk.{l}.ffn_gate.weight"),
            "wu": dq(f"blk.{l}.ffn_up.weight"),
            "wd": dq(f"blk.{l}.ffn_down.weight"),
        })

    probes = dla.build_curated_probes()
    stats = {
        dom: {l: {"attn_dmargin": 0.0, "mlp_dmargin": 0.0, "n": 0} for l in range(n_layers)}
        for dom in probes
    }

    for dom, items in probes.items():
        for prompt, good_s, bad_s in items:
            ids = torch.tensor([encode_simple(prompt)], dtype=torch.long)
            T = ids.shape[1]
            pos = torch.arange(T)
            h = E[ids]  # (1, T, 2048)
            mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)[None, None, :, :]
            yg = tok_id(good_s)
            yb = tok_id(bad_s)

            for l in range(n_layers):
                L = layers[l]
                h_in = h[:, -1, :]
                hn = rms_norm(h, L["an"])
                q = F.linear(hn, L["wq"]).view(1, T, n_heads, d_head).transpose(1, 2)
                k = F.linear(hn, L["wk"]).view(1, T, n_kv, d_head).transpose(1, 2)
                v = F.linear(hn, L["wv"]).view(1, T, n_kv, d_head).transpose(1, 2)
                q = apply_rope(q, pos)
                k = apply_rope(k, pos)
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)
                att = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(d_head) + mask, dim=-1) @ v
                att_out = F.linear(att.transpose(1, 2).contiguous().view(1, T, -1), L["wo"])
                h_mid_seq = h + att_out
                h_mid = h_mid_seq[:, -1, :]

                fn = rms_norm(h_mid_seq, L["fn"])
                mlp_out = F.linear(F.silu(F.linear(fn, L["wg"])) * F.linear(fn, L["wu"]), L["wd"])
                h = h_mid_seq + mlp_out
                h_out = h[:, -1, :]

                lp_in = F.log_softmax(F.linear(rms_norm(h_in, out_norm_w), out_w), dim=-1)[0]
                lp_mid = F.log_softmax(F.linear(rms_norm(h_mid, out_norm_w), out_w), dim=-1)[0]
                lp_out = F.log_softmax(F.linear(rms_norm(h_out, out_norm_w), out_w), dim=-1)[0]

                s = stats[dom][l]
                s["attn_dmargin"] += float((lp_mid[yg] - lp_mid[yb]) - (lp_in[yg] - lp_in[yb]))
                s["mlp_dmargin"] += float((lp_out[yg] - lp_out[yb]) - (lp_mid[yg] - lp_mid[yb]))
                s["n"] += 1

    print(f"\n=== DRAGON GGUF LAYER ATTRIBUTION ({n_layers} layers) ===")
    summary: Dict[str, object] = {"gguf": gguf_path, "n_layers": n_layers, "domains": {}}
    for dom in probes:
        rows = []
        for l in range(n_layers):
            s = stats[dom][l]
            n = max(1, s["n"])
            rows.append({
                "layer": l,
                "attn_dmargin": round(s["attn_dmargin"] / n, 4),
                "mlp_dmargin": round(s["mlp_dmargin"] / n, 4),
            })
        wa = sorted(rows, key=lambda r: r["attn_dmargin"])[:3]
        wm = sorted(rows, key=lambda r: r["mlp_dmargin"])[:3]
        print(f"[{dom}] worst attn_dmargin: {[(r['layer'], r['attn_dmargin']) for r in wa]}")
        print(f"[{dom}] worst  mlp_dmargin: {[(r['layer'], r['mlp_dmargin']) for r in wm]}")
        summary["domains"][dom] = rows

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    analyze_dragon_gguf(args.gguf, args.out_json)
