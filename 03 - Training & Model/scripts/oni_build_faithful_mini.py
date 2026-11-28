#!/usr/bin/env python3
"""Diagnostic faithful-Qwen mini transplant into the native ONI body.

Purpose: isolate whether full-rank shape-matched donor weights restore coherent
generation. This is NOT the owner's 4->2->1 deliverable and makes no score claim.
Qwen hidden 896 == ONI mini hidden, so q/o and biases are exact; k/v are GQA
replicated 2 heads -> 14 heads; FFN is evenly downsampled 4864 -> 1792.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni

EXPORTS = Path(r"F:\02_QUILLAN\exports")
OUT = EXPORTS / "oni_transfer"
QWEN = Path(r"F:\quillan_cache\donors\qwen.safetensors")
Q2D = EXPORTS / "quillan_token_maps_q2d.pt"
ALIGN = OUT / "oni_donor_alignment_rebuilt.pt"
TOK = EXPORTS / "quillan_tokenizer_v54_fourdonor" / "tokenizer.json"
FINAL = OUT / "quillan_mini_oni_faithful_qwen.pt"
TMP = OUT / "quillan_mini_oni_faithful_qwen.pt.tmp"

H, F, L = 896, 1792, 6
N_REACHABLE = 151643
DONOR_LAYERS = [0, 5, 9, 14, 19, 23]


def get_opt(fq, name: str, shape) -> torch.Tensor:
    try:
        return fq.get_tensor(name).float()
    except Exception:
        return torch.zeros(shape, dtype=torch.float32)


def even_select(t: torch.Tensor, n: int, dim: int) -> torch.Tensor:
    if t.shape[dim] == n:
        return t
    idx = torch.linspace(0, t.shape[dim] - 1, n).long()
    return torch.index_select(t, dim, idx)


def gqa_tile(t: torch.Tensor) -> torch.Tensor:
    # (128,896) -> (896,896) or (128,) -> (896,): repeat 2 GQA heads 7x.
    reps = H // t.shape[0]
    if t.dim() == 1:
        return t.repeat(reps)[:H]
    return t.repeat(reps, 1)[:H]


def main() -> None:
    print("loading maps", flush=True)
    q2d = torch.load(Q2D, map_location="cpu", weights_only=True)["qwen"]
    align = torch.load(ALIGN, map_location="cpu", weights_only=True, mmap=True)
    slots = align["slot_for_token_id"]
    print("building model", flush=True)
    model = QuillanRoninOni(owner_config("mini")).eval()
    sd = model.state_dict()
    cfg = model.cfg

    with safe_open(str(QWEN), framework="pt", device="cpu") as fq:
        # embedding: exact donor row per Quillan token string where available
        emb = fq.get_tensor("model.embed_tokens.weight").float()
        did = torch.full((N_REACHABLE,), -1, dtype=torch.long)
        s = slots[:N_REACHABLE]
        ok = (s >= 0) & (s < q2d.shape[0])
        did[ok] = q2d[s[ok]]
        good = (did >= 0) & (did < emb.shape[0])
        rows = torch.where(good, did, 0)
        with torch.no_grad():
            sd["wte.weight"][:N_REACHABLE].copy_(emb[rows])
        print(f"embedding mapped={int(good.sum())}/{N_REACHABLE}", flush=True)
        del emb, did, s, ok, good, rows

        for i, dl in enumerate(DONOR_LAYERS):
            pre = f"model.layers.{dl}."
            q = fq.get_tensor(pre + "self_attn.q_proj.weight").float()
            k = fq.get_tensor(pre + "self_attn.k_proj.weight").float()
            v = fq.get_tensor(pre + "self_attn.v_proj.weight").float()
            o = fq.get_tensor(pre + "self_attn.o_proj.weight").float()
            gate = fq.get_tensor(pre + "mlp.gate_proj.weight").float()
            up = fq.get_tensor(pre + "mlp.up_proj.weight").float()
            down = fq.get_tensor(pre + "mlp.down_proj.weight").float()
            qb = get_opt(fq, pre + "self_attn.q_proj.bias", (H,))
            kb = get_opt(fq, pre + "self_attn.k_proj.bias", (128,))
            vb = get_opt(fq, pre + "self_attn.v_proj.bias", (128,))
            ob = get_opt(fq, pre + "self_attn.o_proj.bias", (H,))
            gb = get_opt(fq, pre + "mlp.gate_proj.bias", (4864,))
            ub = get_opt(fq, pre + "mlp.up_proj.bias", (4864,))
            db = get_opt(fq, pre + "mlp.down_proj.bias", (H,))
            with torch.no_grad():
                cat = sd[f"h.{i}.attn.c_attn.weight"]
                cat[0:H].copy_(q)
                cat[H : 2 * H].copy_(gqa_tile(k))
                cat[2 * H : 3 * H].copy_(gqa_tile(v))
                cpro = sd[f"h.{i}.attn.c_proj.weight"]
                cpro.copy_(o)
                cb = sd[f"h.{i}.attn.c_attn.bias"]
                cb[0:H].copy_(qb)
                cb[H : 2 * H].copy_(gqa_tile(kb))
                cb[2 * H : 3 * H].copy_(gqa_tile(vb))
                sd[f"h.{i}.attn.c_proj.bias"].copy_(ob)
                fc = sd[f"h.{i}.moe.c_fc.weight"]
                fc[0:F].copy_(even_select(gate, F, 0))
                fc[F : 2 * F].copy_(even_select(up, F, 0))
                sd[f"h.{i}.moe.c_proj.weight"].copy_(even_select(down, F, 1))
                fcb = sd[f"h.{i}.moe.c_fc.bias"]
                fcb[0:F].copy_(even_select(gb, F, 0))
                fcb[F : 2 * F].copy_(even_select(ub, F, 0))
                sd[f"h.{i}.moe.c_proj.bias"].copy_(db)
            print(f"layer {i} <- donor {dl}", flush=True)
            del q, k, v, o, gate, up, down, qb, kb, vb, ob, gb, ub, db

    metadata = {
        "build": "diagnostic_faithful_qwen_mini_full_rank",
        "knowledge_source": "Qwen3/Qwen2.5 0.5B donor, full-rank shape-matched with GQA replication and even FFN downsample",
        "NOT_owner_deliverable": True,
        "NOT_a_raw_state_dict_transplant": True,
        "no_score_claim": True,
        "model": "mini",
        "param_count": sum(p.numel() for p in model.parameters()),
        "owner_config": {
            "vocab_size": int(cfg.vocab_size),
            "max_seq_len": int(cfg.max_seq_len),
            "hidden_dim": int(cfg.hidden_dim),
            "ffn_dim": int(cfg.ffn_dim),
            "n_layer": int(cfg.n_layer),
            "n_head": int(cfg.n_head),
            "num_experts": int(cfg.num_experts),
        },
        "recommended_inference_flags": {"compute_aux": False, "deliberation": False},
        "limitations": [
            "Single-donor diagnostic; not the owner's 4->2->1 merge.",
            "GQA heads replicated and FFN evenly downsampled; no learned alignment.",
            "No factual or benchmark claim.",
        ],
    }
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, TMP)
    os.replace(TMP, FINAL)
    print(f"saved {FINAL} {FINAL.stat().st_size / 1e9:.3f} GB", flush=True)


if __name__ == "__main__":
    main()
