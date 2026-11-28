#!/usr/bin/env python3
"""24-layer Qwen -> native ONI Main body, exact functional transplant.

Measured size: H896/F4864/L24 flags_off = 1,158,468,313 params (fits 1.0-1.2B).
All 24 Qwen layers are mapped 1:1, avoiding the depth truncation that caused
the 6-layer parity model to emit punctuation garbage.
"""

from __future__ import annotations
import os
import sys
from dataclasses import replace
from pathlib import Path
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni, RotaryEmbedding

QWEN = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors"
)
ALIGN = Path(r"F:\02_QUILLAN\exports\oni_transfer\oni_donor_alignment_rebuilt.pt")
Q2D = Path(r"F:\02_QUILLAN\exports\quillan_token_maps_q2d.pt")
OUT = Path(r"F:\02_QUILLAN\exports\oni_transfer")
FINAL = OUT / "quillan_main_qwen_transplant.pt"
TMP = FINAL.with_suffix(".pt.tmp")
H, F, L, NH, HD = 896, 4864, 24, 14, 64
V, N_REACHABLE = 151648, 151643


def log(s: str) -> None:
    print(f"[QWEN-MAIN] {s}", flush=True)


def main() -> None:
    cfg = replace(
        owner_config("main"),
        hidden_dim=H,
        ffn_dim=F,
        n_layer=L,
        n_head=NH,
        head_dim=HD,
        vocab_size=V,
        use_evo_moe=False,
        use_mod=False,
        use_agent_evolution=False,
        use_real_swarm=False,
        use_world_model=False,
        couil_sparse_heads=False,
        use_memory_attention=False,
        use_fa3=False,
        use_moba=False,
    )
    log("building native ONI Main body")
    model = QuillanRoninOni(cfg).eval()
    nparams = sum(int(p.numel()) for p in model.parameters())
    log(f"params={nparams}")
    if not (1_000_000_000 <= nparams <= 1_200_000_000):
        raise RuntimeError(f"Main param count {nparams} outside 1.0-1.2B")

    for blk in model.h:
        blk.attn.rope = RotaryEmbedding(HD, cfg.max_seq_len * 4, base=1_000_000.0)
        blk.ln_1.eps = 1e-6
        blk.ln_2.eps = 1e-6
    model.ln_f.eps = 1e-6
    sd = model.state_dict()
    sovereign_before = {
        k: v.clone()
        for k, v in sd.items()
        if k.startswith(("tristack.", "throne_swarm."))
    }

    align = torch.load(ALIGN, map_location="cpu", weights_only=True, mmap=True)
    slots = align["slot_for_token_id"][:N_REACHABLE]
    q2d = torch.load(Q2D, map_location="cpu", weights_only=True)["qwen"]
    did = torch.full((N_REACHABLE,), -1, dtype=torch.long)
    ok = (slots >= 0) & (slots < q2d.shape[0])
    did[ok] = q2d[slots[ok]]

    writes = 0
    with safe_open(str(QWEN), framework="pt", device="cpu") as fq:
        emb = fq.get_tensor("model.embed_tokens.weight").float()
        with torch.no_grad():
            sd["wte.weight"][:N_REACHABLE].copy_(emb[torch.where(did >= 0, did, 0)])
        del emb
        writes += 1
        for i in range(L):
            p = f"model.layers.{i}."

            def g(n):
                return fq.get_tensor(p + n).float()

            q, qb = g("self_attn.q_proj.weight"), g("self_attn.q_proj.bias")
            k, kb = g("self_attn.k_proj.weight"), g("self_attn.k_proj.bias")
            v, vb = g("self_attn.v_proj.weight"), g("self_attn.v_proj.bias")
            o = g("self_attn.o_proj.weight")
            gate, up, down = (
                g("mlp.gate_proj.weight"),
                g("mlp.up_proj.weight"),
                g("mlp.down_proj.weight"),
            )
            ln1, ln2 = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
            kr = k.view(2, HD, H).repeat_interleave(NH // 2, dim=0).reshape(NH * HD, H)
            kbr = kb.view(2, HD).repeat_interleave(NH // 2, dim=0).reshape(NH * HD)
            vr = v.view(2, HD, H).repeat_interleave(NH // 2, dim=0).reshape(NH * HD, H)
            vbr = vb.view(2, HD).repeat_interleave(NH // 2, dim=0).reshape(NH * HD)
            with torch.no_grad():
                c = sd[f"h.{i}.attn.c_attn.weight"]
                c[0:H].copy_(q)
                c[H : 2 * H].copy_(kr)
                c[2 * H : 3 * H].copy_(vr)
                cb = sd[f"h.{i}.attn.c_attn.bias"]
                cb[0:H].copy_(qb)
                cb[H : 2 * H].copy_(kbr)
                cb[2 * H : 3 * H].copy_(vbr)
                sd[f"h.{i}.attn.c_proj.weight"].copy_(o)
                sd[f"h.{i}.attn.c_proj.bias"].zero_()
                sd[f"h.{i}.ln_1.weight"].copy_(ln1)
                sd[f"h.{i}.ln_2.weight"].copy_(ln2)
                fc = sd[f"h.{i}.moe.c_fc.weight"]
                fc[0:F].copy_(gate)
                fc[F : 2 * F].copy_(up)
                sd[f"h.{i}.moe.c_fc.bias"].zero_()
                sd[f"h.{i}.moe.c_proj.weight"].copy_(down)
                sd[f"h.{i}.moe.c_proj.bias"].zero_()
            writes += 12
            if (i + 1) % 6 == 0:
                log(f"layer {i + 1}/{L}")
        lnf = fq.get_tensor("model.norm.weight").float()
        with torch.no_grad():
            sd["ln_f.weight"].copy_(lnf)
        writes += 1

    neutral = [
        "q1_bridge.weight",
        "q2_bridge.weight",
        "ingest_gate.weight",
        "ingest_gate.bias",
        "recirc_proj.weight",
        "quillan_finalizer_q1.weight",
        "quillan_finalizer_q1.bias",
        "quillan_finalizer_q2.weight",
        "quillan_finalizer_q2.bias",
        "quillan_comm_gate.weight",
        "quillan_comm_gate.bias",
    ]
    with torch.no_grad():
        for k in neutral:
            if k in sd:
                sd[k].zero_()
        for i in range(L):
            for suf in (
                "attn.prism.w_gate.weight",
                "moe.moe_gate.weight",
                "moe.moe_gate.bias",
                "moe.pull_gate.gate.weight",
                "attn.c_proj.bias",
                "moe.c_fc.bias",
                "moe.c_proj.bias",
            ):
                k = f"h.{i}.{suf}"
                if k in sd:
                    sd[k].zero_()
    for k, v in sovereign_before.items():
        if not torch.equal(sd[k], v):
            raise RuntimeError(f"sovereign changed: {k}")

    model.load_state_dict(sd, strict=False)
    meta = {
        "build": "exact_shape_qwen_functional_transplant_full_depth_main",
        "knowledge_source": "Qwen2.5-0.5B-Instruct all 24 layers",
        "no_backprop": True,
        "param_count": nparams,
        "owner_config": {
            "vocab_size": V,
            "max_seq_len": cfg.max_seq_len,
            "hidden_dim": H,
            "ffn_dim": F,
            "n_layer": L,
            "n_head": NH,
        },
        "recommended_inference_flags": {"compute_aux": False, "deliberation": False},
        "limitations": [
            "Single-donor functional transplant; not yet the owner's four-donor merge.",
            "No score claim without held-out evaluation.",
        ],
        "writes": writes,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": meta}, TMP)
    os.replace(TMP, FINAL)
    log(f"saved {FINAL} ({FINAL.stat().st_size / 1e9:.3f} GB) writes={writes}")


if __name__ == "__main__":
    main()
