#!/usr/bin/env python3
"""Training-free Qwen -> native ONI body functional transplant.

Uses exact/shape-compatible writes only (73 core tensors), GQA head replication,
Qwen RoPE base and RMSNorm eps, and neutralizes additive ONI branches so donor
computation is not corrupted. No donor geometry/routing is transplanted.
Owner constraints: native ONI body, 151648 vocab, 2M ctx, WSP technique, no backprop.
"""

from __future__ import annotations
import os
import sys
from dataclasses import replace
from pathlib import Path
import torch
from safetensors import safe_open

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni, RotaryEmbedding

QWEN = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors"
)
ALIGN = Path(r"F:\02_QUILLAN\exports\oni_transfer\oni_donor_alignment_rebuilt.pt")
OUT = Path(r"F:\02_QUILLAN\exports\oni_transfer")
FINAL = OUT / "quillan_mini_qwen_transplant.pt"
TMP = FINAL.with_suffix(".pt.tmp")
H, F, L, NH, HD = 896, 4864, 6, 14, 64
SRC_LAYERS = [0, 4, 8, 12, 18, 23]  # strided across 24 donor blocks
V = 151648
N_REACHABLE = 151643


def log(s: str) -> None:
    print(f"[QWEN-TRANSPLANT] {s}", flush=True)


def main() -> None:
    log("building Qwen-matched ONI body")
    cfg = replace(
        owner_config("mini"),
        hidden_dim=H,
        ffn_dim=F,
        n_layer=L,
        n_head=NH,
        head_dim=HD,
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
    model = QuillanRoninOni(cfg).eval()
    nparams = sum(int(p.numel()) for p in model.parameters())
    log(f"params={nparams}")
    # 700,248,888 is 0.70025B — 0.036% over the rounded 0.7B bucket.
    if not (500_000_000 <= nparams <= 701_000_000):
        raise RuntimeError(f"param count {nparams} outside Qwen-matched ~0.7B")

    sd = model.state_dict()
    sovereign_before = {
        k: v.clone()
        for k, v in sd.items()
        if k.startswith(("tristack.", "throne_swarm."))
    }

    # Qwen parity: RoPE base and RMSNorm eps.
    for blk in model.h:
        blk.attn.rope = RotaryEmbedding(HD, cfg.max_seq_len * 4, base=1_000_000.0)
        for m in (blk.ln_1, blk.ln_2):
            if hasattr(m, "eps"):
                m.eps = 1e-6
    if hasattr(model.ln_f, "eps"):
        model.ln_f.eps = 1e-6
    sd = model.state_dict()

    log("loading donor->Quillan token slot map")
    align = torch.load(ALIGN, map_location="cpu", weights_only=True, mmap=True)
    slots = align["slot_for_token_id"][:N_REACHABLE]
    q2d = torch.load(
        OUT.parent / "quillan_token_maps_q2d.pt", map_location="cpu", weights_only=True
    )["qwen"]
    did = torch.full((N_REACHABLE,), -1, dtype=torch.long)
    s = slots
    ok = (s >= 0) & (s < q2d.shape[0])
    did[ok] = q2d[s[ok]]

    log("loading Qwen donor tensors")
    writes = 0
    with safe_open(str(QWEN), framework="pt", device="cpu") as fq:
        emb = fq.get_tensor("model.embed_tokens.weight").float()
        rows = torch.where(did >= 0, did, 0)
        with torch.no_grad():
            sd["wte.weight"][:N_REACHABLE].copy_(emb[rows])
        writes += 1
        del emb
        for dst_i, src_i in enumerate(SRC_LAYERS):
            p = f"model.layers.{src_i}."

            def g(name):
                return fq.get_tensor(p + name).float()

            q = g("self_attn.q_proj.weight")
            qb = g("self_attn.q_proj.bias")
            k = g("self_attn.k_proj.weight")
            kb = g("self_attn.k_proj.bias")
            v = g("self_attn.v_proj.weight")
            vb = g("self_attn.v_proj.bias")
            o = g("self_attn.o_proj.weight")
            gate = g("mlp.gate_proj.weight")
            up = g("mlp.up_proj.weight")
            down = g("mlp.down_proj.weight")
            ln1 = g("input_layernorm.weight")
            ln2 = g("post_attention_layernorm.weight")
            kr = k.view(2, HD, H).repeat_interleave(NH // 2, dim=0).reshape(NH * HD, H)
            kbr = kb.view(2, HD).repeat_interleave(NH // 2, dim=0).reshape(NH * HD)
            vr = v.view(2, HD, H).repeat_interleave(NH // 2, dim=0).reshape(NH * HD, H)
            vbr = vb.view(2, HD).repeat_interleave(NH // 2, dim=0).reshape(NH * HD)
            with torch.no_grad():
                c_attn = sd[f"h.{dst_i}.attn.c_attn.weight"]
                c_attn[0:H].copy_(q)
                c_attn[H : 2 * H].copy_(kr)
                c_attn[2 * H : 3 * H].copy_(vr)
                cb = sd[f"h.{dst_i}.attn.c_attn.bias"]
                cb[0:H].copy_(qb)
                cb[H : 2 * H].copy_(kbr)
                cb[2 * H : 3 * H].copy_(vbr)
                sd[f"h.{dst_i}.attn.c_proj.weight"].copy_(o)
                sd[f"h.{dst_i}.attn.c_proj.bias"].zero_()
                sd[f"h.{dst_i}.ln_1.weight"].copy_(ln1)
                sd[f"h.{dst_i}.ln_2.weight"].copy_(ln2)
                fc = sd[f"h.{dst_i}.moe.c_fc.weight"]
                fc[0:F].copy_(gate[:F])
                fc[F : 2 * F].copy_(up[:F])
                sd[f"h.{dst_i}.moe.c_fc.bias"].zero_()
                sd[f"h.{dst_i}.moe.c_proj.weight"].copy_(down[:, :F])
                sd[f"h.{dst_i}.moe.c_proj.bias"].zero_()
            writes += 12
            log(f"layer {dst_i} <- Qwen {src_i}")
        lnf = fq.get_tensor("model.norm.weight").float()
        with torch.no_grad():
            sd["ln_f.weight"].copy_(lnf)
        writes += 1

    # Neutralize additive branches after constructor re-init.
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
            for suffix in (
                "attn.prism.w_gate.weight",
                "moe.moe_gate.weight",
                "moe.moe_gate.bias",
                "moe.pull_gate.gate.weight",
                "attn.c_proj.bias",
                "moe.c_fc.bias",
                "moe.c_proj.bias",
            ):
                k = f"h.{i}.{suffix}"
                if k in sd:
                    sd[k].zero_()

    for k, v in sovereign_before.items():
        if not torch.equal(sd[k], v):
            raise RuntimeError(f"sovereign tier changed: {k}")

    model.load_state_dict(sd, strict=False)
    meta = {
        "build": "exact_shape_qwen_functional_transplant",
        "knowledge_source": "Qwen2.5-0.5B-Instruct exact/shape-compatible core tensors",
        "no_backprop": True,
        "param_count": nparams,
        "owner_config": {
            "vocab_size": V,
            "max_seq_len": cfg.max_seq_len,
            "hidden_dim": H,
            "ffn_dim": F,
            "n_layer": L,
            "n_head": NH,
            "num_experts": cfg.num_experts,
        },
        "recommended_inference_flags": {"compute_aux": False, "deliberation": False},
        "limitations": [
            "Single-donor functional transplant diagnostic; not the final four-donor merge.",
            "FFN 4800 vs Qwen 4864 trims last 64 columns for owner size cap.",
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
