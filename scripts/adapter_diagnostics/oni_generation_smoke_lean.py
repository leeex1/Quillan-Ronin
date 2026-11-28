#!/usr/bin/env python3
"""Memory-lean greedy generation smoke for an ONI checkpoint."""

import sys
from dataclasses import replace
from pathlib import Path
import torch
from tokenizers import Tokenizer

sys.path.insert(0, r"F:\02_QUILLAN\03 - Training & Model\scripts")
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni

model_key = sys.argv[1] if len(sys.argv) > 1 else "main"
path = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path(
        rf"F:\02_QUILLAN\exports\oni_transfer\quillan_{model_key}_oni_transfer.pt"
    )
)
tok_path = Path(r"F:\02_QUILLAN\exports\quillan_tokenizer_v54_fourdonor\tokenizer.json")

blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
sd, oc = blob["state_dict"], blob["metadata"]["owner_config"]
cfg = replace(
    owner_config(model_key),
    vocab_size=int(oc["vocab_size"]),
    max_seq_len=int(oc["max_seq_len"]),
    hidden_dim=int(oc["hidden_dim"]),
    ffn_dim=int(oc["ffn_dim"]),
    n_layer=int(oc["n_layer"]),
    n_head=int(oc["n_head"]),
    num_experts=int(oc.get("num_experts", 34)),
)
if blob["metadata"].get("build") in (
    "exact_shape_qwen_functional_transplant",
    "exact_shape_qwen_functional_transplant_full_depth_main",
):
    cfg = replace(
        cfg,
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
if blob["metadata"].get("build") in (
    "exact_shape_qwen_functional_transplant",
    "exact_shape_qwen_functional_transplant_full_depth_main",
):
    for blk in model.h:
        from quillan_v5_4_oni import RotaryEmbedding

        blk.attn.rope = RotaryEmbedding(
            cfg.head_dim, cfg.max_seq_len * 4, base=1_000_000.0
        )
        blk.ln_1.eps = 1e-6
        blk.ln_2.eps = 1e-6
    model.ln_f.eps = 1e-6
model.load_state_dict(sd, strict=False)
tok = Tokenizer.from_file(str(tok_path))
print(
    f"loaded {model_key} params={sum(p.numel() for p in model.parameters())}",
    flush=True,
)

for txt in ("2 + 2 =", "The capital of France is", "Hello"):
    ids = tok.encode(txt).ids
    x = torch.tensor([ids])
    with torch.inference_mode():
        for _ in range(8):
            out = model(x, return_aux=False, compute_aux=False, deliberation=False)
            logits = out[0] if isinstance(out, tuple) else out
            nxt = int(logits[0, -1].argmax())
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    gen = tok.decode(x[0, len(ids) :].tolist(), skip_special_tokens=True)
    print(repr(txt), "->", repr(gen), flush=True)
