#!/usr/bin/env python3
"""Smoke test a separate readout adapter: logits += adapter(hidden).

No ONI weight is modified. The adapter is applied by an lm_head forward hook.
"""

import sys
from dataclasses import replace
from pathlib import Path
import torch
from tokenizers import Tokenizer

sys.path.insert(0, r"F:\02_QUILLAN\03 - Training & Model\scripts")
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni

base = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(r"F:\02_QUILLAN\exports\oni_transfer\quillan_mini_oni_transfer.pt")
)
adapter_path = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path(
        r"F:\02_QUILLAN\exports\oni_transfer\knowledge_readout\quillan_mini_token_knowledge_adapter.pt"
    )
)

blob = torch.load(base, map_location="cpu", weights_only=True, mmap=True)
oc = blob["metadata"]["owner_config"]
cfg = replace(
    owner_config("mini"),
    vocab_size=int(oc["vocab_size"]),
    max_seq_len=int(oc["max_seq_len"]),
    hidden_dim=int(oc["hidden_dim"]),
    ffn_dim=int(oc["ffn_dim"]),
    n_layer=int(oc["n_layer"]),
    n_head=int(oc["n_head"]),
    num_experts=int(oc["num_experts"]),
)
model = QuillanRoninOni(cfg).eval()
model.load_state_dict(blob["state_dict"], strict=False)

ad = torch.load(adapter_path, map_location="cpu", weights_only=True)
W = ad["adapter_weight"].float()  # (V, H)
captured = {}
hook_h = model.ln_f.register_forward_hook(
    lambda m, inp, out: captured.__setitem__("h", out.detach())
)


def head_hook(module, inp, out):
    h = captured.get("h")
    if h is None:
        return out
    # h: (B,T,H); adapter: (V,H) -> (B,T,V), aligned with lm_head output.
    delta = torch.einsum("bth,vh->btv", h, W)
    return out + delta


head_handle = model.lm_head.register_forward_hook(head_hook)

tok = Tokenizer.from_file(
    r"F:\02_QUILLAN\exports\quillan_tokenizer_v54_fourdonor\tokenizer.json"
)
print(
    f"loaded adapter {tuple(W.shape)} params={sum(p.numel() for p in model.parameters())}",
    flush=True,
)
for txt in ("2 + 2 =", "The capital of France is", "Hello", "cat"):
    ids = tok.encode(txt).ids
    x = torch.tensor([ids])
    with torch.inference_mode():
        for _ in range(8):
            captured.clear()
            out = model(x, return_aux=False, compute_aux=False, deliberation=False)
            logits = out[0] if isinstance(out, tuple) else out
            nxt = int(logits[0, -1].argmax())
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    gen = tok.decode(x[0, len(ids) :].tolist(), skip_special_tokens=True)
    print(repr(txt), "->", repr(gen), flush=True)

head_handle.remove()
hook_h.remove()
