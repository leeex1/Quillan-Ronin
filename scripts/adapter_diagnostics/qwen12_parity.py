#!/usr/bin/env python3
"""Test whether 12-layer Qwen mappings remain coherent. Read-only."""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SNAP = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
torch.manual_seed(0)
torch.set_num_threads(1)
tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    SNAP, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"
).eval()
orig = list(model.model.layers)
assert len(orig) == 24
maps = {
    "full24_control": list(range(24)),
    "first12": list(range(12)),
    "last12": list(range(12, 24)),
    "strided12": list(range(0, 24, 2)),
    "outer12": list(range(6)) + list(range(18, 24)),
}
for name, idx in maps.items():
    model.model.layers = torch.nn.ModuleList([orig[i] for i in idx])
    model.config.num_hidden_layers = len(idx)
    print(f"=== {name} map={idx}", flush=True)
    for p in ("2 + 2 =", "The capital of France is"):
        ids = tok(p, return_tensors="pt").input_ids
        with torch.inference_mode():
            out = model.generate(
                ids, do_sample=False, max_new_tokens=16, pad_token_id=tok.eos_token_id
            )
        cont = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()
        alnum = sum(ch.isalnum() for ch in cont)
        ratio = alnum / max(1, len(cont.replace(" ", "")))
        punct_run = bool(re.search(r"([^\w\s])\1{5,}", cont))
        anchor = ("4" in cont) if p.startswith("2 + 2") else ("paris" in cont.lower())
        ok = alnum > 0 and ratio >= 0.5 and not punct_run and anchor
        print(
            f"  {p!r} -> {cont!r} alnum={alnum} ratio={ratio:.2f} "
            f"punct_run={punct_run} anchor={anchor} PASS={ok}",
            flush=True,
        )
