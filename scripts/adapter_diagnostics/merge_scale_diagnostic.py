#!/usr/bin/env python3
"""Diagnose WHY the Llama+Falcon merge collapsed: activation/logit scale."""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from heldout_fresh40 import prompts

torch.set_num_threads(1)


def main():
    paths = [
        ROOT / "exports" / "quillan-flagship-hifi-g4-exact.safetensors",
        Path(
            r"E:\quillan_quarantine_20261124\quillan-flagship-llama-falcon-a03.safetensors"
        ),
    ]
    for path in paths:
        if not path.exists():
            print(f"MISSING {path}", flush=True)
            continue
        m = load_model(path)
        tok = tok_for(m.cfg.vocab_size)
        lids = letter_ids(tok, m.cfg.vocab_size)
        cap = {}
        hk = m.norm.register_forward_hook(
            lambda mod, inp, out: cap.__setitem__("h", out.detach().float())
        )
        hn = []
        lgmax = []
        gaps = []
        with torch.inference_mode():
            for seq in prompts(tok)[:10]:
                m(torch.tensor([seq]))
                h = cap["h"][0, -1, :]
                Ew = m.embed_tokens.unpack_float().float()
                logits = torch.nn.functional.linear(cap["h"][:, -1, :], Ew)[0].float()
                ll = logits[lids]
                hn.append(h.norm().item())
                lgmax.append(logits.max().item())
                gaps.append((ll.max() - ll.min()).item())
        hk.remove()
        print(
            f"{path.name}: h_norm={sum(hn) / len(hn):.2f} logit_max={sum(lgmax) / len(lgmax):.2f} letter_gap={sum(gaps) / len(gaps):.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
