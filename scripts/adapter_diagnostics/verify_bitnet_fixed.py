#!/usr/bin/env python3
"""Verify the corrected BitNet eager loader produces sane hidden states + logits."""

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
from bitnet_loader import load_bitnet


def main():
    print("loading BitNet with corrected eager unpack...", flush=True)
    m = load_bitnet(r"E:\source_models\bitnet_2b_4t")
    print("loaded", flush=True)
    # check a weight
    for name, mod in m.named_modules():
        if hasattr(mod, "weight") and mod.weight is not None:
            print(
                f"  {name}: weight absmax={mod.weight.detach().float().abs().max().item():.4f} mean={mod.weight.detach().float().mean().item():.6f}"
            )
            break
    # forward on a real prompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        r"E:\source_models\bitnet_2b_4t", local_files_only=True
    )
    text = "The capital of France is"
    ids = tok.encode(text, add_special_tokens=False)
    print(f"prompt ids={ids[:10]}...", flush=True)
    cap = {}
    h = m.model.layers[0].register_forward_hook(
        lambda mod, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o)
    )
    with torch.inference_mode():
        out = m(torch.tensor([ids]))
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits.float(), -1)
        top = torch.topk(probs, 5)
    h.remove()
    print(f"layer0 hidden norm={cap['h'][0, -1, :].norm().item():.3f}", flush=True)
    print(
        f"top5 tokens={[(tok.decode([int(t)]), round(float(p), 3)) for t, p in zip(top.indices, top.values)]}",
        flush=True,
    )
    # MCQ letter probe
    lids = [tok.encode(f" {L}", add_special_tokens=False) for L in "ABCD"]
    print("letter ids:", lids, flush=True)


if __name__ == "__main__":
    main()
