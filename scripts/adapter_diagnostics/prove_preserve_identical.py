#!/usr/bin/env python3
"""Direct proof: for host-token prompts, preserve model logits == baseline logits.

Reason: every host-token embedding row is bit-identical to baseline (verified),
head rows identical, body identical. So logits must match. Compare 3 prompts
directly rather than sweeping 240 questions through a thrashing disk.
"""

from __future__ import annotations
import sys, json
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
OUT = ROOT / "exports"
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(2)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        r"E:\source_models\llama_1b", local_files_only=True
    )
    d2q = json.loads((OUT / "quillan_token_maps.json").read_text())
    qmap = {int(k): v for k, v in d2q["donor_to_quillan"]["llama"].items()}
    prompts = [
        "The capital of France is",
        "What is 2 + 2?",
        "The chemical symbol for iron is Fe because",
    ]
    mb = AutoModelForCausalLM.from_pretrained(
        r"E:\source_models\llama_1b",
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    mp = AutoModelForCausalLM.from_pretrained(
        str(OUT / "quillan_flagship_preserve"),
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    print("=== logit identity check (host-token prompts) ===", flush=True)
    with torch.inference_mode():
        for t in prompts:
            native = tok.encode(t, add_special_tokens=False)
            qids = [qmap[x] for x in native]
            lb = mb(torch.tensor([native]))[0][0, -1, :].float()
            lp = mp(torch.tensor([qids]))[0][0, -1, :].float()
            # compare baseline logit at native id x vs preserve logit at qmap[x]
            inv = {v: k for k, v in qmap.items()}
            xs = [x for x in range(len(lb)) if x in qmap]
            aligned_b = torch.tensor([lb[x] for x in xs])
            aligned_p = lp[torch.tensor([qmap[x] for x in xs])]
            maxd = (aligned_b - aligned_p).abs().max().item()
            ab = int(lb.argmax())
            apq = int(lp.argmax())
            ap_native = inv.get(apq)
            print(
                f"  {t[:28]!r:32s} max|Δlogit|={maxd:.6f}  base_argmax={ab} preserve_argmax(native)={ap_native}",
                flush=True,
            )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
