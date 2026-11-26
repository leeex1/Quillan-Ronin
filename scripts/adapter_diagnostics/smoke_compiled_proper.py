#!/usr/bin/env python3
"""Proper smoke test: encode withHOST tokenizer, map native->Quillan IDs, decode back.

The earlier greedy raw-string encoder could not match byte-level tokens (e.g.
'Ġthe' against plain text ' the'), producing garbage IDs. This uses the actual
tokenizer so input IDs are exactly what the model's embedding expects.
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
torch.set_num_threads(1)

MODELS = {
    "flagship": (
        OUT / "quillan_flagship_compiled",
        r"E:\source_models\llama_1b",
        "llama",
    ),
    "mini": (OUT / "quillan_mini_compiled", r"E:\source_models\qwen_0.5b", "qwen"),
}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    d2q = json.loads((OUT / "quillan_token_maps.json").read_text(encoding="utf-8"))
    prompts = [
        "The capital of France is",
        "2 + 2 =",
        "Hello, my name is",
        "The chemical symbol for water is",
    ]
    for name, (path, tokpath, donor) in MODELS.items():
        print(f"\n=== {name} ===", flush=True)
        tok = AutoTokenizer.from_pretrained(tokpath, local_files_only=True)
        qmap = {int(k): v for k, v in d2q["donor_to_quillan"][donor].items()}
        # inverse: quillan id -> donor id -> text
        inv = {v: k for k, v in qmap.items()}
        m = AutoModelForCausalLM.from_pretrained(
            str(path),
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        with torch.inference_mode():
            for p in prompts:
                native = tok.encode(p, add_special_tokens=False)
                ids = [qmap[x] for x in native]
                out = m.generate(
                    torch.tensor([ids]),
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=0,
                )
                gen = out[0, len(ids) :].tolist()
                # decode via donor ids where available
                native_gen = [inv[t] for t in gen if t in inv]
                txt = (
                    tok.decode(native_gen, skip_special_tokens=True)
                    if native_gen
                    else f"<ids {gen}>"
                )
                print(f"  {p!r} -> {txt!r}", flush=True)
        del m


if __name__ == "__main__":
    main()
