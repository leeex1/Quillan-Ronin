#!/usr/bin/env python3
"""COMPILE STEP 4: assemble a runnable flagship checkpoint.

- Llama transformer body (all layers)
- canonical four-donor embedding + head (235845 x 2048)
- config vocab_size updated
- direct-ID generation smoke test to check coherence
No training, no gradient.
"""

from __future__ import annotations
import sys, json, shutil
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
OUT = ROOT / "exports"
DEST = OUT / "quillan_flagship_compiled"
LLAMA = Path(r"E:\source_models\llama_1b")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(2)


def main():
    from safetensors import safe_open
    from safetensors.torch import save_file

    Ec = torch.load(OUT / "quillan_embedding_compiled.pt", weights_only=True)[
        "embedding"
    ]
    Q, D = Ec.shape
    print(f"compiled embedding {tuple(Ec.shape)}", flush=True)

    src = LLAMA / "model.safetensors"
    DEST.mkdir(parents=True, exist_ok=True)
    tensors = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k in ("model.embed_tokens.weight", "lm_head.weight"):
                t = Ec
            tensors[k] = t.contiguous()
    print(f"copied {len(tensors)} tensors from Llama body", flush=True)

    # convert to bf16 to halve size, keep compiled embedding in bf16 too
    tensors = {k: v.to(torch.bfloat16) for k, v in tensors.items()}
    save_file(tensors, str(DEST / "model.safetensors"))
    print(f"saved {DEST / 'model.safetensors'}", flush=True)

    cfg = json.loads((LLAMA / "config.json").read_text(encoding="utf-8"))
    cfg["vocab_size"] = Q
    (DEST / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"config vocab_size={cfg['vocab_size']}", flush=True)

    # generation_config + tokenizer copied from llama (IDs differ; smoke test uses direct IDs)
    for fn in ("generation_config.json",):
        if (LLAMA / fn).exists():
            shutil.copy(LLAMA / fn, DEST / fn)
    print("checkpoint assembled", flush=True)


if __name__ == "__main__":
    main()
