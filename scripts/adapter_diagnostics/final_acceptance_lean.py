#!/usr/bin/env python3
"""Memory-lean acceptance: one checkpoint at a time, mmap, hard release."""

from pathlib import Path
from dataclasses import replace
import gc
import json
import sys
import torch

SCRIPTS = Path(r"F:\02_QUILLAN\03 - Training & Model\scripts")
sys.path.insert(0, str(SCRIPTS))
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni

TRANSFER = Path(r"F:\02_QUILLAN\exports\oni_transfer")
TARGETS = {"mini": (0.5e9, 0.7e9), "main": (1.0e9, 1.2e9)}
DIMS = {"mini": (896, 1792, 6, 14), "main": (1152, 2304, 12, 18)}


def check_one(name: str) -> dict:
    path = TRANSFER / f"quillan_{name}_oni_transfer.pt"
    if not path.exists():
        return {"status": "FAIL", "reason": f"missing {path}"}
    blob = None
    model = None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        sd = blob["state_dict"]
        cfgd = blob["metadata"]["owner_config"]
        vocab, ctx = int(cfgd["vocab_size"]), int(cfgd["max_seq_len"])
        if vocab != 151648 or ctx != 2097152:
            return {"status": "FAIL", "reason": f"vocab/ctx {vocab}/{ctx}"}
        H, F, L, NH = DIMS[name]
        cfg = replace(
            owner_config(name),
            vocab_size=vocab,
            max_seq_len=ctx,
            hidden_dim=H,
            ffn_dim=F,
            n_layer=L,
            n_head=NH,
            head_dim=64,
            num_experts=34,
        )
        model = QuillanRoninOni(cfg).eval()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        params = sum(p.numel() for p in model.parameters())
        ids = torch.randint(1, 1000, (1, 4))
        with torch.inference_mode():
            out = model(ids, return_aux=False, compute_aux=False)
        logits = out[0] if isinstance(out, tuple) else out
        lo, hi = TARGETS[name]
        ok = (
            lo <= params <= hi
            and bool(torch.isfinite(logits).all())
            and all(k.startswith(("tristack.", "throne_swarm.")) for k in missing)
            and len(unexpected) == 0
        )
        return {
            "status": "PASS" if ok else "FAIL",
            "params": params,
            "target": [lo, hi],
            "logits": list(logits.shape),
            "finite": bool(torch.isfinite(logits).all()),
            "missing": sorted(missing)[:8],
            "unexpected": sorted(unexpected)[:8],
            "has_tristack": model.tristack is not None,
            "has_throne_swarm": model.throne_swarm is not None,
        }
    except Exception as e:
        return {"status": "FAIL", "reason": f"{type(e).__name__}: {e}"}
    finally:
        del model, blob
        gc.collect()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = ("mini", "main") if which == "all" else (which,)
    results = {n: check_one(n) for n in names}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
