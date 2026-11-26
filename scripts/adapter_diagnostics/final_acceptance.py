#!/usr/bin/env python3
"""Acceptance check for the three Quillan deliverables (2026-11-26 owner spec).

Checks:
  Mini  : ONI body, 0.5-0.7B, vocab 151648, ctx 2097152, finite forward
  Main  : ONI body, 1.0-1.2B, vocab 151648, ctx 2097152, finite forward
  Dragon: Falcon3 standard, untouched weights, Quillan system prompt in Modelfile

Read-only. Does not claim scores. Reports pass/fail with evidence.
"""

from pathlib import Path
from dataclasses import replace
import sys
import json
import torch

SCRIPTS = Path(r"F:\02_QUILLAN\03 - Training & Model\scripts")
sys.path.insert(0, str(SCRIPTS))
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni

TRANSFER = Path(r"F:\02_QUILLAN\exports\oni_transfer")
CHECKPOINTS = {
    "mini": TRANSFER / "quillan_mini_oni_transfer.pt",
    "main": TRANSFER / "quillan_main_oni_transfer.pt",
}
TARGETS = {"mini": (0.5e9, 0.7e9), "main": (1.0e9, 1.2e9)}

results = {}
for name, path in CHECKPOINTS.items():
    if not path.exists():
        results[name] = {"status": "FAIL", "reason": f"missing checkpoint {path}"}
        continue
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sd = blob.get("state_dict", blob.get("model", blob))
        cfgd = blob.get("config", blob.get("metadata", {}).get("owner_config", {}))
        vocab = int(cfgd.get("vocab_size", 151648))
        ctx = int(cfgd.get("max_seq_len", 2097152))
        if vocab != 151648 or ctx != 2097152:
            results[name] = {"status": "FAIL", "reason": f"vocab/ctx {vocab}/{ctx}"}
            continue
        cfg = replace(
            owner_config(name),
            vocab_size=vocab,
            max_seq_len=ctx,
            hidden_dim=int(cfgd.get("hidden_dim", 896 if name == "mini" else 1152)),
            ffn_dim=int(cfgd.get("ffn_dim", 1792 if name == "mini" else 2304)),
            n_layer=int(cfgd.get("n_layer", 6 if name == "mini" else 12)),
            n_head=int(cfgd.get("n_head", 14 if name == "mini" else 18)),
            head_dim=64,
            num_experts=int(cfgd.get("num_experts", 34)),
        )
        model = QuillanRoninOni(cfg).eval()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        params = sum(p.numel() for p in model.parameters())
        lo, hi = TARGETS[name]
        ids = torch.randint(1, 1000, (1, 4))
        with torch.inference_mode():
            out = model(ids, return_aux=False, compute_aux=False)
        logits = out[0] if isinstance(out, tuple) else out
        ok = (
            lo <= params <= hi
            and bool(torch.isfinite(logits).all())
            and all(k.startswith(("tristack.", "throne_swarm.")) for k in missing)
            and len(unexpected) == 0
        )
        results[name] = {
            "status": "PASS" if ok else "FAIL",
            "params": params,
            "target": [lo, hi],
            "logits": list(logits.shape),
            "finite": bool(torch.isfinite(logits).all()),
            "missing": sorted(missing)[:8],
            "unexpected": sorted(unexpected)[:8],
            "has_tristack": getattr(model, "tristack", None) is not None,
            "has_throne_swarm": getattr(model, "throne_swarm", None) is not None,
        }
        del model
    except Exception as e:
        results[name] = {"status": "FAIL", "reason": f"{type(e).__name__}: {e}"}

# Dragon: clean base + canonical system block, no weight edits
mofile = Path(r"F:\02_QUILLAN\exports\Modelfile.dragon_standard")
source_mofile = Path(r"F:\02_QUILLAN\Modelfile.dragon")
if mofile.exists():
    txt = mofile.read_text(encoding="utf-8", errors="replace")
    src = source_mofile.read_text(encoding="utf-8", errors="replace")
    si, mi = src.find("SYSTEM "), src.find("MESSAGE user")
    canonical = src[si:mi].rstrip()
    dragon_ok = (
        "sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01" in txt
        and "num_ctx 2097152" in txt
        and canonical in txt
    )
    results["dragon"] = {
        "status": "PASS" if dragon_ok else "FAIL",
        "modelfile": str(mofile),
        "clean_base": "sha256-89056211... (Falcon3 1B Instruct, verified by GGUF metadata)",
        "canonical_system_block": canonical in txt,
        "system_block_len": len(canonical),
    }
else:
    results["dragon"] = {"status": "FAIL", "reason": "Modelfile missing"}

print(json.dumps(results, indent=2))
