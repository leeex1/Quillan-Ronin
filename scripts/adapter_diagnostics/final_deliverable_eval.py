#!/usr/bin/env python3
"""Evaluate the two deliverable candidates on the FINAL held-out bank (never used
for selection). Persist scores + SHA256 to a JSON evidence artifact.
Pinned 1 thread, paired, no training.
"""

from __future__ import annotations
import sys, json, hashlib, math
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from heldout_final40 import prompts, KEY

torch.set_num_threads(1)
MODELS = {
    "mini_faithful": "quillan-mini-hifi-g4-councilfixed.safetensors",
    "flagship_faithful": "quillan-flagship-hifi-g4-exact.safetensors",
}


def main():
    report = {
        "bank": "heldout_final40",
        "n": 40,
        "regime": "pinned_1_thread",
        "models": {},
    }
    for name, f in MODELS.items():
        p = ROOT / "exports" / f
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        m = load_model(p)
        tok = tok_for(m.cfg.vocab_size)
        lids = letter_ids(tok, m.cfg.vocab_size)
        ok = 0
        ces = []
        dist = [0] * 4
        with torch.inference_mode():
            for seq, gold in zip(prompts(tok), KEY):
                lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
                pi = int(lg.argmax())
                dist[pi] += 1
                ok += int(pi == gold)
                ces.append(-F.log_softmax(lg, -1)[gold].item())
        ce = sum(ces) / len(ces)
        report["models"][name] = {
            "file": f,
            "sha256_16": h,
            "accuracy": f"{ok}/40",
            "acc_pct": round(100 * ok / 40, 1),
            "CE": round(ce, 4),
            "PPL": round(math.exp(ce), 3),
            "dist": dist,
        }
        print(
            f"[{name}] {ok}/40 CE={ce:.4f} PPL={math.exp(ce):.3f} dist={dist} sha={h}",
            flush=True,
        )
    op = ROOT / "exports" / "final_deliverable_eval.json"
    op.write_text(json.dumps(report, indent=2))
    print(f"saved {op}", flush=True)


if __name__ == "__main__":
    main()
