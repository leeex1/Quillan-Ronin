#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C18-SHEPHERD: in-memory pos_embed RESCALE CONTROL (read-only, no file edits).

Forge (#906) showed pos_embed mean-abs 0.622575 vs txt_emb 0.011981 (51.96x)
and that pos_embed*=0.02 changes top-1 from BOSE to digits but not coherence.
This script removes the scale confound PROPERLY (scale = txt_emb_mean_abs /
pos_embed_mean_abs, computed from the checkpoint) and re-runs the SAME functional
transfer metrics against the clean donor. Decisive question:

    After the pos_embed scale mismatch is removed in memory, does the custom
    checkpoint's output distribution show ANY functional correlation with
    the donor's on the same prompts?

If NO -> the divergence is NOT explainable by scaling -> functional transfer
is FALSIFIED and the root cause is semantic/graph mapping, not just magnitude.

Writes ONLY _acts/custom_rescaled_acts.pt and rescale_control_results.json.
"""

from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
ACTORS = DIAG / "_acts"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "03 - Training & Model"))
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch
import torch.nn.functional as F
from probe_longctx_quality import build_real_model

PROMPTS = (
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
    "Quillan-Ronin is a sovereign",
    "The quick brown fox jumps over the lazy dog and then",
)


def cos_rows(a, b):
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    c = F.cosine_similarity(a, b, dim=-1)
    return round(float(c.mean()), 6)


def linear_cka(x, y):
    x = x.float().reshape(-1, x.shape[-1])
    y = y.float().reshape(-1, y.shape[-1])
    if x.shape[0] < 3 or x.shape[1] != y.shape[1]:
        return None
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xtx = x @ x.t()
    yty = y @ y.t()
    den = math.sqrt(float((xtx * xtx).sum()) * float((yty * yty).sum()))
    return round(float((xtx * yty).sum()) / den, 6) if den > 1e-12 else None


def main():
    d = torch.load(
        str(ACTORS / "donor_acts.pt"), map_location="cpu", weights_only=False
    )
    model, meta = build_real_model()
    with torch.no_grad():
        txt_mean = float(model.ingestion.txt_emb.weight.float().abs().mean())
        pos_mean = float(model.ingestion.pos_embed.float().abs().mean())
        scale = txt_mean / pos_mean
        model.ingestion.pos_embed.mul_(scale)  # IN-MEMORY ONLY
    print(
        f"in-memory pos_embed scale = {scale:.6f} (txt {txt_mean:.6f} / pos {pos_mean:.6f})",
        flush=True,
    )

    modules = {
        "ingestion": model.ingestion,
        "decomposition": model.decomposition,
        "diffusion_core": model.diffusion_core,
        "moe": model.moe,
        "final_norm": model.final_norm,
        "logits": model.txt_dec,
    }
    cap = {}
    for p in PROMPTS:
        ids = d[p]["ids"]
        acts = {}
        handles = []

        def mk(name):
            def hook(mod, inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(t, torch.Tensor):
                    acts[name] = t.detach().float().clone()

            return hook

        for name, mod in modules.items():
            handles.append(mod.register_forward_hook(mk(name)))

        def dec_in(mod, inp, out):
            t = inp[0] if isinstance(inp, (tuple, list)) else inp
            if isinstance(t, torch.Tensor):
                acts["decoder_input"] = t.detach().float().clone()

        handles.append(model.txt_dec.register_forward_hook(dec_in))
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            _ = model(x)
        for h in handles:
            h.remove()
        cap[p] = acts
        del x, acts, handles
    del model
    gc.collect()
    torch.save(cap, str(ACTORS / "custom_rescaled_acts.pt"))

    R: dict = {
        "owner": "c18-shepherd",
        "brief": 911,
        "in_memory_scale": round(scale, 6),
        "txt_emb_mean_abs": round(txt_mean, 6),
        "pos_embed_mean_abs_before": round(pos_mean, 6),
        "pos_embed_mean_abs_after": round(pos_mean * scale, 6),
        "note": "in-memory only; NO checkpoint/model file was modified",
    }
    R["final_hidden_cosine_vs_donor"] = {
        p: cos_rows(cap[p]["decoder_input"], d[p]["final_hidden"]) for p in PROMPTS
    }
    all_c = torch.cat([cap[p]["decoder_input"] for p in PROMPTS], dim=1).squeeze(0)
    R["final_hidden_CKA_by_donor_layer"] = {
        f"layer_{i}": linear_cka(
            all_c,
            torch.cat([d[p]["hidden_states"][i] for p in PROMPTS], dim=1).squeeze(0),
        )
        for i in range(len(d[PROMPTS[0]]["hidden_states"]))
    }
    R["logits_cosine_vs_donor"] = {
        p: cos_rows(cap[p]["logits"], d[p]["logits"]) for p in PROMPTS
    }

    # top-k overlap (last position, all prompts)
    ov = {}
    for k in (1, 5, 10, 20):
        hits = []
        for p in PROMPTS:
            a = set(cap[p]["logits"][0, -1, :].topk(k).indices.tolist())
            b = set(d[p]["logits"][0, -1, :].topk(k).indices.tolist())
            hits.append(len(a & b))
        ov[f"top{k}"] = {
            "hits_per_prompt": hits,
            "mean": round(sum(hits) / len(hits), 4),
            "chance": round(k * k / 151936, 8),
        }
    R["topk_overlap_after_rescale"] = ov

    ranks = []
    for p in PROMPTS:
        dt1 = int(d[p]["logits"][0, -1, :].argmax().item())
        row = cap[p]["logits"][0, -1, :].float()
        ranks.append(
            {
                "prompt": p,
                "donor_top1_id": dt1,
                "rank_in_custom_after_rescale": int((row > row[dt1]).sum().item()) + 1,
            }
        )
    R["donor_top1_rank_in_custom_after_rescale"] = ranks

    # prompt discrimination after rescale
    disc = {}
    for stage in ("ingestion", "decomposition", "final_norm", "logits"):
        vals = []
        for i in range(len(PROMPTS)):
            for j in range(i + 1, len(PROMPTS)):
                a = cap[PROMPTS[i]][stage][0, -1, :].float().reshape(1, -1)
                b = cap[PROMPTS[j]][stage][0, -1, :].float().reshape(1, -1)
                vals.append(round(float(F.cosine_similarity(a, b)[0]), 6))
        disc[stage] = round(sum(vals) / len(vals), 6)
    R["prompt_discrimination_after_rescale"] = disc

    op = DIAG / "rescale_control_results.json"
    op.write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", op)
    print(
        json.dumps(
            {
                k: v
                for k, v in R.items()
                if k not in ("final_hidden_CKA_by_donor_layer",)
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
