#!/usr/bin/env python3
"""Structural pathology scan of the sovereign model's STORED weights.

No tuning, no optimizer, no gold-answer direction. Pure diagnosis: find
malformed parameters (dead rows, NaN/Inf, norm outliers, collapsed scales)
that can be corrected by direct hand surgery.

Reports per tensor:
  - NaN / Inf counts
  - zero fraction
  - dead rows/columns (all exactly zero)
  - norm distribution outliers (rows > 5x median or < 0.2x median)
Only unpacks one matrix at a time to stay within RAM.
"""

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
from wsp_math_eval import load_model

torch.set_num_threads(1)


def stats(name, t):
    f = t.detach().float()
    n = f.numel()
    nan = int(torch.isnan(f).sum())
    inf = int(torch.isinf(f).sum())
    z = int((f == 0).sum())
    line = [f"{name}", f"n={n}", f"nan={nan}", f"inf={inf}", f"zero={z / n:.4f}"]
    if f.dim() >= 2 and n <= 64_000_000:
        rownorm = f.reshape(f.shape[0], -1).abs().amax(dim=1)
        med = rownorm.median().clamp_min(1e-12)
        hi = int((rownorm > 5 * med).sum())
        lo = int((rownorm < 0.2 * med).sum())
        dead = int((rownorm == 0).sum())
        line += [
            f"rowmax_med={med:.4g}",
            f"dead_rows={dead}",
            f"hi5x={hi}",
            f"lo0.2x={lo}",
        ]
    print("  ".join(line), flush=True)


def main():
    ck = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
    m = load_model(ck)
    print(
        f"=== MODEL {ck.name} vocab={m.cfg.vocab_size} layers={len(m.layers)} ===",
        flush=True,
    )

    print("\n-- NON-MATRIX TENSORS --", flush=True)
    for name, t in m.named_parameters():
        if t.dim() < 2:
            stats(name, t)
    for name, t in m.named_buffers():
        if t.dim() < 2 or name.endswith("weight_scale"):
            stats(name, t)

    print("\n-- EMBEDDING ROWS (letter tokens + stats) --", flush=True)
    E = m.embed_tokens.unpack_float()
    stats("embed_tokens.unpack_float", E)
    tok_ids = {"A": None, "B": None, "C": None, "D": None}
    # derive letter ids from the qwen tokenizer path used by the harness
    from wsp_math_eval import tok_for, letter_ids

    tok = tok_for(m.cfg.vocab_size)
    lid = letter_ids(tok, m.cfg.vocab_size)
    for L, i in zip("ABCD", lid):
        r = E[i]
        print(
            f"  letter {L} id={i} norm={r.norm().item():.4f} amax={r.abs().max().item():.4f} "
            f"mean={r.mean().item():.5f} zerofrac={(r == 0).float().mean().item():.3f}",
            flush=True,
        )
    # embedding norm percentiles over full table
    en = E.norm(dim=1)
    for q in (0.1, 1, 5, 50, 95, 99, 99.9):
        print(
            f"  embed_norm_p{q}={torch.quantile(en.float(), q / 100).item():.4f}",
            flush=True,
        )

    print("\n-- PACKED LINEAR GAMMAS (scale pathologies) --", flush=True)
    for li, layer in enumerate(m.layers):
        for sub in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            mod = getattr(layer, sub, None)
            if mod is None or not hasattr(mod, "gamma"):
                continue
            stats(f"L{li}.{sub}.gamma", mod.gamma)
        for sub in ("input_layernorm", "post_attention_layernorm"):
            mod = getattr(layer, sub, None)
            if mod is not None:
                stats(f"L{li}.{sub}.weight", mod.weight)
    stats("final_norm.weight", m.norm.weight)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
