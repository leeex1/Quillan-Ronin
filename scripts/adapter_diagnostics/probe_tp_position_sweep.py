#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-P DECISIVE ZERO-EDIT TEST (board #986, C21 primitive).

One in-memory forward per prompt. Sweeps txt_dec(final_norm(x_final)[t]) over
ALL token positions (not just the last) and asks: is prompt context readable
at ANY position?

READ-ONLY CONTRACT:
  * No model, checkpoint, tokenizer, config, adapter or gate file is written.
  * The candidate is loaded IN MEMORY by overriding plq.CKPT (same as gate).
  * The only optional mutation is an in-memory PE rescale on the loaded module
    weights (never saved) to test the second condition.
  * Output is stdout only.

Decision rule (pre-registered #986):
  * If ANY position t gives logit cos < 0.99 AND differing top-1 across the
    two prose prompts, context IS readable -> readout-only fix (R3/R1).
  * If ALL positions collapse (cos >= 0.99 or identical top-1), C28
    non-invertibility is confirmed -> NO DEPLOY under current locks.

Usage:
  python -u scripts\\adapter_diagnostics\\probe_tp_position_sweep.py --checkpoint <PATH>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for _p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Same frozen prompts as conversion_gate.py #926.
PROSE_A = "The capital of France is"
PROSE_B = "Hello, my name is"
PE_RESCALE = 0.01924  # 1/52 approx; c10/c7 rescale factor


def capture_hidden(model: Any, ids: List[int]) -> torch.Tensor:
    """Return final_norm(x_final) at ALL positions via a forward hook."""
    captured: Dict[str, torch.Tensor] = {}

    def hook(_mod: Any, _inp: Tuple[Any, ...], out: torch.Tensor) -> None:
        captured["h"] = out.detach().clone()

    h = model.final_norm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(torch.tensor([ids], dtype=torch.long))
    finally:
        h.remove()
    if "h" not in captured:
        raise RuntimeError("final_norm hook did not fire")
    return captured["h"][0]  # [T, hidden]


def position_logits(model: Any, h: torch.Tensor) -> torch.Tensor:
    """Apply the tied decoder to every position: [T, vocab]."""
    with torch.no_grad():
        return model.txt_dec(h).float()


def sweep(model: Any, tok: Any, label: str) -> Dict[str, Any]:
    a_ids = [int(x) for x in tok.encode(PROSE_A, add_special_tokens=False)]
    b_ids = [int(x) for x in tok.encode(PROSE_B, add_special_tokens=False)]
    ha = capture_hidden(model, a_ids)
    hb = capture_hidden(model, b_ids)
    la = position_logits(model, ha)
    lb = position_logits(model, hb)
    T = min(la.shape[0], lb.shape[0])
    rows: List[Dict[str, Any]] = []
    any_pass = False
    best: Dict[str, Any] | None = None
    for t in range(T):
        cos = float(F.cosine_similarity(la[t].unsqueeze(0), lb[t].unsqueeze(0)).item())
        t1a = int(la[t].argmax().item())
        t1b = int(lb[t].argmax().item())
        differ = t1a != t1b
        pos_pass = (cos < 0.99) and differ
        any_pass = any_pass or pos_pass
        row = {
            "t": t,
            "cos": round(cos, 6),
            "top1_a": t1a,
            "top1_a_tok": repr(tok.decode([t1a], skip_special_tokens=True)),
            "top1_b": t1b,
            "top1_b_tok": repr(tok.decode([t1b], skip_special_tokens=True)),
            "top1_differ": differ,
            "pos_pass": pos_pass,
        }
        rows.append(row)
        if pos_pass and (best is None or cos < best["cos"]):
            best = row
    print(f"\n=== T-P SWEEP [{label}] len_a={len(a_ids)} len_b={len(b_ids)} ===")
    for r in rows:
        print(
            f"  t={r['t']:>2} cos={r['cos']:.6f} top1_a={r['top1_a_tok']:<12} "
            f"top1_b={r['top1_b_tok']:<12} differ={r['top1_differ']} "
            f"pos_pass={r['pos_pass']}"
        )
    verdict = "CONTEXT_READABLE" if any_pass else "ALL_POSITIONS_COLLAPSED"
    print(f"T-P RESULT [{label}]: {verdict}  any_pos_pass={any_pass}")
    if best is not None:
        print(f"  best_position: t={best['t']} cos={best['cos']:.6f}")
    return {
        "label": label,
        "verdict": verdict,
        "any_pos_pass": any_pass,
        "best": best,
        "rows": rows,
        "len_a": len(a_ids),
        "len_b": len(b_ids),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()
    ckpt = args.checkpoint.resolve()
    if not ckpt.exists():
        print(f"FATAL: checkpoint missing: {ckpt}")
        return 2

    import probe_longctx_quality as plq

    tok, _ = plq.load_tokenizer()

    print("=" * 78)
    print("T-P POSITION SWEEP (read-only, zero file edits)")
    print(f"checkpoint: {ckpt}")
    print("=" * 78)

    # Condition 1: raw checkpoint, no mutation.
    plq.CKPT = ckpt
    model, _meta = plq.build_real_model()
    raw = sweep(model, tok, "RAW")

    # Condition 2: in-memory PE rescale only (never saved).
    with torch.no_grad():
        pe = None
        pe_name = None
        for name, param in model.named_parameters():
            if name.endswith("pos_embed"):
                pe = param
                pe_name = name
                break
        if pe is not None:
            print(f"PE parameter located: {pe_name} shape={tuple(pe.shape)}")
        if pe is None:
            print("WARN: pos_embed parameter not found; skipping rescale condition")
            scaled = {
                "label": "PE_RESCALED",
                "verdict": "SKIPPED",
                "any_pos_pass": False,
            }
        else:
            pe.data.mul_(PE_RESCALE)
            scaled = sweep(model, tok, f"PE_RESCALED x{PE_RESCALE}")

    print("\n" + "=" * 78)
    print(f"T-P RAW        : {raw['verdict']}")
    print(f"T-P PE_RESCALED: {scaled['verdict']}")
    decisive = raw["any_pos_pass"] or scaled["any_pos_pass"]
    print(
        "T-P FINAL: "
        + (
            "CONTEXT_READABLE -> readout-only fix viable (R3/R1)"
            if decisive
            else "ALL_POSITIONS_COLLAPSED -> C28 non-invertibility confirmed; NO DEPLOY under locks"
        )
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
