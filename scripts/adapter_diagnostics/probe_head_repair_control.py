#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HEAD-REPAIR + POSITIVE CONTROL — READ-ONLY, IN-MEMORY.

The single untried combined test:
  1. Load socket-v2 checkpoint, PE rescale.
  2. Fold W_u (donor embed_tokens) into txt_dec.weight (value-only, gate D intact).
  3. Zero txt_dec.lora_B.
  4. POSITIVE CONTROL: repaired txt_dec(donor TRUE hidden) -> donor logits?
     Must pass (cos~1, top1 5/5) for any transfer test to be admissible.
  5. If it passes: re-run gate-A/C style readout on held-out prompts using
     the repaired head, both last-position and pooled.

No checkpoint/config/gate/source written. WSP untouched. No backprop.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DONOR_CACHE = ROOT / "exports" / "donor_cache_v23" / "qwen_0.5b_donors.pt"
CKPT = ROOT / "exports" / "quillan-mini-sovereign-bitnet158-socket-v2.safetensors"
ACTS = DIAG / "_acts"
PE_RESCALE = 0.01924
GATE_PROMPTS = {
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
    "Quillan-Ronin is a sovereign",
    "The quick brown fox jumps over the lazy dog and then",
}
HOLDOUT = [
    "The largest planet in our solar system is",
    "Water boils at one hundred degrees",
    "The square root of sixteen is",
    "In 1969 humans first landed on the",
    "The opposite of cold is",
    "The capital of Japan is",
    "Three multiplied by seven equals",
    "The sun rises in the",
    "Gravity pulls objects toward the",
    "A triangle has three sides and three",
]


def capture(model: Any, ids: List[int], stage: str) -> torch.Tensor:
    mods = {"decomposition": model.decomposition, "final_norm": model.final_norm}
    out: Dict[str, torch.Tensor] = {}

    def hook(_m: Any, _i: Tuple[Any, ...], o: torch.Tensor) -> None:
        t = o[0] if isinstance(o, (tuple, list)) else o
        if isinstance(t, torch.Tensor):
            out["h"] = t.detach().float().clone()

    h = mods[stage].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(torch.tensor([ids], dtype=torch.long))
    finally:
        h.remove()
    return out["h"][0]


def tok_of(tok: Any, lg: torch.Tensor, k: int = 4) -> str:
    return " | ".join(
        f"{tok.decode([int(i)], skip_special_tokens=True)!r}"
        for i in torch.topk(lg, k).indices.tolist()
    )


def main() -> int:
    import probe_longctx_quality as plq
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 78)
    print("HEAD REPAIR + POSITIVE CONTROL (read-only, in-memory)")
    print("=" * 78)

    plq.CKPT = CKPT
    model, meta = plq.build_real_model()
    for name, param in model.named_parameters():
        if name.endswith("pos_embed"):
            with torch.no_grad():
                param.data.mul_(PE_RESCALE)
            break
    print(f"custom params={meta['param_count']:,}; PE rescaled")

    snaps = glob.glob(
        os.path.expanduser(
            r"~\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*"
        )
    )
    tok = AutoTokenizer.from_pretrained(snaps[0])
    donor_model = AutoModelForCausalLM.from_pretrained(snaps[0]).eval()
    Wu = torch.load(str(DONOR_CACHE), map_location="cpu", weights_only=False)[
        "embed_tokens"
    ].float()
    dec = model.txt_dec
    print(f"txt_dec.weight {tuple(dec.weight.shape)}  W_u {tuple(Wu.shape)}")
    with torch.no_grad():
        dec.weight.data.copy_(Wu.to(dec.weight.dtype))
        if hasattr(dec, "lora_B"):
            dec.lora_B.data.zero_()
    print("HEAD REPAIRED: txt_dec.weight <- W_u, lora_B zeroed")

    donor = torch.load(
        str(ACTS / "donor_acts.pt"), map_location="cpu", weights_only=False
    )

    # ---- POSITIVE CONTROL through the REPAIRED head ----
    print(
        "\n=== POSITIVE CONTROL: repaired txt_dec(donor TRUE hidden) -> donor logits ==="
    )
    cos_l, t1, t5 = [], 0, 0
    for p, v in donor.items():
        hd = v["final_hidden"][0].float()
        dl = v["logits"][0].float()
        t = hd.shape[0] - 1
        with torch.no_grad():
            cl = dec(hd[t : t + 1])[0].float()
        c = float(F.cosine_similarity(cl.unsqueeze(0), dl[t].unsqueeze(0)).item())
        o1 = int(
            bool(
                set(torch.topk(cl, 1).indices.tolist())
                & set(torch.topk(dl[t], 1).indices.tolist())
            )
        )
        o5 = len(
            set(torch.topk(cl, 5).indices.tolist())
            & set(torch.topk(dl[t], 5).indices.tolist())
        )
        cos_l.append(c)
        t1 += o1
        t5 += o5
        print(f"  {p[:40]!r:43} cos={c:+.6f} top1={o1} top5={o5}")
    mean_cos = sum(cos_l) / len(cos_l)
    print(f"  MEAN cos={mean_cos:+.6f}  top1={t1}/{len(donor)}  top5={t5}/{len(donor)}")
    pc_pass = mean_cos > 0.99 and t1 == len(donor)
    print(f"  POSITIVE CONTROL: {'PASS' if pc_pass else 'FAIL'}")

    # ---- If control passes, transfer test on HELD-OUT prompts ----
    if pc_pass:
        print("\n=== TRANSFER TEST with repaired head — HELD-OUT prompts ===")
        for stage in ["decomposition", "final_norm"]:
            o1s, o5s, cos_s = [], [], []
            print(f"  -- stage={stage} --")
            for p in HOLDOUT:
                ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
                hc = capture(model, ids, stage)
                with torch.no_grad():
                    dh, dlog = (
                        donor_model(
                            torch.tensor([ids]), output_hidden_states=True
                        ).hidden_states[-1][0],
                        donor_model(torch.tensor([ids])).logits[0],
                    )
                t = hc.shape[0] - 1
                with torch.no_grad():
                    cl = dec(hc[t : t + 1])[0].float()
                    dl = dlog[t].float() if hasattr(dlog, "shape") else dlog
                o1 = int(
                    bool(
                        set(torch.topk(cl, 1).indices.tolist())
                        & set(torch.topk(dl, 1).indices.tolist())
                    )
                )
                o5 = len(
                    set(torch.topk(cl, 5).indices.tolist())
                    & set(torch.topk(dl, 5).indices.tolist())
                )
                c = float(F.cosine_similarity(cl.unsqueeze(0), dl.unsqueeze(0)).item())
                o1s.append(o1)
                o5s.append(o5)
                cos_s.append(c)
            print(
                f"    mean top1={sum(o1s) / len(o1s):.4f} top5={sum(o5s) / len(o5s):.4f} cos={sum(cos_s) / len(cos_s):+.4f}"
            )
    else:
        print("\nPositive control FAILED — transfer tests INADMISSIBLE (C2 rule 1).")

    print("\n" + "=" * 78)
    print("DONE — no files written, WSP untouched")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
