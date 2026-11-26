#!/usr/bin/env python3
"""GLASS BOX surgical section editor (no backprop, no optimizer).

Implements the user's protocol:
  - FREEZE: skip sections already in the ledger's frozen set.
  - UNFREZE one section; apply a closed-form SVD residual edit ONLY to that
    section's matrices (weights restored from the stock donor residual).
  - Measure on the section's OWN domain gate + all OTHER domains as protected
    baseline (regression guard).
  - ACCEPT (freeze) only if own-domain improves AND no sibling regresses;
    else REVERT and log the failure reason.
  - Log every edit to glassbox_ledger.json: what, why, what changed, outcome.

The actual weights are stored in buffers (uint8 packed + fp16 gamma); we edit
via unpack -> closed-form write -> repack. No requires_grad needed.
"""

from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import datetime

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import load_model
from glassbox_ledger import load_ledger, save_ledger, LEDGER
from domain_score import score_all  # will add

import torch

# thread count owned by top-level script/env (Predator audit 2026-11-24)

QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
BASE = ROOT / "exports" / "quillan-mini-glassbox-base.safetensors"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
# section -> (domain gate, matrix filter by submodule name)
SECTION_MAP = {
    "attention": ("geography", ATTN),
    "math_ffn": ("math", ("gate_proj", "up_proj", "down_proj")),
    "identity": ("science", ()),  # embedding/norm handled specially
    "council": ("history", ("council_moe",)),
}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True, choices=list(SECTION_MAP))
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    L = load_ledger()
    if a.section in L["frozen"]:
        print(f"SKIP {a.section}: already FROZEN (change-controlled)")
        return

    domain, subs = SECTION_MAP[a.section]
    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    m = load_model(BASE)

    # baseline domain scores
    base_scores = score_all(m)
    print(f"baseline domains: {base_scores}", flush=True)

    n = 0
    if a.section == "identity":
        # SVD residual on embedding table only
        We = q.get_input_embeddings().weight.detach().float()
        Wq = m.embed_tokens.unpack_float().float()
        if We.shape == Wq.shape:
            R = We - Wq
            r = min(a.rank, min(R.shape))
            U, S, Vh = torch.linalg.svd(R, full_matrices=False)
            A = Vh[:r].contiguous()
            B = (U[:, :r] * S[:r]).contiguous()
            # fold residual directly into embedding buffer (identity section)
            m.embed_tokens.set_weights_from_float(Wq + B @ A)
            n += 1
    else:
        for li, layer in enumerate(m.layers):
            ql = q.model.layers[li]
            for sub in subs:
                parent = ql.self_attn if sub in ATTN else ql.mlp
                qq = getattr(parent, sub, None)
                mm = getattr(layer, sub, None)
                if (
                    qq is None
                    or mm is None
                    or not hasattr(mm, "unpack_effective_weight")
                ):
                    continue
                We = qq.weight.detach().float()
                Wq = mm.unpack_effective_weight().float()
                if tuple(We.shape) != tuple(Wq.shape):
                    continue
                R = We - Wq
                r = min(a.rank, mm.lora_A.shape[0], min(R.shape))
                U, S, Vh = torch.linalg.svd(R, full_matrices=False)
                A = Vh[:r].contiguous()
                B = (U[:, :r] * S[:r]).contiguous()
                with torch.no_grad():
                    mm.lora_A.copy_(A.to(mm.lora_A.dtype))
                    mm.lora_B.copy_(B.to(mm.lora_B.dtype))
                    mm._wc = None
                n += 1

    new_scores = score_all(m)
    print(f"after {a.section} edit: {new_scores}", flush=True)
    own_gain = new_scores[domain] - base_scores[domain]
    regressed = [
        k for k in new_scores if k != domain and new_scores[k] < base_scores[k]
    ]
    accept = own_gain > 0 and not regressed

    L["edits"].append(
        {
            "time": datetime.now().isoformat(),
            "section": a.section,
            "rank": a.rank,
            "modules_edited": n,
            "domain_gate": domain,
            "own_gain": own_gain,
            "regressed": regressed,
            "accept": accept,
            "base_scores": base_scores,
            "new_scores": new_scores,
            "why": "SVD residual restore donor weights in this section only",
            "outcome": "FROZEN (improvement)"
            if accept
            else "REVERTED (no gain or regression)",
        }
    )
    if accept:
        L["frozen"].append(a.section)
        m.save_packed_safetensors(ROOT / "exports" / a.out)
        print(
            f"ACCEPT {a.section}: own_gain={own_gain} -> FROZEN, saved {a.out}",
            flush=True,
        )
    else:
        print(
            f"REJECT {a.section}: own_gain={own_gain} regressed={regressed} -> REVERTED (not saved)",
            flush=True,
        )
    save_ledger(L)


if __name__ == "__main__":
    main()
