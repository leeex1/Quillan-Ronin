#!/usr/bin/env python3
"""Build a sovereign ONI scaffold checkpoint (no donor splice).

Owner spec 2026-11-26: native ONI body, 151648 vocab, 2M ctx, owner dims.
This variant deliberately contains NO donor knowledge. It is the clean control
for testing whether donor projection causes mini incoherence. It must not be
presented as knowledge-bearing or coherent beyond surface token statistics.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni

OUT = Path(r"F:\02_QUILLAN\exports\oni_transfer")
EXPECTED = {"mini": 650_666_808, "main": 1_076_774_730}


def build(model_key: str) -> None:
    model = QuillanRoninOni(owner_config(model_key)).eval()
    nparams = sum(int(p.numel()) for p in model.parameters())
    if nparams != EXPECTED[model_key]:
        raise RuntimeError(f"{model_key} params {nparams} != {EXPECTED[model_key]}")
    cfg = model.cfg
    metadata = {
        "build": "sovereign_scaffold_no_donor_splice",
        "NOT_a_raw_state_dict_transplant": True,
        "knowledge_content": "none; fresh owner_config init",
        "model": model_key,
        "param_count": nparams,
        "expected_param_count": EXPECTED[model_key],
        "owner_config": {
            "vocab_size": int(cfg.vocab_size),
            "max_seq_len": int(cfg.max_seq_len),
            "hidden_dim": int(cfg.hidden_dim),
            "ffn_dim": int(cfg.ffn_dim),
            "n_layer": int(cfg.n_layer),
            "n_head": int(cfg.n_head),
            "num_experts": int(cfg.num_experts),
        },
        "recommended_inference_flags": {"compute_aux": False, "deliberation": False},
        "limitations": [
            "No donor knowledge; control artifact only.",
            "No factual or semantic coherence claim.",
            "Tri-stack and throne-swarm are zero-init identity.",
        ],
    }
    final = OUT / f"quillan_{model_key}_oni_scaffold.pt"
    tmp = OUT / f"quillan_{model_key}_oni_scaffold.pt.tmp"
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, tmp)
    os.replace(tmp, final)
    print(f"saved {final} {final.stat().st_size / 1e9:.3f} GB params={nparams}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "mini")
