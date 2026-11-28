#!/usr/bin/env python3
"""Finish the main ONI checkpoint from surviving merged tensors, atomically.

No donor reload, no alignment recompute. Rebuilds only the state_dict write
that was interrupted. Owner spec 2026-11-26: 1152/2304/L12, vocab 151648,
ctx 2097152, expected 1,076,774,730 params.
"""

from __future__ import annotations
import gc
import hashlib
import os
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni

OUT = Path(r"F:\02_QUILLAN\exports\oni_transfer")
MERGED = OUT / "merged_tensors_main.pt"
FINAL = OUT / "quillan_main_oni_transfer.pt"
TMP = OUT / "quillan_main_oni_transfer.pt.tmp"
EXPECTED = 1_076_774_730
N_REACHABLE = 151643
H, F, L = 1152, 2304, 12


def digest(t: torch.Tensor) -> str:
    flat = t.detach().contiguous().reshape(-1)
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> None:
    t0 = time.time()
    print(f"loading merged tensors {MERGED}", flush=True)
    merged = torch.load(MERGED, map_location="cpu", weights_only=True, mmap=True)
    print(
        f"loaded: embed={tuple(merged['embedding'].shape)} layers={len(merged['layers'])}",
        flush=True,
    )

    print("instantiating fresh main model", flush=True)
    model = QuillanRoninOni(owner_config("main")).eval()
    nparams = sum(int(p.numel()) for p in model.parameters())
    print(f"params={nparams} expected={EXPECTED}", flush=True)
    if nparams != EXPECTED:
        raise RuntimeError(f"param mismatch {nparams} != {EXPECTED}")

    sd = model.state_dict()
    writable = {"wte.weight", "lm_head.weight"}
    for i in range(L):
        writable.add(f"h.{i}.attn.c_attn.weight")
        writable.add(f"h.{i}.attn.c_proj.weight")
        writable.add(f"h.{i}.moe.c_fc.weight")
        writable.add(f"h.{i}.moe.c_proj.weight")
    before = {k: digest(v) for k, v in sd.items() if k not in writable}

    with torch.no_grad():
        sd["wte.weight"][:N_REACHABLE].copy_(merged["embedding"][:N_REACHABLE])
        for i in range(L):
            src = merged["layers"][i]
            sd[f"h.{i}.attn.c_attn.weight"][0:H].copy_(src["q"])
            sd[f"h.{i}.attn.c_attn.weight"][H : 2 * H].copy_(src["k"])
            sd[f"h.{i}.attn.c_attn.weight"][2 * H : 3 * H].copy_(src["v"])
            sd[f"h.{i}.attn.c_proj.weight"].copy_(src["o"])
            sd[f"h.{i}.moe.c_fc.weight"][0:F].copy_(src["gate"])
            sd[f"h.{i}.moe.c_fc.weight"][F : 2 * F].copy_(src["up"])
            sd[f"h.{i}.moe.c_proj.weight"].copy_(src["down"])

    after = {k: digest(v) for k, v in model.state_dict().items() if k not in writable}
    changed = [k for k in before if before[k] != after.get(k)]
    if changed:
        raise RuntimeError(f"sovereign tensors changed: {changed[:8]}")
    print(f"sovereign tensors verified unchanged: {len(before)}", flush=True)

    gate_info = {
        k: {"nonzero": int((v != 0).sum().item()), "maxabs": float(v.abs().max())}
        for k, v in sd.items()
        if k.startswith("tristack.gate_") or k == "throne_swarm.coupling"
    }
    metadata = {
        "build": "finish_main_from_surviving_merged_tensors",
        "NOT_a_raw_state_dict_transplant": True,
        "model": "main",
        "param_count": nparams,
        "expected_param_count": EXPECTED,
        "owner_config": {
            "vocab_size": 151648,
            "max_seq_len": 2097152,
            "hidden_dim": H,
            "ffn_dim": F,
            "n_layer": L,
            "n_head": 18,
            "num_experts": 34,
        },
        "recommended_inference_flags": {"compute_aux": False, "deliberation": False},
        "recommended_inference_note": (
            "Donor geometries do not match ONI; clean path is the stable path. "
            "Falsification: clean-aux max logit delta 1.152; top-1 flips 750 -> 739 -> 602."
        ),
        "pairs": [["qwen", "bitnet"], ["llama", "falcon"]],
        "embedding": {
            "method": "surviving merged_tensors_main.pt",
            "special_rows": list(range(151643, 151648)),
            "special_rows_policy": "model-init preserved; no donor/noise write",
        },
        "role_projection": {
            "method": "Y@P donor->ONI, Y@Q reverse least squares, rank-k SVD shape resize",
            "layer_map": "round(i*(n_donor-1)/(n_oni-1)) clamped",
        },
        "sovereign_tensors_verified_unchanged": len(before),
        "zero_init_tier_gates": gate_info,
        "state_dict_sha256_of_sovereign_keys": hashlib.sha256(
            "".join(sorted(before.values())).encode()
        ).hexdigest()[:24],
        "limitations": [
            "Closed-form knowledge projection only; not raw weight identity.",
            "No claim of coherence, quality, or benchmark improvement.",
            "No backprop; WSP tuning deferred by owner spec.",
            "Rebuilt from surviving merged tensors after corrupted interrupted save.",
        ],
    }

    del merged
    gc.collect()
    print(f"atomic save -> {FINAL}", flush=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, TMP)
    os.replace(TMP, FINAL)
    print(
        f"saved {FINAL.stat().st_size / 1e9:.3f} GB in {time.time() - t0:.1f}s",
        flush=True,
    )

    print("verifying reload", flush=True)
    ckpt = torch.load(FINAL, map_location="cpu", weights_only=True, mmap=True)
    print(
        f"reload keys={list(ckpt.keys())} state_keys={len(ckpt['state_dict'])}",
        flush=True,
    )
    print(
        f"metadata params={ckpt['metadata']['param_count']} vocab={ckpt['metadata']['owner_config']['vocab_size']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
