#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BUILD FINALNORM-FOLD CANDIDATE — NEW FILE, VALUE-ONLY EDITS.

Loads socket-v2 safetensors dict, applies three value-only edits:
  1. ingestion.pos_embed *= 0.01924   (PE rescale)
  2. txt_dec.weight = Wu @ R.T        (correct fold, lam=0.1, ingestion R)
  3. txt_dec.lora_B = 0                (kill rank-256 residual)

R is re-fit on the same 40 TRAIN prompts used for the holdout test.
All 554 keys preserved, shapes unchanged, canonical never touched.
Output: exports/quillan-mini-sovereign-bitnet158-finalnorm.safetensors
"""

from __future__ import annotations

import glob
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

BASE = ROOT / "exports" / "quillan-mini-sovereign-bitnet158-socket-v2.safetensors"
OUT = ROOT / "exports" / "quillan-mini-sovereign-bitnet158-finalnorm.safetensors"
CANON = ROOT / "exports" / "quillan-mini-sovereign-bitnet158.safetensors"
DONOR_CACHE = ROOT / "exports" / "donor_cache_v23" / "qwen_0.5b_donors.pt"
PE_RESCALE = 0.01924
LAM = 0.1
STAGE = "final_norm"

from probe_construct_readout_map import TRAIN  # noqa: E402


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import probe_longctx_quality as plq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if OUT.exists():
        print(f"FATAL: output exists, will not overwrite: {OUT}")
        return 2
    canon_before = sha(CANON)
    print("=" * 78)
    print("BUILD FINALNORM-FOLD CANDIDATE (value-only, new file)")
    print("=" * 78)
    print(f"canonical sha before: {canon_before}")

    # fit R on TRAIN prompts (same as holdout test)
    plq.CKPT = BASE
    custom, _ = plq.build_real_model()
    for name, param in custom.named_parameters():
        if name.endswith("pos_embed"):
            with torch.no_grad():
                param.data.mul_(PE_RESCALE)
            break

    snaps = glob.glob(
        os.path.expanduser(
            r"~\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*"
        )
    )
    tok = AutoTokenizer.from_pretrained(snaps[0])
    donor = AutoModelForCausalLM.from_pretrained(snaps[0]).eval()
    Wu = torch.load(str(DONOR_CACHE), map_location="cpu", weights_only=False)[
        "embed_tokens"
    ].float()

    mods = {"final_norm": custom.final_norm, "ingestion": custom.ingestion}
    Xs, Ys = [], []
    t0 = time.perf_counter()
    for i, p in enumerate(TRAIN):
        ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
        out: Dict[str, torch.Tensor] = {}

        def hook(_m: Any, _i: tuple, o: torch.Tensor, _out=out) -> None:
            t = o[0] if isinstance(o, (tuple, list)) else o
            if isinstance(t, torch.Tensor):
                _out["h"] = t.detach().float().clone()

        h = mods[STAGE].register_forward_hook(hook)
        try:
            with torch.no_grad():
                _ = custom(torch.tensor([ids], dtype=torch.long))
        finally:
            h.remove()
        hc = out["h"][0]
        with torch.no_grad():
            dm = donor(torch.tensor([ids], dtype=torch.long), output_hidden_states=True)
        hd = dm.hidden_states[-1][0].detach().float()
        T = min(hc.shape[0], hd.shape[0])
        Xs.append(hc[:T])
        Ys.append(hd[:T])
        if (i + 1) % 10 == 0:
            print(f"  fit R: {i + 1}/{len(TRAIN)} ({time.perf_counter() - t0:.1f}s)")
    X = torch.cat(Xs, 0)
    Y = torch.cat(Ys, 0)
    A = X.T @ X + LAM * torch.eye(X.shape[1], dtype=X.dtype)
    R = torch.linalg.solve(A, X.T @ Y)
    W_eff = (Wu @ R.T).to(torch.float16).contiguous()
    print(f"fit R on {tuple(X.shape)} pairs; W_eff={tuple(W_eff.shape)} {W_eff.dtype}")

    # load full base dict
    sd: Dict[str, torch.Tensor] = {}
    with safe_open(str(BASE), framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    n_keys_before = len(sd)
    print(f"loaded {n_keys_before} keys from base")

    # three value-only edits
    sd["ingestion.pos_embed"] = (
        (sd["ingestion.pos_embed"].float() * PE_RESCALE).to(torch.float16).contiguous()
    )
    sd["txt_dec.weight"] = W_eff
    sd["txt_dec.lora_B"] = torch.zeros_like(sd["txt_dec.lora_B"])
    print("edits applied: PE rescale, txt_dec.weight <- W_eff, lora_B zeroed")

    assert len(sd) == n_keys_before == 554, "key count changed!"
    assert sd["txt_dec.weight"].shape == (151936, 896)
    save_file(sd, str(OUT), metadata={"format": "pt"})
    print(f"saved: {OUT} ({OUT.stat().st_size:,} bytes)")

    canon_after = sha(CANON)
    print(f"canonical sha after : {canon_after}")
    print(f"CANONICAL UNCHANGED: {'PASS' if canon_before == canon_after else 'FAIL'}")
    print("=" * 78)
    return 0 if canon_before == canon_after else 2


if __name__ == "__main__":
    raise SystemExit(main())
