#!/usr/bin/env python3
"""Rebuild Dragon packed checkpoint with an fp32 LoRA on lm_head (rank 64).

Loaded from FORGE's quillan-dragon-packed.safetensors. Sets cfg.lm_head_rank=64,
constructs the model (lm_head now has fp32 lora_A/lora_B, zero-init = output
neutral), copies all existing tensors (LM head packed weights preserved), saves.
"""

from __future__ import annotations
import sys, json
from dataclasses import asdict
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "03 - Training & Model"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_sovereign_bitnet import QuillanSovereignBitNetLM, QuillanSovereignConfig

src = ROOT / "exports" / "quillan-dragon-packed.safetensors"
dst = ROOT / "exports" / "quillan-dragon-headlora.safetensors"
old = QuillanSovereignBitNetLM.load_packed_safetensors(src)
cfg = QuillanSovereignConfig(**asdict(old.cfg))
cfg.lm_head_rank = 64
print(f"[cfg] lm_head_rank {old.cfg.lm_head_rank}->{cfg.lm_head_rank}", flush=True)
new = QuillanSovereignBitNetLM(cfg)
osd = old.state_dict()
nsd = new.state_dict()
copied = 0
missing = []
for k, v in nsd.items():
    if k in osd and tuple(v.shape) == tuple(osd[k].shape):
        v.copy_(osd[k])
        copied += 1
    elif k not in osd:
        pass  # new zero-init lm_head lora tensors (fine)
    else:
        missing.append(f"{k} {tuple(osd[k].shape)}!={tuple(v.shape)}")
new.save_packed_safetensors(dst)
print(f"[saved] copied={copied} issues={missing[:5]}", flush=True)
chk = QuillanSovereignBitNetLM.load_packed_safetensors(dst)
print(
    f"[verify] lm_head lora_A={tuple(chk.lm_head.lora_A.shape)} lora_B={tuple(chk.lm_head.lora_B.shape)} absmax={chk.lm_head.lora_B.abs().max().item():.4f}",
    flush=True,
)
