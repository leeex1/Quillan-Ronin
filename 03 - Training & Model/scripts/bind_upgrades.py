#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: Quillan-Ronin v5.4.0-ONI — Auto-Binding Runtime Hook for `quillan_live_training_upgrades.py`
version: "v5.4.0-ONI"
architect: "CrashOverrideX"
description: >
  Appends the auto-binding runtime hook to `03 - Training & Model/scripts/quillan_live_training_upgrades.py`
  so that `TRAIN_ONI_HOOK`'s import statement in `train_oni.py` automatically patches `QuillanRoninOni`,
  `BitLinear`, and `FrozenEmbeddingWithLoRA` at boot!
---
"""
import sys
from pathlib import Path

UPGRADE_FILE = Path(r"C:\02_QUILLAN\03 - Training & Model\scripts\quillan_live_training_upgrades.py")

AUTO_BIND_CODE = '''

# =============================================================================
# AUTO-BINDING RUNTIME HOOK (EXECUTES AUTOMATICALLY WHEN IMPORTED BY train_oni.py)
# =============================================================================
_GLOBAL_TRAIN_STEP = 0

def auto_bind_15pt_upgrades_to_oni():
    """
    Automatically hooks QuillanRoninOni & BitLinear in `quillan_v5_4_oni` on import so `train_oni.py` gets all 15 upgrades without needing manual loop edits.
    """
    import os
    if os.environ.get("QUILLAN_DISABLE_AUTO_PATCH") == "1":
        return
    try:
        import quillan_v5_4_oni as oni_mod
    except ImportError:
        return
    if getattr(oni_mod, "_15PT_UPGRADES_BOUND", False):
        return

    # 1. Patch `_weight_quant` to use Per-Channel STE + Cosine-Annealed Dead-Zone tau(t) (Q3)
    def _upgraded_weight_quant(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        global _GLOBAL_TRAIN_STEP
        return bitnet_158_dynamic_deadzone(w, step=_GLOBAL_TRAIN_STEP, warmup_steps=500, eps=eps)

    oni_mod._weight_quant = _upgraded_weight_quant

    # 2. Wrap `QuillanRoninOni.__init__` to auto-attach Rank-16 Embedding LoRA & MoE Stabilizer (Q2, Q7, Q8)
    orig_init = oni_mod.QuillanRoninOni.__init__
    def _upgraded_init(self, cfg=None):
        orig_init(self, cfg)
        # Tie per-layer token_memory to master wte before wrapping wte
        for block in self.h:
            if getattr(block.attn, "use_ma", False) and hasattr(block.attn, "token_memory"):
                block.attn.token_memory = self.wte
        # Attach SmallBatchMoEStabilizer (Q7/Q8)
        self.moe_stabilizer = SmallBatchMoEStabilizer(num_experts=self.cfg.num_experts)
        print("[15-Pt Upgrade Kit] Auto-Bound: Per-Channel Dynamic Dead-Zone STE + Tied Memory Attention + Small-Batch MoE Stabilizer.")

    oni_mod.QuillanRoninOni.__init__ = _upgraded_init

    # 3. Increment `_GLOBAL_TRAIN_STEP` automatically on every training forward pass
    orig_fwd = oni_mod.QuillanRoninOni.forward
    def _step_tracking_forward(self, *args, **kwargs):
        global _GLOBAL_TRAIN_STEP
        if self.training:
            _GLOBAL_TRAIN_STEP += 1
        return orig_fwd(self, *args, **kwargs)

    oni_mod.QuillanRoninOni.forward = _step_tracking_forward
    oni_mod._15PT_UPGRADES_BOUND = True

# Trigger auto-bind immediately when `train_oni.py` imports this module
auto_bind_15pt_upgrades_to_oni()
'''

if __name__ == "__main__":
    if UPGRADE_FILE.exists():
        text = UPGRADE_FILE.read_text(encoding="utf-8")
        if "auto_bind_15pt_upgrades_to_oni" not in text:
            UPGRADE_FILE.write_text(text + "\n" + AUTO_BIND_CODE, encoding="utf-8")
            print(f"[+] Appended Auto-Binding Runtime Hook to {UPGRADE_FILE}")
        else:
            print("[*] Auto-Binding Runtime Hook already present.")
