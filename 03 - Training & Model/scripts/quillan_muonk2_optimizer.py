#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4.0-ONI — SOVEREIGN MUON-K2 + SOPHIA-H + ADAMW + CCRL HYBRID OPTIMIZER
------------------------------------------------------------------------------------------
Canonical implementation forwarding to QuillanFusedOptimizer for 100% native PyTorch compatibility.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add parent to path to find quillan_fused_optimizer
_PARENT_DIR = Path(__file__).resolve().parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from quillan_fused_optimizer import (
    centralize_gradient,
    newton_schulz,
    PIDGovernor,
    NoiseScaleEMA,
    QuillanFusedOptimizer,
    create_quillan_fused_optimizer,
    SovereignMuonK2AdamW,
    create_quillan_muonk2_optimizer,
)

__all__ = [
    "centralize_gradient",
    "newton_schulz",
    "PIDGovernor",
    "NoiseScaleEMA",
    "QuillanFusedOptimizer",
    "create_quillan_fused_optimizer",
    "SovereignMuonK2AdamW",
    "create_quillan_muonk2_optimizer",
]
