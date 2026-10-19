#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
👑 QUILLAN-RONIN v5.4-ONI COMPATIBILITY SHIM (FORMER v10 SCRIPT)
=============================================================================
NOTICE: Version 10 was an artifact ahead of canonical v5.4-ONI.
In accordance with sovereign version governance, all versions have converged
on canonical v5.4-ONI (quillan_v5_4_oni.py).

This file provides backward-compatible shims and re-exports from quillan_v5_4_oni.py
to maintain 100% backward compatibility for all training and diagnostic scripts.
=============================================================================
"""

import sys
from pathlib import Path

# Add script directory to sys.path
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import torch
import torch.nn as nn

from quillan_v5_4_oni import (
    QuillanOniConfig,
    QuillanRoninOni,
    RotaryEmbedding,
    CausalSelfAttention,
    CouncilExpertSwarm,
    CouncilExpert,
    UnrolledCouncilMoEBlock,
    NineVectorPrismDecomposition,
    NineVectorPrism,
    Conv1D,
    EthicalImpactConstraintEngine,
    MARTAThermodynamicGating,
    DynamicQuantumSwarmOscillation,
    PrimeCovenantFramework,
    CCRLFramework,
    QuantumFormulasEngine,
    ComplexityRouter,
    QuillanAgenticExecutor,
    QuillanMemory,
    ModalityIsolatedThermoDiffusion,
    DistillationHead,
    LeeMach6Governor,
    LeeMach6VelocityGovernor,
)

# Canonical config aliases
QuillanArchConfig = QuillanOniConfig
QuillanUnrolledConfig = QuillanOniConfig
QuillanConfig = QuillanOniConfig


class QuillanLegacyAdapterModel(nn.Module):
    """
    Backward-compatible adapter wrapping canonical QuillanRoninOni.
    When invoked with labels, defaults return_aux=False so callers receive
    (logits, total_loss) as expected by legacy scripts, while maintaining
    100% v5.4-ONI underlying parameters and architecture.
    """
    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = QuillanOniConfig(**kwargs)
        self.cfg = cfg
        self.inner = QuillanRoninOni(cfg)

    def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False,
                path_override=None, recirc_state=None, deliberation=True,
                teacher_tokens=None, proxy_logits=None, persona_id=None, return_aux=False):
        return self.inner(
            input_ids, labels=labels, past_key_values=past_key_values, use_cache=use_cache,
            path_override=path_override, recirc_state=recirc_state, deliberation=deliberation,
            teacher_tokens=teacher_tokens, proxy_logits=proxy_logits, persona_id=persona_id,
            return_aux=return_aux,
        )

    def generate(self, *args, **kwargs):
        return self.inner.generate(*args, **kwargs)

    def deliberate(self, *args, **kwargs):
        return self.inner.deliberate(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


QuillanRoninSovereign = QuillanLegacyAdapterModel
QuillanUnrolledSovereign = QuillanLegacyAdapterModel
QuillanSovereignUnifiedModel = QuillanLegacyAdapterModel

__all__ = [
    "QuillanOniConfig",
    "QuillanRoninOni",
    "QuillanRoninSovereign",
    "QuillanArchConfig",
    "QuillanUnrolledConfig",
    "QuillanUnrolledSovereign",
    "QuillanConfig",
    "QuillanSovereignUnifiedModel",
    "RotaryEmbedding",
    "CausalSelfAttention",
    "CouncilExpertSwarm",
    "CouncilExpert",
    "UnrolledCouncilMoEBlock",
    "NineVectorPrismDecomposition",
    "NineVectorPrism",
    "Conv1D",
    "EthicalImpactConstraintEngine",
    "MARTAThermodynamicGating",
    "DynamicQuantumSwarmOscillation",
    "PrimeCovenantFramework",
    "CCRLFramework",
    "QuantumFormulasEngine",
    "ComplexityRouter",
    "QuillanAgenticExecutor",
    "QuillanMemory",
    "ModalityIsolatedThermoDiffusion",
    "DistillationHead",
    "LeeMach6Governor",
    "LeeMach6VelocityGovernor",
]
