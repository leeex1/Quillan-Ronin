#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Tools Pack (checklist rows 28, 30, 31, 40)

Wires four previously standalone engines into the model as one opt-in pack:

  Row 28  sol_pi_harness.SoLPiHarness                 (SoL-Pi, arXiv:2609.20519)
  Row 30  ai_consciousness_evaluator.DifferentiableConsciousnessHead
                                                       (Cacophony to Hierarchy, arXiv:2609.35618)
  Row 31  ax_prover_engine.DifferentiableProverScorer  (Ax-Prover, MCP/Lean)
  Row 40  inverse_cryptography_engine.ZeroPriorAgentHandshake
                                                       (Callimahos inverse cryptography)

Integration contract (documented gap, not hidden):
  * Differentiable regularizers only. forward() never changes the model logits.
  * The consciousness score is a detached telemetry value, not a training signal.
  * The Lean MCP client, the SoL-Pi tool-output path and the handshake challenge
    are plain-Python APIs. They are exposed as methods and are not called per token.
  * Off by default in ModelConfig (use_standalone_tools = False).

Complexity: forward is O(B * (D*37 + num_tactics*D*D)) for pooled input (B, D).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ai_consciousness_evaluator import DifferentiableConsciousnessHead
from ax_prover_engine import DifferentiableProverScorer
from inverse_cryptography_engine import ZeroPriorAgentHandshake
from sol_pi_harness import SoLPiHarness


class StandaloneToolsPack(nn.Module):
    """Opt-in pack combining SoL-Pi, consciousness head, prover scorer and handshake."""

    def __init__(self, hidden_dim: int, num_tactics: int = 12, num_indicators: int = 37) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.num_tactics = int(num_tactics)
        self.consciousness = DifferentiableConsciousnessHead(hidden_dim, num_indicators)
        self.prover_scorer = DifferentiableProverScorer(hidden_dim, num_tactics)
        self.harness = SoLPiHarness()
        self.handshake = ZeroPriorAgentHandshake()
        # Arithmetic-invariant handshake is deterministic; evaluate once at construction.
        self.handshake_ok: bool = bool(
            self.handshake.verify_response(self.handshake.generate_challenge())
        )

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute regularizer and telemetry from pooled hidden states.

        Args:
            pooled: (B, hidden_dim) mean-pooled hidden state.
        Returns:
            Dict with ``loss`` (scalar, differentiable), ``consciousness_score`` (B,),
            ``tactic_entropy`` (B,) and ``indicator_probs`` (B, num_indicators).
        """
        if pooled.dim() != 2 or pooled.size(-1) != self.hidden_dim:
            raise ValueError(
                f"expected pooled of shape (B, {self.hidden_dim}), got {tuple(pooled.shape)}"
            )
        pooled = pooled.to(self.prover_scorer.goal_proj.weight.dtype)
        indicator_probs, score = self.consciousness(pooled)
        tactic_probs, _ = self.prover_scorer(pooled)
        entropy = -(tactic_probs * tactic_probs.clamp_min(1e-9).log()).sum(-1)
        norm_entropy = entropy / math.log(max(self.num_tactics, 2))
        harness_loss = self.harness.compute_harness_loss(indicator_probs)
        loss = harness_loss + 0.1 * norm_entropy.mean()
        return {
            "loss": loss,
            "consciousness_score": score.detach(),
            "tactic_entropy": entropy.detach(),
            "indicator_probs": indicator_probs.detach(),
        }

    def process_tool_output(
        self, command: str, exit_code: int, raw_output: str, current_turn: int
    ) -> str:
        """SoL-Pi tool-output path: evidence-preserving reduce plus observation pack."""
        return self.harness.process_tool_output(command, exit_code, raw_output, current_turn)

    def telemetry(self, aux: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Compact JSON-safe telemetry."""
        info: Dict[str, Any] = {"handshake_ok": self.handshake_ok}
        info.update(self.harness.get_telemetry())
        if aux is not None:
            info["consciousness_score"] = round(float(aux["consciousness_score"].mean()), 4)
            info["tactic_entropy"] = round(float(aux["tactic_entropy"].mean()), 4)
        return info
