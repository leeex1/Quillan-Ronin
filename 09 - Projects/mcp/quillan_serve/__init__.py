"""Serving path for the custom-architecture Quillan-Ronin-Mini checkpoint.

This package does NOT register a llama.cpp graph and does NOT proxy to Ollama.
It runs ``quillan_v8_saturated.QuillanRoninSovereign`` directly in PyTorch.
"""

from .engine import (
    ACTIVE_WINDOW_TOKENS,
    DECLARED_MAX_POSITIONS,
    MODEL_NAME,
    ContextWindowError,
    QuillanServingEngine,
)

__all__ = [
    "ACTIVE_WINDOW_TOKENS",
    "DECLARED_MAX_POSITIONS",
    "MODEL_NAME",
    "ContextWindowError",
    "QuillanServingEngine",
]
