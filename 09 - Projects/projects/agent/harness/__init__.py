"""
Quillan Multi-Agent Harness Package
===================================
10 Sovereign Agent Variants with Role-Based Tool Gating and Capability Sandboxing.
"""

from .base import BaseAgent, AgentConfig, AgentResponse
from .variants import AGENT_VARIANTS, get_agent, list_variants
from .registry import AgentRegistry

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentResponse",
    "AGENT_VARIANTS",
    "get_agent",
    "list_variants",
    "AgentRegistry",
]
