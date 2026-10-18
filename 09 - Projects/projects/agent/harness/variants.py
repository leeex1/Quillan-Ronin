"""
Quillan Agent Variants & Council Registry
=========================================
Exports the full 34-Chamber Council of Quillan-Ronin (C0-C33) with legacy alias support.
"""

from typing import Dict, List, Any, Optional
from .base import AgentConfig, BaseAgent
from .council_members import COUNCIL_SPECS, build_council_configs

# Build complete dictionary of 34 variants + aliases
AGENT_VARIANTS: Dict[str, AgentConfig] = build_council_configs()

def get_agent(name: str, api_key: Optional[str] = None) -> BaseAgent:
    """Retrieve an initialized agent instance by chamber ID, persona name, or alias."""
    clean = name.lower().strip()
    if clean not in AGENT_VARIANTS:
        available_chambers = [s["id"] for s in COUNCIL_SPECS]
        available_personas = [s["name"] for s in COUNCIL_SPECS]
        raise ValueError(
            f"Unknown agent variant '{name}'.\n"
            f"Available Chamber IDs: {', '.join(available_chambers)}\n"
            f"Available Personas   : {', '.join(available_personas)}"
        )
    return BaseAgent(AGENT_VARIANTS[clean], api_key=api_key)

def list_variants() -> List[Dict[str, Any]]:
    """List metadata for all 34 canonical Council chambers."""
    return [
        {
            "id": s["id"],
            "name": s["name"],
            "chamber": s["chamber"],
            "role_title": s["title"],
            "description": s["desc"],
            "authorized_tools": s["tools"],
            "temperature": s["temp"],
        }
        for s in COUNCIL_SPECS
    ]
