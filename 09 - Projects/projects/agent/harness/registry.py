"""
Quillan Agent Registry & Orchestration Bus
==========================================
Manages lifecycle, capabilities, and cross-agent delegation dispatching
across all 34 Council Chambers with Top-K sparse council deliberation.
"""

from typing import Dict, List, Any, Optional
from .base import BaseAgent, AgentResponse
from .variants import AGENT_VARIANTS, get_agent, list_variants

class AgentRegistry:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._cache: Dict[str, BaseAgent] = {}

    def get(self, name: str) -> BaseAgent:
        clean = name.lower().strip()
        if clean not in self._cache:
            self._cache[clean] = get_agent(clean, api_key=self.api_key)
        return self._cache[clean]

    def list_all(self) -> List[Dict[str, Any]]:
        return list_variants()

    def dispatch(self, agent_name: str, task: str) -> AgentResponse:
        """Dispatch a specific task to an individual agent or chamber."""
        agent = self.get(agent_name)
        return agent.run(task)

    def deliberate(self, task: str, chamber_ids: Optional[List[str]] = None) -> Dict[str, AgentResponse]:
        """
        Execute Top-4 Council Deliberation:
        By default, routes to the core synthesis triad + specialist:
        C0-ASTRA (Pattern), C6-LOGOS (Logic), C1-VIR (Ethics), and C9-CODEWEAVER (Execution).
        """
        chambers = chamber_ids or ["c0", "c6", "c1", "c9"]
        deliberations = {}
        for c in chambers:
            deliberations[c] = self.dispatch(c, task)
        return deliberations

    def broadcast(self, task: str, agent_names: Optional[List[str]] = None) -> Dict[str, AgentResponse]:
        """Run multiple agents across a shared prompt and collect perspectives."""
        targets = agent_names or ["c0", "c1", "c6", "c9"]
        results = {}
        for name in targets:
            results[name] = self.dispatch(name, task)
        return results

# Default singleton instance
default_registry = AgentRegistry()
