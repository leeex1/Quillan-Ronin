"""PASS 2 tool-consolidation shims (audit #886, coordinator brief #896).

Collapses three HIGH-confidence duplicate families behind canonical tools:

    session_replay x17 -> mcp_session_replay
    session_list   x17 -> mcp_session_list
    cross-MCP route x9 -> mcp_route

Every legacy name stays resolvable through a namespaced shim for a 90-day /
one-minor-release deprecation window. Nothing in this package edits, imports,
or reconfigures any live server or facade.

Explicitly NOT consolidated (see DEPRECATION.md):
    * prism_decompose - different signatures in quillan_prism and
      quillan_thinking_engine; kept as two namespaced, distinct callables.
    * The six thinking_engine @mcp.tool(name=...) overrides, including the
      typo'd "Genuis_level" and "adversarial_dual_council_diliberation".
      Those strings are preserved verbatim - call sites bind to them.
"""

from __future__ import annotations

from .namespace_registry import (
    CANONICAL_NAMES,
    LEGACY_TOOLS,
    MERGED_NAMES,
)
from .shims import SHIMS, is_legacy_name, resolve

DEPRECATION_WINDOW_DAYS: int = 90
DEPRECATION_MINOR_RELEASES: int = 1
DEPRECATED_SINCE_VERSION: str = "5.4.0-ONI"
REMOVAL_TARGET_VERSION: str = "5.5.0-ONI"

__all__ = [
    "CANONICAL_NAMES",
    "LEGACY_TOOLS",
    "MERGED_NAMES",
    "SHIMS",
    "resolve",
    "is_legacy_name",
    "DEPRECATION_WINDOW_DAYS",
    "DEPRECATION_MINOR_RELEASES",
    "DEPRECATED_SINCE_VERSION",
    "REMOVAL_TARGET_VERSION",
]
