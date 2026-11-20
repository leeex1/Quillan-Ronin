"""Namespace registry - single source of truth for PASS 2 tool consolidation.

Derived from ``mcp_tool_registry.json`` (19 quillan servers, snapshot
2026-10-21) plus the empirical findings of audit #886 and the live facade
inspection performed under brief #896.

Every legacy name below is a REAL exposed tool name. Nothing is invented.

Scope discipline (audit #886):
  * HIGH-confidence merges implemented here: session_replay (x17),
    session_list (x17), cross-MCP route templates (x9). 43 names -> 3 canonical
    interfaces.
  * PRESERVED, never merged: prism_decompose (two different signatures in
    quillan_prism / quillan_thinking_engine), the six thinking_engine
    ``@mcp.tool(name=...)`` overrides (including the typo'd exposed strings
    "Genuis_level" and "adversarial_dual_council_diliberation"), and the three
    pure *_alias tools whose call sites bind to those exact strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

# ---------------------------------------------------------------------------
# Canonical tool names (the new single interfaces)
# ---------------------------------------------------------------------------
CANONICAL_SESSION_REPLAY: str = "mcp_session_replay"
CANONICAL_SESSION_LIST: str = "mcp_session_list"
CANONICAL_MCP_ROUTE: str = "mcp_route"

CANONICAL_NAMES: FrozenSet[str] = frozenset(
    {CANONICAL_SESSION_REPLAY, CANONICAL_SESSION_LIST, CANONICAL_MCP_ROUTE}
)

# ---------------------------------------------------------------------------
# Namespace -> original server directory. All 19 sessions/ dirs were verified
# present on disk; the 17 below are exactly the servers that expose the
# session_replay / session_list family.
# ---------------------------------------------------------------------------
SESSION_NAMESPACES: Dict[str, str] = {
    "audio": "quillan_audio",
    "codeweaver": "quillan_code",
    "council": "quillan_council",
    "gamedev": "quillan_gamedev",
    "governor": "quillan_governor",
    "media": "quillan_media",
    "model_lab": "quillan_model_lab",
    "ops": "quillan_ops",
    "orchestrator": "quillan_orchestrator",
    "orochi": "quillan_orochi",
    "prism": "quillan_prism",
    "rag": "quillan_rag",
    "re": "quillan_re",
    "samurai": "quillan_samurai_engine",
    "spider": "quillan_spider",
    "team": "quillan_team",
    "web": "quillan_web",
}

SESSION_SERVER_TO_NAMESPACE: Dict[str, str] = {
    server: ns for ns, server in SESSION_NAMESPACES.items()
}

# ---------------------------------------------------------------------------
# Cross-MCP route profile table. These 9 are the byte-identical template
# family (identical docstring, control flow, target/suggested_tool mapping,
# and error string). Only the per-source `note` text and default target vary,
# so the canonical tool keeps them as data rather than as 9 copies of code.
#
# DELIBERATELY EXCLUDED: orochi_route ("Cross-MCP routing for Orochi evolved
# artifacts.") - different description/semantics, NOT part of the identical
# template. Also excluded: spider_route_findings, skill_route, complexity_route.
# ---------------------------------------------------------------------------
ROUTE_PROFILES: Dict[str, Dict[str, object]] = {
    "audio": {
        "default_target": "code",
        "code_tool": "codeweaver_full_pipeline",
        "notes": {
            "council": "Route audio design objective to council deliberation.",
            "code": "Route audio implementation to code scan pipeline.",
            "buddy": "Route audio cue to mirror-collaboration.",
        },
    },
    "gamedev": {
        "default_target": "code",
        "code_tool": "codeweaver_full_pipeline",
        "notes": {
            "council": "Route gamedev objective to council deliberation.",
            "code": "Route game code to code scan pipeline.",
            "buddy": "Route game design to mirror-collaboration.",
        },
    },
    "governor": {
        "default_target": "council",
        "code_tool": "codeweaver_full_pipeline",
        "notes": {
            "council": "Route governor resource decision to council deliberation.",
            "code": "Route governor infrastructure setup to code scan pipeline.",
            "buddy": "Route governor tuning to mirror-collaboration.",
        },
    },
    "media": {
        "default_target": "code",
        "code_tool": "codeweaver_full_pipeline",
        "notes": {
            "council": "Route media design objective to council deliberation.",
            "code": "Route media asset implementation to code scan pipeline.",
            "buddy": "Route media creative brief to mirror-collaboration.",
        },
    },
    "model_lab": {
        "default_target": "council",
        "code_tool": "codeweaver_full_pipeline",
        "notes": {
            "council": "Route model lab diagnostics to council deliberation.",
            "code": "Route model lab findings to code scan pipeline.",
            "buddy": "Route model lab results to mirror-collaboration.",
        },
    },
    "rag": {
        "default_target": "council",
        "code_tool": "codeweaver_full_pipeline",
        "notes": {
            "council": "Route RAG findings to council deliberation.",
            "code": "Route RAG code docs to code scan pipeline.",
            "buddy": "Route RAG answer to mirror-collaboration.",
        },
    },
    "re": {
        "default_target": "council",
        # re_route is the one structural variant in this family: its "code"
        # branch suggests the security scanner, not the full pipeline.
        "code_tool": "codeweaver_security_scan",
        "notes": {
            "council": "Route RE findings to council deliberation.",
            "code": "Route RE findings to code scan pipeline.",
            "buddy": "Route RE findings to mirror-collaboration.",
        },
    },
    "team": {
        "default_target": "council",
        "code_tool": "codeweaver_full_pipeline",
        "notes": {
            "council": "Route crew objective to council deliberation.",
            "code": "Route code artifacts to code scan pipeline.",
            "buddy": "Route proposal to mirror-collaboration.",
        },
    },
    "web": {
        "default_target": "council",
        "code_tool": "codeweaver_full_pipeline",
        "notes": {
            "council": "Route web project objective to council deliberation.",
            "code": "Route web code to code scan pipeline.",
            "buddy": "Route web design to mirror-collaboration.",
        },
    },
}

# ---------------------------------------------------------------------------
# Legacy name tables: exposed name -> namespace
# ---------------------------------------------------------------------------
LEGACY_SESSION_REPLAY: Dict[str, str] = {
    "audio_session_replay": "audio",
    "codeweaver_session_replay": "codeweaver",
    "council_session_replay": "council",
    "gamedev_session_replay": "gamedev",
    "governor_session_replay": "governor",
    "media_session_replay": "media",
    "model_lab_session_replay": "model_lab",
    "ops_session_replay": "ops",
    "orchestrator_session_replay": "orchestrator",
    "orochi_session_replay": "orochi",
    "prism_session_replay": "prism",
    "rag_session_replay": "rag",
    "re_session_replay": "re",
    "samurai_session_replay": "samurai",
    "spider_session_replay": "spider",
    "team_session_replay": "team",
    "web_session_replay": "web",
}

LEGACY_SESSION_LIST: Dict[str, str] = {
    name[: -len("_session_replay")] + "_session_list": ns
    for name, ns in LEGACY_SESSION_REPLAY.items()
}

LEGACY_MCP_ROUTE: Dict[str, str] = {
    "audio_route": "audio",
    "gamedev_route": "gamedev",
    "governor_route": "governor",
    "media_route": "media",
    "model_lab_route": "model_lab",
    "rag_route": "rag",
    "re_route": "re",
    "team_route": "team",
    "web_route": "web",
}

# ---------------------------------------------------------------------------
# PRESERVED names - never merged, never string-fixed.
# ---------------------------------------------------------------------------
# The three pure aliases in quillan_thinking_engine. The RIGHT-HAND SIDE
# strings are the typo'd exposed names. DO NOT CORRECT THEM: call sites bind
# to them verbatim (audit #886 / brief #896).
LEGACY_ALIASES: Dict[str, str] = {
    "deep_reasoning_alias": "Deep_reasoning",
    "genius_level_alias": "Genuis_level",  # typo preserved on purpose
    "adversarial_dual_council_deliberation_alias": (
        "adversarial_dual_council_diliberation"  # typo preserved on purpose
    ),
}

# The six @mcp.tool(name=...) overrides in quillan_thinking_engine
# (source lines 300/374/384/449/459/523 at inspection time).
PRESERVED_NAME_OVERRIDES: FrozenSet[str] = frozenset(
    {
        "Deep_reasoning",
        "deep_reasoning",
        "Genuis_level",
        "genius_level",
        "adversarial_dual_council_diliberation",
        "adversarial_dual_council_deliberation",
    }
)

# Same exposed name, DIFFERENT signature -> keep namespaced keys, never a
# bare ``prism_decompose`` shim.
PRESERVED_DISTINCT: Dict[str, str] = {
    "prism_decompose@quillan_prism": "quillan_prism",
    "prism_decompose@quillan_thinking_engine": "quillan_thinking_engine",
}


@dataclass(frozen=True)
class LegacyTool:
    """One legacy exposed name and how it resolves under the shim layer."""

    name: str
    category: str  # session_replay | session_list | mcp_route | alias | name_override | distinct
    canonical: Optional[str]  # None means preserved, not merged
    namespace: Optional[str]  # frozen namespace argument for the canonical tool
    target: Optional[str] = None  # lazy-import target server for preserved names
    note: str = ""


LEGACY_TOOLS: Dict[str, LegacyTool] = {}

for _name, _ns in LEGACY_SESSION_REPLAY.items():
    LEGACY_TOOLS[_name] = LegacyTool(
        name=_name,
        category="session_replay",
        canonical=CANONICAL_SESSION_REPLAY,
        namespace=_ns,
        note="byte-identical clone family (13/17 byte-identical per audit #886)",
    )
for _name, _ns in LEGACY_SESSION_LIST.items():
    LEGACY_TOOLS[_name] = LegacyTool(
        name=_name,
        category="session_list",
        canonical=CANONICAL_SESSION_LIST,
        namespace=_ns,
        note="session_list clone family (17 servers)",
    )
for _name, _ns in LEGACY_MCP_ROUTE.items():
    LEGACY_TOOLS[_name] = LegacyTool(
        name=_name,
        category="mcp_route",
        canonical=CANONICAL_MCP_ROUTE,
        namespace=_ns,
        note="byte-identical cross-MCP route template",
    )
for _name, _target in LEGACY_ALIASES.items():
    LEGACY_TOOLS[_name] = LegacyTool(
        name=_name,
        category="alias",
        canonical=None,
        namespace=None,
        target="quillan_thinking_engine",
        note=f"pure alias -> {_target} (string preserved verbatim)",
    )
for _name in PRESERVED_NAME_OVERRIDES:
    LEGACY_TOOLS[_name] = LegacyTool(
        name=_name,
        category="name_override",
        canonical=None,
        namespace=None,
        target="quillan_thinking_engine",
        note="@mcp.tool(name=...) override; call sites bind to this string",
    )
for _key, _server in PRESERVED_DISTINCT.items():
    LEGACY_TOOLS[_key] = LegacyTool(
        name=_key,
        category="distinct",
        canonical=None,
        namespace=_server,
        target=_server,
        note="prism_decompose signature conflict - namespaced, NEVER merged",
    )


MERGED_NAMES: FrozenSet[str] = frozenset(
    set(LEGACY_SESSION_REPLAY) | set(LEGACY_SESSION_LIST) | set(LEGACY_MCP_ROUTE)
)


def category_names(category: str) -> Tuple[str, ...]:
    """Return sorted legacy names belonging to ``category``."""
    return tuple(
        sorted(name for name, spec in LEGACY_TOOLS.items() if spec.category == category)
    )


def iter_legacy() -> Iterable[LegacyTool]:
    """Yield every registered legacy tool spec in deterministic name order."""
    for name in sorted(LEGACY_TOOLS):
        yield LEGACY_TOOLS[name]
