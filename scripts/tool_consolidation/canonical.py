"""Canonical consolidated tools for PASS 2 (audit #886 / brief #896).

Three public callables replace 43 duplicate legacy tools:

  * ``mcp_session_replay(session_id, namespace=None)``  <- 17 x *_session_replay
  * ``mcp_session_list(namespace=None)``                <- 17 x *_session_list
  * ``mcp_route(target_mcp, payload=None, *, source)``  <- 9 x *_route templates

Design invariants:
  * Pure, deterministic, side-effect free (reads only; no writes).
  * No broad ``except Exception: pass``. Only expected I/O and JSON errors are
    caught and returned as structured error dicts.
  * Per-namespace behavior (session dir, route note text, defaults) is DATA in
    ``namespace_registry``, not duplicated control flow.
  * A ``namespace`` argument scopes playback/listing to one original server;
    ``None`` searches all consolidated namespaces in deterministic order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .namespace_registry import ROUTE_PROFILES, SESSION_NAMESPACES

# Original MCP root. Resolved relative to this file so the shim works from any
# CWD: scripts/tool_consolidation -> scripts -> 02_QUILLAN -> 09 - Projects/mcp
_MCP_ROOT: Path = Path(__file__).resolve().parents[2] / "09 - Projects" / "mcp"

# Domain summary keys used by the original per-server session_list tools.
# The canonical list is a superset: {session_id, timestamp, file, namespace,
# summary}. Keeping these keys preserves the useful part of each legacy shape.
_LIST_SUMMARY_KEYS: Dict[str, Tuple[str, ...]] = {
    "audio": ("cue",),
    "codeweaver": ("task", "objective", "overall"),
    "council": ("proposition", "overall"),
    "gamedev": ("project",),
    "governor": ("overall",),
    "media": ("asset", "brief"),
    "model_lab": ("checkpoint",),
    "ops": ("goal", "overall"),
    "orchestrator": ("objective",),
    "orochi": ("type",),
    "prism": ("task", "overall", "score"),
    "rag": ("overall",),
    "re": ("result", "path"),
    "samurai": ("query", "fusion"),
    "spider": ("label", "trail_events", "memory_nodes"),
    "team": ("objective",),
    "web": ("project", "scaffold"),
}

_MAX_SUMMARY_CHARS = 120


def sessions_root() -> Path:
    """Return the original MCP root directory holding every sessions/ dir."""
    return _MCP_ROOT


def _sessions_dir(namespace: str) -> Path:
    server = SESSION_NAMESPACES[namespace]
    return _MCP_ROOT / server / "sessions"


def _ordered_namespaces(namespace: Optional[str]) -> List[str]:
    if namespace is None:
        return list(SESSION_NAMESPACES)
    if namespace not in SESSION_NAMESPACES:
        raise ValueError(
            f"Unknown namespace '{namespace}'. "
            f"Valid namespaces: {sorted(SESSION_NAMESPACES)}"
        )
    return [namespace]


def _short(value: Any) -> Any:
    """Bound summary values so list responses stay small and deterministic."""
    if isinstance(value, str):
        return value[:_MAX_SUMMARY_CHARS]
    if isinstance(value, (list, tuple)):
        return list(value[:8])
    if isinstance(value, dict):
        return {str(k): _short(v) for k, v in list(value.items())[:8]}
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    # Legacy tools unwrap a {"result": {...}} envelope when present.
    if isinstance(raw, dict) and isinstance(raw.get("result"), dict):
        return raw["result"]
    return raw


def mcp_session_list(namespace: Optional[str] = None) -> Dict[str, Any]:
    """List persisted sessions across one namespace or all 17.

    Args:
        namespace: one of ``SESSION_NAMESPACES``; ``None`` aggregates all.

    Returns:
        ``{"count": int, "sessions": [...], "namespaces": [...]}``. Every
        entry carries ``session_id``, ``timestamp``, ``file``, ``namespace``
        and a bounded domain ``summary``.
    """
    namespaces = _ordered_namespaces(namespace)
    sessions: List[Dict[str, Any]] = []
    for ns in namespaces:
        sdir = _sessions_dir(ns)
        try:
            paths = sorted(
                sdir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            continue
        for path in paths:
            try:
                data = _read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            summary: Dict[str, Any] = {}
            for key in _LIST_SUMMARY_KEYS.get(ns, ()):
                if key in data and data[key] is not None:
                    summary[key] = _short(data[key])
            sessions.append(
                {
                    "session_id": data.get("session_id"),
                    "timestamp": data.get("timestamp"),
                    "file": str(path),
                    "namespace": ns,
                    "summary": summary,
                }
            )
    return {"count": len(sessions), "sessions": sessions, "namespaces": namespaces}


def mcp_session_replay(
    session_id: str, namespace: Optional[str] = None
) -> Dict[str, Any]:
    """Replay a persisted session by ID, optionally scoped to a namespace.

    Faithful to the legacy implementations: exact ``<id>.json`` first, then a
    ``<id>*.json`` prefix match (first hit), with structured error dicts.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        return {"error": "session_id must be a non-empty string"}
    namespaces = _ordered_namespaces(namespace)
    errors: List[str] = []
    for ns in namespaces:
        sdir = _sessions_dir(ns)
        path = sdir / f"{session_id}.json"
        if not path.exists():
            try:
                matches = sorted(sdir.glob(f"{session_id}*.json"))
            except OSError as exc:
                errors.append(f"{ns}: glob failed: {exc}")
                continue
            if not matches:
                continue
            path = matches[0]
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{ns}: read failed for {path.name}: {exc}")
            continue
        # Preserve the legacy return contract: the full JSON document.
        result: Dict[str, Any] = dict(payload)
        result.setdefault("namespace", ns)
        result.setdefault("file", str(path))
        return result
    if errors:
        return {"error": f"Session not found: {session_id}", "details": errors}
    return {"error": f"Session not found: {session_id}"}


def mcp_route(
    target_mcp: str = "council",
    payload: Optional[Dict[str, Any]] = None,
    *,
    source: str = "audio",
) -> Dict[str, Any]:
    """Cross-MCP routing payload - one implementation for 9 legacy templates.

    The per-source ``note`` text and default target are preserved verbatim from
    the original tools via ``ROUTE_PROFILES``.
    """
    profile = ROUTE_PROFILES.get(source)
    if profile is None:
        return {
            "error": (
                f"Unknown route source '{source}'. "
                f"Valid sources: {sorted(ROUTE_PROFILES)}"
            )
        }
    key = str(target_mcp).lower()
    notes: Dict[str, str] = profile["notes"]  # type: ignore[assignment]
    code_tool: str = profile["code_tool"]  # type: ignore[assignment]
    if key == "council":
        return {
            "target": "QuillanCouncil",
            "suggested_tool": "council_full_deliberation",
            "payload": payload or {},
            "note": notes["council"],
        }
    if key == "code":
        return {
            "target": "QuillanCode",
            "suggested_tool": code_tool,
            "payload": payload or {},
            "note": notes["code"],
        }
    if key == "buddy":
        return {
            "target": "QuillanBuddy",
            "suggested_tool": "buddy_mirror_collaborate",
            "payload": payload or {},
            "note": notes["buddy"],
        }
    return {"error": f"Unknown target '{target_mcp}'. Use council, code, or buddy."}


__all__ = [
    "mcp_session_list",
    "mcp_session_replay",
    "mcp_route",
    "sessions_root",
]
