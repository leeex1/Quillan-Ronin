"""Namespaced deprecation shims for every PASS 2 legacy tool name.

Each of the 43 collapsed names resolves to a thin wrapper around one of the
three canonical tools with its namespace FROZEN in, so legacy call sites keep
their exact per-server behavior during the 90-day / one-minor-release window.

Preserved names (aliases, name overrides, prism_decompose) resolve to a
``PreservedTool`` that lazy-loads the original server attribute. Their strings
are never rewritten - in particular "Genuis_level" and
"adversarial_dual_council_diliberation" stay typo'd on purpose.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, Callable, Dict, Optional

from .canonical import mcp_route, mcp_session_list, mcp_session_replay
from .namespace_registry import (
    CANONICAL_MCP_ROUTE,
    CANONICAL_SESSION_LIST,
    CANONICAL_SESSION_REPLAY,
    LEGACY_ALIASES,
    LEGACY_TOOLS,
    ROUTE_PROFILES,
    LegacyTool,
)

_DEPRECATION_MSG = (
    "Legacy MCP tool '{name}' is deprecated since {since}; use '{canonical}' "
    "with namespace={namespace!r}. Removal no earlier than {removal} "
    "(90 days / one minor release)."
)
_SINCE = "5.4.0-ONI"
_REMOVAL = "5.5.0-ONI"

_CANONICAL_FNS: Dict[str, Callable[..., Dict[str, Any]]] = {
    CANONICAL_SESSION_REPLAY: mcp_session_replay,
    CANONICAL_SESSION_LIST: mcp_session_list,
    CANONICAL_MCP_ROUTE: mcp_route,
}


class PreservedTool:
    """Callable placeholder that lazy-loads a preserved legacy tool.

    Used ONLY for names that must not be merged or string-corrected. Resolution
    is local and deterministic; the underlying server is imported on first
    invocation, so importing this module never imports any server.
    """

    __slots__ = ("name", "server", "target", "note")

    def __init__(
        self, name: str, server: str, target: Optional[str] = None, note: str = ""
    ) -> None:
        self.name = name
        self.server = server
        self.target = target or name
        self.note = note

    def _load(self) -> Callable[..., Any]:
        try:
            module = importlib.import_module(f"{self.server}.server")
        except Exception as exc:  # re-raise with actionable context, never swallow
            raise RuntimeError(
                f"Preserved tool '{self.name}' needs '{self.server}.server' "
                f"on PYTHONPATH (import failed: {exc})"
            ) from exc
        fn = getattr(module, self.target, None)
        if fn is None or not callable(fn):
            raise RuntimeError(
                f"Preserved tool '{self.name}' could not find callable "
                f"'{self.target}' in '{self.server}.server'"
            )
        return fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"Preserved legacy tool '{self.name}' ({self.note}). "
            f"Call sites bind to this exact string; do not rename.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._load()(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"PreservedTool(name={self.name!r}, server={self.server!r}, target={self.target!r})"


def _warn_wrapper(name: str, spec: LegacyTool) -> None:
    warnings.warn(
        _DEPRECATION_MSG.format(
            name=name,
            since=_SINCE,
            removal=_REMOVAL,
            canonical=spec.canonical,
            namespace=spec.namespace,
        ),
        DeprecationWarning,
        stacklevel=3,
    )


def _make_session_replay_shim(name: str, spec: LegacyTool) -> Callable[..., Any]:
    fn = _CANONICAL_FNS[spec.canonical]  # type: ignore[index]
    ns = spec.namespace

    def shim(session_id: str) -> Dict[str, Any]:
        _warn_wrapper(name, spec)
        return fn(session_id, namespace=ns)

    shim.__name__ = name
    shim.__qualname__ = name
    shim.__doc__ = f"DEPRECATED shim -> {spec.canonical}(namespace={ns!r})."
    return shim


def _make_session_list_shim(name: str, spec: LegacyTool) -> Callable[..., Any]:
    fn = _CANONICAL_FNS[spec.canonical]  # type: ignore[index]
    ns = spec.namespace

    def shim() -> Dict[str, Any]:
        _warn_wrapper(name, spec)
        return fn(namespace=ns)

    shim.__name__ = name
    shim.__qualname__ = name
    shim.__doc__ = f"DEPRECATED shim -> {spec.canonical}(namespace={ns!r})."
    return shim


def _make_route_shim(name: str, spec: LegacyTool) -> Callable[..., Any]:
    fn = _CANONICAL_FNS[spec.canonical]  # type: ignore[index]
    ns = spec.namespace
    default_target = ROUTE_PROFILES[ns]["default_target"]  # type: ignore[index]

    def shim(
        target_mcp: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _warn_wrapper(name, spec)
        return fn(
            default_target if target_mcp is None else target_mcp,
            payload,
            source=ns,
        )

    shim.__name__ = name
    shim.__qualname__ = name
    shim.__doc__ = f"DEPRECATED shim -> {spec.canonical}(source={ns!r})."
    return shim


def _build_shims() -> Dict[str, Callable[..., Any]]:
    built: Dict[str, Callable[..., Any]] = {}
    for name in sorted(LEGACY_TOOLS):
        spec = LEGACY_TOOLS[name]
        if spec.category == "session_replay":
            built[name] = _make_session_replay_shim(name, spec)
        elif spec.category == "session_list":
            built[name] = _make_session_list_shim(name, spec)
        elif spec.category == "mcp_route":
            built[name] = _make_route_shim(name, spec)
        elif spec.category == "alias":
            built[name] = PreservedTool(
                name=name,
                server=spec.target or "quillan_thinking_engine",
                target=LEGACY_ALIASES[name],
                note=spec.note,
            )
        elif spec.category == "name_override":
            built[name] = PreservedTool(
                name=name,
                server=spec.target or "quillan_thinking_engine",
                target=name,
                note=spec.note,
            )
        elif spec.category == "distinct":
            # spec.name is "prism_decompose@<server>"; the original attribute
            # name is the part before '@'. Never expose a bare prism_decompose.
            attr = name.split("@", 1)[0]
            built[name] = PreservedTool(
                name=name,
                server=spec.namespace or spec.target or "quillan_prism",
                target=attr,
                note=spec.note,
            )
        else:  # pragma: no cover - guarded by registry construction
            raise ValueError(f"Unhandled legacy category for '{name}': {spec.category}")
    return built


SHIMS: Dict[str, Callable[..., Any]] = _build_shims()


def resolve(name: str) -> Callable[..., Any]:
    """Resolve a legacy tool name to a callable shim.

    Raises:
        KeyError: if ``name`` is not a registered legacy tool.
    """
    return SHIMS[name]


def is_legacy_name(name: str) -> bool:
    """Return True when ``name`` has a registered legacy shim."""
    return name in SHIMS


__all__ = ["SHIMS", "PreservedTool", "resolve", "is_legacy_name"]
