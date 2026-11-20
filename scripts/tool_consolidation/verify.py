"""Verifier for the PASS 2 tool-consolidation shim layer.

Confirms, empirically, that:
  1. all three canonical tools resolve and are callable;
  2. EVERY registered legacy tool name still resolves to a callable shim;
  3. category counts match audit #886 exactly (17 / 17 / 9 merged, 3 / 6 / 2
     preserved);
  4. dangerous strings are untouched (typo'd exposed names, two distinct
     prism_decompose signatures);
  5. the canonical tools actually execute against the live sessions/ layout
     and return deterministic, structured results.

Run:  python C:\\02_QUILLAN\\scripts\\tool_consolidation\\verify.py
Exit: 0 = all checks pass, 1 = at least one failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow both ``python verify.py`` and ``python -m tool_consolidation.verify``.
_PKG_PARENT = Path(__file__).resolve().parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from tool_consolidation import (  # noqa: E402
    CANONICAL_NAMES,
    DEPRECATION_MINOR_RELEASES,
    DEPRECATION_WINDOW_DAYS,
    LEGACY_TOOLS,
    MERGED_NAMES,
    SHIMS,
    is_legacy_name,
    resolve,
)
from tool_consolidation.canonical import (  # noqa: E402
    mcp_route,
    mcp_session_list,
    mcp_session_replay,
    sessions_root,
)
from tool_consolidation.namespace_registry import (  # noqa: E402
    LEGACY_ALIASES,
    LEGACY_MCP_ROUTE,
    LEGACY_SESSION_LIST,
    LEGACY_SESSION_REPLAY,
    PRESERVED_DISTINCT,
    PRESERVED_NAME_OVERRIDES,
    ROUTE_PROFILES,
    SESSION_NAMESPACES,
    category_names,
)

# Live registered tool count at the time of audit #886 / brief #896. The shim
# layer is NOT wired into the facades, so this number does not change here.
LIVE_REGISTERED_TOOLS = 539

results: List[Dict[str, Any]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    results.append({"check": label, "pass": bool(ok), "detail": detail})
    return bool(ok)


def run() -> int:
    # 1. canonical tools
    canonicals = {
        "mcp_session_replay": mcp_session_replay,
        "mcp_session_list": mcp_session_list,
        "mcp_route": mcp_route,
    }
    check(
        "canonical names resolve and are callable",
        all(callable(fn) for fn in canonicals.values())
        and set(canonicals) == set(CANONICAL_NAMES),
        f"canonicals={sorted(CANONICAL_NAMES)}",
    )

    # 2. every legacy name resolves
    unresolved = [
        name
        for name in sorted(LEGACY_TOOLS)
        if not is_legacy_name(name) or not callable(resolve(name))
    ]
    check(
        "every registered legacy tool name resolves to a callable",
        not unresolved,
        f"registered={len(LEGACY_TOOLS)} unresolved={unresolved}",
    )
    check("SHIMS keys match registry", set(SHIMS) == set(LEGACY_TOOLS))

    # 3. counts
    expected = {
        "session_replay": 17,
        "session_list": 17,
        "mcp_route": 9,
        "alias": 3,
        "name_override": 6,
        "distinct": 2,
    }
    actual = {cat: len(category_names(cat)) for cat in expected}
    check(
        "audit #886 category counts match",
        actual == expected,
        f"actual={actual}",
    )
    check(
        "43 high-confidence names collapse to 3 canonical interfaces",
        len(MERGED_NAMES) == 43
        and len(MERGED_NAMES)
        == len(LEGACY_SESSION_REPLAY)
        + len(LEGACY_SESSION_LIST)
        + len(LEGACY_MCP_ROUTE),
        f"merged={len(MERGED_NAMES)}",
    )

    # 4. dangerous strings untouched
    check(
        "typo'd alias targets preserved verbatim",
        LEGACY_ALIASES.get("genius_level_alias") == "Genuis_level"
        and LEGACY_ALIASES.get("adversarial_dual_council_deliberation_alias")
        == "adversarial_dual_council_diliberation",
        f"aliases={LEGACY_ALIASES}",
    )
    check(
        "typo'd name overrides still present",
        {"Genuis_level", "adversarial_dual_council_diliberation"}
        <= set(PRESERVED_NAME_OVERRIDES)
        and len(PRESERVED_NAME_OVERRIDES) == 6,
    )
    check(
        "bare prism_decompose is NOT exposed (namespaced only)",
        "prism_decompose" not in SHIMS
        and all(k.startswith("prism_decompose@") for k in PRESERVED_DISTINCT),
        f"distinct={sorted(PRESERVED_DISTINCT)}",
    )

    # orochi_route must NOT be part of the 9-template merge
    check(
        "orochi_route correctly excluded from route template family",
        "orochi_route" not in LEGACY_MCP_ROUTE and len(ROUTE_PROFILES) == 9,
    )

    # 5. behavioral smoke test (read-only, deterministic)
    check(
        "sessions root resolves on disk",
        sessions_root().is_dir(),
        f"root={sessions_root()}",
    )
    try:
        listing = mcp_session_list(namespace="audio")
        list_ok = (
            isinstance(listing, dict) and "count" in listing and "sessions" in listing
        )
    except Exception as exc:  # noqa: BLE001 - verifier reports, never hides
        list_ok = False
        listing = {"error": repr(exc)}
    check("mcp_session_list executes", list_ok, f"audio count={listing.get('count')}")

    missing = mcp_session_replay("__definitely_missing_session__", namespace="audio")
    check(
        "mcp_session_replay returns structured error for missing id",
        isinstance(missing, dict) and "error" in missing,
    )

    route_ok = True
    route_samples: Dict[str, str] = {}
    for source in ROUTE_PROFILES:
        out = mcp_route("council", {"probe": True}, source=source)
        route_samples[source] = out.get("note", "")
        route_ok = route_ok and out.get("target") == "QuillanCouncil"
    check(
        "mcp_route executes for all 9 sources",
        route_ok,
        f"sources={len(route_samples)}",
    )

    # shim preserves per-source default target and note
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            shim_out = resolve("audio_route")("code", {"x": 1})
        shim_ok = (
            shim_out.get("suggested_tool") == "codeweaver_full_pipeline"
            and shim_out.get("note")
            == "Route audio implementation to code scan pipeline."
        )
    except Exception as exc:  # noqa: BLE001
        shim_ok = False
        shim_out = {"error": repr(exc)}
    check(
        "namespaced shim preserves legacy route behavior", shim_ok, str(shim_out)[:160]
    )

    # deprecation policy
    check(
        "deprecation policy = 90 days / one minor release",
        DEPRECATION_WINDOW_DAYS == 90 and DEPRECATION_MINOR_RELEASES == 1,
    )

    passed = all(r["pass"] for r in results)
    report: Dict[str, Any] = {
        "verdict": "PASS" if passed else "FAIL",
        "checks_run": len(results),
        "checks_passed": sum(1 for r in results if r["pass"]),
        "legacy_names_registered": len(LEGACY_TOOLS),
        "legacy_names_resolved": len(LEGACY_TOOLS) - len(unresolved),
        "merged_legacy_names": len(MERGED_NAMES),
        "canonical_interfaces": len(CANONICAL_NAMES),
        "net_interface_reduction": len(MERGED_NAMES) - len(CANONICAL_NAMES),
        "preserved_names": len(LEGACY_TOOLS) - len(MERGED_NAMES),
        "live_registered_tools_before": LIVE_REGISTERED_TOOLS,
        "live_registered_tools_after": LIVE_REGISTERED_TOOLS,
        "live_count_changed": False,
        "honest_note": (
            "43 duplicate names collapse to 3 canonical interfaces at the shim "
            "layer (40-name reduction in the shim API surface). Live MCP tool "
            "count is UNCHANGED at 539: this deliverable adds NEW modules only "
            "and does not edit facades or configs. No 536->50 claim is made."
        ),
        "results": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
