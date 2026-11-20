#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-process verification of the 3 consolidated Quillan MCP facades.

For every facade this script:
  1. imports the facade module in-process (no MCP client, no config switch),
  2. enumerates EVERY tool registered on its FastMCP instance,
  3. asserts each child's expected tool count is present under its native
     ``<child_prefix>_<original_tool>`` names (FastMCP prefix separator = "_"),
  4. asserts the facade health tool exists,
  5. asserts total tool count == sum(children) + 1,
  6. asserts no duplicate names and no unprefixed tool leakage.

If a child cannot be imported the script names that child, the exact exception
type/message recorded by the facade, and FAILS. It never reports a false PASS.

Exit code 0 only when all three facades fully pass; 1 otherwise.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import traceback
from pathlib import Path
from typing import Any

BASE = Path(os.environ.get("QUILLAN_BASE", r"C:\02_QUILLAN"))
MCP_ROOT = BASE / "09 - Projects" / "mcp"

# Facade env must match production before any child module is imported.
_REQUIRED_PP = [
    str(MCP_ROOT),
    str(BASE / "09 - Projects" / "projects" / "oni"),
    str(BASE / "03 - Training & Model" / "scripts"),
    str(BASE),
]
_existing = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
_merged: list[str] = []
for _p in _REQUIRED_PP + _existing:
    if _p and _p not in _merged:
        _merged.append(_p)
os.environ["PYTHONPATH"] = os.pathsep.join(_merged)
os.environ.setdefault("QUILLAN_BASE", str(BASE))
os.environ.setdefault(
    "QUILLAN_RAG_DB", str(BASE / "07 - Memory & LanceDB" / "quillan_rag_db")
)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _p in reversed(_merged):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FACADES: tuple[dict[str, Any], ...] = (
    {
        "group": "quillan_cortex",
        "health": "quillan_cortex_health",
        "child_total": 238,
        "total": 239,
        "children": {
            "quillan_thinking_engine": 61,
            "quillan_samurai_engine": 20,
            "quillan_orochi": 24,
            "quillan_prism": 20,
            "quillan_council": 15,
            "quillan_governor": 22,
            "quillan_orchestrator": 14,
            "quillan_buddy": 33,
            "quillan_team": 29,
        },
    },
    {
        "group": "quillan_archive",
        "health": "quillan_archive_health",
        "child_total": 168,
        "total": 169,
        "children": {
            "quillan_rag": 30,
            "quillan_model_lab": 36,
            "quillan_re": 39,
            "quillan_spider": 33,
            "quillan_ops": 30,
        },
    },
    {
        "group": "quillan_sensory",
        "health": "quillan_sensory_health",
        "child_total": 130,
        "total": 131,
        "children": {
            "quillan_audio": 29,
            "quillan_media": 25,
            "quillan_gamedev": 30,
            "quillan_web": 24,
            "quillan_code": 22,
        },
    },
)


def out(message: str = "") -> None:
    """Write straight to the real stdout (children monkeypatch builtins.print)."""
    sys.__stdout__.write(message + "\n")
    sys.__stdout__.flush()


def load_facade(group: str):
    path = MCP_ROOT / group / "server.py"
    if not path.is_file():
        raise FileNotFoundError(f"facade not found: {path}")
    mod_name = f"verify_facade_{group}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # facade bootstrap runs here
    return module


def verify_one(spec: dict[str, Any]) -> bool:
    group = spec["group"]
    out("=" * 78)
    out(f"FACADE: {group}")
    out("=" * 78)

    try:
        module = load_facade(group)
    except BaseException as exc:  # noqa: BLE001 - report facade-level failure verbatim
        out(f"  [FAIL] facade import raised {type(exc).__name__}: {exc}")
        tb = traceback.format_exc().strip().splitlines()
        for line in tb[-6:]:
            out(f"         {line}")
        out(f"  RESULT: FAIL ({group} could not be instantiated)")
        return False

    mcp = getattr(module, "mcp", None)
    if mcp is None:
        out("  [FAIL] facade defines no module-level `mcp`")
        out(f"  RESULT: FAIL ({group})")
        return False

    try:
        tools = asyncio.run(mcp.list_tools())
    except BaseException as exc:  # noqa: BLE001
        out(f"  [FAIL] mcp.list_tools() raised {type(exc).__name__}: {exc}")
        out(f"  RESULT: FAIL ({group})")
        return False

    names = [t.name for t in tools]
    name_set = set(names)
    bootstrap = getattr(module, "_BOOTSTRAP", {}) or {}
    child_status = {c["name"]: c for c in bootstrap.get("children", [])}
    ok = True

    # 1. health tool
    if spec["health"] in name_set:
        out(f"  [PASS] health tool registered: {spec['health']}")
    else:
        out(f"  [FAIL] health tool MISSING: {spec['health']}")
        ok = False

    # 2. per-child counts under native prefix
    out("  child tool counts (prefix separator '_'):")
    total_child_found = 0
    for child, expected in spec["children"].items():
        prefixed = [n for n in names if n.startswith(child + "_")]
        found = len(prefixed)
        total_child_found += found
        status = child_status.get(child, {})
        state = status.get("status", "UNKNOWN")
        error = status.get("error")
        if found == expected and state == "OK":
            out(f"    [PASS] {child:<26} {found:>3}/{expected:<3} status={state}")
        else:
            ok = False
            out(f"    [FAIL] {child:<26} {found:>3}/{expected:<3} status={state}")
            if error:
                out(f"           child error: {error}")
            if not error and state == "UNKNOWN":
                out("           child error: no bootstrap status recorded")
            if prefixed:
                out(f"           sample: {prefixed[:3]}")

    # 3. total count
    expected_total = spec["total"]
    if len(names) == expected_total:
        out(
            f"  [PASS] total tools = {len(names)} (sum children {spec['child_total']} + 1 health)"
        )
    else:
        ok = False
        out(
            f"  [FAIL] total tools = {len(names)}, expected {expected_total} "
            f"(children {spec['child_total']} + 1 health)"
        )

    # 4. child subtotal consistency
    if total_child_found == spec["child_total"]:
        out(f"  [PASS] prefixed child tool subtotal = {total_child_found}")
    else:
        ok = False
        out(
            f"  [FAIL] prefixed child tool subtotal = {total_child_found}, "
            f"expected {spec['child_total']}"
        )

    # 5. duplicates
    if len(names) == len(name_set):
        out("  [PASS] no duplicate tool names")
    else:
        ok = False
        dupes = sorted({n for n in names if names.count(n) > 1})
        out(f"  [FAIL] duplicate tool names: {dupes[:10]}")

    # 6. unprefixed leakage
    prefixes = tuple(spec["children"].keys())
    leaked = [n for n in names if n != spec["health"] and not n.startswith(prefixes)]
    if not leaked:
        out("  [PASS] no unprefixed/legacy tool names leaked")
    else:
        ok = False
        out(f"  [FAIL] unprefixed tool names leaked: {leaked[:10]}")

    # 7. bootstrap self-report cross-check
    reported = bootstrap.get("total_tools")
    if reported == len(names):
        out(f"  [PASS] facade health self-report total_tools = {reported}")
    else:
        ok = False
        out(
            f"  [FAIL] facade health self-report total_tools = {reported}, "
            f"actual = {len(names)}"
        )

    out(f"  RESULT: {'PASS' if ok else 'FAIL'} ({group}: {len(names)} tools)")
    return ok


def main() -> int:
    out("Quillan MCP consolidation verification (in-process, no config switch)")
    out(f"fastmcp root: {MCP_ROOT}")
    results = [verify_one(spec) for spec in FACADES]
    expected_grand = sum(s["total"] for s in FACADES)
    out("=" * 78)
    out(
        "OVERALL: "
        + ("PASS" if all(results) else "FAIL")
        + f"  ({sum(results)}/{len(results)} facades, expected grand total {expected_grand} tools)"
    )
    out("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
