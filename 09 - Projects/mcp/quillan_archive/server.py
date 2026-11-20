#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quillan-Archive consolidated FastMCP facade (5 children / 168 tools + health).

MECHANISM
    For every child this calls ``await mcp.import_server(child.mcp, prefix=<child
    directory name>)``. FastMCP 3.4.7's native prefix separator is a SINGLE
    UNDERSCORE (source: ``f"{prefix}_{tool.name}"`` in FastMCP.import_server),
    so a child tool ``foo`` becomes ``quillan_rag_foo``. No child tool is
    rewritten, renamed by hand, or dropped.

HONEST LIMITATIONS (verified against installed fastmcp 3.4.7 source)
    * import_server is EAGER: it calls ``await child.list_tools()`` immediately.
      Every child module and its heavy libraries (torch, transformers, chromadb,
      lancedb, PIL, ...) load when this facade starts. True lazy loading is NOT
      possible with this API; do not claim otherwise. RAM implication documented
      in the cutover plan.
    * import_server is deprecated in favour of mount(); it is used here because
      the consolidation contract mandates it. Its deprecation warning is
      suppressed locally at each call with warnings.catch_warnings() and is
      disclosed here rather than hidden.
    * import_server does not import a child's FastMCP lifespan. The 19 children
      use module-level state rather than lifespan hooks, so nothing is lost; the
      tool-count assertions in scripts/verify_mcp_consolidation.py prove it.
    * Bootstrap runs at module import via asyncio.run() (with a same-result
      thread fallback if a host harness already has a running loop), so tools are
      registered before mcp.run() opens the stdio transport.
"""

from __future__ import annotations

import asyncio
import builtins
import importlib.util
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# --- stdio guard: children redirect print() to stderr; match that here -------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_orig_print = builtins.print


def _quillan_safe_print(*args: Any, **kwargs: Any) -> Any:
    if "file" not in kwargs or kwargs["file"] is None or kwargs["file"] is sys.stdout:
        kwargs["file"] = sys.stderr
    return _orig_print(*args, **kwargs)


builtins.print = _quillan_safe_print

BASE = Path(os.environ.get("QUILLAN_BASE", r"C:\02_QUILLAN"))
MCP_ROOT = BASE / "09 - Projects" / "mcp"

# Special-env child in this group:
#   quillan_rag -> needs mcp root on PYTHONPATH + QUILLAN_RAG_DB
_REQUIRED_PYTHONPATH = [
    str(MCP_ROOT),
    str(BASE / "09 - Projects" / "projects" / "oni"),
    str(BASE / "03 - Training & Model" / "scripts"),
    str(BASE),
]
_existing_pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
_merged_pp: list[str] = []
for _p in _REQUIRED_PYTHONPATH + _existing_pp:
    if _p and _p not in _merged_pp:
        _merged_pp.append(_p)
os.environ["PYTHONPATH"] = os.pathsep.join(_merged_pp)
os.environ.setdefault("QUILLAN_BASE", str(BASE))
os.environ.setdefault(
    "QUILLAN_RAG_DB", str(BASE / "07 - Memory & LanceDB" / "quillan_rag_db")
)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _p in reversed(_merged_pp):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastmcp import FastMCP  # noqa: E402
from fastmcp.exceptions import FastMCPDeprecationWarning  # noqa: E402

# import_server is deprecated in 3.4.7; suppression is applied locally around
# each call below (a module-level filter would be displaced by FastMCP's own
# filter registration at import time). Disclosed, not hidden.

GROUP = "quillan_archive"
EXPECTED_CHILD_TOTAL = 168
EXPECTED_TOTAL = EXPECTED_CHILD_TOTAL + 1  # + this facade's health tool

CHILDREN: tuple[tuple[str, int], ...] = (
    ("quillan_rag", 30),
    ("quillan_model_lab", 36),
    ("quillan_re", 39),
    ("quillan_spider", 33),
    ("quillan_ops", 30),
)

mcp = FastMCP(
    "QuillanArchive",
    instructions="Consolidated retrieval / analysis / model-ops facade (5 servers, 168 tools).",
)

_BOOTSTRAP: dict[str, Any] = {"total_tools": 0, "children": [], "done": False}


@mcp.tool()
def quillan_archive_health() -> dict[str, Any]:
    """Health report for the Quillan-Archive consolidation facade."""
    children = _BOOTSTRAP["children"]
    ok = [c["name"] for c in children if c["status"] == "OK"]
    bad = [
        {
            "name": c["name"],
            "expected": c["expected"],
            "loaded": c["loaded"],
            "error": c["error"],
        }
        for c in children
        if c["status"] != "OK"
    ]
    return {
        "facade": GROUP,
        "ready": bool(ok) and not bad and _BOOTSTRAP["done"],
        "child_tool_total": sum(c["loaded"] for c in children),
        "expected_child_total": EXPECTED_CHILD_TOTAL,
        "total_tools": _BOOTSTRAP["total_tools"],
        "expected_total": EXPECTED_TOTAL,
        "loaded_children": ok,
        "failed_children": bad,
        "children": children,
    }


def _load_child_module(name: str):
    """Import one child server.py from an explicit path under a unique module name."""
    path = MCP_ROOT / name / "server.py"
    if not path.is_file():
        raise FileNotFoundError(f"missing child server: {path}")
    mod_name = f"{GROUP}_child_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    child = getattr(module, "mcp", None)
    if child is None:
        raise AttributeError(f"{name}/server.py defines no module-level `mcp`")
    return child


async def _bootstrap_children() -> None:
    for name, expected in CHILDREN:
        entry: dict[str, Any] = {
            "name": name,
            "prefix": name,
            "expected": expected,
            "loaded": 0,
            "status": "PENDING",
            "error": None,
        }
        _BOOTSTRAP["children"].append(entry)
        try:
            child = _load_child_module(name)
            child_tools = await child.list_tools()
            entry["loaded"] = len(child_tools)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FastMCPDeprecationWarning)
                await mcp.import_server(child, prefix=name)
            if len(child_tools) == expected:
                entry["status"] = "OK"
            else:
                entry["status"] = "COUNT_MISMATCH"
                entry["error"] = f"expected {expected} tools, found {len(child_tools)}"
            logging.info(
                "[%s] imported %s: %d tools (expected %d)",
                GROUP,
                name,
                len(child_tools),
                expected,
            )
        except Exception as exc:  # noqa: BLE001 - per-child diagnosis keeps the facade alive
            entry["status"] = "IMPORT_ERROR"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            logging.error("[%s] FAILED to import %s: %s", GROUP, name, entry["error"])
    _BOOTSTRAP["total_tools"] = len(await mcp.list_tools())
    _BOOTSTRAP["done"] = True
    logging.info(
        "[%s] bootstrap complete: %d tools (expected %d)",
        GROUP,
        _BOOTSTRAP["total_tools"],
        EXPECTED_TOTAL,
    )


def _run_bootstrap() -> None:
    try:
        asyncio.run(_bootstrap_children())
    except RuntimeError as exc:
        # Defensive: a harness may import this module from inside a live loop.
        if "running event loop" not in str(exc).lower():
            raise
        box: dict[str, BaseException] = {}

        def _worker() -> None:
            try:
                asyncio.run(_bootstrap_children())
            except BaseException as inner:  # noqa: BLE001
                box["error"] = inner

        import threading

        thread = threading.Thread(
            target=_worker, name=f"{GROUP}-bootstrap", daemon=True
        )
        thread.start()
        thread.join()
        if "error" in box:
            raise box["error"]


_run_bootstrap()

if __name__ == "__main__":
    mcp.run(transport="stdio")
