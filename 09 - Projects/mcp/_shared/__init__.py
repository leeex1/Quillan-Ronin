"""
Quillan-Ronin Shared MCP Utilities
==================================
Common utilities for all Quillan MCP servers.
Provides: stdio guard, error handling, input validation, standard response format.
"""

import sys
import os
import logging
import builtins
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from functools import wraps

# ─── Stdio Guard ─────────────────────────────────────────────────────────────
# Prevents stdout corruption in JSON-RPC stdio transport

_ORIG_PRINT = builtins.print

def _safe_print(*args, **kwargs):
    """Redirect all prints to stderr to prevent JSON-RPC corruption."""
    if "file" not in kwargs or kwargs["file"] is None or kwargs["file"] is sys.stdout:
        kwargs["file"] = sys.stderr
    return _ORIG_PRINT(*args, **kwargs)

def install_stdio_guard():
    """Install the stdio guard globally. Call at module load time."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                       format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    builtins.print = _safe_print
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# ─── Cross-Platform Path Injection ───────────────────────────────────────────

_BASE_PATHS = [
    r"C:\02_QUILLAN\09 - Projects\mcp",
    r"C:\02_QUILLAN\09 - Projects\projects\oni",
    r"C:\02_QUILLAN\03 - Training & Model\scripts",
    r"C:\02_QUILLAN",
]

def inject_paths():
    """Inject standard Quillan paths into sys.path."""
    for _bp in _BASE_PATHS:
        if os.name != "nt":
            if _bp.startswith("C:") or _bp.startswith("c:"):
                _bp = "/mnt/c/" + _bp[3:].replace("\\", "/")
        if os.path.exists(_bp) and _bp not in sys.path:
            sys.path.insert(0, _bp)

# ─── Error Handling Decorator ─────────────────────────────────────────────────

def tool_guard(func: Callable) -> Callable:
    """Decorator that wraps MCP tool functions with error handling and validation."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, dict) and "error" not in result:
                result.setdefault("_meta", {})
                result["_meta"]["status"] = "OK"
                result["_meta"]["timestamp"] = time.time()
            return result
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc() if os.environ.get("QUILLAN_DEBUG") else None,
                "_meta": {"status": "ERROR", "timestamp": time.time()},
            }
    return wrapper

# ─── Input Validation ────────────────────────────────────────────────────────

def validate_string(value: Any, name: str = "input", min_len: int = 1, max_len: int = 100000) -> str:
    """Validate and sanitize string input."""
    if value is None:
        raise ValueError(f"{name} is required")
    s = str(value).strip()
    if len(s) < min_len:
        raise ValueError(f"{name} must be at least {min_len} characters")
    if len(s) > max_len:
        raise ValueError(f"{name} exceeds maximum length of {max_len}")
    return s

def validate_path(value: Any, name: str = "path") -> str:
    """Validate and sanitize file path input."""
    s = validate_string(value, name)
    # Remove quotes and whitespace
    s = s.strip().strip('"').strip("'").strip()
    return s

def validate_int(value: Any, name: str = "value", min_val: Optional[int] = None,
                 max_val: Optional[int] = None) -> int:
    """Validate integer input."""
    try:
        v = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be an integer")
    if min_val is not None and v < min_val:
        raise ValueError(f"{name} must be >= {min_val}")
    if max_val is not None and v > max_val:
        raise ValueError(f"{name} must be <= {max_val}")
    return v

# ─── Standard Response Format ────────────────────────────────────────────────

def ok_response(data: Any, message: str = "") -> Dict[str, Any]:
    """Create a standard success response."""
    return {"status": "OK", "data": data, "message": message, "timestamp": time.time()}

def error_response(message: str, error_type: str = "Error") -> Dict[str, Any]:
    """Create a standard error response."""
    return {"status": "ERROR", "error": message, "error_type": error_type, "timestamp": time.time()}

# ─── File Reading Helper ─────────────────────────────────────────────────────

def read_file_safe(path: str, max_size: int = 10_000_000) -> str:
    """Safely read a file with size limit and encoding fallback."""
    p = Path(path.strip().strip('"').strip("'"))
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Not a file: {path}")
    if p.stat().st_size > max_size:
        raise ValueError(f"File too large: {p.stat().st_size} bytes (max {max_size})")
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8", errors="replace")

# ─── Auto-Setup ───────────────────────────────────────────────────────────────

def setup(server_name: str = "QuillanMCP"):
    """One-call setup for any MCP server. Returns configured FastMCP instance."""
    install_stdio_guard()
    inject_paths()
    from fastmcp import FastMCP
    mcp = FastMCP(server_name)
    return mcp

__all__ = [
    "install_stdio_guard", "inject_paths", "tool_guard",
    "validate_string", "validate_path", "validate_int",
    "ok_response", "error_response", "read_file_safe", "setup",
]
