#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin v5.4.0-ONI — Shared MCP Kernel (_shared/__init__.py)
==================================================================
Upgrades all 19 Quillan MCP servers with:
1. Bulletproof Stdio Guard (prevents JSON-RPC stdout corruption)
2. Cross-platform Path Injection (C:\\02_QUILLAN & ONI engine paths)
3. Thread-safe LRU + TTL Cache for expensive file/DB/network calls
4. Token-Bucket Rate Limiter per tool to prevent runaway agent loops
5. Automatic Parameter Validation (validate_string, validate_path, validate_int)
6. Automatic MCP Docstring Enrichment for all 196 tools
7. Global QuillanErrorMiddleware + Standardized E_ICE Telemetry Envelopes
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

__version__ = "5.4.0-ONI"

# ============================================================================
# 1. SAFE PRINT STDIO GUARD (JSON-RPC Protection)
# ============================================================================


class _StdioRedirector(io.TextIOBase):
    """Redirects rogue print() calls to sys.stderr so sys.stdout stays pure JSON-RPC.
    Delegates all other attribute access to the original stdout to preserve
    .buffer, .reconfigure(), and other properties needed by MCP stdio transport."""

    def __init__(self, stderr_stream: Any, original_stdout: Any):
        self._stderr = stderr_stream
        self._original = original_stdout

    def write(self, s: str) -> int:
        if s:
            return self._stderr.write(s)
        return 0

    def flush(self) -> None:
        self._stderr.flush()

    @property
    def encoding(self) -> str:
        return getattr(self._stderr, "encoding", "utf-8")

    def reconfigure(self, **kwargs: Any) -> Any:
        """Delegate reconfigure to the original stdout."""
        return self._original.reconfigure(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate any missing attribute to the original stdout (e.g., .buffer)."""
        return getattr(self._original, name)


_ORIGINAL_STDOUT = sys.stdout
_STDIO_GUARD_INSTALLED = False


def install_stdio_guard() -> Any:
    """
    Redirects sys.stdout to sys.stderr for all user/library code while returning
    the original stdout descriptor for the MCP stdio transport server.
    """
    global _STDIO_GUARD_INSTALLED
    if not _STDIO_GUARD_INSTALLED:
        sys.stdout = _StdioRedirector(sys.stderr, _ORIGINAL_STDOUT)
        _STDIO_GUARD_INSTALLED = True
    return _ORIGINAL_STDOUT


# ============================================================================
# 2. CROSS-PLATFORM PATH INJECTION
# ============================================================================


def inject_paths(extra_paths: Optional[List[Union[str, Path]]] = None) -> Path:
    """Injects C:\\02_QUILLAN, mcp root, and ONI project paths into sys.path."""
    mcp_dir = Path(__file__).resolve().parent.parent
    quillan_root = (
        mcp_dir.parent.parent
        if mcp_dir.parent.name == "09 - Projects"
        else Path(os.environ.get("BASE_DIR", "C:/02_QUILLAN"))
    )

    candidate_paths = [
        mcp_dir,
        quillan_root,
        quillan_root / "09 - Projects" / "projects" / "oni",
        quillan_root / "07 - Memory & LanceDB",
    ]
    if extra_paths:
        candidate_paths.extend(Path(p) for p in extra_paths)

    for p in candidate_paths:
        p_str = str(p)
        if p.exists() and p_str not in sys.path:
            sys.path.insert(0, p_str)

    return quillan_root


# ============================================================================
# 3. INPUT VALIDATION PRIMITIVES (C13-WARDEN Security Gate)
# ============================================================================

_FORBIDDEN_PATH_PATTERNS = ("..", "~", "$", "|", ";", "`")


def validate_string(
    value: Any,
    name: str = "value",
    min_len: int = 0,
    max_len: int = 100_000,
    allow_empty: bool = True,
) -> str:
    """Validates and sanitizes string parameters."""
    if value is None:
        if allow_empty and min_len == 0:
            return ""
        raise ValueError(f"[WARDEN] Parameter '{name}' must not be None.")
    if not isinstance(value, str):
        value = str(value)
    if not allow_empty and not value.strip():
        raise ValueError(f"[WARDEN] Parameter '{name}' cannot be empty.")
    if len(value) < min_len:
        raise ValueError(
            f"[WARDEN] Parameter '{name}' length ({len(value)}) is below minimum ({min_len})."
        )
    if len(value) > max_len:
        raise ValueError(
            f"[WARDEN] Parameter '{name}' length ({len(value)}) exceeds maximum ({max_len})."
        )
    # Strip null bytes
    return value.replace("\x00", "")


def validate_int(
    value: Any,
    name: str = "value",
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    default: Optional[int] = None,
) -> int:
    """Validates and bounds integer parameters."""
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"[WARDEN] Parameter '{name}' expected int, got bool.")
    try:
        ival = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"[WARDEN] Parameter '{name}' must be an integer, got {type(value).__name__}."
        ) from exc

    if min_val is not None and ival < min_val:
        raise ValueError(
            f"[WARDEN] Parameter '{name}' ({ival}) is below minimum ({min_val})."
        )
    if max_val is not None and ival > max_val:
        raise ValueError(
            f"[WARDEN] Parameter '{name}' ({ival}) exceeds maximum ({max_val})."
        )
    return ival


def validate_path(
    path_val: Union[str, Path],
    name: str = "path",
    must_exist: bool = False,
    allowed_roots: Optional[List[Union[str, Path]]] = None,
) -> Path:
    """Validates filesystem paths and prevents CWE-22 path traversal attacks."""
    raw = validate_string(
        path_val, name=name, min_len=1, max_len=4096, allow_empty=False
    )
    if "\x00" in raw:
        raise ValueError(f"[WARDEN] Null byte detected in path '{name}'.")

    resolved = Path(raw).expanduser().resolve()

    if allowed_roots:
        roots = [Path(r).expanduser().resolve() for r in allowed_roots]
        if not any(resolved == r or r in resolved.parents for r in roots):
            raise PermissionError(
                f"[WARDEN] Path '{resolved}' is outside permitted roots: {[str(r) for r in roots]}"
            )

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"[WARDEN] Path '{resolved}' does not exist.")

    return resolved


def read_file_safe(
    path_val: Union[str, Path],
    max_bytes: int = 5 * 1024 * 1024,
    encoding: str = "utf-8",
    allowed_roots: Optional[List[Union[str, Path]]] = None,
) -> str:
    """Safely reads a file with size limits and C13-WARDEN path validation."""
    p = validate_path(
        path_val, name="file_path", must_exist=True, allowed_roots=allowed_roots
    )
    if not p.is_file():
        raise IsADirectoryError(f"[WARDEN] Path '{p}' is not a regular file.")
    size = p.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"[WARDEN] File '{p.name}' ({size:,} bytes) exceeds safe limit ({max_bytes:,} bytes)."
        )
    return p.read_text(encoding=encoding, errors="replace")


# ============================================================================
# 4. C14-KAIDO LRU + TTL CACHE & TOKEN-BUCKET RATE LIMITER
# ============================================================================


class TTLCache:
    """Thread-safe LRU + TTL Cache for expensive MCP operations (file I/O, RAG, network)."""

    def __init__(self, maxsize: int = 512, default_ttl: float = 60.0):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self._store: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(
        self, func_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> str:
        payload = json.dumps(
            {"f": func_name, "a": args, "k": kwargs}, sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Tuple[bool, Any]:
        now = time.monotonic()
        with self._lock:
            if key in self._store:
                expires_at, val = self._store[key]
                if now < expires_at:
                    self._store.move_to_end(key)
                    self.hits += 1
                    return True, val
                del self._store[key]
            self.misses += 1
            return False, None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (expires_at, value)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._store),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
            }


class TokenBucketRateLimiter:
    """Thread-safe Token-Bucket Rate Limiter to protect resource-intensive MCP tools."""

    def __init__(self, rate_per_sec: float = 25.0, capacity: float = 50.0):
        self.rate = rate_per_sec
        self.capacity = capacity
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> Tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            tokens, last_time = self._buckets.get(key, (self.capacity, now))
            elapsed = max(0.0, now - last_time)
            tokens = min(self.capacity, tokens + elapsed * self.rate)
            if tokens >= cost:
                tokens -= cost
                self._buckets[key] = (tokens, now)
                return True, 0.0
            retry_after = (cost - tokens) / max(self.rate, 1e-6)
            self._buckets[key] = (tokens, now)
            return False, round(retry_after, 3)


GLOBAL_CACHE = TTLCache(maxsize=1024, default_ttl=60.0)
GLOBAL_RATE_LIMITER = TokenBucketRateLimiter(rate_per_sec=30.0, capacity=60.0)


# ============================================================================
# 5. STANDARDIZED RESPONSE FORMATTERS & ERROR MIDDLEWARE
# ============================================================================


def ok_response(
    data: Any = None,
    server: str = "QuillanMCP",
    tool: str = "unknown",
    latency_ms: Optional[float] = None,
    cached: bool = False,
    **extra: Any,
) -> Dict[str, Any]:
    """Standard v5.4.0-ONI success response envelope."""
    res: Dict[str, Any] = {
        "ok": True,
        "status": "success",
        "version": __version__,
        "server": server,
        "tool": tool,
        "cached": cached,
        "data": data,
    }
    if latency_ms is not None:
        res["latency_ms"] = round(latency_ms, 3)
    if extra:
        res.update(extra)
    return res


def error_response(
    error: Union[str, Exception],
    server: str = "QuillanMCP",
    tool: str = "unknown",
    code: str = "EXECUTION_ERROR",
    **extra: Any,
) -> Dict[str, Any]:
    """Standard v5.4.0-ONI structured error response envelope."""
    msg = str(error)
    err_type = type(error).__name__ if isinstance(error, Exception) else "Error"
    res: Dict[str, Any] = {
        "ok": False,
        "status": "error",
        "version": __version__,
        "server": server,
        "tool": tool,
        "error_code": code,
        "error_type": err_type,
        "error": msg,
    }
    if extra:
        res.update(extra)
    return res


def _auto_validate_bound_args(
    sig: inspect.Signature, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    """Automatically applies C13-WARDEN validation based on parameter names and type annotations."""
    try:
        bound = sig.bind_partial(*args, **kwargs)
    except TypeError:
        return

    for param_name, val in bound.arguments.items():
        if val is None or param_name in ("self", "cls", "ctx"):
            continue
        param = sig.parameters[param_name]
        ann = param.annotation
        pname_lower = param_name.lower()

        if "path" in pname_lower or "file" in pname_lower or "dir" in pname_lower:
            if isinstance(val, (str, Path)) and str(val).strip():
                validate_path(val, name=param_name, must_exist=False)
        elif (
            ann is int
            or ann == "int"
            or pname_lower.endswith(
                ("_count", "_limit", "top_k", "max_tokens", "steps")
            )
        ):
            if not isinstance(val, bool) and isinstance(val, (int, str)):
                validate_int(val, name=param_name, min_val=0, max_val=10_000_000)
        elif ann is str or ann == "str" or isinstance(val, str):
            validate_string(val, name=param_name, max_len=500_000)


def enrich_docstring(func: Callable) -> Callable:
    """Ensures every MCP tool has a rich, LLM-friendly docstring with parameter metadata."""
    sig = inspect.signature(func)
    existing = (func.__doc__ or "").strip()
    param_lines = []
    for name, p in sig.parameters.items():
        if name in ("self", "cls", "ctx"):
            continue
        ann_str = (
            p.annotation.__name__
            if hasattr(p.annotation, "__name__")
            else str(p.annotation)
        )
        if p.annotation is inspect.Parameter.empty:
            ann_str = "Any"
        default_str = (
            f" (default={p.default!r})"
            if p.default is not inspect.Parameter.empty
            else " (required)"
        )
        param_lines.append(f"  - {name} ({ann_str}){default_str}")

    if not existing:
        existing = f"Executes Quillan-Ronin v5.4.0-ONI tool '{func.__name__}'."

    if "Parameters:" not in existing and param_lines:
        func.__doc__ = f"{existing}\n\nParameters:\n" + "\n".join(param_lines)
    else:
        func.__doc__ = existing
    return func


def tool_guard(
    _func: Optional[Callable] = None,
    *,
    server_name: str = "QuillanMCP",
    cache_ttl: Optional[float] = None,
    rate_limit: Optional[Tuple[float, float]] = None,
    auto_validate: bool = True,
) -> Callable:
    """
    Universal decorator for all 196 Quillan MCP tools.
    Provides:
      - Automatic parameter validation (validate_path, validate_int, validate_string)
      - Optional LRU+TTL caching (cache_ttl in seconds)
      - Optional Token-Bucket rate limiting (rate_per_sec, capacity)
      - Automatic docstring enrichment for MCP schema discovery
      - Structured exception catching (never crashes the MCP server)
    """
    limiter = TokenBucketRateLimiter(*rate_limit) if rate_limit else GLOBAL_RATE_LIMITER

    def decorator(func: Callable) -> Callable:
        enrich_docstring(func)
        sig = inspect.signature(func)
        tool_name = func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                allowed, retry_after = limiter.allow(f"{server_name}:{tool_name}")
                if not allowed:
                    return error_response(
                        f"Rate limit exceeded for {tool_name}. Retry after {retry_after}s.",
                        server=server_name,
                        tool=tool_name,
                        code="RATE_LIMITED",
                        retry_after_sec=retry_after,
                    )
                try:
                    if auto_validate:
                        _auto_validate_bound_args(sig, args, kwargs)
                    if cache_ttl is not None and cache_ttl > 0:
                        ckey = GLOBAL_CACHE._make_key(
                            f"{server_name}:{tool_name}", args, kwargs
                        )
                        hit, cval = GLOBAL_CACHE.get(ckey)
                        if hit:
                            return cval
                    result = await func(*args, **kwargs)
                    if cache_ttl is not None and cache_ttl > 0:
                        GLOBAL_CACHE.set(ckey, result, ttl=cache_ttl)
                    return result
                except Exception as exc:
                    logging.getLogger(server_name).error(
                        f"[{server_name}:{tool_name}] {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    )
                    return error_response(
                        exc,
                        server=server_name,
                        tool=tool_name,
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                    )

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                allowed, retry_after = limiter.allow(f"{server_name}:{tool_name}")
                if not allowed:
                    return error_response(
                        f"Rate limit exceeded for {tool_name}. Retry after {retry_after}s.",
                        server=server_name,
                        tool=tool_name,
                        code="RATE_LIMITED",
                        retry_after_sec=retry_after,
                    )
                try:
                    if auto_validate:
                        _auto_validate_bound_args(sig, args, kwargs)
                    if cache_ttl is not None and cache_ttl > 0:
                        ckey = GLOBAL_CACHE._make_key(
                            f"{server_name}:{tool_name}", args, kwargs
                        )
                        hit, cval = GLOBAL_CACHE.get(ckey)
                        if hit:
                            return cval
                    result = func(*args, **kwargs)
                    if cache_ttl is not None and cache_ttl > 0:
                        GLOBAL_CACHE.set(ckey, result, ttl=cache_ttl)
                    return result
                except Exception as exc:
                    logging.getLogger(server_name).error(
                        f"[{server_name}:{tool_name}] {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    )
                    return error_response(
                        exc,
                        server=server_name,
                        tool=tool_name,
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                    )

            return sync_wrapper

    if _func is not None:
        return decorator(_func)
    return decorator


class QuillanErrorMiddleware:
    """
    Wraps a FastMCP / MCP server instance so every registered tool automatically
    gains @tool_guard protection, input validation, rate limiting, and docstring enrichment.
    """

    def __init__(
        self,
        mcp_app: Any,
        server_name: str = "QuillanMCP",
        default_cache_ttl: Optional[float] = None,
    ):
        self.app = mcp_app
        self.server_name = server_name
        self.default_cache_ttl = default_cache_ttl
        self._wrap_existing_and_future_tools()

    def _wrap_existing_and_future_tools(self) -> None:
        # 1. Intercept future @mcp.tool() registrations
        if hasattr(self.app, "tool"):
            orig_tool = self.app.tool

            @functools.wraps(orig_tool)
            def guarded_tool(*t_args: Any, **t_kwargs: Any) -> Callable:
                base_decorator = orig_tool(*t_args, **t_kwargs)

                def inner(fn: Callable) -> Any:
                    ttl = (
                        self.default_cache_ttl
                        if fn.__name__.startswith(
                            ("get_", "list_", "read_", "search_", "query_")
                        )
                        else None
                    )
                    wrapped = tool_guard(
                        fn, server_name=self.server_name, cache_ttl=ttl
                    )
                    return base_decorator(wrapped)

                return inner

            self.app.tool = guarded_tool

        # 2. Wrap already-registered tools in FastMCP _tool_manager
        tm = getattr(self.app, "_tool_manager", None)
        tools_dict = getattr(tm, "_tools", None) if tm else None
        if isinstance(tools_dict, dict):
            for tname, tobj in tools_dict.items():
                if hasattr(tobj, "fn") and callable(tobj.fn):
                    ttl = (
                        30.0
                        if tname.startswith(("get_", "list_", "read_", "search_"))
                        else None
                    )
                    tobj.fn = tool_guard(
                        tobj.fn, server_name=self.server_name, cache_ttl=ttl
                    )
                    if not getattr(tobj, "description", None) and tobj.fn.__doc__:
                        tobj.description = tobj.fn.__doc__.strip()


# ============================================================================
# 6. ONE-CALL SERVER INITIALIZATION (setup)
# ============================================================================


def setup(
    server_name: str = "QuillanMCP",
    mcp_app: Optional[Any] = None,
    extra_paths: Optional[List[Union[str, Path]]] = None,
    log_level: int = logging.INFO,
) -> Dict[str, Any]:
    """
    One-call initialization for all 19 Quillan MCP servers:
      1. Installs the stdio guard (protects JSON-RPC stdout)
      2. Injects C:\\02_QUILLAN and ONI paths
      3. Configures stderr-only structured logging
      4. Attaches QuillanErrorMiddleware if mcp_app is passed
    """
    orig_stdout = install_stdio_guard()

    # Parent-death reaper: exit when the client that spawned us goes away.
    # Never fatal -- a failure here must not stop the server from serving.
    try:
        from . import lifecycle as _lifecycle

        _lifecycle.install(server_name)
    except Exception:
        pass

    root_dir = inject_paths(extra_paths)

    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s | [{server_name}] | %(levelname)s | %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(server_name)

    if mcp_app is not None:
        QuillanErrorMiddleware(mcp_app, server_name=server_name)

    return {
        "server_name": server_name,
        "version": __version__,
        "root_dir": root_dir,
        "logger": logger,
        "orig_stdout": orig_stdout,
        "cache": GLOBAL_CACHE,
        "rate_limiter": GLOBAL_RATE_LIMITER,
    }


__all__ = [
    "__version__",
    "install_stdio_guard",
    "inject_paths",
    "validate_string",
    "validate_int",
    "validate_path",
    "read_file_safe",
    "TTLCache",
    "TokenBucketRateLimiter",
    "GLOBAL_CACHE",
    "GLOBAL_RATE_LIMITER",
    "ok_response",
    "error_response",
    "enrich_docstring",
    "tool_guard",
    "QuillanErrorMiddleware",
    "setup",
]
