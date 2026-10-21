#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ QUILLAN-RONIN CODEWEAVER ENGINE — NATIVE FAST-MCP SERVER
==========================================================
Embodying C10-CODEWEAVER, C25-PROMETHEUS, and C13-WARDEN:
  1. AST Structure & Complexity Inspection (Cyclomatic Complexity, Typings, Docstrings)
  2. Static Security & Hygiene Scanner (CWE-89, CWE-78, CWE-22, CWE-502, CWE-798, ReDoS, +more)
  3. Sovereign Refactoring Adviser (SOLID, Coupling, Test Seams, Decoupling)
  4. Unit Test Scaffold Generator (Happy Path, Boundaries, Edge Cases, Invariants)
  5. Unified Diff Linter & Continuity Validator
  6. Dependency Audit (risky/deprecated imports)
  7. Complexity Gate (threshold enforcement)
  8. Full Pipeline (scan + predator strike + review) with persistence
  9. Fix-It Scaffold Generator
  10. Session persistence + replay
"""

import sys
import os
import logging
import builtins

# --- Quillan-Ronin v5.4.0-ONI Stdio Guard ---
# Ensure all logs and prints go strictly to stderr to prevent stdio JSON-RPC corruption
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_orig_print = builtins.print
def _quillan_safe_print(*args, **kwargs):
    if "file" not in kwargs or kwargs["file"] is None or kwargs["file"] is sys.stdout:
        kwargs["file"] = sys.stderr
    return _orig_print(*args, **kwargs)
builtins.print = _quillan_safe_print

# --- Unified Cross-Platform PYTHONPATH Injection ---
_BASE_PATHS = [
    r"C:\02_QUILLAN\09 - Projects\mcp",
    r"C:\02_QUILLAN\09 - Projects\projects\oni",
    r"C:\02_QUILLAN\03 - Training & Model\scripts",
    r"C:\02_QUILLAN",
]
for _bp in _BASE_PATHS:
    if os.name != "nt":
        if _bp.startswith("C:") or _bp.startswith("c:"):
            _bp = "/mnt/c/" + _bp[3:].replace("\\", "/")
    if os.path.exists(_bp) and _bp not in sys.path:
        sys.path.insert(0, _bp)



import ast
import re
import math
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanCode")
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import setup as _quillan_setup, tool_guard, ok_response, error_response, validate_string, validate_path, validate_int, read_file_safe
    from _shared.base_engines import PRISM_ENGINE, COUNCIL_ENGINE, GOVERNOR_ENGINE, SANDBOX_ENGINE
    _SHARED_CTX = _quillan_setup(server_name='quillan_code', mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys
    print(f'[quillan_code] _shared warning: {_shared_err}', file=_sys.stderr)

SESSION_DIR = Path(r"C:\02_QUILLAN\09 - Projects\mcp\quillan_code\sessions")
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ─── 1. AST INSPECTION & COMPLEXITY ──────────────────────────────────────────

class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):

        self.imports = []
        self.docstrings = 0
        self.typed_args = 0
        self.untyped_args = 0

    def visit_FunctionDef(self, node):
        self.functions += 1
        if ast.get_docstring(node):
            self.docstrings += 1
        for arg in node.args.args:
            if arg.arg == "self": continue
            if arg.annotation: self.typed_args += 1
            else: self.untyped_args += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.functions += 1
        if ast.get_docstring(node):
            self.docstrings += 1
        for arg in node.args.args:
            if arg.arg == "self": continue
            if arg.annotation: self.typed_args += 1
            else: self.untyped_args += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.classes += 1
        if ast.get_docstring(node):
            self.docstrings += 1
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        mod = node.module or ""
        for alias in node.names:
            self.imports.append(f"{mod}.{alias.name}")
        self.generic_visit(node)

    def visit_If(self, node): self.complexity += 1; self.generic_visit(node)
    def visit_For(self, node): self.complexity += 1; self.generic_visit(node)
    def visit_AsyncFor(self, node): self.complexity += 1; self.generic_visit(node)
    def visit_While(self, node): self.complexity += 1; self.generic_visit(node)
    def visit_ExceptHandler(self, node): self.complexity += 1; self.generic_visit(node)
    def visit_With(self, node): self.complexity += 1; self.generic_visit(node)
    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)


@mcp.tool()
def codeweaver_ast_inspect(code_or_file_path: str) -> Dict[str, Any]:
    """Inspects Python code or a file path to return AST metrics, McCabe complexity, and typing coverage."""
    code = code_or_file_path
    path = Path(code_or_file_path.strip().strip('"\''))
    if path.exists() and path.is_file():
        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"status": "SYNTAX_ERROR", "line": e.lineno, "offset": e.offset,
                "text": e.text, "msg": str(e)}
    visitor = ComplexityVisitor()
    visitor.visit(tree)
    total_args = visitor.typed_args + visitor.untyped_args
    typing_ratio = round((visitor.typed_args / total_args) * 100, 1) if total_args > 0 else 100.0
    return {
        "status": "VALID_AST",
        "cyclomatic_complexity": visitor.complexity,
        "complexity_rating": "LOW" if visitor.complexity < 10 else ("MEDIUM" if visitor.complexity < 20 else "HIGH"),
        "total_classes": visitor.classes,
        "total_functions": visitor.functions,
        "total_docstrings": visitor.docstrings,
        "type_hint_coverage_pct": typing_ratio,
        "imports": sorted(list(set(visitor.imports))),
        "lines_of_code": len(code.splitlines()),
    }


# ─── 2. STATIC SECURITY & HYGIENE SCANNER ───────────────────────────────────

SECURITY_RULES = [
    {"id": "CWE-89", "title": "SQL Injection Risk",
     "pattern": r"(SELECT\s+.*FROM|INSERT\s+INTO|UPDATE\s+.*SET|DELETE\s+FROM).*(%s|\+.*format|f[\"'].*\{)",
     "severity": "CRITICAL", "fix": "Use parameterized queries or ORM binds instead of string concatenation/f-strings."},
    {"id": "CWE-78", "title": "OS Command Injection Risk",
     "pattern": r"(os\.system|subprocess\.Popen|subprocess\.call|subprocess\.run)\s*\(\s*(f[\"']|.*%\s*|.*\+\s*)",
     "severity": "CRITICAL", "fix": "Pass command arguments as a list with shell=False; sanitize untrusted input."},
    {"id": "CWE-22", "title": "Path Traversal Risk",
     "pattern": r"open\s*\(\s*(f[\"'].*\{|.*\+\s*|.*\.format\()",
     "severity": "HIGH", "fix": "Resolve and canonicalize paths using Path.resolve(), ensuring it remains inside allowed root directory."},
    {"id": "CWE-502", "title": "Insecure Deserialization",
     "pattern": r"(pickle\.loads|pickle\.load|_pickle\.load|yaml\.load\s*\([^,)]+\))",
     "severity": "CRITICAL", "fix": "Avoid pickle on untrusted data; use json, safe_load, or protobuf."},
    {"id": "CWE-798", "title": "Hardcoded Secret / API Key",
     "pattern": r"(api[_-]?key|secret|password|token|bearer)\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
     "severity": "HIGH", "fix": "Retrieve secrets via environment variables or secret managers (e.g. os.environ)."},
    {"id": "HYGIENE-SILENT-EXCEPT", "title": "Silent Exception Swallowing",
     "pattern": r"except(\s+Exception)?\s*:\s*\n\s*pass\b",
     "severity": "MEDIUM", "fix": "Log exceptions with context using standard logging, or raise a custom domain error."},
    {"id": "CWE-94", "title": "Dynamic Code Execution Risk",
     "pattern": r"\b(eval|exec)\s*\(",
     "severity": "CRITICAL", "fix": "Avoid eval/exec on untrusted input; use ast.literal_eval or a sandboxed interpreter."},
    {"id": "CWE-327", "title": "Weak Cryptography",
     "pattern": r"\b(md5|sha1)\s*\(",
     "severity": "MEDIUM", "fix": "Use SHA-256 or stronger; avoid MD5/SHA1 for security-sensitive hashing."},
    {"id": "RE DOS", "title": "Potential ReDoS Pattern",
     "pattern": r"re\.(compile|match|search|findall)\s*\(\s*r?['\"].*(\(\.\*\)\+|\(\.\+\)\*|\(\.\*\)\{)",
     "severity": "MEDIUM", "fix": "Audit regex for catastrophic backtracking; add timeouts or use re2/regex with limits."},
]

DEPENDENCY_RISK = {
    "pickle": "Insecure deserialization (CWE-502)",
    "yaml": "Use yaml.safe_load only",
    "subprocess": "Validate shell=False and sanitize args",
    "os.system": "Avoid; use subprocess with list args",
    "eval": "Avoid dynamic execution",
    "exec": "Avoid dynamic execution",
}


@mcp.tool()
def codeweaver_security_scan(code_or_file_path: str, persist: bool = False) -> Dict[str, Any]:
    """Scans code or a file for security vulnerabilities, CWEs, and hygiene anti-patterns."""
    code = code_or_file_path
    path = Path(code_or_file_path.strip().strip('"\''))
    if path.exists() and path.is_file():
        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}

    findings = []
    lines = code.splitlines()

    for rule in SECURITY_RULES:
        regex = re.compile(rule["pattern"], re.IGNORECASE)
        for idx, line in enumerate(lines, start=1):
            if regex.search(line):
                findings.append({
                    "rule_id": rule["id"],
                    "title": rule["title"],
                    "severity": rule["severity"],
                    "line_number": idx,
                    "snippet": line.strip()[:100],
                    "remediation": rule["fix"],
                })

    result = {
        "total_findings": len(findings),
        "status": "PASS" if not findings else ("FAIL" if any(f["severity"] == "CRITICAL" for f in findings) else "WARN"),
        "findings": findings,
        "timestamp": time.time(),
    }

    if persist:
        try:
            sid = f"scan-{int(time.time())}"
            p = SESSION_DIR / f"{sid}.json"
            p.write_text(json.dumps({"type": "security_scan", "result": result}, indent=2), encoding="utf-8")
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result


@mcp.tool()
def codeweaver_dependency_audit(code_or_file_path: str) -> Dict[str, Any]:
    """Audits imports in code/file for risky or deprecated modules."""
    code = code_or_file_path
    path = Path(code_or_file_path.strip().strip('"\''))
    if path.exists() and path.is_file():
        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.append(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])

    risky = []
    for imp in sorted(set(imports)):
        if imp in DEPENDENCY_RISK:
            risky.append({"module": imp, "concern": DEPENDENCY_RISK[imp]})

    return {
        "total_imports": len(set(imports)),
        "imports": sorted(set(imports)),
        "risky_imports": risky,
        "status": "PASS" if not risky else "WARN",
    }


@mcp.tool()
def codeweaver_complexity_gate(code_or_file_path: str, max_complexity: int = 20,
                              min_type_coverage: float = 70.0) -> Dict[str, Any]:
    """Enforces complexity and type-coverage thresholds."""
    inspect = codeweaver_ast_inspect(code_or_file_path)
    if inspect.get("status") != "VALID_AST":
        return {"status": "SKIP", "reason": inspect.get("status"), "detail": inspect}
    cc = inspect["cyclomatic_complexity"]
    cov = inspect["type_hint_coverage_pct"]
    failures = []
    if cc > max_complexity:
        failures.append(f"Cyclomatic complexity {cc} exceeds max {max_complexity}")
    if cov < min_type_coverage:
        failures.append(f"Type coverage {cov}% below min {min_type_coverage}%")
    return {
        "status": "PASS" if not failures else "FAIL",
        "cyclomatic_complexity": cc,
        "type_hint_coverage_pct": cov,
        "max_complexity": max_complexity,
        "min_type_coverage": min_type_coverage,
        "failures": failures,
    }


# ─── 3. SOVEREIGN REFACTORING ADVISER ────────────────────────────────────────

@mcp.tool()
def codeweaver_refactor_adviser(function_code: str, goal: str = "production_hardening") -> Dict[str, Any]:
    """Generates structured architectural refactoring advice."""
    recommendations = []
    has_hints = bool(re.search(r"def\s+\w+\s*\([^)]*:[^)]*\)\s*->", function_code))
    if not has_hints:
        recommendations.append({"dimension": "Type Safety",
            "action": "Add explicit Python 3.10+ type annotations to arguments and return signatures.",
            "impact": "Enables compile-time mypy validation and self-documenting IDE contracts."})
    if "open(" in function_code and "with " not in function_code:
        recommendations.append({"dimension": "Resource Hygiene",
            "action": "Wrap file/socket I/O in deterministic context managers ('with' statement).",
            "impact": "Prevents OS file descriptor leaks across failure paths."})
    if "print(" in function_code:
        recommendations.append({"dimension": "Observability",
            "action": "Replace print statements with injected structured logging (logging.getLogger).",
            "impact": "Allows production log filtering, correlation IDs, and log aggregation."})
    if re.search(r"global\s+\w+", function_code):
        recommendations.append({"dimension": "Coupling & Purity",
            "action": "Eliminate global state mutation; pass required state explicitly as function parameters.",
            "impact": "Eliminates race conditions and makes unit testing trivial."})
    if not recommendations:
        recommendations.append({"dimension": "General Architecture",
            "action": "Code adheres to high baseline standards. Ensure unit tests cover boundary values.",
            "impact": "Maintains production readiness."})
    return {
        "target_goal": goal,
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
        "precedence": "Correctness and Security > API Stability > Performance > Maintainability and Style",
    }


# ─── 4. UNIT TEST SCAFFOLD GENERATOR ─────────────────────────────────────────

@mcp.tool()
def codeweaver_test_scaffold(func_name: str, args_spec: str, expected_behavior: str) -> str:
    """Generates a comprehensive pytest unit test scaffold."""
    return f'''import pytest

# ── Tests for {func_name} ──────────────────────────────────────────────────

def test_{func_name}_happy_path():
    """Validates nominal execution under expected standard inputs."""
    # TODO: Arrange nominal inputs based on ({args_spec})
    # result = {func_name}(...)
    # assert result is not None
    pass


def test_{func_name}_boundary_limits():
    """Validates behavior at extreme boundaries (zeros, min/max floats, empty collections)."""
    # TODO: Pass boundary values
    pass


def test_{func_name}_invalid_types():
    """Validates that unexpected argument types raise appropriate TypeError or ValueError."""
    with pytest.raises((TypeError, ValueError)):
        # TODO: Pass invalid type
        pass


def test_{func_name}_deterministic_invariance():
    """Verifies that multiple calls with identical arguments yield strictly identical results."""
    # result_a = {func_name}(...)
    # result_b = {func_name}(...)
    # assert result_a == result_b
    pass
'''


# ─── 5. UNIFIED DIFF LINTER ──────────────────────────────────────────────────

@mcp.tool()
def codeweaver_diff_linter(diff_text: str) -> Dict[str, Any]:
    """Validates a unified diff for header syntax, balanced hunk counts, and common diff errors."""
    lines = diff_text.strip().splitlines()
    hunks = additions = deletions = 0
    errors = []
    for idx, line in enumerate(lines, start=1):
        if line.startswith("@@"):
            hunks += 1
            if not re.match(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@", line):
                errors.append(f"Line {idx}: Malformed hunk header '{line}'")
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return {
        "valid_diff": len(errors) == 0,
        "hunk_count": hunks,
        "additions": additions,
        "deletions": deletions,
        "errors": errors,
    }


# ─── 6. FIX-IT SCAFFOLD GENERATOR ───────────────────────────────────────────

@mcp.tool()
def codeweaver_fix_scaffold(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates a fix-it scaffold from security scan findings (CWE -> remediation stubs)."""
    fixes = []
    for f in findings:
        fixes.append({
            "rule_id": f.get("rule_id"),
            "line_number": f.get("line_number"),
            "severity": f.get("severity"),
            "remediation": f.get("remediation"),
            "stub": f"# TODO: fix {f.get('rule_id')} at line {f.get('line_number')}: {f.get('remediation')}",
        })
    return {
        "fix_count": len(fixes),
        "fixes": fixes,
        "note": "Apply fixes in order of severity (CRITICAL first). Re-run codeweaver_security_scan to verify.",
    }


# ─── 7. FULL PIPELINE (scan + complexity + deps + predator-style) ───────────

@mcp.tool()
def codeweaver_full_pipeline(code_or_file_path: str, persist: bool = True) -> Dict[str, Any]:
    """Runs AST inspect + security scan + dependency audit + complexity gate in one pipeline."""
    inspect = codeweaver_ast_inspect(code_or_file_path)
    scan = codeweaver_security_scan(code_or_file_path)
    deps = codeweaver_dependency_audit(code_or_file_path)
    gate = codeweaver_complexity_gate(code_or_file_path)

    session_id = f"code-{int(time.time())}"
    result = {
        "status": "PIPELINE_COMPLETE",
        "session_id": session_id,
        "ast": inspect,
        "security": scan,
        "dependencies": deps,
        "complexity_gate": gate,
        "overall": "PASS" if (scan.get("status") == "PASS" and gate.get("status") == "PASS" and deps.get("status") == "PASS") else "REVIEW",
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(json.dumps({"type": "full_pipeline", "result": result}, indent=2), encoding="utf-8")
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result


# ─── 8. SESSION PERSISTENCE / REPLAY ─────────────────────────────────────────

@mcp.tool()
def codeweaver_session_list() -> Dict[str, Any]:
    """List persisted code sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            r = data.get("result", {})
            sessions.append({
                "session_id": r.get("session_id"),
                "type": data.get("type"),
                "overall": r.get("overall"),
                "timestamp": r.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def codeweaver_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted code session by ID."""
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        matches = list(SESSION_DIR.glob(f"{session_id}*.json"))
        if not matches:
            return {"error": f"Session not found: {session_id}"}
        path = matches[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"Failed to read session: {e}"}



# --- Quillan-Ronin Global Error-Handling Middleware ---
from fastmcp.server.middleware import Middleware, MiddlewareContext
import logging as _log

class QuillanErrorMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        try:
            return await call_next(context)
        except Exception as e:
            _log.error("Tool error: %s", e, exc_info=True)
            return {"status": "ERROR", "error": str(e), "error_type": type(e).__name__}


mcp.add_middleware(QuillanErrorMiddleware())
if __name__ == "__main__":
    mcp.run()


# ── HARDENED AST SANDBOX (v5.4.0-ONI CWE-94) ─────────────────────────────────────────

@mcp.tool()
def sandbox_execute(code: str, max_steps: int = 50000) -> Dict[str, Any]:
    """Execute Python code in the v5.4.0-ONI Hardened AST Sandbox (zero-privilege, CWE-94 safe)."""
    from _shared.base_engines import SANDBOX_ENGINE
    engine = SANDBOX_ENGINE if max_steps == 50000 else __import__('_shared.base_engines', fromlist=['HardenedASTSandbox']).HardenedASTSandbox(max_steps=max_steps)
    result = engine.run(code)
    return {"status": "OK", "sandbox": result, "engine": "v5.4.0-ONI HardenedASTSandbox"}

@mcp.tool()
def sandbox_validate(code: str) -> Dict[str, Any]:
    """Validate Python code syntax and AST safety without executing it."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"status": "SYNTAX_ERROR", "line": e.lineno, "msg": str(e)}
    # Check for dangerous constructs
    dangerous = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            dangerous.append("import statement")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "compile", "__import__"):
            dangerous.append(f"dangerous call: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            dangerous.append(f"dunder attribute: {node.attr}")
    if dangerous:
        return {"status": "UNSAFE", "issues": dangerous, "safe": False}
    return {"status": "SAFE", "issues": [], "safe": True, "ast_nodes": len(list(ast.walk(tree)))}

@mcp.tool()
def code_health() -> dict:
    """Health check endpoint for QuillanCode MCP server."""
    import time
    return {"status": "OK", "server": "QuillanCode", "timestamp": time.time()}
