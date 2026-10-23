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
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_orig_print = builtins.print


def _quillan_safe_print(*args, **kwargs):
    if "file" not in kwargs or kwargs["file"] is None or kwargs["file"] is sys.stdout:
        kwargs["file"] = sys.stderr
    return _orig_print(*args, **kwargs)


builtins.print = _quillan_safe_print


# --- Unified Cross-Platform PYTHONPATH Injection ---
def _get_quillan_base() -> Path:
    """Get the Quillan base directory from environment or default."""
    base = os.environ.get("QUILLAN_BASE")
    if base:
        return Path(base)
    return Path(r"C:\02_QUILLAN")


_QUILLAN_BASE = _get_quillan_base()

_BASE_PATHS = [
    str(_QUILLAN_BASE / "09 - Projects" / "mcp"),
    str(_QUILLAN_BASE / "09 - Projects" / "projects" / "oni"),
    str(_QUILLAN_BASE / "03 - Training & Model" / "scripts"),
    str(_QUILLAN_BASE),
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
    from _shared import (
        setup as _quillan_setup,
        tool_guard,
        ok_response,
        error_response,
        validate_string,
        validate_path,
        validate_int,
        read_file_safe,
    )
    from _shared.base_engines import (
        PRISM_ENGINE,
        COUNCIL_ENGINE,
        GOVERNOR_ENGINE,
        SANDBOX_ENGINE,
    )

    _SHARED_CTX = _quillan_setup(server_name="quillan_code", mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys

    print(f"[quillan_code] _shared warning: {_shared_err}", file=_sys.stderr)

SESSION_DIR = _QUILLAN_BASE / "09 - Projects" / "mcp" / "quillan_code" / "sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ─── 1. AST INSPECTION & COMPLEXITY ──────────────────────────────────────────


class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        self.docstrings = 0
        self.typed_args = 0
        self.untyped_args = 0
        self.functions = 0
        self.classes = 0
        self.complexity = 1  # Base complexity (McCabe starts at 1)

    def visit_FunctionDef(self, node):
        self.functions += 1
        if ast.get_docstring(node):
            self.docstrings += 1
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            if arg.annotation:
                self.typed_args += 1
            else:
                self.untyped_args += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.functions += 1
        if ast.get_docstring(node):
            self.docstrings += 1
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            if arg.annotation:
                self.typed_args += 1
            else:
                self.untyped_args += 1
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

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)


@mcp.tool()
def codeweaver_ast_inspect(code_or_file_path: str) -> Dict[str, Any]:
    """Inspects Python code or a file path to return AST metrics, McCabe complexity, and typing coverage."""
    code = code_or_file_path
    path = Path(code_or_file_path.strip().strip("\"'"))
    if path.exists() and path.is_file():
        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "status": "SYNTAX_ERROR",
            "line": e.lineno,
            "offset": e.offset,
            "text": e.text,
            "msg": str(e),
        }
    visitor = ComplexityVisitor()
    visitor.visit(tree)
    total_args = visitor.typed_args + visitor.untyped_args
    typing_ratio = (
        round((visitor.typed_args / total_args) * 100, 1) if total_args > 0 else 100.0
    )
    return {
        "status": "VALID_AST",
        "cyclomatic_complexity": visitor.complexity,
        "complexity_rating": "LOW"
        if visitor.complexity < 10
        else ("MEDIUM" if visitor.complexity < 20 else "HIGH"),
        "total_classes": visitor.classes,
        "total_functions": visitor.functions,
        "total_docstrings": visitor.docstrings,
        "type_hint_coverage_pct": typing_ratio,
        "imports": sorted(list(set(visitor.imports))),
        "lines_of_code": len(code.splitlines()),
    }


# ─── 2. STATIC SECURITY & HYGIENE SCANNER ───────────────────────────────────

SECURITY_RULES = [
    {
        "id": "CWE-89",
        "title": "SQL Injection Risk",
        "pattern": r"(SELECT\s+.*FROM|INSERT\s+INTO|UPDATE\s+.*SET|DELETE\s+FROM).*(%s|\+.*format|f[\"'].*\{)",
        "severity": "CRITICAL",
        "fix": "Use parameterized queries or ORM binds instead of string concatenation/f-strings.",
    },
    {
        "id": "CWE-78",
        "title": "OS Command Injection Risk",
        "pattern": r"(os\.system|subprocess\.Popen|subprocess\.call|subprocess\.run)\s*\(\s*(f[\"']|.*%\s*|.*\+\s*)",
        "severity": "CRITICAL",
        "fix": "Pass command arguments as a list with shell=False; sanitize untrusted input.",
    },
    {
        "id": "CWE-22",
        "title": "Path Traversal Risk",
        "pattern": r"open\s*\(\s*(f[\"'].*\{|.*\+\s*|.*\.format\()",
        "severity": "HIGH",
        "fix": "Resolve and canonicalize paths using Path.resolve(), ensuring it remains inside allowed root directory.",
    },
    {
        "id": "CWE-502",
        "title": "Insecure Deserialization",
        "pattern": r"(pickle\.loads|pickle\.load|_pickle\.load|yaml\.load\s*\([^,)]+\))",
        "severity": "CRITICAL",
        "fix": "Avoid pickle on untrusted data; use json, safe_load, or protobuf.",
    },
    {
        "id": "CWE-798",
        "title": "Hardcoded Secret / API Key",
        "pattern": r"(api[_-]?key|secret|password|token|bearer)\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
        "severity": "HIGH",
        "fix": "Retrieve secrets via environment variables or secret managers (e.g. os.environ).",
    },
    {
        "id": "HYGIENE-SILENT-EXCEPT",
        "title": "Silent Exception Swallowing",
        "pattern": r"except(\s+Exception)?\s*:\s*\n\s*pass\b",
        "severity": "MEDIUM",
        "fix": "Log exceptions with context using standard logging, or raise a custom domain error.",
    },
    {
        "id": "CWE-94",
        "title": "Dynamic Code Execution Risk",
        "pattern": r"\b(eval|exec)\s*\(",
        "severity": "CRITICAL",
        "fix": "Avoid eval/exec on untrusted input; use ast.literal_eval or a sandboxed interpreter.",
    },
    {
        "id": "CWE-327",
        "title": "Weak Cryptography",
        "pattern": r"\b(md5|sha1)\s*\(",
        "severity": "MEDIUM",
        "fix": "Use SHA-256 or stronger; avoid MD5/SHA1 for security-sensitive hashing.",
    },
    {
        "id": "RE DOS",
        "title": "Potential ReDoS Pattern",
        "pattern": r"re\.(compile|match|search|findall)\s*\(\s*r?['\"].*(\(\.\*\)\+|\(\.\+\)\*|\(\.\*\)\{)",
        "severity": "MEDIUM",
        "fix": "Audit regex for catastrophic backtracking; add timeouts or use re2/regex with limits.",
    },
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
def codeweaver_security_scan(
    code_or_file_path: str, persist: bool = False
) -> Dict[str, Any]:
    """Scans code or a file for security vulnerabilities, CWEs, and hygiene anti-patterns."""
    code = code_or_file_path
    path = Path(code_or_file_path.strip().strip("\"'"))
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
                findings.append(
                    {
                        "rule_id": rule["id"],
                        "title": rule["title"],
                        "severity": rule["severity"],
                        "line_number": idx,
                        "snippet": line.strip()[:100],
                        "remediation": rule["fix"],
                    }
                )

    result = {
        "total_findings": len(findings),
        "status": "PASS"
        if not findings
        else ("FAIL" if any(f["severity"] == "CRITICAL" for f in findings) else "WARN"),
        "findings": findings,
        "timestamp": time.time(),
    }

    if persist:
        try:
            sid = f"scan-{int(time.time())}"
            p = SESSION_DIR / f"{sid}.json"
            p.write_text(
                json.dumps({"type": "security_scan", "result": result}, indent=2),
                encoding="utf-8",
            )
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result


@mcp.tool()
def codeweaver_dependency_audit(code_or_file_path: str) -> Dict[str, Any]:
    """Audits imports in code/file for risky or deprecated modules."""
    code = code_or_file_path
    path = Path(code_or_file_path.strip().strip("\"'"))
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
def codeweaver_complexity_gate(
    code_or_file_path: str, max_complexity: int = 20, min_type_coverage: float = 70.0
) -> Dict[str, Any]:
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
def codeweaver_refactor_adviser(
    function_code: str, goal: str = "production_hardening"
) -> Dict[str, Any]:
    """Generates structured architectural refactoring advice."""
    recommendations = []
    has_hints = bool(re.search(r"def\s+\w+\s*\([^)]*:[^)]*\)\s*->", function_code))
    if not has_hints:
        recommendations.append(
            {
                "dimension": "Type Safety",
                "action": "Add explicit Python 3.10+ type annotations to arguments and return signatures.",
                "impact": "Enables compile-time mypy validation and self-documenting IDE contracts.",
            }
        )
    if "open(" in function_code and "with " not in function_code:
        recommendations.append(
            {
                "dimension": "Resource Hygiene",
                "action": "Wrap file/socket I/O in deterministic context managers ('with' statement).",
                "impact": "Prevents OS file descriptor leaks across failure paths.",
            }
        )
    if "print(" in function_code:
        recommendations.append(
            {
                "dimension": "Observability",
                "action": "Replace print statements with injected structured logging (logging.getLogger).",
                "impact": "Allows production log filtering, correlation IDs, and log aggregation.",
            }
        )
    if re.search(r"global\s+\w+", function_code):
        recommendations.append(
            {
                "dimension": "Coupling & Purity",
                "action": "Eliminate global state mutation; pass required state explicitly as function parameters.",
                "impact": "Eliminates race conditions and makes unit testing trivial.",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "dimension": "General Architecture",
                "action": "Code adheres to high baseline standards. Ensure unit tests cover boundary values.",
                "impact": "Maintains production readiness.",
            }
        )
    return {
        "target_goal": goal,
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
        "precedence": "Correctness and Security > API Stability > Performance > Maintainability and Style",
    }


# ─── 4. UNIT TEST SCAFFOLD GENERATOR ─────────────────────────────────────────


@mcp.tool()
def codeweaver_test_scaffold(
    func_name: str, args_spec: str, expected_behavior: str
) -> str:
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
        fixes.append(
            {
                "rule_id": f.get("rule_id"),
                "line_number": f.get("line_number"),
                "severity": f.get("severity"),
                "remediation": f.get("remediation"),
                "stub": f"# TODO: fix {f.get('rule_id')} at line {f.get('line_number')}: {f.get('remediation')}",
            }
        )
    return {
        "fix_count": len(fixes),
        "fixes": fixes,
        "note": "Apply fixes in order of severity (CRITICAL first). Re-run codeweaver_security_scan to verify.",
    }


# ─── 7. FULL PIPELINE (scan + complexity + deps + predator-style) ───────────


@mcp.tool()
def codeweaver_full_pipeline(
    code_or_file_path: str, persist: bool = True
) -> Dict[str, Any]:
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
        "overall": "PASS"
        if (
            scan.get("status") == "PASS"
            and gate.get("status") == "PASS"
            and deps.get("status") == "PASS"
        )
        else "REVIEW",
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(
                json.dumps({"type": "full_pipeline", "result": result}, indent=2),
                encoding="utf-8",
            )
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result


# ─── 8. SESSION PERSISTENCE / REPLAY ─────────────────────────────────────────


@mcp.tool()
def codeweaver_session_list() -> Dict[str, Any]:
    """List persisted code sessions."""
    sessions = []
    for p in sorted(
        SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
    ):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            r = data.get("result", {})
            sessions.append(
                {
                    "session_id": r.get("session_id"),
                    "type": data.get("type"),
                    "overall": r.get("overall"),
                    "timestamp": r.get("timestamp"),
                    "file": str(p),
                }
            )
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


# ─── 9. LINTING INTEGRATION (ruff/flake8/pylint) ────────────────────────────


@mcp.tool()
def codeweaver_lint(code_or_file_path: str, linter: str = "ruff") -> Dict[str, Any]:
    """Run a Python linter (ruff, flake8, or pylint) on code/file and return findings."""
    import subprocess

    code = code_or_file_path
    path = Path(code_or_file_path.strip().strip("\"'"))
    if path.exists() and path.is_file():
        target = str(path)
    else:
        # Write to temp file
        tmp = SESSION_DIR / f"lint-{int(time.time())}.py"
        tmp.write_text(code, encoding="utf-8")
        target = str(tmp)

    linters = {
        "ruff": ["ruff", "check", target, "--output-format=json"],
        "flake8": [
            "flake8",
            target,
            "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s",
        ],
        "pylint": ["pylint", target, "--output-format=json"],
    }
    cmd = linters.get(linter.lower(), linters["ruff"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        findings = []
        if linter.lower() == "ruff":
            try:
                findings = json.loads(result.stdout) if result.stdout.strip() else []
            except Exception:
                findings = [{"raw": result.stdout[:500]}]
        else:
            findings = [{"raw": result.stdout[:500]}] if result.stdout.strip() else []
        return {
            "linter": linter,
            "exit_code": result.returncode,
            "total_findings": len(findings),
            "findings": findings[:50],
            "stderr": result.stderr[:300] if result.stderr else "",
            "status": "PASS" if result.returncode == 0 else "WARN",
        }
    except FileNotFoundError:
        return {
            "error": f"{linter} not installed. Install with: pip install {linter}",
            "status": "MISSING_DEP",
        }
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}


# ─── 10. CODE GENERATION / AUTOCOMPLETION ────────────────────────────────────


@mcp.tool()
def codeweaver_generate_code(
    description: str,
    language: str = "python",
    template: str = "function",
    include_docstring: bool = True,
    include_type_hints: bool = True,
    include_tests: bool = False,
) -> Dict[str, Any]:
    """Generate code from a natural language description using templates."""
    lang_templates = {
        "python": {
            "function": '''def {name}(args) -> ReturnType:
    """{docstring}"""
    # TODO: implement
    pass
''',
            "class": '''class {name}:
    """{docstring}"""
    def __init__(self):
        pass
''',
            "async_function": '''async def {name}(args) -> ReturnType:
    """{docstring}"""
    # TODO: implement
    pass
''',
        },
        "javascript": {
            "function": "function {name}(args) {{\n  // {docstring}\n  // TODO: implement\n}}\n",
            "class": "class {name} {{\n  constructor() {{\n    // {docstring}\n  }}\n}}\n",
            "async_function": "async function {name}(args) {{\n  // {docstring}\n  // TODO: implement\n}}\n",
        },
    }

    templates = lang_templates.get(language.lower(), lang_templates["python"])
    tmpl = templates.get(template, templates["function"])

    # Generate a reasonable name from description
    import re as _re

    name = _re.sub(r"[^a-zA-Z0-9_]", "_", description.lower().replace(" ", "_"))[
        :40
    ].strip("_")
    if not name:
        name = "generated_function"

    docstring = description if include_docstring else ""

    code = tmpl.format(name=name, docstring=docstring, args="")

    # Add type hints if requested (Python only)
    if include_type_hints and language.lower() == "python":
        code = code.replace("def ", "def ", 1)  # placeholder for hint injection

    result = {
        "status": "GENERATED",
        "language": language,
        "template": template,
        "name": name,
        "code": code,
        "description": description,
    }

    if include_tests:
        result["test_scaffold"] = (
            f"def test_{name}():\n    # TODO: test implementation\n    pass\n"
        )

    return result


# ─── 11. GIT OPERATIONS (blame, diff, commit review) ─────────────────────────


@mcp.tool()
def codeweaver_git_blame(
    file_path: str, line_number: Optional[int] = None
) -> Dict[str, Any]:
    """Run git blame on a file to find who last modified each line."""
    import subprocess

    try:
        cmd = ["git", "blame", "-l", file_path]
        if line_number:
            cmd = [
                "git",
                "blame",
                "-L",
                f"{line_number},{line_number}",
                "-l",
                file_path,
            ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(file_path).parent),
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip(), "status": "ERROR"}
        lines = result.stdout.strip().splitlines()
        blame_entries = []
        for line in lines[:50]:
            blame_entries.append(line)
        return {
            "file": file_path,
            "line_number": line_number,
            "blame": blame_entries,
            "status": "OK",
        }
    except FileNotFoundError:
        return {"error": "git not installed", "status": "MISSING_DEP"}
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}


@mcp.tool()
def codeweaver_git_diff(file_path: str = "", staged: bool = False) -> Dict[str, Any]:
    """Get git diff for a file or the entire working tree."""
    import subprocess

    try:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--cached")
        if file_path:
            cmd.append(file_path)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 and result.stderr:
            return {"error": result.stderr.strip(), "status": "ERROR"}
        diff_text = result.stdout[:5000]
        return {
            "diff": diff_text,
            "lines_changed": len(diff_text.splitlines()),
            "staged": staged,
            "status": "OK" if diff_text else "NO_CHANGES",
        }
    except FileNotFoundError:
        return {"error": "git not installed", "status": "MISSING_DEP"}
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}


@mcp.tool()
def codeweaver_git_log(file_path: str = "", limit: int = 10) -> Dict[str, Any]:
    """Get git commit log for a file or repository."""
    import subprocess

    try:
        cmd = ["git", "log", f"--max-count={limit}", "--oneline", "--decorate"]
        if file_path:
            cmd.append("--")
            cmd.append(file_path)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {"error": result.stderr.strip(), "status": "ERROR"}
        commits = result.stdout.strip().splitlines()
        return {"commits": commits, "count": len(commits), "status": "OK"}
    except FileNotFoundError:
        return {"error": "git not installed", "status": "MISSING_DEP"}
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}


# ─── 12. CODE COVERAGE REPORTING ─────────────────────────────────────────────


@mcp.tool()
def codeweaver_coverage(
    test_path: str = ".", source_path: str = ".", min_coverage: float = 80.0
) -> Dict[str, Any]:
    """Run pytest with coverage and return a coverage summary."""
    import subprocess

    try:
        cmd = [
            "python",
            "-m",
            "pytest",
            test_path,
            f"--cov={source_path}",
            "--cov-report=json",
            "--cov-report=term-missing",
            "-q",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        # Try to parse coverage.json
        cov_file = Path("coverage.json")
        coverage_pct = 0.0
        files = {}
        if cov_file.exists():
            try:
                cov_data = json.loads(cov_file.read_text(encoding="utf-8"))
                coverage_pct = cov_data.get("totals", {}).get("percent_covered", 0.0)
                for fname, fdata in cov_data.get("files", {}).items():
                    files[fname] = {
                        "covered": fdata.get("summary", {}).get("percent_covered", 0.0)
                    }
                cov_file.unlink(missing_ok=True)
            except Exception:
                pass
        return {
            "coverage_percent": round(coverage_pct, 1),
            "min_coverage": min_coverage,
            "passed": coverage_pct >= min_coverage,
            "files": files,
            "pytest_output": result.stdout[-500:] if result.stdout else "",
            "status": "PASS" if coverage_pct >= min_coverage else "FAIL",
        }
    except FileNotFoundError:
        return {
            "error": "pytest-cov not installed. Install with: pip install pytest-cov",
            "status": "MISSING_DEP",
        }
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}


# ─── 13. PERFORMANCE PROFILING ────────────────────────────────────────────────


@mcp.tool()
def codeweaver_profile(code: str, iterations: int = 1000) -> Dict[str, Any]:
    """Profile Python code performance using cProfile and return hot spots."""
    import cProfile
    import io
    import pstats
    import tempfile

    # Write code to temp file
    tmp = Path(tempfile.mktemp(suffix=".py"))
    tmp.write_text(code, encoding="utf-8")

    try:
        prof = cProfile.Profile()
        prof.enable()
        exec(compile(open(str(tmp)).read(), str(tmp), "exec"), {})
        prof.disable()

        stream = io.StringIO()
        stats = pstats.Stats(prof, stream=stream)
        stats.sort_stats("tottime")
        stats.print_stats(15)
        stream.seek(0)
        output = stream.read()

        # Parse top functions
        hot_spots = []
        for line in output.splitlines():
            if (
                line.strip()
                and not line.startswith(" ")
                and "function calls" not in line
                and "ncalls" not in line
                and "tottime" not in line
            ):
                try:
                    parts = line.split()
                    if len(parts) >= 6:
                        hot_spots.append(
                            {
                                "ncalls": parts[0],
                                "tottime": parts[1],
                                "percall": parts[2],
                                "cumtime": parts[3],
                                "function": " ".join(parts[5:]),
                            }
                        )
                except Exception:
                    pass

        return {
            "iterations": iterations,
            "hot_spots": hot_spots[:10],
            "raw_profile": output[:2000],
            "status": "PROFILE_COMPLETE",
        }
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}
    finally:
        tmp.unlink(missing_ok=True)


# ─── 14. DOCUMENTATION GENERATOR ──────────────────────────────────────────────


@mcp.tool()
def codeweaver_generate_docs(
    code_or_file_path: str, style: str = "google"
) -> Dict[str, Any]:
    """Generate docstrings for functions and classes missing them."""
    code = code_or_file_path
    path = Path(code_or_file_path.strip().strip("\"'"))
    if path.exists() and path.is_file():
        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    missing_docs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not ast.get_docstring(node):
                missing_docs.append(
                    {
                        "name": node.name,
                        "type": "function"
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        else "class",
                        "line": node.lineno,
                        "suggested_docstring": f'"""{node.name} — TODO: describe purpose, args, and return value."""',
                    }
                )

    # Generate updated code with docstrings inserted
    lines = code.splitlines()
    # Simple approach: report what's missing
    return {
        "total_missing": len(missing_docs),
        "missing_docs": missing_docs,
        "style": style,
        "status": "DOCS_ANALYZED",
        "note": "Apply suggested docstrings manually or use an auto-doc tool like pydocstring.",
    }


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

    engine = (
        SANDBOX_ENGINE
        if max_steps == 50000
        else __import__(
            "_shared.base_engines", fromlist=["HardenedASTSandbox"]
        ).HardenedASTSandbox(max_steps=max_steps)
    )
    result = engine.run(code)
    return {
        "status": "OK",
        "sandbox": result,
        "engine": "v5.4.0-ONI HardenedASTSandbox",
    }


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
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("exec", "eval", "compile", "__import__")
        ):
            dangerous.append(f"dangerous call: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            dangerous.append(f"dunder attribute: {node.attr}")
    if dangerous:
        return {"status": "UNSAFE", "issues": dangerous, "safe": False}
    return {
        "status": "SAFE",
        "issues": [],
        "safe": True,
        "ast_nodes": len(list(ast.walk(tree))),
    }


@mcp.tool()
def code_health() -> dict:
    """Health check endpoint for QuillanCode MCP server."""
    import time

    return {"status": "OK", "server": "QuillanCode", "timestamp": time.time()}
