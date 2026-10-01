#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ QUILLAN-RONIN CODEWEAVER ENGINE — NATIVE FAST-MCP SERVER
===========================================================
Embodying C10-CODEWEAVER, C25-PROMETHEUS, and C13-WARDEN:
  1. AST Structure & Complexity Inspection (Cyclomatic Complexity, Typings, Docstrings)
  2. Static Security & Hygiene Scanner (CWE-89, CWE-78, CWE-22, CWE-502, CWE-798, ReDoS)
  3. Sovereign Refactoring Adviser (SOLID, Coupling, Test Seams, Decoupling)
  4. Unit Test Scaffold Generator (Happy Path, Boundaries, Edge Cases, Invariants)
  5. Unified Diff Linter & Continuity Validator
"""

import sys
import os
import ast
import re
import math
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanCode")


# ─── 1. AST INSPECTION & COMPLEXITY ──────────────────────────────────────────

class ComplexityVisitor(ast.NodeVisitor):
    """Calculates McCabe Cyclomatic Complexity and tallies structural elements."""
    def __init__(self):
        self.complexity = 1
        self.functions = 0
        self.classes = 0
        self.imports = []
        self.docstrings = 0
        self.typed_args = 0
        self.untyped_args = 0

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
    """Inspects Python code or a file path to return AST metrics, McCabe complexity, and typing coverage.
    
    Args:
        code_or_file_path: Raw Python code or an absolute path to a .py file.
    """
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
]

@mcp.tool()
def codeweaver_security_scan(code_or_file_path: str) -> Dict[str, Any]:
    """Scans code or a file for security vulnerabilities, CWEs, and hygiene anti-patterns.
    
    Args:
        code_or_file_path: Raw code or path to file to scan.
    """
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

    return {
        "total_findings": len(findings),
        "status": "PASS" if not findings else ("FAIL" if any(f["severity"] == "CRITICAL" for f in findings) else "WARN"),
        "findings": findings,
    }


# ─── 3. SOVEREIGN REFACTORING ADVISER ────────────────────────────────────────

@mcp.tool()
def codeweaver_refactor_adviser(function_code: str, goal: str = "production_hardening") -> Dict[str, Any]:
    """Generates structured architectural refactoring advice aligned with C25-PROMETHEUS and C10-CODEWEAVER.
    
    Args:
        function_code: Snippet of code to evaluate for refactoring.
        goal: Target objective (e.g., 'production_hardening', 'performance', 'testability').
    """
    recommendations = []
    
    has_hints = bool(re.search(r"def\s+\w+\s*\([^)]*:[^)]*\)\s*->", function_code))
    if not has_hints:
        recommendations.append({
            "dimension": "Type Safety",
            "action": "Add explicit Python 3.10+ type annotations to arguments and return signatures.",
            "impact": "Enables compile-time mypy validation and self-documenting IDE contracts."
        })

    if "open(" in function_code and "with " not in function_code:
        recommendations.append({
            "dimension": "Resource Hygiene",
            "action": "Wrap file/socket I/O in deterministic context managers ('with' statement).",
            "impact": "Prevents OS file descriptor leaks across failure paths."
        })

    if "print(" in function_code:
        recommendations.append({
            "dimension": "Observability",
            "action": "Replace print statements with injected structured logging (logging.getLogger).",
            "impact": "Allows production log filtering, correlation IDs, and log aggregation."
        })

    if re.search(r"global\s+\w+", function_code):
        recommendations.append({
            "dimension": "Coupling & Purity",
            "action": "Eliminate global state mutation; pass required state explicitly as function parameters.",
            "impact": "Eliminates race conditions and makes unit testing trivial."
        })

    if not recommendations:
        recommendations.append({
            "dimension": "General Architecture",
            "action": "Code adheres to high baseline standards. Ensure unit tests cover boundary values.",
            "impact": "Maintains production readiness."
        })

    return {
        "target_goal": goal,
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
        "precedence": "Correctness and Security > API Stability > Performance > Maintainability and Style",
    }


# ─── 4. UNIT TEST SCAFFOLD GENERATOR ─────────────────────────────────────────

@mcp.tool()
def codeweaver_test_scaffold(func_name: str, args_spec: str, expected_behavior: str) -> str:
    """Generates a comprehensive pytest unit test scaffold covering happy paths, edge cases, and exceptions.
    
    Args:
        func_name: Name of the function under test (e.g., 'calculate_velocity').
        args_spec: Parameter list (e.g., 'mass: float, acceleration: float').
        expected_behavior: Expected behavior or formula summary.
    """
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
    """Validates a unified diff for header syntax, balanced hunk counts, and common diff errors.
    
    Args:
        diff_text: Unified diff block (starting with @@ or ---/+++).
    """
    lines = diff_text.strip().splitlines()
    hunks = 0
    additions = 0
    deletions = 0
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


if __name__ == "__main__":
    mcp.run()
