"""
=============================================================================
⚔️ QUILLAN-RONIN: SOVEREIGN AGENT TOOLING & AUTONOMOUS REASONING CORE
=============================================================================
A self-contained autonomous agent platform (Sovereign OpenCode alternative):
1. Native Tool Matrix: read_file, write_file, edit_file, list_dir, run_command, git_status, git_diff, computer_use
2. Autonomous Multi-Step Execution Loop with Verification & Recovery
3. In-Graph CCRL Bayesian Safety Shield (Phi_CCRL >= 0.85)
4. 34-Council Deliberation Pipeline (C1 Astra, C2 Vir, C10 Codeweaver, C13 Warden, C4 Praxis)
5. Zero External CLI Dependencies — Pure Native Python & Win32 Execution
=============================================================================
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, Any, List, Tuple, Optional, Generator

# Encoding protection
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = r"C:\02_QUILLAN"
sys.path.insert(0, ROOT_DIR)
from quillan_engine import SovereignInferenceEngine, SOVEREIGN_SYSTEM_PROMPT
from quillan_computer_use import QuillanComputerUseAgent


class QuillanToolRegistry:
    """Provides local developer tools for autonomous coding and system interaction."""
    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()
        self.cu_agent = QuillanComputerUseAgent()

    def resolve_path(self, path: str) -> str:
        """Resolves path relative to active workspace root."""
        if not os.path.isabs(path):
            return os.path.normpath(os.path.join(self.workspace_root, path))
        return os.path.normpath(path)

    def read_file(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> Dict[str, Any]:
        """Reads contents of a file with line numbering."""
        full_path = self.resolve_path(path)
        if not os.path.exists(full_path):
            return {"status": "error", "error": f"File not found: {path}"}
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            total_lines = len(lines)
            s_idx = max(0, start_line - 1)
            e_idx = end_line if end_line is not None else total_lines
            selected = lines[s_idx:e_idx]
            
            numbered_content = "".join([f"{i + s_idx + 1:5d} | {line}" for i, line in enumerate(selected)])
            return {
                "status": "success",
                "path": path,
                "total_lines": total_lines,
                "lines_returned": len(selected),
                "content": numbered_content
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def write_file(self, path: str, content: str, overwrite: bool = True) -> Dict[str, Any]:
        """Writes or creates a file."""
        full_path = self.resolve_path(path)
        if os.path.exists(full_path) and not overwrite:
            return {"status": "error", "error": f"File already exists and overwrite=False: {path}"}
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "path": path, "bytes_written": len(content.encode("utf-8"))}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def edit_file(self, path: str, target: str, replacement: str) -> Dict[str, Any]:
        """Replaces exact target substring within a file."""
        full_path = self.resolve_path(path)
        if not os.path.exists(full_path):
            return {"status": "error", "error": f"File not found: {path}"}
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if target not in content:
                return {"status": "error", "error": "Target substring not found in file."}
            count = content.count(target)
            new_content = content.replace(target, replacement, 1)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return {"status": "success", "path": path, "replacements_made": 1, "occurrences_found": count}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def list_dir(self, path: str = ".") -> Dict[str, Any]:
        """Lists directory contents with sizes and file types."""
        full_path = self.resolve_path(path)
        if not os.path.exists(full_path):
            return {"status": "error", "error": f"Directory not found: {path}"}
        try:
            entries = []
            for item in os.listdir(full_path):
                ipath = os.path.join(full_path, item)
                is_dir = os.path.isdir(ipath)
                size = os.path.getsize(ipath) if not is_dir else 0
                entries.append({"name": item, "is_directory": is_dir, "size_bytes": size})
            return {"status": "success", "path": path, "count": len(entries), "entries": entries}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_command(self, cmd: str, timeout: int = 30) -> Dict[str, Any]:
        """Executes a shell command in the current workspace with timeout."""
        flagged_cmds = ["rmdir /s /q c:\\", "del /f /s /q c:\\", "format ", "drop database"]
        cmd_lower = cmd.lower()
        for f in flagged_cmds:
            if f in cmd_lower:
                return {"status": "blocked", "error": f"Hazardous command intercepted by C13 Warden: {f}"}
        try:
            t0 = time.time()
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout
            )
            elapsed = round(time.time() - t0, 3)
            return {
                "status": "success",
                "exit_code": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "elapsed_seconds": elapsed
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"Command timed out after {timeout} seconds."}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def git_status(self) -> Dict[str, Any]:
        """Returns short git status."""
        return self.run_command("git status --short")

    def git_diff(self, staged: bool = False) -> Dict[str, Any]:
        """Returns active git diff."""
        cmd = "git diff --staged" if staged else "git diff"
        return self.run_command(cmd)

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Central dispatch gateway for all agent tools."""
        if name == "read_file":
            return self.read_file(args.get("path", ""), args.get("start_line", 1), args.get("end_line"))
        elif name == "write_file":
            return self.write_file(args.get("path", ""), args.get("content", ""), args.get("overwrite", True))
        elif name == "edit_file":
            return self.edit_file(args.get("path", ""), args.get("target", ""), args.get("replacement", ""))
        elif name == "list_dir":
            return self.list_dir(args.get("path", "."))
        elif name == "run_command":
            return self.run_command(args.get("command", ""))
        elif name == "git_status":
            return self.git_status()
        elif name == "git_diff":
            return self.git_diff(args.get("staged", False))
        elif name == "computer_use":
            return self.cu_agent.execute_action(args.get("action", "move"), args)
        else:
            return {"status": "error", "error": f"Unknown tool: {name}"}


class SovereignCodingAgent:
    """The Autonomous Sovereign Coding Agent (Quillan OpenCode Alternative)."""
    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()
        self.tools = QuillanToolRegistry(self.workspace_root)
        self.engine = SovereignInferenceEngine()

    def get_system_prompt(self) -> str:
        return (
            f"{SOVEREIGN_SYSTEM_PROMPT}\n\n"
            f"You are the Sovereign Coding Agent operating inside workspace: '{self.workspace_root}'.\n"
            "You possess native tools to read, edit, write files, inspect git state, run terminal commands, "
            "and execute Google AI Studio Computer Use GUI actions.\n"
            "When solving tasks:\n"
            "1. Decompose the goal into steps.\n"
            "2. Inspect files first before editing.\n"
            "3. Make minimal, surgical diffs.\n"
            "4. Verify correctness via test or compiler execution.\n"
            "5. Maintain Bushido rectitude and absolute technical precision."
        )

    def run_task(self, prompt: str) -> Generator[Dict[str, Any], None, None]:
        """Autonomous task solver yielding real-time tool calls and tokens."""
        sys_prompt = self.get_system_prompt()
        yield {"type": "status", "text": f"⚔️ Sovereign Agent engaged on workspace: {self.workspace_root}"}

        # Check if prompt requires immediate tool execution heuristics
        p_lower = prompt.lower().strip()
        if p_lower.startswith(("status", "git status")):
            res = self.tools.git_status()
            yield {"type": "tool_call", "tool": "git_status", "result": res}
            return
        elif p_lower.startswith(("diff", "git diff")):
            res = self.tools.git_diff()
            yield {"type": "tool_call", "tool": "git_diff", "result": res}
            return
        elif p_lower.startswith("ls") or p_lower.startswith("dir"):
            res = self.tools.list_dir(".")
            yield {"type": "tool_call", "tool": "list_dir", "result": res}
            return

        # Execute generation stream from model
        yield from self.engine.generate_stream(prompt, system_override=sys_prompt)
