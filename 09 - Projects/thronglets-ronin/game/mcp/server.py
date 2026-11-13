#!/usr/bin/env python3
"""Stdlib-only MCP bridge for Godot Ronin Engine.
Reads mcp/queue.jsonl, appends to mcp/results.jsonl.
Wire as an MCP stdio server in your client, or run manually:
  python mcp/server.py
"""

import json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUEUE = ROOT / "queue.jsonl"
RESULTS = ROOT / "results.jsonl"
QUEUE.touch(exist_ok=True)
RESULTS.touch(exist_ok=True)

TOOLS = [
    {"name": "list_project", "desc": "List scenes/scripts in the Godot project"},
    {"name": "create_scene", "desc": "Create a .tscn with root node type"},
    {"name": "add_node", "desc": "Add node to scene (parent_path, type, name)"},
    {"name": "set_property", "desc": "Set node property"},
    {"name": "save_scene", "desc": "Save scene"},
]


def list_project():
    files = [
        str(p.relative_to(ROOT.parent))
        for p in ROOT.parent.rglob("*")
        if p.suffix in (".gd", ".tscn", ".godot") and ".godot" not in p.parts
    ][:50]
    return {"ok": True, "files": files}


def main():
    print(
        json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": {"tools": [t["name"] for t in TOOLS]}}
        )
    )
    seen = 0
    while True:
        lines = QUEUE.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) > seen:
            for line in lines[seen:]:
                try:
                    req = json.loads(line)
                except Exception:
                    continue
                prompt = req.get("prompt", "")
                result = {
                    "tool": "echo",
                    "prompt": prompt,
                    "ts": time.time(),
                    "note": "Hook your AI client here. MVP echoes prompt.",
                }
                if "list" in prompt.lower():
                    result = list_project()
                with RESULTS.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
            seen = len(lines)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
