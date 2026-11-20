#!/usr/bin/env python3
"""Reversible MCP cutover: 19 Quillan servers -> 3 facades across 4 configs + registry.

Safety:
  - Backs up every touched file to C:\\02_QUILLAN\\exports\\mcp_cutover_backups\\<timestamp>\\
  - Writes an inverse rollback script + manifest in the same backup dir
  - Idempotent: re-running detects already-cut-over configs
  - Never touches third-party MCP entries

Usage:
  python mcp_cutover.py --apply
  python mcp_cutover.py --rollback <backup_dir>
  python mcp_cutover.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_ROOT = Path(r"C:\02_QUILLAN\exports\mcp_cutover_backups") / STAMP
PY314 = r"C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe"
MCP_ROOT = r"C:\02_QUILLAN\09 - Projects\mcp"

# The 19 Quillan child names as they appear in EACH config (aliases differ).
CONFIGS: dict[str, dict[str, Any]] = {
    "opencode": {
        "path": Path(r"C:\Users\Admin\.config\opencode\opencode.jsonc"),
        "format": "jsonc",
        "section": "mcp",
        "remove": [
            "QuillanThinkingEngine",
            "QuillanSamuraiEngine",
            "QuillanBuddy",
            "QuillanRAG",
            "QuillanAudio",
            "QuillanCode",
            "QuillanCouncil",
            "QuillanGameDev",
            "QuillanGovernor",
            "QuillanMedia",
            "QuillanModelLab",
            "QuillanOps",
            "QuillanPrism",
            "QuillanSpider",
            "QuillanTeam",
            "QuillanWeb",
        ],  # 16 present of 19 (RE, Orchestrator, Orochi absent here)
        "facades": {
            "QuillanCortex": {
                "type": "local",
                "command": [PY314, MCP_ROOT + r"\quillan_cortex\server.py"],
                "enabled": True,
            },
            "QuillanArchive": {
                "type": "local",
                "command": [PY314, MCP_ROOT + r"\quillan_archive\server.py"],
                "enabled": True,
            },
            "QuillanSensory": {
                "type": "local",
                "command": [PY314, MCP_ROOT + r"\quillan_sensory\server.py"],
                "enabled": True,
            },
        },
    },
    "gemini": {
        "path": Path(r"C:\Users\Admin\.gemini\config\mcp_config.json"),
        "format": "json",
        "section": "mcpServers",
        "remove": [
            "QuillanRAG",
            "ThinkingEngine",
            "SamuraiEngine",
            "RoninBuddy",
            "QuillanCode",
            "QuillanModelLab",
            "QuillanCouncil",
            "QuillanGovernor",
            "QuillanPrism",
            "QuillanAudio",
            "QuillanGameDev",
            "QuillanMedia",
            "QuillanOps",
            "QuillanSpider",
            "QuillanTeam",
            "QuillanWeb",
            "QuillanRE",
            "QuillanOrchestrator",
            "QuillanOrochi",
        ],
        "facades": {
            "QuillanCortex": {
                "transport": "stdio",
                "command": PY314,
                "args": [MCP_ROOT + r"\quillan_cortex\server.py"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": MCP_ROOT + r";C:\02_QUILLAN",
                },
                "origin": "quillan",
            },
            "QuillanArchive": {
                "transport": "stdio",
                "command": PY314,
                "args": [MCP_ROOT + r"\quillan_archive\server.py"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": MCP_ROOT + r";C:\02_QUILLAN",
                    "QUILLAN_RAG_DB": r"C:\02_QUILLAN\07 - Memory & LanceDB\quillan_rag_db",
                },
                "origin": "quillan",
            },
            "QuillanSensory": {
                "transport": "stdio",
                "command": PY314,
                "args": [MCP_ROOT + r"\quillan_sensory\server.py"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": MCP_ROOT + r";C:\02_QUILLAN",
                },
                "origin": "quillan",
            },
        },
    },
    "qwen": {
        "path": Path(r"C:\Users\Admin\.qwen\mcp_config.json"),
        "format": "json",
        "section": "mcpServers",
        "remove": [
            "QuillanRAG",
            "ThinkingEngine",
            "RoninBuddy",
            "SamuraiEngine",
            "QuillanCode",
            "QuillanModelLab",
            "QuillanCouncil",
            "QuillanGovernor",
            "QuillanPrism",
            "QuillanAudio",
            "QuillanGameDev",
            "QuillanMedia",
            "QuillanOps",
            "QuillanSpider",
            "QuillanTeam",
            "QuillanWeb",
        ],
        "facades": {
            "QuillanCortex": {
                "transport": "stdio",
                "command": PY314,
                "args": [MCP_ROOT + r"\quillan_cortex\server.py"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": MCP_ROOT + r";C:\02_QUILLAN",
                },
                "origin": "quillan",
            },
            "QuillanArchive": {
                "transport": "stdio",
                "command": PY314,
                "args": [MCP_ROOT + r"\quillan_archive\server.py"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": MCP_ROOT + r";C:\02_QUILLAN",
                    "QUILLAN_RAG_DB": r"C:\02_QUILLAN\07 - Memory & LanceDB\quillan_rag_db",
                },
                "origin": "quillan",
            },
            "QuillanSensory": {
                "transport": "stdio",
                "command": PY314,
                "args": [MCP_ROOT + r"\quillan_sensory\server.py"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": MCP_ROOT + r";C:\02_QUILLAN",
                },
                "origin": "quillan",
            },
        },
    },
    "agents": {
        "path": Path(r"C:\02_QUILLAN\.agents\mcp_config.json"),
        "format": "json",
        "section": "mcpServers",
        "remove": [
            "QuillanPrism",
            "QuillanGovernor",
            "QuillanCouncil",
            "QuillanCode",
            "QuillanModelLab",
            "QuillanRAG",
            "QuillanAudio",
            "QuillanGameDev",
            "QuillanMedia",
            "QuillanOps",
            "QuillanOrchestrator",
            "QuillanOrochi",
            "QuillanRE",
            "QuillanSamuraiEngine",
            "QuillanSpider",
            "QuillanTeam",
            "QuillanThinkingEngine",
            "QuillanWeb",
            "QuillanBuddy",
        ],
        "facades": {
            "QuillanCortex": {
                "transport": "stdio",
                "command": PY314,
                "args": [MCP_ROOT + r"\quillan_cortex\server.py"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": MCP_ROOT + r";C:\02_QUILLAN",
                },
                "origin": "quillan",
            },
            "QuillanArchive": {
                "transport": "stdio",
                "command": PY314,
                "args": [MCP_ROOT + r"\quillan_archive\server.py"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": MCP_ROOT + r";C:\02_QUILLAN",
                    "QUILLAN_RAG_DB": r"C:\02_QUILLAN\07 - Memory & LanceDB\quillan_rag_db",
                },
                "origin": "quillan",
            },
            "QuillanSensory": {
                "transport": "stdio",
                "command": PY314,
                "args": [MCP_ROOT + r"\quillan_sensory\server.py"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": MCP_ROOT + r";C:\02_QUILLAN",
                },
                "origin": "quillan",
            },
        },
    },
    "registry": {
        "path": Path(MCP_ROOT + r"\mcp_registry.json"),
        "format": "json",
        "section": "servers",
        "remove": [
            "QuillanAudio",
            "QuillanBuddy",
            "QuillanCode",
            "QuillanCouncil",
            "QuillanGameDev",
            "QuillanGovernor",
            "QuillanMedia",
            "QuillanModelLab",
            "QuillanOps",
            "QuillanOrchestrator",
            "QuillanOrochi",
            "QuillanPrism",
            "QuillanRAG",
            "QuillanRE",
            "QuillanSamuraiEngine",
            "QuillanSpider",
            "QuillanTeam",
            "QuillanThinkingEngine",
            "QuillanWeb",
        ],
        "facades": {
            "QuillanCortex": {
                "transport": "stdio",
                "command": PY314,
                "args": [MCP_ROOT + r"\quillan_cortex\server.py"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": MCP_ROOT + r";C:\02_QUILLAN",
                },
                "origin": "quillan",
            },
            "QuillanArchive": {
                "transport": "stdio",
                "command": PY314,
                "args": [MCP_ROOT + r"\quillan_archive\server.py"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": MCP_ROOT + r";C:\02_QUILLAN",
                    "QUILLAN_RAG_DB": r"C:\02_QUILLAN\07 - Memory & LanceDB\quillan_rag_db",
                },
                "origin": "quillan",
            },
            "QuillanSensory": {
                "transport": "stdio",
                "command": PY314,
                "args": [MCP_ROOT + r"\quillan_sensory\server.py"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": MCP_ROOT + r";C:\02_QUILLAN",
                },
                "origin": "quillan",
            },
        },
    },
}

FACADE_KEYS = {"QuillanCortex", "QuillanArchive", "QuillanSensory"}


def load_jsonc(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8-sig")
    # Config is standard JSON with .jsonc extension; strip // line comments if any.
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        lines.append(line)
    return json.loads("\n".join(lines))


def apply() -> None:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for name, cfg in CONFIGS.items():
        path: Path = cfg["path"]
        original = path.read_bytes()
        data = (
            load_jsonc(path)
            if cfg["format"] == "jsonc"
            else json.loads(original.decode("utf-8-sig"))
        )
        section = data[cfg["section"]]
        removed = [k for k in cfg["remove"] if k in section]
        if FACADE_KEYS.issubset(section.keys()) and not removed:
            print(f"[{name}] SKIP already cut over")
            continue
        backup = BACKUP_ROOT / f"{name}{path.suffix}"
        backup.write_bytes(original)
        for k in removed:
            del section[k]
        section.update(cfg["facades"])
        out = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        path.write_text(out, encoding="utf-8")
        manifest.append(
            {
                "config": name,
                "path": str(path),
                "backup": str(backup),
                "removed": ",".join(removed),
                "added": ",".join(cfg["facades"]),
            }
        )
        print(
            f"[{name}] removed {len(removed)} -> {len(section)} entries; backup={backup.name}"
        )
    (BACKUP_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    rollback = BACKUP_ROOT / "rollback.py"
    rollback.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\nimport json, shutil\n"
        f"M = json.loads(Path(r'{BACKUP_ROOT / 'manifest.json'}').read_text())\n"
        "for e in M:\n"
        "    shutil.copy2(e['backup'], e['path'])\n"
        "    print('restored', e['config'])\n",
        encoding="utf-8",
    )
    print(f"\nAPPLY COMPLETE. Backup: {BACKUP_ROOT}")
    print(
        "New sessions use 3 facades; THIS session keeps already-loaded tools until restart."
    )
    print(f"Rollback: python {rollback}")


def rollback(backup_dir: Path) -> None:
    m = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    for e in m:
        shutil.copy2(e["backup"], e["path"])
        print("restored", e["config"])
    print("ROLLBACK COMPLETE — restart all IDEs/agents")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rollback", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.rollback:
        rollback(a.rollback)
    elif a.apply:
        apply()
    else:
        print("DRY RUN — would modify:")
        for name, cfg in CONFIGS.items():
            if cfg["path"].exists():
                data = (
                    load_jsonc(cfg["path"])
                    if cfg["format"] == "jsonc"
                    else json.loads(cfg["path"].read_text(encoding="utf-8-sig"))
                )
                sec = data[cfg["section"]]
                print(
                    f"  {name}: remove {len([k for k in cfg['remove'] if k in sec])}, add 3"
                )


if __name__ == "__main__":
    main()
