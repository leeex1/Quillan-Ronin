"""Patch every active MCP config and agent/system-prompt surface from the registry.

* JSON configs: merge-only (add missing servers, never remove or alter existing
  entries) except the generated repo opencode.json / .idx/mcp.json.
* Prompt files: insert or refresh a marker-delimited MCP routing block.
All touched files are backed up to scripts/_backups/<timestamp>/ first.
Usage: python patch_mcp_surfaces.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

LOG = logging.getLogger("mcp-patch")
REPO = Path(r"C:\02_QUILLAN")
HOME = Path.home()
REGISTRY = REPO / "09 - Projects" / "mcp" / "mcp_registry.json"
BACKUP = REPO / "scripts" / "_backups" / time.strftime("%Y%m%d-%H%M%S")

BEGIN, END = "<!-- MCP-REGISTRY:BEGIN -->", "<!-- MCP-REGISTRY:END -->"

JSON_MERGE = [
    REPO / ".agents" / "mcp_config.json",
    REPO / ".antigravity" / "mcp.json",
    REPO / ".claude" / "mcp_config.json",
    REPO / ".devin" / "mcp_config.json",
    REPO / "03 - Training & Model" / "mcp" / "quillan-mcp-config.json",
]
OPENCODE_PROJECT = REPO / ".opencode" / "opencode.jsonc"

PROMPTS = [
    REPO / "quillan.md",
    REPO / ".opencode" / "agent" / "quillan.md",
    HOME / ".config" / "opencode" / "agent" / "quillan.md",
    REPO / "CLAUDE.md",
    REPO / "GEMINI.md",
    REPO / "AGENTS.md",
    REPO / ".gemini" / "GEMINI.md",
    REPO / ".antigravity" / "rules.md",
    REPO / ".qwen" / "system_prompt.md",
    REPO / "06 - Deployment & Platforms" / "system prompts" / "Quillan-Samurai.md",
    REPO / "06 - Deployment & Platforms" / "Platforms" / "Claude" / "Quillan-Samurai.md",
    REPO / "06 - Deployment & Platforms" / "Platforms" / "Gemini" / "Quillan-Samurai.md",
    REPO / "06 - Deployment & Platforms" / "Platforms" / "Mistral" / "Quillan-Samurai.md",
]

ROUTING: list[tuple[str, list[str]]] = [
    ("Repo structure, callers, impact, git history", ["GitNexus", "Git", "Filesystem"]),
    ("Past decisions, prior sessions, durable memory", ["MemPalace", "Memory"]),
    ("Quillan corpus, knowledge files, formulas, vector search", ["QuillanRAG", "LocalRAG"]),
    ("Structured thinking, reasoning modes, traces", ["QuillanThinkingEngine", "QuillanSamuraiEngine", "SequentialThinking", "QuillanPrism"]),
    ("Council deliberation, adversarial/duo debate, evolution", ["QuillanCouncil", "QuillanBuddy", "QuillanOrochi"]),
    ("Pipeline orchestration, governance, teams, ops/deploy", ["QuillanOrchestrator", "QuillanGovernor", "QuillanTeam", "QuillanOps"]),
    ("Code, web, game, model-lab, reverse-engineering work", ["QuillanCode", "QuillanWeb", "QuillanGameDev", "QuillanModelLab", "QuillanRE"]),
    ("Audio and media generation/analysis", ["QuillanAudio", "QuillanMedia"]),
    ("Web research, docs, models, crawling", ["WebSearch", "Fetch", "DeepWiki", "HuggingFace", "QuillanSpider"]),
    ("Live browser inspection/automation", ["ChromeDevTools", "Playwright", "Puppeteer"]),
    ("Windows GUI automation", ["ComputerUse"]),
    ("Current time / timezone", ["Time"]),
]


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["servers"]


def backup(path: Path) -> None:
    tag = hashlib.sha1(str(path).encode()).hexdigest()[:8]
    BACKUP.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, BACKUP / f"{tag}_{path.name}")


def build_block(servers: dict) -> str:
    covered = {n for _, ns in ROUTING for n in ns}
    missing = set(servers) - covered
    if missing:
        raise SystemExit(f"routing table misses registry servers: {sorted(missing)}")
    rows = []
    for need, names in ROUTING:
        cells = []
        for n in names:
            s = servers[n]
            tag = "" if s["origin"] == "quillan" else " *(3rd-party)*"
            alias = s.get("aliases", {}).get("antigravity")
            cells.append(f"`{n}`" + (f" (Antigravity: `{alias}`)" if alias else "") + tag)
        rows.append(f"| {need} | {', '.join(cells)} |")
    return "\n".join([
        BEGIN,
        f"## MCP Registry — {len(servers)} servers (generated from `09 - Projects/mcp/mcp_registry.json`)",
        "",
        "Call the tool instead of guessing. Names below are the OpenCode names; aliases are the Antigravity names.",
        "",
        "| Need | Tool |",
        "|---|---|",
        *rows,
        END,
    ])


def read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes().decode("utf-8")
    return raw, ("\r\n" if "\r\n" in raw else "\n")


def patch_prompt(path: Path, block: str, dry: bool) -> str:
    text, nl = read_text(path)
    blk = block.replace("\n", nl)
    if BEGIN in text and END in text:
        new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _m: blk, text, count=1, flags=re.S)
        action = "refreshed"
    elif re.search(r"(?m)^## Routing table[^\n]*\n", text) and re.search(r"(?m)^## Behavior rules", text):
        new = re.sub(r"(?ms)^## Routing table.*?(?=^## Behavior rules)", lambda _m: blk + nl + nl, text, count=1)
        action = "replaced old routing table"
    else:
        m = re.match(r"---\r?\n.*?\r?\n---\r?\n", text, flags=re.S)
        pos = m.end() if m else 0
        new = text[:pos] + nl + blk + nl + nl + text[pos:]
        action = "inserted"
    if new == text:
        return "unchanged"
    if not dry:
        backup(path)
        path.write_bytes(new.encode("utf-8"))
    return action


def patch_json_merge(path: Path, servers: dict, dry: bool) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    cur = data.setdefault("mcpServers", {})
    added = []
    for n, s in servers.items():
        if n in cur:
            continue
        # Antigravity-named equivalent already present?
        alias = s.get("aliases", {})
        if any(a in cur for a in alias.values()):
            continue
        cur[n] = {"url": s["url"]} if s["transport"] == "remote" else {
            "command": s["command"], "args": s["args"], "env": s["env"]}
        added.append(n)
    if isinstance(data.get("mcp"), dict):
        for n in added:
            data["mcp"][n] = cur[n]
    if not added:
        return "unchanged"
    if not dry:
        backup(path)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return f"added {len(added)}: {', '.join(added)}"


def patch_opencode_project(path: Path, servers: dict, dry: bool) -> str:
    text = path.read_text(encoding="utf-8")
    if re.search(r"(?m)^\s*//", text) or "/*" in text:
        return "SKIPPED (contains comments; rewriting would drop them)"
    data = json.loads(text)
    mcp = data.setdefault("mcp", {})
    added = []
    for n, s in servers.items():
        if n in mcp:
            continue
        if s["transport"] == "remote":
            mcp[n] = {"type": "remote", "url": s["url"], "enabled": True}
        else:
            mcp[n] = {"type": "local", "command": [s["command"], *s["args"]], "enabled": True,
                      **({"environment": s["env"]} if s["env"] else {})}
        added.append(n)
    if not added:
        return "unchanged"
    if not dry:
        backup(path)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return f"added {len(added)}: {', '.join(added)}"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    dry = ap.parse_args().dry_run
    servers = load_registry()
    block = build_block(servers)
    for p in JSON_MERGE:
        LOG.info("%-70s %s", p.relative_to(REPO) if p.is_relative_to(REPO) else p,
                 patch_json_merge(p, servers, dry) if p.exists() else "MISSING")
    LOG.info("%-70s %s", OPENCODE_PROJECT.relative_to(REPO), patch_opencode_project(OPENCODE_PROJECT, servers, dry))
    for p in PROMPTS:
        LOG.info("%-70s %s", p.relative_to(REPO) if p.is_relative_to(REPO) else p,
                 patch_prompt(p, block, dry) if p.exists() else "MISSING")
    return 0


if __name__ == "__main__":
    sys.exit(main())
