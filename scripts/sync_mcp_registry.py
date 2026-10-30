"""Build the canonical MCP registry and regenerate repo-level MCP configs.

Source of truth: the live OpenCode config (~/.config/opencode/opencode.jsonc),
plus Puppeteer from the Antigravity config. Secrets are never copied: only
allowlisted, non-sensitive environment keys survive into generated files.

Usage:  python sync_mcp_registry.py [--dry-run]
Live user configs are never rewritten by this script.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

LOG = logging.getLogger("mcp-sync")

REPO = Path(r"C:\02_QUILLAN")
LIVE_OPENCODE = Path.home() / ".config" / "opencode" / "opencode.jsonc"
ANTIGRAVITY = Path.home() / ".gemini" / "config" / "mcp_config.json"
REGISTRY = REPO / "09 - Projects" / "mcp" / "mcp_registry.json"
REPO_OPENCODE = REPO / "opencode.json"
IDX_MCP = REPO / ".idx" / "mcp.json"

# Environment keys safe to persist in committed files.
ENV_ALLOWLIST = {
    "PYTHONIOENCODING", "PYTHONPATH", "WINDOWS_MCP_SCREENSHOT_SCALE",
    "WINDOWS_MCP_DISABLE_FLASH", "WINDOWS_MCP_WATCHDOG", "ANONYMIZED_TELEMETRY",
    "BASE_DIR", "QUILLAN_RAG_DB",
}

# canonical (OpenCode) name -> names used on other platforms
ALIASES: dict[str, dict[str, str]] = {
    "QuillanThinkingEngine": {"antigravity": "ThinkingEngine"},
    "QuillanSamuraiEngine": {"antigravity": "SamuraiEngine"},
    "QuillanBuddy": {"antigravity": "RoninBuddy", "legacy": "RoninBuddy"},
    "SequentialThinking": {"antigravity": "Sequential-Thinking"},
}

# Servers authored by Quillan that don't carry the "Quillan" name prefix.
# SequentialThinking is the stock upstream server and is NOT Quillan's.
QUILLAN_NATIVE = {"ComputerUse"}


def load_jsonc(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^\s*//.*$", "", text)
    return json.loads(text)


def safe_env(env: dict[str, str] | None) -> dict[str, str]:
    return {k: v for k, v in (env or {}).items() if k in ENV_ALLOWLIST}


def build_registry() -> dict[str, dict[str, Any]]:
    live = load_jsonc(LIVE_OPENCODE).get("mcp", {})
    servers: dict[str, dict[str, Any]] = {}
    for name, entry in live.items():
        if entry.get("type") == "remote" or entry.get("url"):
            servers[name] = {"transport": "remote", "url": entry["url"]}
            continue
        cmd = entry.get("command") or []
        if not cmd:
            LOG.warning("skipping %s: empty command", name)
            continue
        servers[name] = {
            "transport": "stdio",
            "command": cmd[0],
            "args": cmd[1:],
            "env": safe_env(entry.get("environment")),
        }
    if "Puppeteer" not in servers and ANTIGRAVITY.exists():
        pup = json.loads(ANTIGRAVITY.read_text(encoding="utf-8"))["mcpServers"].get("Puppeteer")
        if pup:
            servers["Puppeteer"] = {
                "transport": "stdio", "command": pup["command"],
                "args": pup.get("args", []), "env": safe_env(pup.get("env")),
            }
    for name, al in ALIASES.items():
        if name in servers:
            servers[name]["aliases"] = al
    for name, s in servers.items():
        s["origin"] = "quillan" if name in QUILLAN_NATIVE or name.startswith("Quillan") else "third-party"
    return dict(sorted(servers.items()))


def to_mcpservers(servers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, s in servers.items():
        if s["transport"] == "remote":
            out[name] = {"url": s["url"]}
        else:
            out[name] = {"command": s["command"], "args": s["args"], "env": s["env"]}
    return out


def write_json(path: Path, data: Any, dry: bool) -> None:
    if dry:
        LOG.info("[dry-run] would write %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    LOG.info("wrote %s", path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    dry = ap.parse_args().dry_run

    servers = build_registry()
    LOG.info("registry: %d servers (%d stdio, %d remote)", len(servers),
             sum(s["transport"] == "stdio" for s in servers.values()),
             sum(s["transport"] == "remote" for s in servers.values()))
    write_json(REGISTRY, {"version": 1, "servers": servers}, dry)

    mcpservers = to_mcpservers(servers)
    write_json(REPO_OPENCODE, {"mcpServers": mcpservers}, dry)

    idx = json.loads(IDX_MCP.read_text(encoding="utf-8")) if IDX_MCP.exists() else {}
    idx["mcpServers"] = mcpservers
    if isinstance(idx.get("mcp"), dict):
        idx["mcp"] = mcpservers
    elif "mcp" in idx:
        idx["mcp"] = sorted(mcpservers)
    write_json(IDX_MCP, idx, dry)
    return 0


if __name__ == "__main__":
    sys.exit(main())

