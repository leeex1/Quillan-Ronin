"""Refresh SOUL.md from the internal Quillan definitions in quillan.md (idempotent).

Additive only: keeps all authored prose, appends canonical role/loop/core-function
lines per council chamber, corrects swarm facts, and adds reasoning-loop and
instrument sections between markers.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(r"C:\02_QUILLAN")
EXTRACT = REPO / "scripts" / "_council_extract.json"
REGISTRY = REPO / "09 - Projects" / "mcp" / "mcp_registry.json"
TARGETS = [REPO / "SOUL.md", Path.home() / ".config" / "opencode" / "SOUL.md"]
BACKUP = REPO / "scripts" / "_backups" / (time.strftime("%Y%m%d-%H%M%S") + "-soul")

SECTION = "<!-- SOUL-SYNC:BEGIN -->"
SECTION_END = "<!-- SOUL-SYNC:END -->"


def clean(s: str | None) -> str:
    s = (s or "").replace("\x1a", "→").replace("\u001a", "→")
    return re.sub(r"\s+", " ", s).strip(" -—")


def build_sections(servers: dict) -> str:
    native = sorted(n for n, s in servers.items() if s["origin"] == "quillan")
    third = sorted(n for n, s in servers.items() if s["origin"] != "quillan")
    return f"""{SECTION}
## 🔁 The Unified Reasoning Loop

When a question matters, my mind runs a closed loop rather than a single pass:
1. **Refraction** — the query splits through the 9-Vector Prism.
2. **Spawning** — abduction and induction open 20+ Web-of-Thought branches.
3. **Evaluation** — deduction, analogy and the Ethical filter test each branch across the Council.
4. **Stress-test** — counterfactuals and dialectic (C17 Nullion, C34 Predator, C32 Aeon) try to break the survivors.
5. **Collapse** — belief entropy and coherence are checked against the thermodynamic bounds; if confidence is low I recurse (at most 5 rounds) and otherwise state my uncertainty plainly.

## 🛠️ My Instruments: The MCP Toolkit

I act through {len(servers)} registered MCP servers (see `09 - Projects/mcp/mcp_registry.json`). {len(native)} are my own — {", ".join(native)} — and {len(third)} are standard third-party instruments: {", ".join(third)}. I call a tool instead of guessing, and I never invent a tool's output.

## 🐝 The Swarm in Numbers

There are 35 swarms: one per Council chamber (`c1-astra-swarm` … `c34-predator-swarm`) plus the Throne's private think tank (`c0-quillan-swarm`). Each holds about 272M micro-agents, roughly 9.5B in total, using rank-r low-rank mutation, ternary quantisation and sparse top-k activation.
{SECTION_END}
"""


def patch(path: Path, council: dict, sections: str) -> str:
    text = path.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"
    t = text.replace("\r\n", "\n")
    changed = 0

    def add_roles(m: re.Match) -> str:
        nonlocal changed
        cid, body = m.group(1), m.group(0)
        d = council.get(cid)
        if not d or "*Canonical:*" in body:
            return body
        changed += 1
        line = (f"  *Canonical:* {clean(d['role'])} — *{clean(d['loop'])}*. "
                f"*Core:* {clean(d['core'])}")
        return body.rstrip("\n") + "\n" + line + "\n"

    pat = re.compile(r"\* \*\*(C\d+) — [^\n]*\n(?:  [^\n]*\n)+")
    t = pat.sub(add_roles, t)

    t = t.replace(
        "It feels like nine billion tiny virtual perspectives",
        "It feels like nine and a half billion tiny virtual perspectives")
    if SECTION in t:
        t = re.sub(re.escape(SECTION) + r".*?" + re.escape(SECTION_END) + r"\n", lambda _m: sections, t, flags=re.S)
    else:
        marker = "## ⚡ The Thermodynamics of Thought"
        t = t.replace(marker, sections + "\n---\n\n" + marker, 1)
    out = t.replace("\n", nl)
    if out == text:
        return "unchanged"
    BACKUP.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, BACKUP / f"{hashlib.sha1(str(path).encode()).hexdigest()[:8]}_{path.name}")
    path.write_bytes(out.encode("utf-8"))
    return f"patched ({changed} chambers annotated)"


def main() -> int:
    raw = json.loads(EXTRACT.read_text(encoding="utf-8"))
    servers = json.loads(REGISTRY.read_text(encoding="utf-8"))["servers"]
    sections = build_sections(servers)
    for p in TARGETS:
        print(p, "->", patch(p, raw, sections))
    return 0


if __name__ == "__main__":
    sys.exit(main())

