"""Targeted prompt/doc sync (idempotent, backs up before writing).

1. Fix the glued 'arbitration.User Query' line in the canonical OpenCode agent file.
2. Make the canonical .opencode/agent/quillan.md the source for the other two
   quillan.md copies (it is a strict superset of both).
3. Correct the stale '19 MCP servers route through the chain bus' claims.
4. Add a native `task` delegation block + task permission to the AGENTS.md files.
"""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
import sys
import time
from pathlib import Path

LOG = logging.getLogger("prompt-sync")
REPO = Path(r"C:\02_QUILLAN")
HOME = Path.home()
CANON = REPO / ".opencode" / "agent" / "quillan.md"
COPIES = [REPO / "quillan.md", HOME / ".config" / "opencode" / "agent" / "quillan.md"]
AGENTS = [REPO / "AGENTS.md", HOME / ".config" / "opencode" / "AGENTS.md"]
RULES = REPO / ".antigravity" / "rules.md"
BACKUP = REPO / "scripts" / "_backups" / (time.strftime("%Y%m%d-%H%M%S") + "-prompts")

T_BEGIN, T_END = "<!-- TASK-DELEGATION:BEGIN -->", "<!-- TASK-DELEGATION:END -->"
TASK_BLOCK = f"""{T_BEGIN}
## Council Delegation (OpenCode native `task` tool)
Delegate to a Council subagent with `task(subagent_type="<id>", prompt="...")`, e.g.
`c1-astra` (patterns), `c2-vir` (ethics), `c7-logos` (logic), `c8-metasynth` (synthesis),
`c13-warden` (security), `c33-typist` (docs), `c34-predator` (adversarial review).
Swarms are named `<council-id>-swarm`. Subagents run in isolated context and return findings
for final arbitration by the root agent.
{T_END}"""

OLD_19 = "All 19 MCP servers route through `09 - Projects/mcp/mcp_chain_bus.py`."
NEW_19 = ("35 MCP servers are registered (19 Quillan-native + ComputerUse + 15 third-party; see "
          "`09 - Projects/mcp/mcp_registry.json`). `09 - Projects/mcp/mcp_chain_bus.py` currently "
          "routes 9 Quillan servers through the 6-stage pipeline.")
OLD_RULE = "**19 Chained MCPs**: Driven via"
NEW_RULE = "**35 registered MCPs (9 chained by the bus)**: Chain driven via"


def backup(p: Path) -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, BACKUP / f"{hashlib.sha1(str(p).encode()).hexdigest()[:8]}_{p.name}")


def write(p: Path, text: str) -> None:
    backup(p)
    p.write_bytes(text.encode("utf-8"))


def rd(p: Path) -> str:
    return p.read_bytes().decode("utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    text = rd(CANON)
    fixed = text.replace("final arbitration.User Query", "final arbitration.\r\n\r\nUser Query")
    if fixed != text:
        write(CANON, fixed)
        LOG.info("canonical: split glued 'User Query' line")
    canon_bytes = rd(CANON)

    for p in COPIES:
        if rd(p) != canon_bytes:
            write(p, canon_bytes)
            LOG.info("synced from canonical: %s", p)
        else:
            LOG.info("already identical: %s", p)

    for p in AGENTS:
        t = rd(p)
        nl = "\r\n" if "\r\n" in t else "\n"
        new = t.replace(OLD_19, NEW_19)
        if T_BEGIN not in new:
            m = re.match(r"---\r?\n.*?\r?\n---\r?\n", new, flags=re.S)
            pos = m.end() if m else 0
            new = new[:pos] + nl + TASK_BLOCK.replace("\n", nl) + nl + new[pos:]
        if p.parent.name == "opencode" and re.search(r"(?m)^steps: 200\r?$", new):
            new = re.sub(r"(?m)^steps: 200(\r?)$", r"steps: 1000\1", new, count=1)
            if not re.search(r"(?m)^  task:", new):
                new = re.sub(r"(?m)^(  todowrite: allow\r?\n)", lambda m: m.group(1) + '  task:' + nl + '    "*": allow' + nl, new, count=1)
        if new != t:
            write(p, new)
            LOG.info("patched: %s", p)
        else:
            LOG.info("unchanged: %s", p)

    t = rd(RULES)
    new = t.replace(OLD_RULE, NEW_RULE)
    if new != t:
        write(RULES, new)
        LOG.info("patched: %s", RULES)
    else:
        LOG.info("unchanged: %s", RULES)
    return 0


if __name__ == "__main__":
    sys.exit(main())

