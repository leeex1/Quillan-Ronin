#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — AUTONOMOUS CHAT HARVESTER DAEMON
===========================================================
Monitors `.gemini/tmp` chat session logs in the background, extracts
new high-quality Q&A and reasoning pairs, and stages them for upcoming training runs.
Runs at low priority with minimal memory footprint.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

REPO_ROOT    = Path(r"C:\02_QUILLAN")
SESSION_DIRS = [
    REPO_ROOT / ".gemini" / "tmp" / "admin" / "chats",
    REPO_ROOT / ".gemini" / "tmp" / "system32" / "chats",
]
OUT_FILE     = REPO_ROOT / "training_data" / "harvested_conversations_gold.jsonl"
STATE_FILE   = REPO_ROOT / "scripts" / "harvester_state.json"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [HARVEST-DAEMON] %(message)s")
LOGGER = logging.getLogger("quillan.daemon.harvester")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"processed_files": [], "total_harvested": 0}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def harvest_session_file(path: Path) -> int:
    """Streamingly parses a session JSONL file and extracts prompt/response pairs."""
    LOGGER.info("Mining chat session: %s (%.1f MB)", path.name, path.stat().st_size / (1024 * 1024))
    harvested = 0
    seen_prompts = set()

    with open(path, "r", encoding="utf-8", errors="ignore") as f, \
         open(OUT_FILE, "a", encoding="utf-8") as out_f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                # Look for user query and assistant response structures
                q = d.get("user_query", d.get("prompt", d.get("question", ""))).strip()
                r = d.get("assistant_response", d.get("response", d.get("output", ""))).strip()
                
                # Check messages array format
                if not q and "messages" in d and isinstance(d["messages"], list):
                    msgs = d["messages"]
                    for idx in range(len(msgs) - 1):
                        if msgs[idx].get("role") == "user" and msgs[idx+1].get("role") == "assistant":
                            q = msgs[idx].get("content", "").strip()
                            r = msgs[idx+1].get("content", "").strip()
                            break

                if len(q) >= 10 and len(r) >= 60:
                    q_prefix = q[:60]
                    if q_prefix not in seen_prompts:
                        seen_prompts.add(q_prefix)
                        record = {"question": q, "response": r, "source": path.name}
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        harvested += 1
                        if harvested >= 2000:
                            break
            except Exception:
                pass

    LOGGER.info("Extracted %d valid dialogue pairs from %s", harvested, path.name)
    return harvested


def main():
    LOGGER.info("Harvester Daemon active. Monitoring session directories...")
    state = load_state()

    for s_dir in SESSION_DIRS:
        if not s_dir.exists():
            continue
        for session_file in sorted(s_dir.glob("*.jsonl")):
            if str(session_file) in state["processed_files"]:
                continue

            # Process file
            count = harvest_session_file(session_file)
            state["total_harvested"] += count
            state["processed_files"].append(str(session_file))
            save_state(state)
            time.sleep(2)  # Yield CPU

    LOGGER.info("Initial pass complete. Total harvested samples: %d", state["total_harvested"])


if __name__ == "__main__":
    main()
