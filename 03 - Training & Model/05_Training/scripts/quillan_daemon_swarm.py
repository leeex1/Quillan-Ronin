#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — AUTONOMOUS DAEMON SWARM ORCHESTRATOR
==============================================================
Manages the concurrent background daemon swarm:
  1. Trainer Watchdog Daemon  -> train_frontier_capability.py (7100 steps)
  2. Auto-Evaluator Daemon    -> daemon_evaluator.py (polls new best checkpoints)
  3. Data Harvester Daemon    -> daemon_harvester.py (mines background chat logs)

Maintains system stability with process priority throttling and automatic telemetry updates.
"""

import sys
import os
import subprocess
import time
import json
import logging
from pathlib import Path

SCRIPTS_DIR = Path(r"C:\02_QUILLAN\scripts")
STATUS_FILE = SCRIPTS_DIR / "swarm_dashboard.json"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [SWARM-ORCHESTRATOR] %(message)s")
LOGGER = logging.getLogger("quillan.daemon.swarm")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DAEMONS = {
    "trainer_watchdog": {
        "script": str(SCRIPTS_DIR / "training_watchdog.py"),
        "proc": None,
        "description": "Primary 7100-step training loop with self-healing auto-restart"
    },
    "auto_evaluator": {
        "script": str(SCRIPTS_DIR / "daemon_evaluator.py"),
        "proc": None,
        "description": "Monitors new best checkpoints and runs automated benchmark probes"
    },
    "data_harvester": {
        "script": str(SCRIPTS_DIR / "daemon_harvester.py"),
        "proc": None,
        "description": "Mines real conversation logs and stages new gold training pairs"
    }
}


def write_dashboard():
    dashboard = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "daemons": {}
    }
    for name, info in DAEMONS.items():
        is_alive = info["proc"] is not None and info["proc"].poll() is None
        pid = info["proc"].pid if is_alive else None
        dashboard["daemons"][name] = {
            "status": "RUNNING" if is_alive else "STOPPED",
            "pid": pid,
            "description": info["description"]
        }

    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2)


def start_daemon(name: str):
    info = DAEMONS[name]
    LOGGER.info("Starting daemon [%s]: %s", name, Path(info["script"]).name)
    
    # Windows low priority flag: BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    creation_flags = 0x00004000 if sys.platform == "win32" else 0

    proc = subprocess.Popen(
        [sys.executable, "-u", info["script"]],
        cwd=str(SCRIPTS_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags
    )
    info["proc"] = proc
    LOGGER.info("Daemon [%s] active with PID %d", name, proc.pid)


def main():
    LOGGER.info("=" * 65)
    LOGGER.info("   👑 QUILLAN-RONIN v5.3.1 — AUTONOMOUS DAEMON SWARM ACTIVE")
    LOGGER.info("=" * 65)

    # 1. Start all daemons in sequence
    for name in ["trainer_watchdog", "auto_evaluator", "data_harvester"]:
        start_daemon(name)
        time.sleep(3)

    write_dashboard()
    LOGGER.info("All daemons deployed. Swarm status written to %s", STATUS_FILE.name)

    # 2. Monitor and maintain swarm health
    try:
        while True:
            for name, info in DAEMONS.items():
                if info["proc"] is None or info["proc"].poll() is not None:
                    # Trainer watchdog should always be kept alive; harvester runs one-shot or periodic
                    if name == "trainer_watchdog" or name == "auto_evaluator":
                        LOGGER.warning("Daemon [%s] stopped. Restarting...", name)
                        start_daemon(name)

            write_dashboard()
            time.sleep(30)
    except KeyboardInterrupt:
        LOGGER.info("Shutting down daemon swarm...")
        for name, info in DAEMONS.items():
            if info["proc"] and info["proc"].poll() is None:
                info["proc"].terminate()
        LOGGER.info("All daemons stopped cleanly.")


if __name__ == "__main__":
    main()
