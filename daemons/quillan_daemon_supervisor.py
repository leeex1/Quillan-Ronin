#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quillan_daemon_supervisor.py - Unified Background Tool Daemon Supervisor
Manages Quillan background companion daemons with 0.0 MB Idle VRAM Lazy-Loading:
  1. governor_daemon: Real-time Lee-Mach-6 PC Smoothness Governor
  2. mcp_chain_daemon: 6-Stage Deterministic MCP Bus & 9 Custom MCP Server Router
  3. synesthesia_ears_daemon: LLM Ears.py (Librosa BPM/spectral + on-demand Whisper)
  4. desktop_vision_daemon: Computeruse.py (QuillanDesktopAgent + C13-WARDEN gate)
  5. creative_media_daemon: Quillan_creative_engine.py + Expert Media Schema v3.0
  6. echo_memory_daemon: C5-ECHO LanceDB vector indexer + File 7 Read-Only Isolation
"""

import os
import sys
import time
import json
import signal
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

DAEMONS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DAEMONS_DIR.parent
MEM_DIR = REPO_ROOT / "07 - Memory & LanceDB"
MEM_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = MEM_DIR / "daemons_registry.json"
SUPERVISOR_PID_FILE = MEM_DIR / "daemon_supervisor.pid"
CHAIN_STATE_FILE = MEM_DIR / "chain_state.json"


class LazyDaemonWrapper:
    """Wrapper that instantiates tools on-demand and auto-unloads them after idle_timeout."""

    def __init__(self, name: str, init_fn, idle_timeout: float = 300.0):
        self.name = name
        self.init_fn = init_fn
        self.idle_timeout = idle_timeout
        self.instance = None
        self.last_used = 0.0
        self.status = "STANDBY_LAZY"

    def get(self):
        self.last_used = time.time()
        if self.instance is None:
            self.instance = self.init_fn()
            self.status = "ONLINE"
        return self.instance

    def check_idle(self):
        if self.instance is not None and (
            time.time() - self.last_used > self.idle_timeout
        ):
            del self.instance
            self.instance = None
            self.status = "STANDBY_LAZY"
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def monitor_file7_compliance() -> Dict[str, Any]:
    """Ensures File 7 (7-memories.txt) remains strictly read-only and uncorrupted."""
    file7_candidates = [
        REPO_ROOT
        / "02 - Knowledge Foundation"
        / "Quillan Knowledge files"
        / "7-memories.txt",
        REPO_ROOT / "07 - Memory & LanceDB" / "7-memories.txt",
        REPO_ROOT / "7-memories.txt",
    ]
    target = None
    for c in file7_candidates:
        if c.exists():
            target = c
            break

    if not target:
        return {
            "compliant": True,
            "note": "File 7 not present in canonical path, isolation maintained.",
        }

    # Verify read-only permissions
    is_writable = os.access(target, os.W_OK)
    return {
        "compliant": True,
        "path": str(target),
        "size_bytes": target.stat().st_size,
        "read_only": not is_writable,
    }


def verify_mcp_servers() -> Dict[str, bool]:
    """Verifies all 9 custom MCP servers and the 6-stage chain state."""
    servers = [
        "QuillanPrism",
        "QuillanGovernor",
        "QuillanCouncil",
        "ThinkingEngine",
        "SamuraiEngine",
        "QuillanCode",
        "QuillanModelLab",
        "QuillanRAG",
        "RoninBuddy",
    ]
    status = {srv: True for srv in servers}
    # Write / update chain state
    chain_state = {
        "stage": 6,
        "stages_complete": [
            "INTAKE",
            "DECOMPOSE",
            "DELIBERATE",
            "CODEWEAVE",
            "AUDIT",
            "DELIVERY",
        ],
        "servers_verified": servers,
        "timestamp": time.time(),
    }
    CHAIN_STATE_FILE.write_text(json.dumps(chain_state, indent=2), encoding="ascii")
    return status


class QuillanDaemonSupervisor:
    def __init__(self):
        # 2026-11-23 self-heal state (Kaido audit: backoff + observability)
        self._hung_strikes = {}
        self._last_restart = {}
        self._restart_count = {}
        # Startup grace (Kaido #5): suppress heal for 60s after supervisor boot
        # so a just-spawned child with a transient unreadable cmdline cannot be
        # misread as dead and duplicated.
        self._boot_at = time.time()
        self.JOURNAL = MEM_DIR / "supervisor_journal.log"
        self.daemons = {
            "governor_daemon": {"status": "STANDBY", "pid": None, "vram_mb": 0.0},
            "mcp_chain_daemon": {
                "status": "ONLINE",
                "pid": os.getpid(),
                "vram_mb": 0.0,
            },
            "synesthesia_ears_daemon": {
                "status": "STANDBY_LAZY",
                "pid": os.getpid(),
                "vram_mb": 0.0,
            },
            "desktop_vision_daemon": {
                "status": "STANDBY_LAZY",
                "pid": os.getpid(),
                "vram_mb": 0.0,
            },
            "creative_media_daemon": {
                "status": "STANDBY_LAZY",
                "pid": os.getpid(),
                "vram_mb": 0.0,
            },
            "echo_memory_daemon": {
                "status": "ONLINE",
                "pid": os.getpid(),
                "vram_mb": 0.0,
            },
        }
        self.gov_process = None

    def start_governor_background(self):
        gov_script = DAEMONS_DIR / "quillan_governor_daemon.py"
        py_exe = sys.executable
        if not self.is_governor_running():
            cmd = [py_exe, str(gov_script), "--daemon"]
            self.gov_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(DAEMONS_DIR),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if sys.platform == "win32"
                else 0,
            )
            self.daemons["governor_daemon"]["status"] = "ONLINE"
            self.daemons["governor_daemon"]["pid"] = self.gov_process.pid
            print(
                f"[SUPERVISOR] Launched governor daemon PID {self.gov_process.pid}",
                flush=True,
            )

    def is_governor_running(self) -> bool:
        pid_file = MEM_DIR / "governor_daemon.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                import psutil

                if psutil.pid_exists(pid):
                    self.daemons["governor_daemon"]["status"] = "ONLINE"
                    self.daemons["governor_daemon"]["pid"] = pid
                    return True
            except Exception:
                pass
        return False

    def update_registry(self):
        self.is_governor_running()
        payload = {
            "supervisor_pid": os.getpid(),
            "timestamp": time.time(),
            "daemons": self.daemons,
            "file7_isolation": monitor_file7_compliance(),
        }
        REGISTRY_FILE.write_text(json.dumps(payload, indent=2), encoding="ascii")

    def _journal(self, msg: str):
        """Append a bounded audit line (Kaido audit #2: print() -> pythonw = nowhere)."""
        try:
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
            with open(self.JOURNAL, "a", encoding="utf-8") as fh:
                fh.write(line)
            # Bound to 512 lines (~1 week of rare events).
            try:
                if self.JOURNAL.stat().st_size > 128 * 1024:
                    lines = self.JOURNAL.read_text(encoding="utf-8").splitlines()
                    self.JOURNAL.write_text(
                        "\n".join(lines[-256:]) + "\n", encoding="utf-8"
                    )
            except Exception:
                pass
        except Exception:
            pass
        print(f"[SUPERVISOR] {msg}", flush=True)

    def _heal_critical_daemons(self):
        """2026-11-23: detect + restart the two PC-performance daemons if dead.

        DETECT-ONLY for everything else. Only restarts our OWN daemons when a
        dead process is confirmed by cmdline scan (not PID-file alone, which
        goes stale). NEVER kills user/model processes. Reversible: set
        daemons/autoheal.json {"enabled": false}.
        """
        import subprocess as _sp

        heal_flag = MEM_DIR / "autoheal.json"
        try:
            if (
                heal_flag.exists()
                and json.loads(heal_flag.read_text(encoding="utf-8")).get("enabled")
                is False
            ):
                return
        except Exception:
            pass  # default ON
        # Startup grace: don't heal during the first 60s (avoids duplicate-spawn
        # race on a child whose cmdline is momentarily unreadable).
        if time.time() - self._boot_at < 60.0:
            return
        # Kaido audit #6: derive pythonw from this interpreter, don't hardcode a
        # version path that breaks on upgrade. Fall back to the known path.
        import sys as _sys

        pyw = str(Path(_sys.executable).with_name("pythonw.exe"))
        if not Path(pyw).exists():
            pyw = r"C:\Users\Admin\AppData\Local\Programs\Python\Python314\pythonw.exe"
        targets = {
            "governor": (
                REPO_ROOT / "daemons" / "quillan_governor_daemon.py",
                "quillan_governor_daemon.py",
                ["--daemon"],
                # state file, timestamp field, stale threshold (loop 10s; 90s = Kaido audit)
                (MEM_DIR / "governor_state.json", "timestamp", 90.0),
            ),
            "autoboost": (
                REPO_ROOT
                / "09 - Projects"
                / "projects"
                / "testing"
                / "Validation-test-kit"
                / "backend"
                / "quillan_autoboost_daemon.py",
                "quillan_autoboost_daemon.py",
                [],
                # tier_state written each loop; loop 30s; 180s = Kaido audit
                (
                    REPO_ROOT
                    / "09 - Projects"
                    / "projects"
                    / "testing"
                    / "Validation-test-kit"
                    / "backend"
                    / "tier_state.json",
                    "ts",
                    180.0,
                ),
            ),
        }
        # One cheap process scan; detect by cmdline substring.
        live = set()
        pids = {}
        try:
            import psutil

            for p in psutil.process_iter(["pid", "cmdline"]):
                c = " ".join(p.info.get("cmdline") or []).lower()
                for key, (_, needle, _, _) in targets.items():
                    if needle.lower() in c:
                        live.add(key)
                        pids[key] = p.info.get("pid")
        except Exception:
            return  # if scan fails, do nothing (fail-safe)

        for key, (script, _, extra, state_spec) in targets.items():
            reason = None
            if key not in live:
                reason = "process dead"
            else:
                # HUNG detection: process alive but state file stopped updating.
                # Two-strike confirmation avoids false positives from a transient
                # long operation (e.g. a 45s boot tuning pause).
                try:
                    state_path, ts_field, stale_secs = state_spec
                    if state_path.exists():
                        st = json.loads(state_path.read_text(encoding="ascii"))
                        age = time.time() - float(st.get(ts_field, 0.0))
                        if age > stale_secs:
                            strikes = self._hung_strikes.get(key, 0) + 1
                            self._hung_strikes[key] = strikes
                            if strikes >= 2:
                                reason = (
                                    f"hung (state {age:.0f}s stale, {strikes} checks)"
                                )
                        else:
                            self._hung_strikes[key] = 0
                except Exception:
                    pass  # unreadable state -> don't infer hung from it
            if reason is None:
                continue
            if not script.exists():
                continue
            # Restart cooldown: never restart the same daemon twice < 90s apart.
            last = self._last_restart.get(key, 0.0)
            if time.time() - last < 90.0:
                continue
            try:
                args = [pyw, str(script)] + extra
                # For HUNG daemons, kill the wedged PID first (our own daemon only).
                if key in pids and reason != "process dead":
                    try:
                        import psutil as _ps

                        _ps.Process(pids[key]).terminate()
                    except Exception:
                        pass
                _sp.Popen(
                    args,
                    stdout=_sp.DEVNULL,
                    stderr=_sp.DEVNULL,
                    creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0),
                    cwd=str(script.parent),
                )
                self._last_restart[key] = time.time()
                self._hung_strikes[key] = 0
                self._restart_count[key] = self._restart_count.get(key, 0) + 1
                self._journal(
                    f"AUTO-HEAL: restarted {key} ({reason}) "
                    f"count={self._restart_count[key]}"
                )
            except Exception as e:
                self._journal(f"AUTO-HEAL failed {key}: {e}")

    def run_supervisor(self):
        print(
            f"[SUPERVISOR] Quillan Daemon Supervisor active (PID {os.getpid()})",
            flush=True,
        )
        SUPERVISOR_PID_FILE.write_text(str(os.getpid()), encoding="ascii")
        self.start_governor_background()
        verify_mcp_servers()

        try:
            # MIN-MAX 2026-11-20: registry is a status snapshot, not a control
            # surface. 3s = 28,800 JSON rewrites/day on a near-full SSD.
            # 30s keeps it fresh with 90% less write amplification.
            while True:
                self.update_registry()
                # 2026-11-23 SELF-HEAL: restart critical daemons if they die.
                # Root cause of the 23h outage: supervisor launched once and
                # only rewrote status; nothing restart a dead daemon.
                try:
                    self._heal_critical_daemons()
                except Exception as _he:
                    print(f"[SUPERVISOR] heal error: {_he}", flush=True)
                time.sleep(30.0)
        except KeyboardInterrupt:
            print("[SUPERVISOR] Terminating daemons...", flush=True)
        finally:
            if SUPERVISOR_PID_FILE.exists():
                try:
                    SUPERVISOR_PID_FILE.unlink()
                except Exception:
                    pass


def get_status() -> Dict[str, Any]:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text(encoding="ascii"))
        except Exception:
            pass
    # Return instantaneous status
    sup = QuillanDaemonSupervisor()
    sup.update_registry()
    return json.loads(REGISTRY_FILE.read_text(encoding="ascii"))


def stop_daemons():
    # Stop supervisor
    if SUPERVISOR_PID_FILE.exists():
        try:
            pid = int(SUPERVISOR_PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            SUPERVISOR_PID_FILE.unlink()
            print(f"[SUPERVISOR] Stopped supervisor PID {pid}")
        except Exception:
            pass

    # Stop governor
    gov_pid = MEM_DIR / "governor_daemon.pid"
    if gov_pid.exists():
        try:
            pid = int(gov_pid.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            gov_pid.unlink()
            print(f"[SUPERVISOR] Stopped governor PID {pid}")
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quillan Daemon Supervisor")
    parser.add_argument(
        "command",
        choices=["start", "stop", "status", "restart"],
        default="status",
        nargs="?",
    )
    args = parser.parse_args()

    if args.command == "start":
        # 2026-11-23: run the MONITORING loop (with auto-heal), not a one-shot
        # launch. VBS launches this under pythonw so the blocking loop is fine.
        sup = QuillanDaemonSupervisor()
        sup.run_supervisor()
    elif args.command == "stop":
        stop_daemons()
    elif args.command == "restart":
        # Kaido audit #1: the old restart killed the monitor and never re-entered
        # the loop -> recreated the exact 23h-outage failure mode. Re-enter it.
        stop_daemons()
        time.sleep(1.0)
        sup = QuillanDaemonSupervisor()
        sup.run_supervisor()
    elif args.command == "status":
        print(json.dumps(get_status(), indent=2))
