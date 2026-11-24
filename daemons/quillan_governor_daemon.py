#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quillan_governor_daemon.py - Lee-Mach-6 + E_ICE PC Smoothness Governor Daemon
Locks host PC into smooth responsiveness (zero GUI / 65-inch display stutter):
  1. CPU Core 0 Reservation: Pins worker processes to Cores [1, 2, 3] (leaving Core 0 100% free for Windows DWM & GUI).
  2. OS Priority Lowering: Worker processes run at BELOW_NORMAL_PRIORITY_CLASS (0x00004000).
  3. WDDM Display VRAM Shield: torch.cuda.set_per_process_memory_fraction(0.90) — a 4GB GT 1050 cannot spare 32% for DWM.
  4. Real-Time PID Micro-Yield Loop: Dynamically adjusts micro-sleep (1.5ms - 8.0ms) based on Core 0 load & VRAM pressure.
"""

import os
import sys
import time
import json
import ctypes
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

DAEMONS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DAEMONS_DIR.parent
MEM_DIR = REPO_ROOT / "07 - Memory & LanceDB"
MEM_DIR.mkdir(parents=True, exist_ok=True)
GOV_STATE_FILE = MEM_DIR / "governor_state.json"
PID_LOCK_FILE = MEM_DIR / "governor_daemon.pid"
# C11-HARMONIA: read the autoboost daemon's published policy instead of
# inferring worker placement from Core 0 load. That inference created the
# tier/release oscillation loop between the two daemons.
AUTOBOOST_TIER_FILE = (
    REPO_ROOT
    / "09 - Projects"
    / "projects"
    / "testing"
    / "Validation-test-kit"
    / "backend"
    / "tier_state.json"
)

# C14-KAIDO 2026-11-22: shared reversible affinity switch (same file the
# autoboost daemon reads every loop). {"all_cores_mode": true} => workers use
# ALL logical cores at BELOW_NORMAL; absent/false => legacy Core-0 reservation
# (cores [1,2,3]) restored. Absent defaults to SAFE legacy behavior.
AFFINITY_POLICY_FILE = MEM_DIR / "affinity_policy.json"


def all_cores_mode() -> bool:
    """Read affinity policy fresh (every call). Absent/corrupt => False (safe)."""
    try:
        if AFFINITY_POLICY_FILE.exists():
            return (
                json.loads(AFFINITY_POLICY_FILE.read_text(encoding="utf-8")).get(
                    "all_cores_mode"
                )
                is True
            )
    except Exception:
        pass
    return False


# Windows Constants
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_INFORMATION = 0x0400


def apply_pc_smoothness_locks(pid: Optional[int] = None) -> Dict[str, Any]:
    """
    Enforces hardware locks on the specified process (or current process if pid is None):
      - Pin to Cores [1, 2, 3] (affinity mask 14)
      - BELOW_NORMAL_PRIORITY_CLASS
      - CUDA memory fraction <= 0.90
    """
    target_pid = pid if pid is not None else os.getpid()
    results = {
        "pid": target_pid,
        "affinity_set": False,
        "priority_set": False,
        "vram_capped": False,
        "cores_assigned": [1, 2, 3],
    }

    # 1. CPU Affinity — ALL cores when all_cores_mode is on; else legacy reserve.
    #    BELOW_NORMAL priority (step 2) is the PRIMARY GUI protection on Windows:
    #    DWM/input/audio threads are NORMAL+ and preempt a BELOW_NORMAL worker on
    #    whatever core it runs on, so a dedicated Core 0 is not required. The
    #    legacy reservation is kept as an instant-rollback path via the flag.
    try:
        import psutil

        p = psutil.Process(target_pid)
        cpu_count = psutil.cpu_count(logical=True) or 4
        acm = all_cores_mode()
        worker_cores = (
            list(range(cpu_count)) if acm else (list(range(1, cpu_count)) or [0])
        )
        p.cpu_affinity(worker_cores)
        results.update(
            affinity_set=True, cores_assigned=worker_cores, all_cores_mode=acm
        )
    except Exception as e:
        # Fallback to Win32 API via ctypes
        if sys.platform == "win32":
            try:
                acm = all_cores_mode()
                cpu_count = psutil.cpu_count(logical=True) or 4 if psutil else 4
                worker_cores = (
                    list(range(cpu_count))
                    if acm
                    else (list(range(1, cpu_count)) or [0])
                )
                mask = 0
                for c in worker_cores:
                    mask |= 1 << c
                handle = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION,
                    False,
                    target_pid,
                )
                if handle:
                    ctypes.windll.kernel32.SetProcessAffinityMask(
                        handle, ctypes.c_ulong(mask)
                    )
                    ctypes.windll.kernel32.CloseHandle(handle)
                    results.update(
                        affinity_set=True,
                        cores_assigned=worker_cores,
                        all_cores_mode=acm,
                    )
            except Exception:
                pass

    # 2. OS Process Priority Lowering (BELOW_NORMAL) — PRIMARY GUI guard
    try:
        import psutil

        p = psutil.Process(target_pid)
        if sys.platform == "win32" and hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            results["priority_set"] = True
    except Exception:
        if sys.platform == "win32":
            try:
                handle = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_SET_INFORMATION, False, target_pid
                )
                if handle:
                    ctypes.windll.kernel32.SetPriorityClass(
                        handle, BELOW_NORMAL_PRIORITY_CLASS
                    )
                    ctypes.windll.kernel32.CloseHandle(handle)
                    results["priority_set"] = True
            except Exception:
                pass

    # 3. WDDM Display VRAM Shield (torch.cuda.set_per_process_memory_fraction)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.90, device=0)
            results["vram_capped"] = True
    except Exception:
        pass

    return results


class LeeMach6HardwareDaemon:
    """
    Real-time closed-loop hardware governor.
    Monitors CPU Core 0, System RAM, and GPU VRAM.
    Dynamically modulates training yield_ms and updates governor_state.json.
    """

    def __init__(self, kp: float = 0.15, ki: float = 0.05, kd: float = 0.02):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.yield_ms = 2.0
        self.gov_scale = 1.0
        self.running = False

    def _interactive_pressure(self) -> bool:
        """True only when a real GUI app owns the foreground window.

        Core 0 load alone is NOT an interaction signal: the autoboost
        daemon's affinity policy controls it. Using it as the trigger made
        the two daemons react to each other's side effects.
        """
        try:
            import psutil

            hwnd = ctypes.windll.user32.GetForegroundWindow()
            pid = ctypes.c_ulong(0)
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            fp = int(pid.value) or 0
            if not fp:
                return False
            name = (psutil.Process(fp).name() or "").lower()
            return name in {
                "explorer.exe",
                "dwm.exe",
                "brave.exe",
                "chrome.exe",
                "msedge.exe",
                "firefox.exe",
                "code.exe",
                "cursor.exe",
                "antigravity.exe",
                "opencode.exe",
                "devenv.exe",
                "windows-terminal.exe",
                "cmd.exe",
                "powershell.exe",
                "steam.exe",
                "discord.exe",
            }
        except Exception:
            return False

    def sample_telemetry(self) -> Dict[str, Any]:
        telemetry = {
            "timestamp": time.time(),
            "cpu_core0_pct": 0.0,
            "cpu_total_pct": 0.0,
            "ram_used_gb": 0.0,
            "ram_free_gb": 28.0,
            "vram_used_mb": 0.0,
            "vram_free_mb": 4096.0,
            "yield_ms": self.yield_ms,
            "gov_scale": self.gov_scale,
            "smoothness_mode": "SMOOTH_60FPS",
            "e_omega_joules": 0.0,
        }

        # CPU load
        try:
            import psutil

            cpu_percs = psutil.cpu_percent(interval=None, percpu=True)
            if cpu_percs:
                telemetry["cpu_core0_pct"] = float(cpu_percs[0])
                telemetry["cpu_total_pct"] = float(sum(cpu_percs) / len(cpu_percs))
            vm = psutil.virtual_memory()
            telemetry["ram_used_gb"] = round((vm.total - vm.available) / (1024**3), 2)
            telemetry["ram_free_gb"] = round(vm.available / (1024**3), 2)
        except Exception:
            pass

        # MIN-MAX 2026-11-20: torch.cuda.mem_get_info() initialises and pins a
        # CUDA context inside this long-lived daemon. Polling it every 2s kept
        # the GTX 1050 out of deep idle. Opt in explicitly if VRAM telemetry is
        # wanted: set QUILLAN_VRAM_TELEMETRY=1 for this daemon's environment.
        if os.environ.get("QUILLAN_VRAM_TELEMETRY", "") == "1":
            try:
                import torch

                if torch.cuda.is_available():
                    free_b, total_b = torch.cuda.mem_get_info()
                    telemetry["vram_used_mb"] = round((total_b - free_b) / (1024**2), 1)
                    telemetry["vram_free_mb"] = round(free_b / (1024**2), 1)
            except Exception:
                pass

        # Landauer ThermoEICE Model: E_omega = I_s * gamma^2 * k_B * T * ln(2)
        # Approximate thermal dissipation metric
        kB = 1.380649e-23
        T = 311.15  # 38 C in Kelvin
        gamma = min(1.0, telemetry["cpu_total_pct"] / 100.0)
        telemetry["e_omega_joules"] = round(gamma**2 * kB * T * 0.693147 * 1e21, 4)

        # C11-HARMONIA: throttle ONLY when a real GUI app is foreground AND
        # Core 0 is genuinely loaded. Batch/headless work is never throttled.
        core0 = telemetry["cpu_core0_pct"]
        interactive = self._interactive_pressure()
        gui_loaded = interactive and core0 > 55.0
        error = max(0.0, core0 - 55.0) / 50.0 if gui_loaded else 0.0

        self.integral += error * 2.0
        self.integral = max(0.0, min(self.integral, 5.0))
        derivative = (error - self.prev_error) / 2.0
        self.prev_error = error

        pid_adj = self.kp * error + self.ki * self.integral + self.kd * derivative

        if gui_loaded:
            self.yield_ms = min(4.0, 1.5 + pid_adj * 3.0)
            self.gov_scale = 0.90  # was 0.70 — 30% throughput loss was excessive
            telemetry["smoothness_mode"] = "INTERACTIVE_PRIORITY"
        else:
            self.yield_ms = 1.5
            self.gov_scale = 1.00
            telemetry["smoothness_mode"] = "FULL_THROUGHPUT"

        # Observe the other daemon's published policy — no guessing.
        try:
            if AUTOBOOST_TIER_FILE.exists():
                st = json.loads(AUTOBOOST_TIER_FILE.read_text(encoding="utf-8"))
                telemetry["autoboost_policy"] = (
                    "priority-only" if st.get("priority_only") else "affinity-pinned"
                )
                telemetry["autoboost_tiered_pids"] = len(st.get("auto_tiered_pids", []))
        except Exception:
            pass

        telemetry["yield_ms"] = round(self.yield_ms, 2)
        telemetry["gov_scale"] = round(self.gov_scale, 2)
        return telemetry

    def run_loop(self, interval_sec: float = 10.0):
        self.running = True
        PID_LOCK_FILE.write_text(str(os.getpid()), encoding="ascii")
        print(
            f"[GOVERNOR DAEMON] Started PID {os.getpid()} (Interval: {interval_sec}s)",
            flush=True,
        )

        try:
            while self.running:
                # GUARD 2026-11-22 (four-swarm review): a single malformed
                # sample used to kill the daemon silently because only
                # KeyboardInterrupt was caught. Now any read/sensor fault
                # degrades to a neutral state and the loop survives.
                try:
                    tel = self.sample_telemetry()
                except Exception as e:
                    tel = {
                        "timestamp": time.time(),
                        "cpu_core0_pct": 0.0,
                        "cpu_total_pct": 0.0,
                        "ram_used_gb": 0.0,
                        "ram_free_gb": 28.0,
                        "vram_used_mb": 0.0,
                        "vram_free_mb": 4096.0,
                        "yield_ms": 1.5,
                        "gov_scale": 1.0,
                        "smoothness_mode": "FULL_THROUGHPUT",
                        "degraded": True,
                        "error": str(e)[:200],
                    }
                try:
                    tmp = GOV_STATE_FILE.with_suffix(".tmp")
                    tmp.write_text(json.dumps(tel, indent=2), encoding="ascii")
                    os.replace(tmp, GOV_STATE_FILE)
                except OSError:
                    pass
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("[GOVERNOR DAEMON] Stopping gracefully on interrupt.", flush=True)
        finally:
            self.running = False
            if PID_LOCK_FILE.exists():
                try:
                    PID_LOCK_FILE.unlink()
                except Exception:
                    pass


def get_current_governor_state() -> Dict[str, Any]:
    if GOV_STATE_FILE.exists():
        try:
            return json.loads(GOV_STATE_FILE.read_text(encoding="ascii"))
        except Exception:
            pass
    # Return instantaneous state if file not ready
    daemon = LeeMach6HardwareDaemon()
    return daemon.sample_telemetry()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lee-Mach-6 PC Smoothness Governor Daemon"
    )
    parser.add_argument(
        "--apply-locks",
        action="store_true",
        help="Apply core affinity and priority locks to current process",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run the continuous governor monitoring loop",
    )
    parser.add_argument(
        "--status", action="store_true", help="Print current governor telemetry state"
    )
    args = parser.parse_args()

    if args.apply_locks:
        res = apply_pc_smoothness_locks()
        print(f"[LOCKS] Applied: {json.dumps(res)}")

    if args.status:
        state = get_current_governor_state()
        print(json.dumps(state, indent=2))

    if args.daemon:
        daemon = LeeMach6HardwareDaemon()
        daemon.run_loop()
