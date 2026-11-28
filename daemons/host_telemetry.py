#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
host_telemetry.py -- Whole-box telemetry snapshot for the daemon pair.

ONE shared, read-only view of the machine that BOTH daemons can consume:

  governor daemon  -> senses this, computes the control decision (QPS/QSSR)
  autoboost daemon -> applies the decision (affinity / priority / power)

DESIGN INVARIANTS
-----------------
- Process-agnostic: no model names, no eval names, no special cases.
- No torch import, no torch.cuda call. GPU sampled via nvidia-smi subprocess
  so the telemetry daemon never pins a CUDA context (that was a past bug).
- Reading this module has no side effects. No daemon, no timer, no loop.
- All samples are bounded with timeouts; a failed sensor yields a neutral
  value, never an exception that could kill a daemon.
- Atomic write (temp + replace) so readers never see a partial snapshot.

USAGE:
    python host_telemetry.py --once      # write snapshot, print summary
    python host_telemetry.py --no-write  # print only
    python host_telemetry.py --status    # show last snapshot age
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import psutil

    PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL = False

_REPO = Path(r"F:\02_QUILLAN")
_SNAPSHOT = _REPO / "07 - Memory & LanceDB" / "system_telemetry.json"

_GPU_TIMEOUT = 3  # seconds; nvidia-smi must never stall a daemon


def _safe(fn, default):  # type: ignore[no-untyped-def]
    try:
        return fn()
    except Exception:
        return default


def _gpu_snapshot() -> dict[str, Any]:
    """Sample GPU via nvidia-smi subprocess. NO torch, NO CUDA context."""
    out: dict[str, Any] = {
        "available": False,
        "utilization_pct": 0.0,
        "memory_util_pct": 0.0,
        "vram_used_mb": 0.0,
        "vram_total_mb": 0.0,
        "temperature_c": 0.0,
        "clocks_sm_mhz": 0,
        "power_w": 0.0,
    }
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,"
                "memory.total,temperature.gpu,clocks.sm,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_GPU_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if r.returncode != 0:
            return out
        line = (r.stdout or "").strip().splitlines()
        if not line:
            return out
        parts = [p.strip() for p in line[0].split(",")]
        if len(parts) < 6:
            return out

        def f(i: int) -> float:
            try:
                return float(parts[i])
            except (ValueError, IndexError):
                return 0.0

        out.update(
            {
                "available": True,
                "utilization_pct": f(0),
                "memory_util_pct": f(1),
                "vram_used_mb": f(2),
                "vram_total_mb": f(3),
                "temperature_c": f(4),
                "clocks_sm_mhz": int(f(5)),
                "power_w": f(6) if len(parts) > 6 else 0.0,
            }
        )
    except Exception:
        pass
    return out


def _cpu_snapshot() -> dict[str, Any]:
    if not PSUTIL:
        return {"per_core_pct": [], "total_pct": 0.0, "logical_cores": 0}
    # FIX 2026-11-22: cpu_percent(interval=None) returns 0.0 on the FIRST
    # call in a fresh process. A one-shot scheduled telemetry run would
    # therefore publish "all cores idle" forever. Prime it with interval,
    # then take the real sample.
    _safe(lambda: psutil.cpu_percent(interval=0.1, percpu=True), None)
    per = _safe(lambda: psutil.cpu_percent(interval=0.1, percpu=True), []) or []
    return {
        "per_core_pct": [round(float(v), 1) for v in per],
        "total_pct": round(sum(per) / len(per), 1) if per else 0.0,
        "logical_cores": psutil.cpu_count(logical=True) or 0,
        "physical_cores": _safe(lambda: psutil.cpu_count(logical=False), 0),
    }


def _memory_snapshot() -> dict[str, Any]:
    if not PSUTIL:
        return {}
    vm = _safe(lambda: psutil.virtual_memory(), None)
    if vm is None:
        return {}
    gb = 1024**3
    return {
        "total_gb": round(vm.total / gb, 2),
        "used_gb": round(vm.used / gb, 2),
        "available_gb": round(vm.available / gb, 2),
        "used_pct": float(vm.percent),
    }


def _commit_snapshot() -> dict[str, Any]:
    if not PSUTIL:
        return {}
    sw = _safe(lambda: psutil.swap_memory(), None)
    if sw is None:
        return {}
    gb = 1024**3
    return {
        "commit_total_gb": round(sw.total / gb, 2),
        "commit_used_gb": round(sw.used / gb, 2),
        "commit_free_gb": round(sw.free / gb, 2),
    }


def _cpu_stats() -> dict[str, Any]:
    if not PSUTIL:
        return {}
    s = _safe(lambda: psutil.cpu_stats(), None)
    if s is None:
        return {}
    return {
        "context_switches": int(s.ctx_switches),
        "interrupts": int(s.interrupts),
        "soft_interrupts": int(s.soft_interrupts),
        "syscalls": int(getattr(s, "syscalls", 0)),
    }


def _disk_snapshot() -> dict[str, Any]:
    if not PSUTIL:
        return {}
    d = _safe(lambda: psutil.disk_io_counters(), None)
    if d is None:
        return {}
    # FIX 2026-11-22: these are CUMULATIVE counters since boot, not rates.
    # A consumer wanting MB/s must diff two snapshots over their dt.
    return {
        "read_bytes_cumulative": int(d.read_bytes),
        "write_bytes_cumulative": int(d.write_bytes),
        "read_mb_cumulative": round(d.read_bytes / (1024**2), 2),
        "write_mb_cumulative": round(d.write_bytes / (1024**2), 2),
        "is_cumulative": True,
    }


def _top_processes(limit: int = 8) -> list[dict[str, Any]]:
    """Top processes by RSS. Name + PID only -- no command-line assumptions."""
    if not PSUTIL:
        return []
    rows: list[dict[str, Any]] = []
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            mi = p.info.get("memory_info")
            rss_mb = round((mi.rss if mi else 0) / (1024**2), 1)
            rows.append(
                {
                    "pid": int(p.info.get("pid") or 0),
                    "name": str(p.info.get("name") or ""),
                    "rss_mb": rss_mb,
                }
            )
        except Exception:
            continue
    rows.sort(key=lambda r: r["rss_mb"], reverse=True)
    return rows[:limit]


def capture() -> dict[str, Any]:
    """Take one whole-box snapshot. Never raises."""
    return {
        "ts": time.time(),
        "source": "host_telemetry",
        "cpu": _cpu_snapshot(),
        "memory": _memory_snapshot(),
        "commit": _commit_snapshot(),
        "cpu_stats": _cpu_stats(),
        "disk": _disk_snapshot(),
        "gpu": _gpu_snapshot(),
        "top_processes": _top_processes(),
        "process_count": _safe(lambda: len(psutil.pids()) if PSUTIL else 0, 0),
    }


def write_snapshot(snap: dict[str, Any]) -> bool:
    """Atomic write so daemon readers never see a partial JSON."""
    try:
        tmp = _SNAPSHOT.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        os.replace(tmp, _SNAPSHOT)
        return True
    except OSError:
        return False


def read_snapshot(max_age_s: float = 75.0) -> dict[str, Any]:
    """Read the shared snapshot with freshness + schema gates.

    Returns {} on ANY failure so consumers treat it as "no data", never
    as "idle box". Guards derived from the four-swarm review 2026-11-22:
      - provenance: source == host_telemetry
      - schema: required nested keys present with sane types
      - freshness: 0 <= age <= max_age_s (reject stale AND future-dated)
    """
    try:
        data = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        if data.get("source") != "host_telemetry":
            return {}
        ts = float(data.get("ts", 0.0))
        age = time.time() - ts
        if age < 0 or age > max_age_s:
            return {}
        cpu = data.get("cpu")
        if not isinstance(cpu, dict):
            return {}
        cores = cpu.get("per_core_pct")
        if not isinstance(cores, list) or not cores:
            return {}
        # Sane ranges; a corrupt snapshot must not feed the controller.
        for v in cores:
            f = float(v)
            if f < 0.0 or f > 100.0:
                return {}
        ncores = int(cpu.get("logical_cores", 0))
        if ncores > 0 and len(cores) != ncores:
            return {}
        return data
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return {}
    except Exception:
        return {}


def summary(snap: dict[str, Any]) -> str:
    cpu = snap.get("cpu", {})
    mem = snap.get("memory", {})
    gpu = snap.get("gpu", {})
    cores = cpu.get("per_core_pct", [])
    return (
        f"CPU total={cpu.get('total_pct', 0)}% "
        f"cores={cores} "
        f"RAM={mem.get('used_gb', 0)}/{mem.get('total_gb', 0)}GB "
        f"GPU={gpu.get('utilization_pct', 0)}% "
        f"VRAM={gpu.get('vram_used_mb', 0)}/{gpu.get('vram_total_mb', 0)}MB "
        f"procs={snap.get('process_count', 0)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Whole-box telemetry snapshot")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        s = read_snapshot()
        if not s:
            print("no snapshot yet")
        else:
            age = round(time.time() - float(s.get("ts", 0.0)), 1)
            print(f"age={age}s | {summary(s)}")
        return 0

    snap = capture()
    if not args.no_write:
        write_snapshot(snap)
    print(json.dumps(snap, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
