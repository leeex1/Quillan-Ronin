#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e_drive_guard.py - DETECT-ONLY guard for the spoofed E: USB drive.

The drive reports 3.8 TB but has ~46 GB real NAND. It is SAFE only because
partition 2 is capped at 46.875 GB (offset 0.016 GB). The failure mode is
someone extending that partition or writing past ~46 GB -> silent file loss
after remount.

This tool NEVER writes to E:, NEVER kills processes, NEVER changes partitions.
It reads geometry via PowerShell/CIM, logs an alert to a JSON on C: if the
safe configuration changes, and prints status. Opt-in scheduling only.

Usage:
  python e_drive_guard.py            # one-shot status + log
  python e_drive_guard.py --check    # same, exits 1 if UNSAFE (for task monitor)
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "07 - Memory & LanceDB" / "e_drive_guard_status.json"
EXPECTED_MAX_PART_GB = 47.0  # real NAND ~46 GB; capped partition is 46.875 GB
EXPECTED_MIN_FREE_GB = 0.2  # warn below this
DISK_NUMBER = 1
PART_NUMBER = 2

PS = r"""
$d = Get-Disk -Number 1 -ErrorAction SilentlyContinue
$p = Get-Partition -DiskNumber 1 -PartitionNumber 2 -ErrorAction SilentlyContinue
$v = Get-Volume -DriveLetter E -ErrorAction SilentlyContinue
[pscustomobject]@{
  disk_size_bytes   = [int64]$d.Size
  part_size_bytes   = [int64]$p.Size
  part_offset_bytes = [int64]$p.Offset
  part_type         = [string]$p.Type
  vol_size_bytes    = [int64]$v.Size
  vol_free_bytes    = [int64]$v.SizeRemaining
  vol_fs            = [string]$v.FileSystemType
  disk_model        = [string]$d.Model
  disk_bus          = [string]$d.BusType
} | ConvertTo-Json -Compress
"""


def read_geometry() -> dict[str, Any]:
    r = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", PS],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"PowerShell geometry read failed: {r.stderr[:200]}")
    return json.loads(r.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if UNSAFE")
    args = ap.parse_args()

    try:
        g = read_geometry()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    gb = 1024**3
    part_gb = g["part_size_bytes"] / gb
    free_gb = g["vol_free_bytes"] / gb
    unalloc_gb = (
        g["disk_size_bytes"] - g["part_size_bytes"] - g["part_offset_bytes"]
    ) / gb

    alerts: list[str] = []
    if part_gb > EXPECTED_MAX_PART_GB:
        alerts.append(
            f"CRITICAL: partition 2 is {part_gb:.2f} GB (> {EXPECTED_MAX_PART_GB} GB cap). "
            "It may have been extended into the fake capacity tail -> silent data loss risk."
        )
    if g["part_type"] != "Basic":
        alerts.append(f"WARN: partition type changed to {g['part_type']!r}.")
    if free_gb < EXPECTED_MIN_FREE_GB:
        alerts.append(
            f"WARN: E: has only {free_gb:.2f} GB free. Nearly full; no room for safe writes."
        )
    if not g["vol_fs"] or g["vol_fs"].upper() != "EXFAT":
        alerts.append(f"WARN: filesystem is {g['vol_fs']!r}, expected exFAT.")

    safe = not any(a.startswith("CRITICAL") for a in alerts)
    status = {
        "timestamp": time.time(),
        "safe": safe,
        "disk_model": g["disk_model"],
        "disk_reported_GB": round(g["disk_size_bytes"] / gb, 2),
        "partition_GB": round(part_gb, 3),
        "partition_offset_GB": round(g["part_offset_bytes"] / gb, 3),
        "unallocated_tail_GB": round(unalloc_gb, 2),
        "free_GB": round(free_gb, 3),
        "filesystem": g["vol_fs"],
        "alerts": alerts,
        "note": "DETECT-ONLY. Never writes to E:, never kills, never changes partitions.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    tmp.replace(OUT)

    print(
        f"disk={g['disk_model']} reported={status['disk_reported_GB']}GB "
        f"partition={part_gb:.3f}GB free={free_gb:.2f}GB fs={g['vol_fs']}"
    )
    print(f"unallocated fake tail = {unalloc_gb:.0f} GB  (NEVER extend into this)")
    if alerts:
        for a in alerts:
            print(f"  {a}")
    else:
        print("  no geometry alerts (partition cap intact)")
    print(f"status -> {OUT}")
    return 0 if safe else 1


if __name__ == "__main__":
    sys.exit(main())
