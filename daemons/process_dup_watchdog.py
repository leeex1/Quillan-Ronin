#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_dup_watchdog.py -- READ-ONLY duplicate process detector.

Reports processes sharing an identical command line (same script, same args)
so you can see duplicate eval/inference runs before they OOM the box.

HARD INVARIANT: THIS MODULE NEVER TERMINATES A PROCESS.
There is no kill/terminate/signal code here. It writes a JSON report only.

WHY THIS EXISTS
---------------
Under load, eval_quillan_trio_bitnet.py was observed running twice with
identical arguments:
  - one interpreter with torch+CUDA (correct path)
  - one interpreter with NO torch (pure CPU, ~50% load, 2.5GB RSS)
Both wrote the same output file. That is the OOM/CPU pattern this detects.

OPT-IN: importing does nothing. No daemon, no timer, no boot entry.
Run explicitly:  python process_dup_watchdog.py --once
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

_REPO = Path(r"F:\02_QUILLAN")
_REPORT = _REPO / "07 - Memory & LanceDB" / "process_dup_report.json"

# Processes to consider. Anything with a real command line and >1MB RSS.
_MIN_RSS_MB = 1.0
# Ignore short command lines (OS noise).
_MIN_CMD_LEN = 12


def _fingerprint(cmdline: list[str]) -> str:
    """Stable hash of the exact command line, order preserved."""
    joined = "\x00".join(cmdline)
    return hashlib.sha256(joined.encode("utf-8", "replace")).hexdigest()[:16]


def _snapshot() -> list[dict[str, Any]]:
    """Read-only process snapshot."""
    if psutil is None:
        return []
    out: list[dict[str, Any]] = []
    for p in psutil.process_iter(
        ["pid", "name", "cmdline", "memory_info", "create_time"]
    ):
        try:
            info = p.info
            cmd = list(info.get("cmdline") or [])
            joined = " ".join(cmd)
            if len(joined) < _MIN_CMD_LEN:
                continue
            rss_mb = float(
                (info.get("memory_info").rss if info.get("memory_info") else 0)
            ) / (1024**2)
            if rss_mb < _MIN_RSS_MB:
                continue
            out.append(
                {
                    "pid": int(info.get("pid") or 0),
                    "name": str(info.get("name") or ""),
                    "cmdline": cmd,
                    "cmd": joined,
                    "fingerprint": _fingerprint(cmd),
                    "rss_mb": round(rss_mb, 1),
                    "created": round(float(info.get("create_time") or 0.0), 2),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            continue
    return out


def detect() -> dict[str, Any]:
    """Group processes by identical command line. Pure read, no mutation."""
    procs = _snapshot()
    groups: dict[str, list[dict[str, Any]]] = {}
    for p in procs:
        groups.setdefault(p["fingerprint"], []).append(p)

    dups = []
    for fp, members in groups.items():
        if len(members) < 2:
            continue
        members_sorted = sorted(members, key=lambda x: x["created"])
        dups.append(
            {
                "fingerprint": fp,
                "count": len(members),
                "combined_rss_mb": round(sum(m["rss_mb"] for m in members), 1),
                "cmd": members_sorted[0]["cmd"][:400],
                "members": [
                    {
                        "pid": m["pid"],
                        "rss_mb": m["rss_mb"],
                        "created": m["created"],
                        "interpreter": m["cmdline"][0] if m["cmdline"] else "",
                    }
                    for m in members_sorted
                ],
            }
        )

    dups.sort(key=lambda d: d["combined_rss_mb"], reverse=True)

    return {
        "ts": time.time(),
        "read_only": True,
        "note": "Detector only. No process is ever terminated by this module.",
        "total_processes_scanned": len(procs),
        "duplicate_groups": len(dups),
        "wasted_rss_mb_estimate": round(
            sum(d["combined_rss_mb"] - d["members"][0]["rss_mb"] for d in dups), 1
        ),
        "duplicates": dups[:20],
    }


def run_once(write_report: bool = True) -> dict[str, Any]:
    report = detect()
    if write_report:
        try:
            _REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError:
            pass
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only duplicate process detector")
    parser.add_argument(
        "--once", action="store_true", help="run one scan and write report"
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print only, do not write report"
    )
    args = parser.parse_args()

    if not args.once and not args.no_write:
        parser.print_help()
        return 0

    report = run_once(write_report=not args.no_write)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
