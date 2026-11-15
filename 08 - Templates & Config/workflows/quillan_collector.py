#!/usr/bin/env python3
"""Compatibility launcher delegating to canonical 06 - Deployment & Platforms/quillan_collector.py."""
from pathlib import Path
import runpy

CANONICAL = Path(__file__).resolve().parents[2] / "06 - Deployment & Platforms" / "quillan_collector.py"
if not CANONICAL.exists():
    raise FileNotFoundError(f"Canonical collector not found: {CANONICAL}")
runpy.run_path(str(CANONICAL), run_name="__main__")
