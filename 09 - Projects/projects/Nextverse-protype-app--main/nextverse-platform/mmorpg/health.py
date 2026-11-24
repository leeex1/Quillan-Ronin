#!/usr/bin/env python3
"""Real health probe for the NextVerse MMORPG service."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchestrator"))
from system import SystemOrchestrator

service = SystemOrchestrator().services()["social-hub"]
print(json.dumps(service, indent=2))
