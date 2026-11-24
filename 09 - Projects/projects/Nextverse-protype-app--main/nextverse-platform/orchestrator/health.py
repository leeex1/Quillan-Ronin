#!/usr/bin/env python3
"""Real health probe for the NextVerse orchestrator service."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from system import SystemOrchestrator

print(json.dumps(SystemOrchestrator().health(), indent=2))
