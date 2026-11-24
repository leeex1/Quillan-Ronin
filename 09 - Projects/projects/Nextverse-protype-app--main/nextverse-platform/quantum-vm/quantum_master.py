#!/usr/bin/env python3
"""Import shim for the quantum VM master module.

The deployable entrypoint is ``quantum-master.py`` (hyphenated). Python cannot
import a hyphenated module name, and the test-suite imports ``quantum_master``.
This shim loads the canonical implementation under a private module name and
re-exports its public symbols so both the entrypoint and the tests reference the
exact same class objects (important for unittest.mock patching).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_IMPL_NAME = "_nextverse_quantum_master_impl"
_IMPL_PATH = Path(__file__).with_name("quantum-master.py")

_spec = importlib.util.spec_from_file_location(_IMPL_NAME, _IMPL_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
    raise ImportError(f"Cannot load quantum VM implementation at {_IMPL_PATH}")
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

QuantumMaster: Any = _impl.QuantumMaster
_SESSION = _impl._SESSION

__all__ = ["QuantumMaster", "_SESSION"]
