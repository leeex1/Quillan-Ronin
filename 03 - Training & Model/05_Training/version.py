"""
👑 QUILLAN-RONIN — SOVEREIGN VERSION & RELEASE AUTHORITY (2026)
Single source of truth for versioning, release metadata, and compatibility contracts.
"""

from typing import Tuple, Dict, Any

__version__: str = "5.4.0-oni"
VERSION_INFO: Tuple[int, int, int, str] = (5, 4, 0, "oni")
RELEASE_YEAR: int = 2026
CODENAME: str = "ONI Sovereign Quantum"
ARCHITECTURE_ERA: str = "2026 Saturated Native"

COMPATIBILITY_MATRIX: Dict[str, Any] = {
    "target_python": ">=3.10,<=3.14.6",
    "target_pytorch": ">=2.4.0,<=2.14.0",
    "council_size": 34,
    "quantization": "BitNet 1.58b STE Ternary",
    "legacy_fallbacks": ["4.2.0", "5.3.1", "6.0.3-pre"],
}

def get_version() -> str:
    """Return canonical Quillan-Ronin version string."""
    return __version__
