"""Build Quantum_Street_Engine_v4.fmx from verified components.

Inputs (all in this folder or the decode temp dir):
  v3 bundle ....... source bundle (structure + signature scaffold)
  ui_v4.html ...... patched React UI (AUTO mode, presets, dB meter, fixed defaults)
  engine_v4.wasm .. rebuilt DSP (clang wasm32, ABI-compatible with v3)
  metadata_v4.json  corrected parameter defaults

The original 128-hex `signature` is preserved verbatim. It is an opaque
provenance stamp from the generator and cannot be regenerated without its
private key; see README_v4.md for the compatibility caveat.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Admin\AppData\Local\Temp\opencode\qse_decode")
from qse_bundle import Bundle  # noqa: E402

HERE = Path(__file__).resolve().parent
V3 = Path(
    r"C:\02_QUILLAN\09 - Projects\projects\Audio Engineer\Quantum_Street_Engine_v3.fmx"
)
UI = Path(r"C:\Users\Admin\AppData\Local\Temp\opencode\qse_decode\ui_v4.html")
WASM = HERE / "engine_v4.wasm"
META = HERE / "metadata_v4.json"
OUT = HERE / "Quantum_Street_Engine_v4.fmx"


def main() -> None:
    for p in (V3, UI, WASM, META):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    b = Bundle(V3)
    b.set_html(UI.read_text(encoding="utf-8"))
    b.set_wasm(WASM.read_bytes())
    b.set_metadata(json.loads(META.read_text(encoding="utf-8")))
    b.data["version"] = "1.4.0"
    b.data["spaceVersion"] = 4
    b.data["name"] = "Quantum Street Engine"

    issues = b.verify()
    if issues:
        raise SystemExit("verification failed: " + "; ".join(issues))

    final = OUT.write_text(json.dumps(b.data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({final} bytes)")
    print("payload digest:", b.digest())
    print("signature (preserved):", b.data["signature"][:16] + "...")


if __name__ == "__main__":
    main()
