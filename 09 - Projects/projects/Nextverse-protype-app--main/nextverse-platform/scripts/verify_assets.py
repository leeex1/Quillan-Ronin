#!/usr/bin/env python3
"""Validate demo GLB assets referenced by the web zone manifest."""

from __future__ import annotations

import json
import struct
from pathlib import Path

ASSET_DIR = (
    Path(__file__).resolve().parent.parent / "web-ui" / "public" / "assets" / "tripio"
)


def validate_glb(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"{path.name}: bad GLB magic")
    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2 or total_length != len(data):
        raise ValueError(f"{path.name}: bad GLB header/version/length")
    chunk_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    if chunk_type != b"JSON":
        raise ValueError(f"{path.name}: first chunk is not JSON")
    document = json.loads(data[20 : 20 + chunk_length].decode("utf-8"))
    return {
        "file": path.name,
        "bytes": len(data),
        "version": version,
        "meshes": len(document.get("meshes", [])),
        "nodes": len(document.get("nodes", [])),
        "skins": len(document.get("skins", [])),
        "animations": len(document.get("animations", [])),
    }


def main() -> int:
    manifest = json.loads((ASSET_DIR / "manifest.json").read_text())
    entries = manifest.get("assets", [])
    if not entries:
        raise RuntimeError("no demo assets registered")
    for entry in entries:
        report = validate_glb(ASSET_DIR / entry["file"])
        print(
            f"[PASS] GLB {report['file']}: {report['bytes']} bytes, "
            f"meshes={report['meshes']} nodes={report['nodes']} "
            f"skins={report['skins']} animations={report['animations']}"
        )
    print(f"[PASS] {len(entries)} demo assets validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
