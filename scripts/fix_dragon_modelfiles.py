#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix stale GGUF references in Quillan Modelfiles (objective part 3).

Problem: #1223 cleanup deleted old failed checkpoints (v20, v15 GGUFs), but several
root Modelfiles still point at them — so `ollama create` from those files fails.
All Modelfiles already have num_ctx 2097152 (correct 2M) — no change needed there.

This script:
  - backs up each Modelfile to .bak_dragonfix_<timestamp>
  - repoints stale FROM lines to the GGUFs that actually exist on disk
  - does NOT touch: GGUF weights, model safetensors, canonical, gates, WSP
  - verifies every FROM path resolves and num_ctx == 2097152

Mapping (stale -> existing):
  quillan-1b-sovereign-wsp-v20-q8_0.gguf  -> quillan-1b-sovereign-wsp-v23-q8_0.gguf
  quillan-mini-0.5b-wsp-v15-q8_0.gguf     -> quillan-mini-0.5b-wsp-v23-q8_0.gguf
  ./quillan-dragon-1b-wsp-v8-q4_k_m.gguf  -> exports/quillan-dragon-1b-wsp-v18b-q8_0.gguf
  ./quillan-1b-sovereign-wsp-v8-q4_k_m.gguf-> exports/quillan-1b-sovereign-wsp-v23-q8_0.gguf
  ./quillan-mini-0.5b-wsp-v8-q4_k_m.gguf  -> exports/quillan-mini-0.5b-wsp-v23-q8_0.gguf
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import re

ROOT = Path(r"C:\02_QUILLAN")
EXPORT = ROOT / "exports"
FILES = [
    "Modelfile",
    "Modelfile.mini",
    "Modelfile.dragon",
    "Modelfile.hf.Quillan-Dragon",
    "Modelfile.hf.Quillan-Ronin",
    "Modelfile.hf.Quillan-Ronin-Mini",
]

REMAP = {
    "quillan-1b-sovereign-wsp-v20-q8_0.gguf": "exports/quillan-1b-sovereign-wsp-v23-q8_0.gguf",
    "quillan-mini-0.5b-wsp-v15-q8_0.gguf": "exports/quillan-mini-0.5b-wsp-v23-q8_0.gguf",
    "quillan-dragon-1b-wsp-v8-q4_k_m.gguf": "exports/quillan-dragon-1b-wsp-v18b-q8_0.gguf",
    "quillan-1b-sovereign-wsp-v8-q4_k_m.gguf": "exports/quillan-1b-sovereign-wsp-v23-q8_0.gguf",
    "quillan-mini-0.5b-wsp-v8-q4_k_m.gguf": "exports/quillan-mini-0.5b-wsp-v23-q8_0.gguf",
}


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in FILES:
        f = ROOT / name
        if not f.exists():
            print(f"[skip] {name} missing")
            continue
        text = f.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        changed = False
        out = []
        for line in lines:
            if line.strip().startswith("FROM "):
                parts = line.split(None, 1)
                target = parts[1].strip() if len(parts) > 1 else ""
                base = target.replace("./", "").split("/")[-1]
                if base in REMAP:
                    new = REMAP[base]
                    indent = line[: len(line) - len(line.lstrip())]
                    out.append(f"{indent}FROM {new}\n")
                    print(f"  {name}: {target} -> {new}")
                    changed = True
                    continue
            out.append(line)
        if changed:
            bak = ROOT / f"{name}.bak_dragonfix_{ts}"
            bak.write_text(text, encoding="utf-8")
            print(f"[backup] {bak.name}")
            f.write_text("".join(out), encoding="utf-8")
            print(f"[fixed] {name}")

    # verify all FROM paths exist + num_ctx 2097152
    print("\n=== VERIFY ===")
    ok = True
    for name in FILES:
        f = ROOT / name
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        frm = re.search(r"^\s*FROM\s+(\S+)", text, re.M)
        ctx = re.search(r"^\s*PARAMETER\s+num_ctx\s+(\d+)", text, re.M)
        target = frm.group(1) if frm else None
        exists = resolve(target).exists() if target else False
        ctxv = ctx.group(1) if ctx else None
        good = exists and ctxv == "2097152"
        ok &= good
        print(
            f"  {name:36} FROM_exists={exists}  num_ctx={ctxv}  {'OK' if good else 'FAIL'}"
        )
    print(f"\n[VERDICT] {'ALL OK' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
