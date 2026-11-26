#!/usr/bin/env python3
"""Generate a REVERSIBLE cleanup plan for exports/.
KEEP: three model bests + all donors + active glassbox work + Modelfiles.
QUARANTINE: everything else (intermediate candidates).
Writes a manifest JSON with full source paths so restoration is a single move.
"""

from __future__ import annotations
from pathlib import Path
import json, shutil
from datetime import datetime

ROOT = Path(r"C:\02_QUILLAN")
EXPORTS = ROOT / "exports"
QDIR = ROOT / "12 - Backups" / "quarantine_checkpoints_20261124"
MANIFEST = QDIR / "manifest.json"

# Explicit KEEP set (relative to exports). Everything else large is quarantined.
KEEP_FILES = {
    # three model bests / bases
    "quillan-mini-sovereign-bitnet.safetensors",
    "quillan-mini-hirank64.safetensors",  # best closed-form gain
    "quillan-mini-glassbox-base.safetensors",  # active protocol base
    "quillan-mini-glassbox-attn.safetensors",  # ACCEPTED/FROZEN attention win
    "quillan-flagship-sovereign-bitnet.safetensors",
    "quillan-flagship-rank64.safetensors",
    "quillan-dragon-1b-wsp-v18b-q8_0.gguf",  # Modelfile.dragon source
    "quillan-mini-0.5b-wsp-v23-q8_0.gguf",  # Modelfile.mini source
    "quillan-1b-sovereign-wsp-v23-q8_0.gguf",  # Modelfile flagship source
    "clean-llama-1b-q8_0.gguf",  # clean donor GGUF
    "clean-qwen-0.5b-q8_0.gguf",  # clean donor GGUF
    "donor_experts_ls.safetensors",  # canonical cross-matched donor bank
}
KEEP_DIRS = {
    "donor_cache_v23",  # all donors (Qwen/Llama/BitNet/Falcon3)
    "quillan_1b_sovereign_hf",  # flagship HF export
    "quillan_mini_hf",  # mini HF export
}
# keep the active ledger regardless of location
KEEP_GLOB = ["glassbox_ledger.json"]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--execute", action="store_true", help="actually move files (default: dry run)"
    )
    a = ap.parse_args()

    entries = []
    for p in EXPORTS.iterdir():
        if p.name in KEEP_DIRS or p.name in KEEP_FILES:
            continue
        if any(p.match(g) for g in KEEP_GLOB):
            continue
        if p.is_file() and p.stat().st_size > 20 * 1024 * 1024:  # >20MB intermediates
            entries.append(
                {
                    "name": p.name,
                    "size_mb": round(p.stat().st_size / 1e6, 1),
                    "src": str(p),
                }
            )

    total = sum(e["size_mb"] for e in entries)
    print(
        f"QUARANTINE PLAN: {len(entries)} files, {total:,.0f} MB ({total / 1024:.1f} GB)",
        flush=True,
    )
    for e in sorted(entries, key=lambda x: -x["size_mb"]):
        print(f"  {e['size_mb']:8.1f} MB  {e['name']}", flush=True)
    print(
        f"KEEP: {len(KEEP_FILES)} model files + {len(KEEP_DIRS)} donor/export dirs",
        flush=True,
    )

    if a.execute:
        QDIR.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(
            json.dumps(
                {
                    "created": datetime.now().isoformat(),
                    "restore": "move each dest back to src",
                    "entries": entries,
                },
                indent=2,
            )
        )
        for e in entries:
            src = Path(e["src"])
            dst = QDIR / src.name
            if dst.exists():
                print(f"SKIP (dest exists) {src.name}", flush=True)
                continue
            shutil.move(str(src), str(dst))
        print(f"MOVED {len(entries)} files -> {QDIR}", flush=True)
        print(f"MANIFEST {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
