#!/usr/bin/env python3
"""
👑 Backup solid v1 checkpoint before refinement
"""
import shutil
from pathlib import Path

REPO_ROOT = Path(r"C:\02_QUILLAN")
ckpt_dir = REPO_ROOT / "checkpoints" / "checkpoints_sft"
src = ckpt_dir / "quillan_thinking_reasoning_master.pt"
dst = ckpt_dir / "quillan_thinking_reasoning_master_v1_solid.pt"

if src.exists():
    shutil.copy2(src, dst)
    print(f"[+] Successfully created golden backup: {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
else:
    print(f"[-] Source checkpoint not found: {src}")
