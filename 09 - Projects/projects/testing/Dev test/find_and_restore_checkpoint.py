import os
import sys
import torch
import shutil
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
ckpt_dir = REPO_ROOT / "checkpoints" / "checkpoints_sft"

print("==================================================================")
print("   👑 FINDING & RESTORING UNCORRUPTED BASE CHECKPOINT")
print("==================================================================")

valid_checkpoints = []

for fpath in ckpt_dir.glob("*"):
    if fpath.is_file():
        print(f"\n[*] Testing: {fpath.name}")
        try:
            ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            loss = ckpt.get("loss", "N/A") if isinstance(ckpt, dict) else "N/A"
            step = ckpt.get("step", "N/A") if isinstance(ckpt, dict) else "N/A"
            print(f"    [+] VALID! Tensors: {len(sd)}, Step: {step}, Loss: {loss}")
            valid_checkpoints.append((fpath, step, loss, len(sd)))
        except Exception as e:
            print(f"    [!] Invalid/Corrupted ({e})")

target_master = ckpt_dir / "quillan_thinking_reasoning_master.pt"

# Find best available valid fallback
if valid_checkpoints:
    best_fallback = valid_checkpoints[0][0]
    print(f"\n[+] Restoring master checkpoint from valid fallback: {best_fallback.name}")
    shutil.copy2(best_fallback, target_master)
    print(f"[+] Restored: {target_master.name}")

print("\n[+] Restoration Audit Complete!")
