import shutil
from pathlib import Path

REPO_ROOT = Path(r"C:\02_QUILLAN")
ckpt_dir = REPO_ROOT / "checkpoints" / "checkpoints_sft"
src = ckpt_dir / "quillan_sft_v3_best.pt"
dst = ckpt_dir / "quillan_thinking_reasoning_master.pt"

shutil.copy2(src, dst)
print(f"[+] Restored {src.name} (Loss: 0.06087, Step 8048) directly to {dst.name}!")
