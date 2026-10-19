"""Print current checkpoint step (source of truth). Slow (~1-2 min on 6.7GB) — use sparingly."""
import torch, sys
p = sys.argv[1] if len(sys.argv) > 1 else r"C:\02_QUILLAN\05_Training\checkpoints\checkpoints_oni\quillan_oni_latest.pt"
try:
    ck = torch.load(p, map_location="cpu", weights_only=True)
    print(int(ck.get("step", 0)))
except Exception:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print(int(ck.get("step", 0)))
