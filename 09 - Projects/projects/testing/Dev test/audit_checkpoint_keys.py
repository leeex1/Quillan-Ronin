import sys, torch
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg)
model_keys = set(model.state_dict().keys())

# Check best checkpoint
ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_sft_v3_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
ckpt_keys = set(sd.keys())

print(f"Model expects {len(model_keys)} tensors")
print(f"Checkpoint has {len(ckpt_keys)} tensors")

missing = model_keys - ckpt_keys
extra = ckpt_keys - model_keys

if missing:
    print(f"\n[!] MISSING from checkpoint ({len(missing)}):")
    for k in sorted(missing):
        print(f"    {k}: {model.state_dict()[k].shape}")

if extra:
    print(f"\n[!] EXTRA in checkpoint not in model ({len(extra)}):")
    for k in sorted(extra):
        print(f"    {k}: {sd[k].shape}")

# Check shape mismatches for matching keys
matched = model_keys & ckpt_keys
mismatches = []
for k in sorted(matched):
    m_shape = model.state_dict()[k].shape
    c_shape = sd[k].shape
    if m_shape != c_shape:
        mismatches.append((k, m_shape, c_shape))

if mismatches:
    print(f"\n[!] SHAPE MISMATCHES ({len(mismatches)}):")
    for k, ms, cs in mismatches:
        print(f"    {k}: model={ms}  ckpt={cs}")
else:
    print(f"\n[+] All {len(matched)} matching keys have correct shapes!")

print(f"\n[+] Audit complete.")
