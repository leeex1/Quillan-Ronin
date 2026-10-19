import sys
from pathlib import Path
import torch
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "09 - Projects" / "projects" / "oni"),
          str(REPO / "scripts"),
          str(REPO / "scripts" / "hf_code"),
          str(REPO / "scripts" / "hf_code_era")]:
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(3)
import quillan_v5_4_oni as Q
print("core file=" + Q.__file__, flush=True)
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
CKPT = REPO / "checkpoints" / "hf_restore" / "quillan_oni_5.4.0_step660_5.22GB.pt"
cfg = QuillanOniConfig(vocab_size=50257, hidden_dim=1024, ffn_dim=2048,
                       n_layer=12, num_experts=34, top_k=4, max_seq_len=512)
m = QuillanRoninOni(cfg)
d = torch.load(CKPT, map_location="cpu", weights_only=True)
sd = d.get("model_state_dict", d.get("model", d))
missing, unexpected = m.load_state_dict(sd, strict=True)
print(f"STRICT missing={len(missing)} unexp={len(unexpected)}", flush=True)
m.eval()
ds = torch.load(REPO / "training_data" / "mini_slice_20k.pt",
                map_location="cpu", weights_only=True)
ids = ds["input_ids"] if isinstance(ds, dict) else ds
tot, cnt = 0.0, 0
with torch.no_grad():
    for i in range(0, min(12, ids.shape[0]), 2):
        x = ids[i:i + 2][:, :160]
        o = m(x)
        logits = o[0] if isinstance(o, tuple) else o
        tot += torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1)).item()
        cnt += 1
        print(f"batch {cnt} loss={tot / cnt:.4f}", flush=True)
print(f"MAIN MEAN={tot / max(1, cnt):.4f}", flush=True)
