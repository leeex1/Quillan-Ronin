import sys
from pathlib import Path
import torch
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts"), str(REPO / "09 - Projects" / "projects" / "oni")]:
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(3)
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
CKPT = REPO / "checkpoints" / "hf_restore" / "quillan_frontier_v2_best_loss0.0789_step2500.pt"
cfg = QuillanOniConfig(vocab_size=50257, hidden_dim=1024, ffn_dim=2048,
                       n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
m = QuillanRoninOni(cfg)
d = torch.load(CKPT, map_location="cpu", weights_only=True)
sd = d.get("model_state_dict", d.get("model", d))
ref = m.state_dict()
sd2 = {}
for k, v in sd.items():
    if k in ref and v.shape != ref[k].shape and v.dim() == 2 and tuple(reversed(v.shape)) == tuple(ref[k].shape):
        sd2[k] = v.t().contiguous()
    else:
        sd2[k] = v
missing, unexpected = m.load_state_dict(sd2, strict=False)
print(f"bound missing={len(missing)} unexp={len(unexpected)} total={len(sd)}",
      flush=True)
m.eval()
ds = torch.load(REPO / "training_data" / "quillan_corpus_CLEAN_V7.pt",
                map_location="cpu", weights_only=True)
ids = ds["input_ids"] if isinstance(ds, dict) else ds
n = ids.shape[0]
tot, cnt = 0.0, 0
with torch.no_grad():
    for i in range(0, min(20, n), 2):
        x = ids[i:i + 2][:, :160]
        o = m(x)
        logits = o[0] if isinstance(o, tuple) else o
        tot += torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1)).item()
        cnt += 1
        print(f"batch {cnt} loss={tot / cnt:.4f}", flush=True)
print(f"MEAN={tot / max(1, cnt):.4f}", flush=True)
