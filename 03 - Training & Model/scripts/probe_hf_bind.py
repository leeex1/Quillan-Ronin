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
sd2, rep = {}, {}
for k, v in sd.items():
    if k in ref and v.shape != ref[k].shape and v.dim() == 2 and tuple(reversed(v.shape)) == tuple(ref[k].shape):
        sd2[k] = v.t().contiguous()
        rep[k] = "transposed"
    elif k in ref and v.shape == ref[k].shape:
        sd2[k] = v
        rep[k] = "direct"
    else:
        rep[k] = f"SKIP ckpt={tuple(v.shape)} model={tuple(ref[k].shape) if k in ref else 'ABSENT'}"
missing, unexpected = m.load_state_dict(sd2, strict=False)
mods = {}
for k, how in rep.items():
    mod = k.split(".")[0] + (".moe" if ".moe." in k else ".attn" if ".attn." in k else ".emb" if "embed" in k or "wte" in k or "wpe" in k else ".other")
    mods.setdefault(mod, {"direct": 0, "transposed": 0, "skip": []})
    if how in ("direct", "transposed"):
        mods[mod][how] += 1
    else:
        mods[mod]["skip"].append(f"{k} {how}")
for mod, s in sorted(mods.items()):
    print(f"{mod}: direct={s['direct']} transposed={s['transposed']}", flush=True)
    for sk in s["skip"][:8]:
        print(f"  SKIP {sk}", flush=True)
m.eval()
ds = torch.load(REPO / "training_data" / "quillan_corpus_CLEAN_V7.pt",
                map_location="cpu", weights_only=True)
ids = ds["input_ids"] if isinstance(ds, dict) else ds
tot, cnt = 0.0, 0
with torch.no_grad():
    for i in range(0, min(20, ids.shape[0]), 2):
        x = ids[i:i + 2][:, :160]
        o = m(x)
        logits = o[0] if isinstance(o, tuple) else o
        tot += torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1)).item()
        cnt += 1
        print(f"batch {cnt} loss={tot / cnt:.4f}", flush=True)
print(f"MEAN={tot / max(1, cnt):.4f}", flush=True)
