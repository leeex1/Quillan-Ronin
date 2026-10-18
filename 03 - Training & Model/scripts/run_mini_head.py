"""Mini full head run: resume lr5e5 best, 160 steps, BEST-only."""
import sys
import time
from pathlib import Path
import torch
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts" / "hf_code_era"),
          str(REPO / "scripts" / "hf_code"),
          str(REPO / "scripts"),
          str(REPO / "09 - Projects" / "projects" / "oni")]:
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
BASE = REPO / "checkpoints" / "hf_restore" / "mini_trirun_lr5e5_best.pt"
SLICE = REPO / "training_data" / "mini_slice_20k.pt"
OUT = REPO / "checkpoints" / "hf_restore" / "mini_head_best.pt"
STEPS, SEQ, LR = 160, 160, 5e-5
ds = torch.load(SLICE, map_location="cpu", weights_only=True)
ids = ds["input_ids"] if isinstance(ds, dict) else ds
ntr = int(len(ids) * 0.9)
tr, va = ids[:ntr], ids[ntr:]
cfg = QuillanOniConfig(vocab_size=50262, hidden_dim=1024, ffn_dim=2048,
                       n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
m = QuillanRoninOni(cfg)
bd = torch.load(BASE, map_location="cpu", weights_only=True)
missing, unexp = m.load_state_dict(bd.get("model_state_dict", bd), strict=False)
print(f"bound missing={len(missing)} unexp={len(unexp)}", flush=True)
for n, p in m.named_parameters():
    p.requires_grad = ("lm_head" in n or "ln_f" in n)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
m.train()
best, t0 = 1e9, time.time()
for s in range(STEPS):
    bi = (s * 2) % (len(tr) - 2)
    x = tr[bi:bi + 2][:, :SEQ]
    opt.zero_grad()
    o = m(x)
    logits = o[0] if isinstance(o, tuple) else o
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    if (s + 1) % 20 == 0:
        m.eval()
        with torch.no_grad():
            vx = va[:4][:, :SEQ]
            vo = m(vx)
            vl = vo[0] if isinstance(vo, tuple) else vo
            vv = torch.nn.functional.cross_entropy(
                vl[:, :-1, :].reshape(-1, vl.size(-1)),
                vx[:, 1:].reshape(-1)).item()
        print(f"step={s + 1} train={loss.item():.4f} val={vv:.4f}", flush=True)
        if vv < best:
            best = vv
            torch.save({"model_state_dict": {k: v.cpu().clone() for k, v in m.state_dict().items()},
                        "meta": {"val": best, "step": s + 1}}, OUT)
            print(f"SAVED val={best:.4f}", flush=True)
        m.train()
print(f"HEAD DONE best={best:.4f} ({time.time() - t0:.0f}s)", flush=True)
