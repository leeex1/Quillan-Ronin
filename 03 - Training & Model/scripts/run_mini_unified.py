"""Mini UNIFIED full run: all params, 80/20 split, winner LR, BEST-only."""
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
CKPT = REPO / "checkpoints" / "hf_restore" / "quillan_6l_vocab62_base.pt"
SLICE = REPO / "training_data" / "mini_slice_unified_v4.pt"
OUT = REPO / "checkpoints" / "hf_restore" / "mini_unified_best.pt"
STEPS, SEQ, LR0, LR1 = 300, 160, 5e-5, 5e-7
ds = torch.load(SLICE, map_location="cpu", weights_only=True)
ids = ds["input_ids"] if isinstance(ds, dict) else ds
ntr = int(len(ids) * 0.8)
tr, va = ids[:ntr], ids[ntr:]
print(f"train={len(tr)} val={len(va)} (80/20)", flush=True)
cfg = QuillanOniConfig(vocab_size=50262, hidden_dim=1024, ffn_dim=2048,
                       n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
m = QuillanRoninOni(cfg)
bd = torch.load(CKPT, map_location="cpu", weights_only=True)
missing, unexp = m.load_state_dict(bd.get("model_state_dict", bd), strict=False)
print(f"bound missing={len(missing)} unexp={len(unexp)}", flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=LR0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR1)
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
    sched.step()
    if (s + 1) % 10 == 0:
        m.eval()
        with torch.no_grad():
            vx = va[:8][:, :SEQ]
            vo = m(vx)
            vl = vo[0] if isinstance(vo, tuple) else vo
            vv = torch.nn.functional.cross_entropy(
                vl[:, :-1, :].reshape(-1, vl.size(-1)),
                vx[:, 1:].reshape(-1)).item()
        print(f"step={s + 1} train={loss.item():.4f} val={vv:.4f} lr={sched.get_last_lr()[0]:.2e}",
              flush=True)
        if vv < best:
            best = vv
            torch.save({"model_state_dict": {k: v.cpu().clone() for k, v in m.state_dict().items()},
                        "meta": {"val": best, "step": s + 1, "split": "80/20"}}, OUT)
            print(f"SAVED val={best:.4f}", flush=True)
        m.train()
print(f"UNIFIED DONE best={best:.4f} ({time.time() - t0:.0f}s)", flush=True)
