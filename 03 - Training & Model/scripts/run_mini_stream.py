"""Mini FULL-STREAM run: resume mega best, 512-length, sharded streaming, 1000 steps, BEST-only. ~7h."""
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
CKPT = REPO / "checkpoints" / "hf_restore" / "mini_mega_best.pt"
MANIFEST = REPO / "training_data" / "mini_full_stream.pt"
OUT = REPO / "checkpoints" / "hf_restore" / "mini_stream_best.pt"
STEPS, SEQ, LR0, LR1, B = 1000, 512, 3e-5, 5e-7, 2
man = torch.load(MANIFEST, map_location="cpu", weights_only=True)
shards = [Path(s) for s in man["shards"]]
nval = max(1, len(shards) // 5)
tr_sh, va_sh = shards[:-nval], shards[-nval:]
print(f"train_shards={len(tr_sh)} val_shards={len(va_sh)} total={man['total']}", flush=True)
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
va_rows = torch.load(va_sh[0], map_location="cpu", weights_only=True)[:8]
step = 0
while step < STEPS:
    for sh in tr_sh:
        rows = torch.load(sh, map_location="cpu", weights_only=True)
        for i in range(0, len(rows) - B, B):
            if step >= STEPS:
                break
            x = rows[i:i + B][:, :SEQ]
            opt.zero_grad()
            o = m(x)
            logits = o[0] if isinstance(o, tuple) else o
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                m.eval()
                with torch.no_grad():
                    vx = va_rows[:, :SEQ]
                    vo = m(vx)
                    vl = vo[0] if isinstance(vo, tuple) else vo
                    vv = torch.nn.functional.cross_entropy(
                        vl[:, :-1, :].reshape(-1, vl.size(-1)),
                        vx[:, 1:].reshape(-1)).item()
                print(f"step={step} train={loss.item():.4f} val={vv:.4f} lr={sched.get_last_lr()[0]:.2e}",
                      flush=True)
                if vv < best:
                    best = vv
                    torch.save({"model_state_dict": {k: v.cpu().clone() for k, v in m.state_dict().items()},
                                "meta": {"val": best, "step": step, "stage": "stream1000", "seq": SEQ}}, OUT)
                    print(f"SAVED val={best:.4f}", flush=True)
                m.train()
        if step >= STEPS:
            break
print(f"STREAM DONE best={best:.4f} ({time.time() - t0:.0f}s)", flush=True)