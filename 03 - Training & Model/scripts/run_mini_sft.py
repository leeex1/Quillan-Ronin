"""Mini SFT: resume stream2 best, structured dialogue+spec, shuffled global index,
1000 steps, LR 2e-5->5e-7, BEST-only. ~8h."""
import sys
import time
import random
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
CKPT = REPO / "checkpoints" / "hf_restore" / "mini_sft_best.pt"
MANIFEST = REPO / "training_data" / "mini_structured.pt"
OUT = REPO / "checkpoints" / "hf_restore" / "mini_sft_best.pt"
STEPS, SEQ, LR0, LR1, B = 1000, 512, 2e-5, 5e-7, 2
man = torch.load(MANIFEST, map_location="cpu", weights_only=True)
shards = [Path(s) for s in man["shards"]]
nval = max(1, len(shards) // 5)
tr_sh, va_sh = shards[:-nval], shards[-nval:]
print(f"train_shards={len(tr_sh)} val_shards={len(va_sh)} total={man['total']}", flush=True)
pairs = []
for si, sh in enumerate(tr_sh):
    n = torch.load(sh, map_location="cpu", weights_only=True).shape[0]
    pairs.extend((si, ri) for ri in range(n))
random.Random(4242).shuffle(pairs)
print(f"pairs={len(pairs)}", flush=True)
cache = {}


def get_row(si, ri):
    if si not in cache:
        cache[si] = torch.load(tr_sh[si], map_location="cpu", weights_only=True)
        if len(cache) > 3:
            cache.pop(next(iter(cache)))
    return cache[si][ri]


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
va_rows = torch.cat([torch.load(va_sh[i], map_location="cpu", weights_only=True)[:8]
                     for i in range(min(2, len(va_sh)))])[:16]
for s in range(STEPS):
    idx = [(s * B + k) % len(pairs) for k in range(B)]
    x = torch.stack([get_row(*pairs[i]) for i in idx])[:, :SEQ]
    opt.zero_grad()
    o = m(x)
    logits = o[0] if isinstance(o, tuple) else o
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    sched.step()
    if (s + 1) % 20 == 0:
        m.eval()
        with torch.no_grad():
            vx = va_rows[:, :SEQ]
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
                        "meta": {"val": best, "step": s + 1, "stage": "sft1000", "seq": SEQ}}, OUT)
            print(f"SAVED val={best:.4f}", flush=True)
        m.train()
print(f"SFT DONE best={best:.4f} ({time.time() - t0:.0f}s)", flush=True)
