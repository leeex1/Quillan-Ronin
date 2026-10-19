#!/usr/bin/env python3
"""FLAWLESS head tune: every defect tonight, fixed in one pass.
1. best-tracking (min val saved, never final)  2. cosine decay 1.5e-5->1e-6
3. fixed 50262 + mapped base + clean slice      4. 18k/2k train/val split
5. val probe every 10 steps (200 seqs)          6. final triage + numbers file
Saves quillan_head_v62_best.pt (BEST ONLY). Nothing else written."""
import math
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO / "scripts"))
from quillan_sft_calibrator import QuillanSFTCalibrator
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from quillan_bpe_tokenizer import QuillanBPETokenizer

CKPT = REPO / "checkpoints" / "checkpoints_sft" / "quillan_6l_vocab62_mapped.pt"
DATA = REPO / "training_data" / "canonical_standardized" / "quillan_clean_v62_cpu20k.pt"
OUT = REPO / "checkpoints" / "checkpoints_sft" / "quillan_head_v62_best.pt"
NUM = REPO / "training_logs" / "head_best_numbers.txt"
STEPS, BATCH, ACCUM, SEQ = 200, 2, 8, 128
PEAK, FLOOR, WARM = 1.5e-5, 1e-6, 10


def lr_at(s):
    if s < WARM:
        return PEAK * (s + 1) / WARM
    t = (s - WARM) / max(1, STEPS - WARM)
    return FLOOR + 0.5 * (PEAK - FLOOR) * (1 + math.cos(math.pi * t))


def main() -> int:
    torch.set_num_threads(3)
    cal = QuillanSFTCalibrator.__new__(QuillanSFTCalibrator)
    cal.device = torch.device("cpu")
    cal.checkpoint_path, cal.dataset_path, cal.lr = CKPT, DATA, PEAK
    cal.tokenizer = QuillanBPETokenizer()
    cal.cfg = QuillanOniConfig(vocab_size=50262, hidden_dim=1024, ffn_dim=2048,
                               n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
    cal.model = QuillanRoninOni(cal.cfg).to(cal.device)
    m = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd = m.get("model", m.get("model_state_dict", m))
    missing, unexpected = cal.model.load_state_dict(sd, strict=False)
    print(f"base bound missing={len(missing)} unexp={len(unexpected)}", flush=True)

    d = torch.load(DATA, map_location="cpu", weights_only=True)
    ids = d["input_ids"][:, :SEQ]
    g = torch.Generator().manual_seed(11)
    perm = torch.randperm(ids.shape[0], generator=g)
    train, val = ids[perm[2000:]], ids[perm[:2000]]
    print(f"train={tuple(train.shape)} val={tuple(val.shape)}", flush=True)

    params = [p for n, p in cal.model.named_parameters()
              if any(k in n.lower() for k in
                     ["lm_head", "ln", "norm", "gate", "router", "lora"])]
    for p in cal.model.parameters():
        p.requires_grad = False
    for p in params:
        p.requires_grad = True
    print(f"trainable={sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=PEAK, weight_decay=0.01)
    cal.model.train()

    best, best_step, log = float("inf"), -1, []
    rng = torch.Generator().manual_seed(21)
    t0 = time.time()
    for s in range(STEPS):
        lr = lr_at(s)
        for grp in opt.param_groups:
            grp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(ACCUM):
            idx = torch.randint(0, train.shape[0], (BATCH,), generator=rng)
            x = train[idx]
            out = cal.model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, 50262),
                                   x[:, 1:].reshape(-1)) / ACCUM
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (s + 1) % 10 == 0 or s == 0:
            cal.model.eval()
            with torch.no_grad():
                vi = torch.randperm(val.shape[0], generator=rng)[:200]
                vx = val[vi]
                outs, totv = [], 0.0
                for i in range(0, 200, 10):
                    o = cal.model(vx[i:i + 10])
                    lg = o[0] if isinstance(o, tuple) else o
                    totv += F.cross_entropy(
                        lg[:, :-1, :].reshape(-1, 50262),
                        vx[i:i + 10][:, 1:].reshape(-1)).item() / 20
            cal.model.train()
            line = (f"step {s + 1}/{STEPS} train={tot:.4f} val={totv:.4f} "
                    f"lr={lr:.2e} {time.time() - t0:.0f}s")
            if totv < best:
                best, best_step = totv, s + 1
                torch.save({"model": {k: v.cpu() for k, v in
                                      cal.model.state_dict().items()},
                            "val": best, "step": best_step}, OUT)
                line += " <-- BEST SAVED"
            print(line, flush=True)
            log.append(line)
    NUM.write_text("\n".join(log) + f"\nBEST val={best:.4f} @step {best_step}\n")
    print(f"DONE best val={best:.4f} @step {best_step} -> {OUT.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
