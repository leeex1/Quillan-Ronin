"""ckpt_state: never-start-over checkpointing. Full state + versioned bests, no overwrites."""
import json
from pathlib import Path
import torch

CKPT_DIR = Path(r"C:\02_QUILLAN\checkpoints\hf_restore")


def save_state(path, model, opt=None, sched=None, best=None, step=None, meta=None):
    d = {"model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
         "best": best, "step": step, "meta": meta or {}}
    if opt is not None:
        d["opt_state_dict"] = opt.state_dict()
    if sched is not None:
        d["sched_state_dict"] = sched.state_dict()
    torch.save(d, path)
    return path


def load_state(path, model, opt=None, sched=None):
    bd = torch.load(path, map_location="cpu", weights_only=True)
    sd = bd.get("model_state_dict", bd)
    missing, unexp = model.load_state_dict(sd, strict=False)
    if opt is not None and "opt_state_dict" in bd:
        opt.load_state_dict(bd["opt_state_dict"])
    if sched is not None and "sched_state_dict" in bd:
        sched.load_state_dict(bd["sched_state_dict"])
    return {"missing": len(missing), "unexp": len(unexp),
            "best": bd.get("best"), "step": bd.get("step", 0),
            "meta": bd.get("meta", {})}


def save_best(tag, model, opt, sched, val, step, meta=None):
    """Versioned best: mini_{tag}_best_step{step}_val{val}.pt + latest pointer. Never overwrites."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"mini_{tag}_best_step{step}_val{val:.4f}.pt"
    p = CKPT_DIR / fname
    save_state(p, model, opt, sched, best=val, step=step, meta=meta)
    ptr = CKPT_DIR / f"mini_{tag}_latest.json"
    ptr.write_text(json.dumps({"file": fname, "val": val, "step": step}), encoding="utf-8")
    return p


def latest(tag):
    ptr = CKPT_DIR / f"mini_{tag}_latest.json"
    if ptr.exists():
        return json.loads(ptr.read_text(encoding="utf-8"))
    return None
