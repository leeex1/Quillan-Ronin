#!/usr/bin/env python3
"""MAKE COMPATIBLE — memory-safe, full-rank residual rotation.

Design (fixes all prior failures):
  - One model loaded at a time; activations cached to disk, model freed.
  - Same-shape donors only (Llama/Falcon: 2048h, 8192ffn).
  - Full-rank ORTHOGONAL Procrustes R on the residual stream (2048x2048),
    fitted on 4096 token positions -> no rank deficiency.
  - FFN space kept IDENTICAL (R_ffn = I), so SiLU/gate nonlinearity is exact:
      gate/up host weight = R^T @ W_donor
      down   host weight = W_donor @ R
  - Alpha-blend donor MLP into host, measure CE. Closed-form, no training.
"""

from __future__ import annotations
import sys, gc
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
CACHE = ROOT / "exports" / "compat_cache"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(2)
LETTERS = ["A", "B", "C", "D"]
HOST_PATH = r"E:\source_models\llama_1b"
DONOR_PATH = r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf"
SEQ_LEN, NSEQ = 32, 128


def load_host_f32():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        HOST_PATH, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()


def load_donor_f32():
    from transformers import AutoModelForCausalLM, AutoConfig
    from safetensors.torch import load_file

    cfg = AutoConfig.from_pretrained(DONOR_PATH, local_files_only=True)
    cfg.torch_dtype = torch.float32
    m = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    sd = load_file(str(Path(DONOR_PATH) / "model.safetensors"))
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    del sd
    return m


def capture_residual(model, seqs, layer_idx):
    cap = {}
    h = model.model.layers[layer_idx].register_forward_hook(
        lambda m, i, o: cap.__setitem__("v", o[0] if isinstance(o, tuple) else o)
    )
    out = []
    with torch.inference_mode():
        for s in seqs:
            model(torch.tensor([s]))
            out.append(cap["v"][0].detach())
    h.remove()
    return torch.cat(out, dim=0).float()


def procrustes(X, Y):
    C = X.T @ Y
    U, _, Vt = torch.linalg.svd(C)
    return U @ Vt


def main():
    from tokenizers import Tokenizer

    CACHE.mkdir(exist_ok=True)
    ht = Tokenizer.from_file(r"E:\source_models\llama_1b\tokenizer.json")
    dt = Tokenizer.from_file(
        r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\tokenizer.json"
    )
    hv, dv = ht.get_vocab(), dt.get_vocab()
    shared = sorted(set(hv) & set(dv))
    seqs = [
        [shared[(i * SEQ_LEN + j) % len(shared)] for j in range(SEQ_LEN)]
        for i in range(NSEQ)
    ]
    hseq = [[hv[s] for s in q] for q in seqs]
    dseq = [[dv[s] for s in q] for q in seqs]
    print(
        f"calibration: {NSEQ}x{SEQ_LEN}={NSEQ * SEQ_LEN} positions, shared={len(shared)}",
        flush=True,
    )

    # --- PHASE 1/2: use cached activations if present ---
    cached = all(
        (CACHE / f"host_{i}.pt").exists() and (CACHE / f"donor_{i}.pt").exists()
        for i in range(16)
    )
    HL, DL = 16, 18
    if cached:
        print("\n[1-2/4] using 32 cached activation tensors", flush=True)
    else:
        print("\n[1/4] host activations...", flush=True)
        m = load_host_f32()
        HL = len(m.model.layers)
        for li in range(HL):
            torch.save(capture_residual(m, hseq, li), CACHE / f"host_{li}.pt")
        del m
        gc.collect()
        print(f"  saved {HL} host activation tensors", flush=True)
        print("\n[2/4] donor activations...", flush=True)
        m = load_donor_f32()
        DL = len(m.model.layers)
        for li in range(HL):
            dl = round(li * (DL - 1) / max(HL - 1, 1))
            torch.save(capture_residual(m, dseq, dl), CACHE / f"donor_{li}.pt")
        del m
        gc.collect()
        print(
            f"  saved {HL} donor activation tensors, host={HL} donor={DL} layers",
            flush=True,
        )

    # --- PHASE 3: fit rotations + build transplanted weights ---
    print("\n[3/4] fit full-rank rotations + transform weights...", flush=True)
    from safetensors import safe_open

    dsd = {}
    with safe_open(
        r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model.safetensors",
        framework="pt",
        device="cpu",
    ) as f:
        for k in ("model.embed_tokens.weight",):
            pass
    # load donor weights one tensor at a time via state dict names from safetensors
    from safetensors.torch import load_file

    donor_sd = load_file(
        r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model.safetensors"
    )
    trans = {}
    for li in range(HL):
        dl = round(li * (DL - 1) / max(HL - 1, 1))
        Xd = torch.load(CACHE / f"donor_{li}.pt", weights_only=True)
        Xh = torch.load(CACHE / f"host_{li}.pt", weights_only=True)
        R = procrustes(Xd, Xh)
        cos = F.cosine_similarity(Xd @ R, Xh, dim=-1).mean().item()
        Rt = R.T
        for sub in ("gate_proj", "up_proj", "down_proj"):
            dk = f"model.layers.{dl}.mlp.{sub}.weight"
            if dk not in donor_sd:
                continue
            Wd = donor_sd[dk].float()
            if sub in ("gate_proj", "up_proj"):
                # h_d = h_h @ R.T ; g = h_d @ Wd.T = h_h @ (Wd @ R).T
                trans[f"model.layers.{li}.mlp.{sub}.weight"] = Wd @ R
            else:
                # y = g @ Wd.T ; host residual y_h = y @ R = g @ (R.T @ Wd).T
                trans[f"model.layers.{li}.mlp.{sub}.weight"] = Rt @ Wd
        if li in (0, 4, 8, 12, 15):
            print(
                f"  L{li}<-{dl}: cos(Xd@R,Xh)={cos:.4f} rank(R)={torch.linalg.matrix_rank(R).item()}",
                flush=True,
            )
    del donor_sd, Xd, Xh
    gc.collect()
    torch.save(trans, CACHE / "transplanted.pt")
    print(f"  built {len(trans)} transplanted MLP tensors", flush=True)

    # --- PHASE 4: load host ONCE, alpha sweep ---
    print("\n[4/4] alpha sweep...", flush=True)
    m = load_host_f32()
    tok_h = None
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HOST_PATH, local_files_only=True)
    lids = [tok.encode(f" {L}", add_special_tokens=False)[0] for L in LETTERS]
    banks = {}
    for bn in ("fresh40", "final40"):
        mod = __import__(
            {"fresh40": "heldout_fresh40", "final40": "heldout_final40"}[bn]
        )
        banks[bn] = (mod.prompts(tok), mod.KEY)
    # only clone the tensors we touch (48 MLP matrices), not the whole 5GB model
    params = dict(m.named_parameters())
    base = {k: params[k].data.clone() for k in trans}
    with torch.inference_mode():
        for alpha in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
            for k, Wt in trans.items():
                params[k].data.copy_(base[k] + alpha * (Wt - base[k]))
            line = [f"alpha={alpha:.2f}"]
            for bn, (seqs_b, key) in banks.items():
                acc = 0
                cl = []
                for s, g in zip(seqs_b, key):
                    lp = F.log_softmax(
                        m(torch.tensor([s]))[0][0, -1, lids].float(), dim=-1
                    )
                    acc += int(lp.argmax() == g)
                    cl.append(-lp[g].item())
                line.append(f"{bn}={acc}/{len(key)} CE={sum(cl) / len(cl):.4f}")
            print("  " + "  ".join(line), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
