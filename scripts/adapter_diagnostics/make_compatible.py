#!/usr/bin/env python3
"""MAKE THEM COMPATIBLE: full-rank activation alignment + weight transplant.

Fixes the earlier broken attempt:
  - used only last-token position; now ALL positions across many sequences
  - 256 samples -> rank-deficient; now >=2048 samples for full rank
  - starts with the two SAME-SHAPE donors (Llama/Falcon: 2048h/8192ffn)
    aligned by orthogonal Procrustes (true change of basis, preserves geometry)

Fit per-layer rotation R: X_donor @ R ~= X_host  (both 2048-d)
Transform donor weights into host basis: W' = R^T @ W @ R
Alpha-blend into host, measure CE. Closed-form, no training.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(2)
LETTERS = ["A", "B", "C", "D"]
HOST_PATH = r"E:\source_models\llama_1b"
DONOR_PATH = r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf"


def procrustes(X, Y):
    """R minimizing ||X R - Y||_F, orthogonal. X,Y:(n,d)."""
    C = X.T @ Y
    U, _, Vt = torch.linalg.svd(C)
    return U @ Vt


def capture_all_positions(model, seqs, layer_idx, sub="h"):
    """Hidden (or gate output) at ALL positions for all sequences."""
    caps = {}
    layers = model.model.layers
    if sub == "h":
        h = layers[layer_idx].register_forward_hook(
            lambda m, i, o: caps.__setitem__("v", o[0] if isinstance(o, tuple) else o)
        )
    else:
        h = layers[layer_idx].mlp.gate_proj.register_forward_hook(
            lambda m, i, o: caps.__setitem__("v", o)
        )
    out = []
    with torch.inference_mode():
        for s in seqs:
            model(torch.tensor([s]))
            out.append(caps["v"][0].detach())
    h.remove()
    return torch.cat(out, dim=0).float()  # (total_tokens, d)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tokenizers import Tokenizer

    ht = Tokenizer.from_file(r"E:\source_models\llama_1b\tokenizer.json")
    dt = Tokenizer.from_file(
        r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\tokenizer.json"
    )
    hv, dv = ht.get_vocab(), dt.get_vocab()
    shared = sorted(set(hv) & set(dv))
    print(f"shared token strings={len(shared)}", flush=True)

    # build diverse sequences from shared tokens (fixed length so all positions align)
    LSEQ = 32
    NSEQ = 160
    seqs = []
    for i in range(NSEQ):
        toks = [shared[(i * LSEQ + j) % len(shared)] for j in range(LSEQ)]
        seqs.append(toks)
    hseq = [[hv[s] for s in q] for q in seqs]
    dseq = [[dv[s] for s in q] for q in seqs]
    print(
        f"sequences={NSEQ} x {LSEQ} = {NSEQ * LSEQ} token positions (full rank for d=2048)",
        flush=True,
    )

    host = AutoModelForCausalLM.from_pretrained(
        HOST_PATH, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    from transformers import AutoConfig

    print("  loading donor config...", flush=True)
    dcfg = AutoConfig.from_pretrained(DONOR_PATH, local_files_only=True)
    dcfg.torch_dtype = torch.float32
    print("  building donor from config...", flush=True)
    donor = AutoModelForCausalLM.from_config(dcfg).to(torch.float32).eval()
    from safetensors.torch import load_file

    print("  loading donor safetensors...", flush=True)
    dsd = load_file(str(Path(DONOR_PATH) / "model.safetensors"))
    donor.load_state_dict({k: v.float() for k, v in dsd.items()}, strict=False)
    HL, DL = len(host.model.layers), len(donor.model.layers)
    print(
        f"host layers={HL} donor layers={DL}  host_ffn=8192 donor_ffn=8192 (same shape)",
        flush=True,
    )

    print("\n=== fit full-rank rotations per layer ===", flush=True)
    Rres = {}
    with torch.inference_mode():
        for li in range(HL):
            dl = round(li * (DL - 1) / max(HL - 1, 1))
            X = capture_all_positions(donor, dseq, dl, "h")
            Y = capture_all_positions(host, hseq, li, "h")
            R = procrustes(X, Y)
            # held-out: split sequences
            n = len(seqs)
            cos = F.cosine_similarity(X @ R, Y, dim=-1).mean().item()
            Rres[li] = (R, dl)
            if li in (0, 4, 8, 12, 15):
                print(
                    f"  L{li}<-{dl}: cos(X@R,Y)={cos:.4f} rank(X)={torch.linalg.matrix_rank(X).item()}",
                    flush=True,
                )

    print("\n=== build transplanted host weights (all 7 matrices) ===", flush=True)
    from safetensors import safe_open

    tensors = {}
    with safe_open(
        r"E:\source_models\llama_1b\model.safetensors", framework="pt", device="cpu"
    ) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).float()
    dstate = donor.state_dict()
    trans = {}
    for li, (R, dl) in Rres.items():
        Rt = R.T
        for sub in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            hk = (
                f"model.layers.{li}.self_attn.{sub}.weight"
                if sub in ("q_proj", "k_proj", "v_proj", "o_proj")
                else f"model.layers.{li}.mlp.{sub}.weight"
            )
            dk = (
                f"model.layers.{dl}.self_attn.{sub}.weight"
                if sub in ("q_proj", "k_proj", "v_proj", "o_proj")
                else f"model.layers.{dl}.mlp.{sub}.weight"
            )
            if hk not in tensors or dk not in dstate:
                continue
            Wd = dstate[dk].float()  # (out, in) host-shaped
            trans[hk] = (Rt @ Wd.T @ R).T  # W' = R^T W R in row convention
    print(f"  transplanted {len(trans)} tensors", flush=True)

    tok = AutoTokenizer.from_pretrained(HOST_PATH, local_files_only=True)
    lids = [tok.encode(f" {L}", add_special_tokens=False)[0] for L in LETTERS]
    banks = {}
    for bn in ("fresh40", "final40"):
        mod = __import__(
            {"fresh40": "heldout_fresh40", "final40": "heldout_final40"}[bn]
        )
        banks[bn] = (mod.prompts(tok), mod.KEY)

    print("\n=== alpha sweep (Falcon -> Llama, full-rank rotation) ===", flush=True)
    with torch.inference_mode():
        for alpha in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
            sd = {k: v.clone() for k, v in tensors.items()}
            for k, Wt in trans.items():
                sd[k] = sd[k] + alpha * (Wt - sd[k])
            host.load_state_dict(sd)
            line = [f"alpha={alpha:.2f}"]
            for bn, (seqs_b, key) in banks.items():
                acc = 0
                cl = []
                for s, g in zip(seqs_b, key):
                    lp = F.log_softmax(
                        host(torch.tensor([s]))[0][0, -1, lids].float(), dim=-1
                    )
                    acc += int(lp.argmax() == g)
                    cl.append(-lp[g].item())
                line.append(f"{bn}={acc}/{len(key)} CE={sum(cl) / len(cl):.4f}")
            print("  " + "  ".join(line), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
