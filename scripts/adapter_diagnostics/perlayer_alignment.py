#!/usr/bin/env python3
"""PROPER activation alignment: fit a SEPARATE ridge map per layer on paired
hidden states (not one embedding P applied everywhere), and validate on HELD-OUT
prompts. This is the correct way to 'adjust donors to match' before import.

If per-layer held-out alignment cosine is high (>0.8), donors CAN be aligned into
the host frame and a weight transplant is viable. If it stays low, alignment
cannot rescue independent bases.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(2)
HOST = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"  # mini host (896)


def load_generic(path, tokpath=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if tokpath is None:
        tokpath = path
    m = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=False
    ).eval()
    return m, tokpath


def capture_hidden(m, seqs, nlayer, device="cpu"):
    caps = {}
    layers = getattr(m, "model", m).layers
    hk = layers[nlayer].register_forward_hook(
        lambda mod, inp, out: caps.__setitem__(
            "h", out[0] if isinstance(out, tuple) else out
        )
    )
    out = []
    import torch as T

    with T.inference_mode():
        for s in seqs:
            m(T.tensor([s]))
            out.append(caps["h"][0, -1, :].detach().cpu())
    hk.remove()
    return T.stack(out)


def fit_ridge(X, Y, ridge=1e-2):
    # y ~= x @ P ; x:(n,dx) y:(n,dy)
    G = X.T @ X + ridge * torch.eye(X.shape[1]) * X.abs().mean().clamp_min(1e-9)
    return torch.linalg.solve(G, X.T @ Y)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", required=True)
    ap.add_argument("--tok", default=None)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--layers", type=int, default=6)
    a = ap.parse_args()
    from tokenizers import Tokenizer

    # host tokenizer for shared strings
    htp = str(Path(HOST) / "tokenizer.json")
    htok = Tokenizer.from_file(htp)
    hv = htok.get_vocab()
    dtok = Tokenizer.from_file(a.tok)
    dv = dtok.get_vocab()
    shared = sorted(set(hv) & set(dv))
    print(f"shared tokens={len(shared)}", flush=True)
    # calibration prompts: single shared tokens embedded in a minimal string
    toks = shared[: a.n]

    def enc(tok, vocab, words):
        return [tok.encode(f"Q: {w} A:", add_special_tokens=False).ids for w in words]

    fit_words = toks[: a.n // 2]
    held_words = toks[a.n // 2 :]
    # load models one at a time, cache hidden
    mh, htp2 = load_generic(HOST)
    hseq = enc(htok, hv, fit_words)
    hseq_h = enc(htok, hv, held_words)
    nlayers_h = len(getattr(mh, "model", mh).layers)
    Hfit = {li: capture_hidden(mh, hseq, li) for li in range(min(a.layers, nlayers_h))}
    Hheld = {li: capture_hidden(mh, hseq_h, li) for li in Hfit}
    del mh
    md, _ = load_generic(a.donor)
    dseq = enc(dtok, dv, fit_words)
    dseq_h = enc(dtok, dv, held_words)
    nlayers_d = len(getattr(md, "model", md).layers)
    print(f"host layers={nlayers_h} donor layers={nlayers_d}", flush=True)
    for li in sorted(Hfit):
        dl = round(li * (nlayers_d - 1) / max(nlayers_h - 1, 1))
        Dfit = capture_hidden(md, dseq, dl)
        Dheld = capture_hidden(md, dseq_h, dl)
        P = fit_ridge(Dfit, Hfit[li])
        # held-out alignment quality
        pred = Dheld @ P
        tgt = Hheld[li]
        cos = torch.nn.functional.cosine_similarity(pred, tgt, dim=-1).mean().item()
        # in-sample for comparison
        cos_in = (
            torch.nn.functional.cosine_similarity(Dfit @ P, Hfit[li], dim=-1)
            .mean()
            .item()
        )
        print(
            f"layer host{li:02d}<-donor{dl:02d}: heldout_cos={cos:.3f} insample_cos={cos_in:.3f} P={tuple(P.shape)}",
            flush=True,
        )
    del md


if __name__ == "__main__":
    main()
