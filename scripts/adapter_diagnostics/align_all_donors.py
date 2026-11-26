#!/usr/bin/env python3
"""Correct + fast donor->host activation alignment test.

Fixes two bugs in perlayer_alignment.py:
  1. shared tokens matched by STRING (token id 123 differs across vocabularies)
  2. all target layers captured in ONE forward pass per prompt (not one per layer)

For each donor, fit a separate ridge map P_l: donor_hidden -> host_hidden at
several depths, and report HELD-OUT cosine. High held-out cos (>0.8) means the
donor representation can be linearly aligned into the host frame -> a
change-of-basis weight transplant is viable. Low cos means it cannot.
No training, no gradient, no weight edits.
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

HOST = r"E:\source_models\llama_1b"  # flagship base = Llama-3.2-1B (2048 hidden)
DONORS = {
    "qwen": r"E:\source_models\qwen_0.5b",
    "falcon": r"E:\source_models\falcon3_1b",
    "bitnet": r"E:\source_models\bitnet_2b_4t",
    "llama": r"E:\source_models\llama_1b",  # control: self-align should be ~1.0
}


def load_model(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    m = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=False
    ).eval()
    return m, tok


def n_layers(m):
    return len(getattr(m, "model", m).layers)


def capture_layers(m, prompts, layer_idxs):
    """One forward pass per prompt; capture hidden at all layer_idxs."""
    layers = getattr(m, "model", m).layers
    caps = {}
    hs = []
    for li in layer_idxs:
        hs.append(
            layers[li].register_forward_hook(
                (
                    lambda k: (
                        lambda mod, inp, out: caps.__setitem__(
                            k, out[0] if isinstance(out, tuple) else out
                        )
                    )
                )(li)
            )
        )
    result = {li: [] for li in layer_idxs}
    with torch.inference_mode():
        for s in prompts:
            ids = tok.encode(s, add_special_tokens=False)
            m(torch.tensor([ids]))
            for li in layer_idxs:
                result[li].append(caps[li][0, -1, :].detach().cpu())
    for h in hs:
        h.remove()
    return {li: torch.stack(v) for li, v in result.items()}


def fit_ridge(X, Y, ridge=1e-2):
    lam = ridge * X.abs().mean().clamp_min(1e-9)
    G = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
    return torch.linalg.solve(G, X.T @ Y)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--donors", default="qwen,falcon,bitnet,llama")
    a = ap.parse_args()

    global tok
    print("loading host...", flush=True)
    mh, tok = load_model(HOST)
    Lh = n_layers(mh)
    host_idxs = sorted(
        set(round(i * (Lh - 1) / max(a.layers - 1, 1)) for i in range(a.layers))
    )
    print(f"host layers={Lh} probe={host_idxs}", flush=True)

    # calibration prompts from shared token STRINGS
    hv = set(tok.get_vocab().keys())
    print(f"host vocab={len(hv)}", flush=True)

    for name in a.donors.split(","):
        path = DONORS[name]
        print(f"\n=== donor {name} ===", flush=True)
        try:
            md, dtok = load_model(path)
        except Exception as e:
            print(f"  LOAD_FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        Ld = n_layers(md)
        dv = set(dtok.get_vocab().keys())
        shared = sorted(hv & dv)
        print(
            f"  donor layers={Ld} vocab={len(dv)} shared_strings={len(shared)}",
            flush=True,
        )
        if len(shared) < a.n:
            print("  SKIP too few shared tokens", flush=True)
            del md
            continue
        words = shared[: a.n]
        fit_w, held_w = words[: len(words) // 2], words[len(words) // 2 :]
        fitp = [f"Q: {w} A:" for w in fit_w]
        heldp = [f"Q: {w} A:" for w in held_w]
        # host capture
        tok = tok  # host tokenizer for host prompts
        Hfit = capture_layers(mh, fitp, host_idxs)
        Hheld = capture_layers(mh, heldp, host_idxs)
        # donor capture (swap tokenizer)
        old_tok = tok
        tok = dtok
        try:
            for li in host_idxs:
                dl = round(li * (Ld - 1) / max(Lh - 1, 1))
                Dfit = capture_layers(md, fitp, [dl])[dl]
                Dheld = capture_layers(md, heldp, [dl])[dl]
                P = fit_ridge(Dfit, Hfit[li])
                cos_h = (
                    torch.nn.functional.cosine_similarity(Dheld @ P, Hheld[li], dim=-1)
                    .mean()
                    .item()
                )
                cos_i = (
                    torch.nn.functional.cosine_similarity(Dfit @ P, Hfit[li], dim=-1)
                    .mean()
                    .item()
                )
                print(
                    f"  h{li:02d}<-d{dl:02d}: heldout_cos={cos_h:.3f} insample_cos={cos_i:.3f}",
                    flush=True,
                )
        except Exception as e:
            print(f"  CAPTURE_FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
        finally:
            tok = old_tok
            del md


if __name__ == "__main__":
    main()
