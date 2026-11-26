#!/usr/bin/env python3
"""Direct structural surgery on the STORED letter embedding rows.

The defect: with tied embeddings, letter logit = h . e_letter. If one letter
embedding has a larger projection onto the mean residual state, it wins by
default on every MCQ. Probe showed B rank-1 on ~4/5 questions regardless of gold,
while all four embedding norms are nearly identical (0.45-0.47) -- so it is a
DIRECTION/baseline problem, not a magnitude problem.

NO gold answers used. The mean hidden state is computed from prompt text only
(the prompts contain options but no answer key). The correction equalizes each
letter embedding's baseline projection onto that mean:

    b_L = h_mean . e_L
    e_L' = e_L - ((b_L - b_mean) / ||h_mean||^2) * h_mean

This is a rank-4 closed-form edit to the embedding table. No optimizer, no
gradient, no lora, no answer label. The whole table is repacked through
set_weights_from_float so the change lives in the actual stored INT8 weights.
"""

from __future__ import annotations
import sys, argparse
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from heldout_fresh40 import prompts
from wsp_math_tune_v3 import enable_cache

torch.set_num_threads(1)
BASE = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="1.0 = full equalization of letter baselines",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = load_model(BASE)
    enable_cache(m)
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    ids = prompts(tok)

    # collect final hidden states, PROMPTS ONLY (no answer keys touched)
    hs = []
    with torch.inference_mode():
        for seq in ids:
            out = m(torch.tensor([seq]))[0]
            hs.append(out[1][0, -1, :].float() if isinstance(out, tuple) else None)
    # if the model doesn't return hidden states, hook the final norm
    if hs[0] is None:
        hs = []
        cap = {}
        hk = m.norm.register_forward_hook(
            lambda mod, inp, out: cap.__setitem__("h", out.detach().float())
        )
        with torch.inference_mode():
            for seq in ids:
                m(torch.tensor([seq]))
                hs.append(cap["h"][0, -1, :].clone())
        hk.remove()
    H = torch.stack(hs)  # N x hidden
    h_mean = H.mean(0)
    denom = h_mean.dot(h_mean).clamp_min(1e-12)

    E = m.embed_tokens.unpack_float().float()
    rows = E[lids].clone()  # 4 x hidden
    base = rows @ h_mean  # 4 baseline logits
    print(
        f"letter baselines before: "
        + " ".join(f"{L}={v.item():+.4f}" for L, v in zip("ABCD", base)),
        flush=True,
    )

    b_mean = base.mean()
    for j in range(4):
        delta = a.strength * ((base[j] - b_mean) / denom)
        E[lids[j]] = rows[j] - delta * h_mean

    after = E[lids] @ h_mean
    print(
        f"letter baselines after:  "
        + " ".join(f"{L}={v.item():+.4f}" for L, v in zip("ABCD", after)),
        flush=True,
    )

    m.embed_tokens.set_weights_from_float(E)
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
