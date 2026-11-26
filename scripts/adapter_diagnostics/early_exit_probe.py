#!/usr/bin/env python3
"""Early-exit probe: accuracy from EACH layer's residual state pushed through
the final RMSNorm + tied embedding (same read head, tapped before later layers).

Pure measurement, no edits. A mid-layer that scores much higher than the final
layer proves later blocks destroy a correct signal.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from heldout_fresh40 import prompts, KEY
from heldout_fresh40b import prompts as prompts_b, KEY as KEY_b

torch.set_num_threads(1)


def measure(m, E, ids, K):
    caps = {li: [] for li in range(len(m.layers))}
    hooks = []
    for li, layer in enumerate(m.layers):

        def mk(idx):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                caps[idx].append(h.detach().float()[:, -1, :].clone())

            return hook

        hooks.append(layer.register_forward_hook(mk(li)))
    with torch.inference_mode():
        for seq in ids:
            m(torch.tensor([seq]))
    for hk in hooks:
        hk.remove()
    Kt = torch.tensor(K)
    out = []
    for li in range(len(m.layers)):
        H = torch.stack(caps[li]).squeeze(1)
        Hn = m.norm(H.unsqueeze(1)).squeeze(1).float()
        lg = Hn @ E.T
        out.append(int((lg.argmax(-1) == Kt).sum()))
    return out


def probe(ck, label):
    m = load_model(ck)
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    E = m.embed_tokens.unpack_float()[lids].float()
    a1 = measure(m, E, prompts(tok), KEY)
    a2 = measure(m, E, prompts_b(tok), KEY_b)
    print(f"\n=== {label} ===", flush=True)
    print("layer | fresh40 | fresh40b | combo", flush=True)
    for li in range(len(a1)):
        print(
            f"{li:>5} | {a1[li]:>3}/40 | {a2[li]:>3}/40 | {a1[li] + a2[li]:>3}/80",
            flush=True,
        )
    bi = max(range(len(a1)), key=lambda i: a1[i] + a2[i])
    print(
        f"BEST layer={bi} combo={a1[bi] + a2[bi]}/80 ; FINAL layer={len(a1) - 1} combo={a1[-1] + a2[-1]}/80",
        flush=True,
    )


def main():
    probe(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors", "MINI base")


if __name__ == "__main__":
    main()
