#!/usr/bin/env python3
"""Where does the correct-answer signal appear, and where is it destroyed?

FIXED capture bug: accumulate hidden states for ALL prompts at EVERY layer
(earlier version overwrote per forward call, leaving one prompt).

For each layer, project the last-position hidden state onto the 4 letter
embeddings and compute:
  margin = logit(correct) - max(logit(wrong))   averaged over prompts
If margin is positive at some layer and collapses later -> a fixable
"signal destroyed downstream" pattern. If it never becomes positive ->
the knowledge is not in the residual stream at all (capacity limit).

NO edits. Pure measurement.
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

torch.set_num_threads(1)


def run(ck_path, label):
    m = load_model(ck_path)
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    E = m.embed_tokens.unpack_float()[lids].float()
    ids = prompts(tok)
    K = torch.tensor(KEY)

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

    print(f"\n=== {label} ({len(ids)} prompts) ===", flush=True)
    print(
        "layer | margin(correct-maxWrong) | argmax_acc | letter_bias(C-A) | h_norm",
        flush=True,
    )
    for li in range(len(m.layers)):
        H = torch.stack(caps[li]).squeeze(1)  # N x hidden
        lg = H @ E.T  # N x 4
        wrong = lg.clone()
        wrong[torch.arange(len(K)), K] = -1e9
        margin = lg[torch.arange(len(K)), K] - wrong.max(-1).values
        acc = int((lg.argmax(-1) == K).sum())
        bias = (lg[:, 2] - lg[:, 0]).mean().item()
        hn = H.norm(dim=-1).mean().item()
        print(
            f"{li:>5} | {margin.mean().item():+8.4f} | {acc:>3}/{len(K)} | "
            f"{bias:+8.4f} | {hn:7.2f}",
            flush=True,
        )


def main():
    run(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors", "MINI base")
    run(
        ROOT / "exports" / "quillan-flagship-sovereign-bitnet.safetensors",
        "FLAGSHIP base",
    )


if __name__ == "__main__":
    main()
