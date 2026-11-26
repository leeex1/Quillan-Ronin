#!/usr/bin/env python3
"""Log each donor's MCQ answer distribution on a calibration set and measure
whether the four donors AGREE. If they agree, a consensus target is meaningful
for closed-form distillation; if they conflict, no merge can be correct.
Light: one forward pass per prompt per donor, no gradients.
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
from heldout_fresh40 import prompts as p1, KEY as K1
from heldout_fresh40b import prompts as p2, KEY as K2

torch.set_num_threads(2)

DONORS = {
    "qwen": r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775",
    "llama": r"E:\source_models\llama_1b",
    "bitnet": r"E:\source_models\bitnet_2b_4t",
    "falcon": r"C:\Users\Admin\AppData\Local\Temp\opencode\falcon32",
}


def letter_ids(tok):
    out = []
    for L in "ABCD":
        for c in (f" {L}", L, L.lower(), f" {L.lower()}"):
            e = tok.encode(c, add_special_tokens=False)
            if len(e) == 1:
                out.append(e[0])
                break
        else:
            out.append(-1)
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # calibration = 40 prompts from bank1
    tasks = []
    from tokenizers import Tokenizer as HFTok

    for name, dpath in DONORS.items():
        tj = str(Path(dpath) / "tokenizer.json")
        htok = HFTok.from_file(tj)

        class Shim:
            def __init__(self, h):
                self.h = h

            def encode(self, text, add_special_tokens=True):
                return self.h.encode(text, add_special_tokens=add_special_tokens).ids

        tok = Shim(htok)
        lids = letter_ids(tok)
        seqs = []
        for pf in (p1, p2):
            for p in pf(tok):
                seqs.append(p)
        tasks.append((dpath, tok, lids, seqs))
    results = {}
    for name, (dpath, tok, lids, seqs) in zip(DONORS, tasks):
        if name == "bitnet":
            print(f"loading {name}... (eager BitNet loader)", flush=True)
            from bitnet_loader import load_bitnet

            m = load_bitnet(dpath)
        else:
            print(f"loading {name}...", flush=True)
            m = AutoModelForCausalLM.from_pretrained(
                dpath,
                local_files_only=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            ).eval()
        preds = []
        with torch.inference_mode():
            for s in seqs:
                lg = m(torch.tensor([s]))[0][0, -1, lids].float()
                preds.append(int(lg.argmax()))
        results[name] = preds
        del m
        print(
            f"  {name}: predicted dist={[preds.count(j) for j in range(4)]}", flush=True
        )
    # pairwise agreement (only loaded donors)
    names = [n for n in DONORS if results[n] and results[n][0] is not None]
    print("\npairwise agreement:", flush=True)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = results[names[i]], results[names[j]]
            agree = sum(1 for x, y in zip(a, b) if x == y)
            print(
                f"  {names[i]:7s} vs {names[j]:7s}: {agree}/{len(a)} ({100 * agree / len(a):.0f}%)",
                flush=True,
            )
    # consensus vote accuracy
    K = K1 + K2
    cons = []
    for t in range(len(K)):
        votes = [results[n][t] for n in names]
        cons.append(max(set(votes), key=votes.count))
    acc = sum(1 for c, g in zip(cons, K) if c == g)
    print(
        f"\nmajority-vote consensus accuracy: {acc}/{len(K)} ({100 * acc / len(K):.1f}%)",
        flush=True,
    )
    # save
    import json

    json.dump(
        {k: [int(x) for x in v] for k, v in results.items()},
        open(ROOT / "exports" / "donor_predictions.json", "w"),
    )
    print("saved donor_predictions.json", flush=True)


if __name__ == "__main__":
    main()
