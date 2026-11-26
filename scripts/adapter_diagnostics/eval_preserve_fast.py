#!/usr/bin/env python3
"""Fast eval: load each model ONCE, score all banks, flush after every line."""

from __future__ import annotations
import sys, json, math
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
OUT = ROOT / "exports"
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(2)
LETTERS = ["A", "B", "C", "D"]
BANKS = ["fresh40", "fresh40c", "final40"]


def load_bank(n):
    m = __import__(
        {
            "fresh40": "heldout_fresh40",
            "fresh40c": "heldout_fresh40c",
            "final40": "heldout_final40",
        }[n]
    )
    return m.QUESTIONS, m.KEY, m.prompts


def letter_ids_native(tok, vocab):
    out = []
    for L in LETTERS:
        for c in (f" {L}", L):
            e = tok.encode(c, add_special_tokens=False)
            if len(e) == 1 and e[0] < vocab:
                out.append(e[0])
                break
        else:
            out.append(tok.encode(f" {L}", add_special_tokens=False)[0] % vocab)
    return out


def score_all(model, tok, qmap=None):
    vocab = model.get_input_embeddings().weight.shape[0]
    native_vocab = tok.vocab_size
    lids = letter_ids_native(tok, native_vocab)
    if qmap:
        lids = [qmap[x] for x in lids]
    res = {}
    with torch.inference_mode():
        for bn in BANKS:
            qs, key, pfn = load_bank(bn)
            seqs = pfn(tok)
            acc = 0
            cl = []
            cv = []
            dist = [0, 0, 0, 0]
            for seq, gold in zip(seqs, key):
                if qmap:
                    seq = [qmap[x] for x in seq]
                lg = model(torch.tensor([seq]))[0][0, -1, :].float()
                lp = F.log_softmax(lg, dim=-1)
                ll = lg[lids]
                lprob = F.log_softmax(ll, dim=-1)
                pi = int(ll.argmax())
                dist[pi] += 1
                acc += int(pi == gold)
                cl.append(-lprob[gold].item())
                cv.append(-lp[lids[gold]].item())
            n = len(key)
            cel = sum(cl) / n
            cev = sum(cv) / n
            res[bn] = (
                f"{acc}/{n}",
                round(cel, 4),
                round(math.exp(min(cel, 20)), 3),
                dist,
            )
            print(
                f"    {bn}: acc={acc}/{n} CE_letter={cel:.4f} PPL={math.exp(min(cel, 20)):.3f} dist={dist}",
                flush=True,
            )
    return res


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        r"E:\source_models\llama_1b", local_files_only=True
    )
    d2q = json.loads((OUT / "quillan_token_maps.json").read_text(encoding="utf-8"))
    qmap = {int(k): v for k, v in d2q["donor_to_quillan"]["llama"].items()}

    print("=== single-donor Llama baseline ===", flush=True)
    mb = AutoModelForCausalLM.from_pretrained(
        r"E:\source_models\llama_1b",
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    score_all(mb, tok, None)
    del mb

    print("=== four-donor PRESERVE flagship ===", flush=True)
    mc = AutoModelForCausalLM.from_pretrained(
        str(OUT / "quillan_flagship_compiled"),
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    score_all(mc, tok, qmap)
    del mc
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

