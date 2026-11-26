#!/usr/bin/env python3
"""Apples-to-apples: same MCQ bank, same body, only embedding/head differs.

BASELINE = original single-donor Llama (E:\\source_models\\llama_1b)
COMPILED = four-donor compiled flagship (same Llama body + compiled emb/head)

Each encodes prompts in its own token space; letter logits read at that space's
letter IDs. Reports accuracy + CE/PPL (letter and full-vocab). No training.
"""

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
torch.set_num_threads(1)
LETTERS = ["A", "B", "C", "D"]

BASELINE = Path(r"E:\source_models\llama_1b")
COMPILED = OUT / "quillan_flagship_compiled"
HOST_TOK = r"E:\source_models\llama_1b"


def load_bank(name):
    mod = {
        "fresh40": "heldout_fresh40",
        "fresh40c": "heldout_fresh40c",
        "final40": "heldout_final40",
        "fresh40b": "heldout_fresh40b",
    }[name]
    m = __import__(mod)
    return m.QUESTIONS, m.KEY, m.prompts


def letter_ids_native(tok, vocab):
    out = []
    for L in LETTERS:
        for cand in (f" {L}", L):
            e = tok.encode(cand, add_special_tokens=False)
            if len(e) == 1 and e[0] < vocab:
                out.append(e[0])
                break
        else:
            out.append(tok.encode(f" {L}", add_special_tokens=False)[0] % vocab)
    return out


def score(model, tok, qs, key, prompts_fn, qids=None, label=""):
    """qids: optional native_id -> model_id map (for compiled Quillan space)."""

    def mid(nid):
        return qids[nid] if qids is not None else nid

    vocab = model.get_input_embeddings().weight.shape[0]
    lids = [mid(x) for x in letter_ids_native(tok, base_vocab_native)]
    seqs = prompts_fn(tok)
    acc = 0
    ce_l = []
    ce_v = []
    dist = [0, 0, 0, 0]
    with torch.inference_mode():
        for seq, gold in zip(seqs, key):
            seq = [mid(x) for x in seq]
            lg = model(torch.tensor([seq]))[0][0, -1, :].float()
            lp = F.log_softmax(lg, dim=-1)
            ll = lg[lids]
            lprob = F.log_softmax(ll, dim=-1)
            pi = int(ll.argmax())
            dist[pi] += 1
            acc += int(pi == gold)
            ce_l.append(-lprob[gold].item())
            ce_v.append(-lp[lids[gold]].item())
    n = len(key)
    return dict(
        label=label,
        n=n,
        acc=f"{acc}/{n}",
        CE_letter=round(sum(ce_l) / n, 4),
        PPL_letter=round(math.exp(min(sum(ce_l) / n, 20)), 3),
        CE_vocab=round(sum(ce_v) / n, 4),
        PPL_vocab=round(math.exp(min(sum(ce_v) / n, 20)), 2),
        dist=dist,
    )


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global base_vocab_native
    tok = AutoTokenizer.from_pretrained(HOST_TOK, local_files_only=True)
    base_vocab_native = tok.vocab_size

    # native -> quillan map
    d2q = json.loads((OUT / "quillan_token_maps.json").read_text(encoding="utf-8"))
    qmap = {int(k): v for k, v in d2q["donor_to_quillan"]["llama"].items()}

    banks = ["fresh40", "fresh40c", "final40"]
    for bname in banks:
        qs, key, pfn = load_bank(bname)
        print(f"\n=== bank {bname} (n={len(key)}) ===", flush=True)
        # baseline
        mb = AutoModelForCausalLM.from_pretrained(
            str(BASELINE),
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        rb = score(mb, tok, qs, key, pfn, qids=None, label="single-donor Llama")
        print("  " + " ".join(f"{k}={v}" for k, v in rb.items()), flush=True)
        del mb
        # compiled
        mc = AutoModelForCausalLM.from_pretrained(
            str(COMPILED),
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        rc = score(mc, tok, qs, key, pfn, qids=qmap, label="four-donor compiled")
        print("  " + " ".join(f"{k}={v}" for k, v in rc.items()), flush=True)
        del mc
