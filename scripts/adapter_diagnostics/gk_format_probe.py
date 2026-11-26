#!/usr/bin/env python3
"""Does GK knowledge exist but get blocked by the A/B/C/D letter format?

Same 25 facts, three probes:
  LETTER  : current MCQ -> argmax over A/B/C/D logits (baseline)
  TEXT    : 'Question: q\\nAnswer:' -> teacher-forced logprob of each option TEXT
  BLANK   : 'q ->' fill-in continuation -> teacher-forced logprob of each option TEXT

If TEXT/BLANK >> LETTER, the knowledge is present and the letter-MCQ mapping is
the broken pattern -- a fixable bottleneck, not absent ability.
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
from general_knowledge import QUESTIONS as GQ, KEY as GK, LETTERS
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache


def score_continuations(m, tok, items):
    """items = [(prompt, [continuation strings])]. Batched by total token length.
    Returns [argmax index] using summed token logprob per continuation."""
    groups = {}
    # precompute token ids
    recs = []  # (orig_i, prompt_ids, cont_ids_list)
    for i, (pr, conts) in enumerate(items):
        pid = tok.encode(pr, add_special_tokens=False)
        cids = [tok.encode(" " + c.strip(), add_special_tokens=False) for c in conts]
        recs.append((i, pid, cids))
    results = [None] * len(items)
    # group by (prompt_len + cont_len) so every seq in a batch is equal length
    buckets = {}
    for i, pid, cids in recs:
        for j, c in enumerate(cids):
            L = len(pid) + len(c)
            buckets.setdefault(L, []).append((i, j, pid, c))
    for L, rows in buckets.items():
        seqs = [pid + c for _, _, pid, c in rows]
        lens = [len(pid) for _, _, pid, _ in rows]
        with torch.inference_mode():
            out = m(torch.tensor(seqs))[0]  # [batch, L, vocab]
        logp = F.log_softmax(out.float(), dim=-1)
        for k, (i, j, pid, c) in enumerate(rows):
            pl = lens[k]
            s = 0.0
            for t in range(pl, L):
                s += float(logp[k, t - 1, seqs[k][t]])
            if results[i] is None:
                results[i] = [None] * len(items[i][1])
            results[i][j] = s
    return [max(range(len(sc)), key=lambda j: sc[j]) if sc else -1 for sc in results]


def score_continuations_norm(m, tok, items):
    """Length-normalized variant: mean logprob per token (avoids short-option bias)."""
    buckets = {}
    recs = []
    for i, (pr, conts) in enumerate(items):
        pid = tok.encode(pr, add_special_tokens=False)
        cids = [tok.encode(" " + c.strip(), add_special_tokens=False) for c in conts]
        recs.append((i, pid, cids))
    results = [None] * len(items)
    for i, pid, cids in recs:
        for j, c in enumerate(cids):
            L = len(pid) + len(c)
            buckets.setdefault(L, []).append((i, j, pid, c))
    for L, rows in buckets.items():
        seqs = [pid + c for _, _, pid, c in rows]
        lens = [len(pid) for _, _, pid, _ in rows]
        with torch.inference_mode():
            out = m(torch.tensor(seqs))[0]
        logp = F.log_softmax(out.float(), dim=-1)
        for k, (i, j, pid, c) in enumerate(rows):
            pl = lens[k]
            Lk = len(seqs[k])
            s = sum(float(logp[k, t - 1, seqs[k][t]]) for t in range(pl, Lk)) / max(
                1, Lk - pl
            )
            if results[i] is None:
                results[i] = [None] * len(items[i][1])
            results[i][j] = s
    return [max(range(len(sc)), key=lambda j: sc[j]) if sc else -1 for sc in results]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path(r"C:\02_QUILLAN\exports\quillan-mini-math40-rank64.safetensors"),
    )
    a = ap.parse_args()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(a.ckpt)
    enable_cache(m)

    # LETTER baseline
    letter_ok = 0
    for i, (q, opts, _) in enumerate(GQ):
        pr = (
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:"
        )
        ids = tok.encode(pr, add_special_tokens=False)
        with torch.inference_mode():
            lg = m(torch.tensor([ids]))[0][0, -1, lids]
        letter_ok += int(int(lg.argmax()) == GK[i])

    # TEXT: no letter symbol
    text_items = [(f"Question: {q}\nAnswer:", opts) for q, opts, _ in GQ]
    text_pred = score_continuations(m, tok, text_items)
    text_ok = sum(int(p == GK[i]) for i, p in enumerate(text_pred))

    # BLANK: direct fill-in
    blank_items = [(f"{q} ->", opts) for q, opts, _ in GQ]
    blank_pred = score_continuations(m, tok, blank_items)
    blank_ok = sum(int(p == GK[i]) for i, p in enumerate(blank_pred))

    text_norm = score_continuations_norm(m, tok, text_items)
    text_norm_ok = sum(int(p == GK[i]) for i, p in enumerate(text_norm))

    n = len(GQ)
    print(f"ckpt={a.ckpt.name}")
    print(f"[LETTER] {letter_ok}/{n} ({100 * letter_ok / n:.0f}%)")
    print(
        f"[TEXT]   {text_ok}/{n} ({100 * text_ok / n:.0f}%)  [length-norm {text_norm_ok}/{n}]"
    )
    print(f"[BLANK]  {blank_ok}/{n} ({100 * blank_ok / n:.0f}%)")
    best = max(text_ok, blank_ok)
    print(
        f"[VERDICT] chance=25% | {'KNOWLEDGE PRESENT - LETTER FORMAT IS THE BOTTLENECK' if best > letter_ok + 4 else 'NO EVIDENCE OF HIDDEN KNOWLEDGE VIA FORMAT'}"
    )


if __name__ == "__main__":
    main()
