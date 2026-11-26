#!/usr/bin/env python3
"""Training-style evaluation metrics for weight surgery (no gradients).

Per the user's specification:
  - accuracy   : fraction of correct answers (argmax == gold)
  - loss       : 1 - mean p(correct)            [closeness to target V]
  - CE         : -log p(correct), letter-normalized  [residual noise]
  - perplexity : exp(CE)                         [model confusion, 1..4]

Also reports full-vocabulary CE/PPL as a degeneracy guard, and the letter
prior distribution (argmax spread) to catch answer-bias artifacts.

Two views:
  - score over the 4 letter tokens only (MCQ decision metric)
  - score over the entire vocabulary (does the model stay a language model?)
"""

from __future__ import annotations
import sys, math, argparse
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache

torch.set_num_threads(1)


def evaluate(ck: Path, bank: str):
    m = load_model(ck)
    enable_cache(m)
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)

    if bank == "fresh40":
        from heldout_fresh40 import prompts, KEY
    elif bank == "fresh40b":
        from heldout_fresh40b import prompts, KEY
    elif bank == "fresh40c":
        from heldout_fresh40c import prompts, KEY
    elif bank == "final40":
        from heldout_final40 import prompts, KEY
    else:
        from general_knowledge import format_q, KEY as KEY25

        prompts = lambda t: [
            t.encode(format_q(i), add_special_tokens=False) for i in range(len(KEY25))
        ]
        KEY = KEY25
    ids = prompts(tok)
    n = len(ids)

    acc = 0
    p_correct = []
    ce_letter = []
    ce_vocab = []
    dist = [0, 0, 0, 0]
    with torch.inference_mode():
        for seq, gold in zip(ids, KEY):
            lg = m(torch.tensor([seq]))[0][0, -1, :].float()  # full vocab
            lp = F.log_softmax(lg, dim=-1)
            # letter-normalized distribution (MCQ decision surface)
            ll = lg[lids]
            lprob = F.log_softmax(ll, dim=-1)
            pi = int(ll.argmax())
            dist[pi] += 1
            acc += int(pi == gold)
            p_correct.append(lprob[gold].exp().item())
            ce_letter.append(-lprob[gold].item())
            ce_vocab.append(
                -lp[lids[gold]].item()
            )  # gold is answer index, not token id

    pl = sum(p_correct) / n
    cel = sum(ce_letter) / n
    cev = sum(ce_vocab) / n
    return {
        "bank": bank,
        "n": n,
        "accuracy": f"{acc}/{n}",
        "acc_pct": 100 * acc / n,
        "loss_1_minus_pcorrect": round(1 - pl, 4),
        "CE_letter": round(cel, 4),
        "PPL_letter": round(math.exp(min(cel, 20)), 3),
        "CE_vocab": round(cev, 4),
        "PPL_vocab": round(math.exp(min(cev, 20)), 2),
        "argmax_dist": dist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument(
        "--bank",
        default="fresh40",
        choices=["fresh40", "fresh40b", "fresh40c", "final40", "gk25"],
    )
    a = ap.parse_args()
    r = evaluate(Path(a.ck), a.bank)
    print(
        f"[{Path(a.ck).name}] " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True
    )


if __name__ == "__main__":
    main()
