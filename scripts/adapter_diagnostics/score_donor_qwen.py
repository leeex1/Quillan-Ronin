#!/usr/bin/env python3
"""Decisive test: score STOCK Qwen 0.5B on the same balanced MCQ banks as the
Quillan mini. If stock donor >> Quillan, knowledge was lost in conversion
(fixable closed-form). If both ~chance, it's a capacity/task-format ceiling.
"""

from __future__ import annotations
import sys, math
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(DIAG),):
    if p not in sys.path:
        sys.path.insert(0, p)
from heldout_fresh40 import prompts as p1, KEY as K1
from heldout_fresh40b import prompts as p2, KEY as K2
from heldout_fresh40c import prompts as p3, KEY as K3

torch.set_num_threads(1)
QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    m = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    lids = []
    for L in "ABCD":
        e = tok.encode(f" {L}", add_special_tokens=False)
        lids.append(e[0] if len(e) == 1 else tok.encode(L, add_special_tokens=False)[0])
    print(
        f"qwen letter_ids={lids} decoded={[tok.decode([i]) for i in lids]}", flush=True
    )

    for name, pf, KEY in (
        ("fresh40", p1, K1),
        ("fresh40b", p2, K2),
        ("fresh40c", p3, K3),
    ):
        ids = pf(tok)
        ok, ces, dist = 0, [], [0, 0, 0, 0]
        with torch.inference_mode():
            for seq, gold in zip(ids, KEY):
                lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
                pi = int(lg.argmax())
                dist[pi] += 1
                ok += int(pi == gold)
                ces.append(-F.log_softmax(lg, -1)[gold].item())
        ce = sum(ces) / len(ces)
        print(
            f"[STOCK QWEN 0.5B] {name}: {ok}/{len(KEY)} CE={ce:.4f} PPL={math.exp(ce):.3f} dist={dist}",
            flush=True,
        )


if __name__ == "__main__":
    main()
