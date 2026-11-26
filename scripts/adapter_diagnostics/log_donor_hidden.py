#!/usr/bin/env python3
"""Phase A of closed-form four-donor distillation: log each donor's final hidden
state on a calibration set. Donors loaded one at a time (RAM-safe). Saves per-donor
hidden states to disk for the WSP-consensus fit in Phase B.

BitNet uses the eager monkeypatch loader; if its forward is unreliable we still
log it but Phase B can weight donors by validity.
"""

from __future__ import annotations
import sys, json
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
from heldout_final40 import prompts as pf, KEY

torch.set_num_threads(2)
DONORS = {
    "qwen": r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775",
    "llama": r"E:\source_models\llama_1b",
    "falcon": r"C:\Users\Admin\AppData\Local\Temp\opencode\falcon32",
    "bitnet": r"E:\source_models\bitnet_2b_4t",
}
OUT = ROOT / "exports" / "donor_hidden_cache"
OUT.mkdir(exist_ok=True)


def load_donor(name, path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if name == "bitnet":
        from bitnet_loader import load_bitnet

        m = load_bitnet(path)
        tok = AutoTokenizer.from_pretrained(
            r"E:\source_models\llama_1b", local_files_only=True
        )
        return m, tok
    from tokenizers import Tokenizer as HFTok

    h = HFTok.from_file(str(Path(path) / "tokenizer.json"))

    class Shim:
        def __init__(self, h):
            self.h = h

        def encode(self, t, add_special_tokens=True):
            return self.h.encode(t, add_special_tokens=add_special_tokens).ids

    tok = Shim(h)
    m = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=False
    ).eval()
    return m, tok


def main():
    # calibration = all 40 final-bank prompts (used only as calibration; validation is separate fresh bank)
    for name, path in DONORS.items():
        print(f"loading {name}...", flush=True)
        try:
            m, tok = load_donor(name, path)
        except Exception as e:
            print(f"  LOAD FAILED {name}: {str(e)[:120]}", flush=True)
            continue
        cap = {}
        # hook last layer output (final hidden before norm)
        last = getattr(m, "model", m).layers[-1]
        hk = last.register_forward_hook(
            lambda mod, inp, out: cap.__setitem__(
                "h", out[0] if isinstance(out, tuple) else out
            )
        )
        seqs = pf(tok)
        H = []
        with torch.inference_mode():
            for s in seqs:
                m(torch.tensor([s]))
                H.append(cap["h"][0, -1, :].detach().float().cpu())
        hk.remove()
        H = torch.stack(H)
        torch.save(H, OUT / f"{name}_hidden.pt")
        print(
            f"  {name}: logged {tuple(H.shape)} norm={H.norm(dim=1).mean().item():.2f}",
            flush=True,
        )
        del m
    print("done", flush=True)


if __name__ == "__main__":
    main()
