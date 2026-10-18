#!/usr/bin/env python3
import os, sys, torch, tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_hyper_tuned_v531.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
model.eval()

prompts = [
    "<|system|>\nYou are Quillan-Ronin.\n<|user|>\nHello! Who are you?\n<|assistant|>\nI am",
    "<|system|>\nYou are Quillan-Ronin.\n<|user|>\nWhat is Python?\n<|assistant|>\nPython is",
    "<|system|>\nYou are Quillan-Ronin.\n<|user|>\nWhat is 2 + 2?\n<|assistant|>\n2 + 2 is"
]

for p in prompts:
    toks = enc.encode(p)
    gen = list(toks)
    with torch.no_grad():
        for _ in range(40):
            inp = torch.tensor([gen[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple): logits = logits[0]
            curr = logits[:, -1, :].clone()
            if len(gen) > 0: curr[0, gen[-1]] -= 50.0
            next_tok = torch.argmax(curr, dim=-1).item()
            gen.append(next_tok)
            if next_tok == 50256: break
    print("="*60, flush=True)
    print("PROMPT:", p.split("<|assistant|>\n")[-1], flush=True)
    print("GENERATED:", enc.decode(gen[len(toks):]), flush=True)
