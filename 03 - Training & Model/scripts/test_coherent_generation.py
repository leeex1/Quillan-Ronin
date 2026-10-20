#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test generation with the newly assembled coherent ternary base model.
"""
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

def main():
    ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_coherent_base_ternary.pt"
    print(f"Loading coherent base checkpoint: {ckpt_path}...")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = QuillanOniConfig.mini_6l()
    model = QuillanRoninOni(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("✓ Model loaded successfully!")

    # Load tokenizer
    llama_tok_path = hf_hub_download(repo_id="unsloth/Llama-3.2-1B-Instruct", filename="tokenizer.json")
    tokenizer = Tokenizer.from_file(llama_tok_path)

    prompt = "The code of the samurai is"
    input_ids = tokenizer.encode(prompt).ids
    input_ids = [min(t, cfg.vocab_size - 1) for t in input_ids]
    x = torch.tensor([input_ids], dtype=torch.long)

    print(f"\nPrompt: '{prompt}'")
    print("Generating next tokens (Greedy/Argmax):")

    generated = list(input_ids)
    with torch.no_grad():
        for _ in range(25):
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            next_token = torch.argmax(logits[0, -1, :]).item()
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)

    decoded = tokenizer.decode(generated)
    print(f"\nFull Output: {decoded}\n")

if __name__ == "__main__":
    main()
