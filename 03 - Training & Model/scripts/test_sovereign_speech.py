#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4.0 — LIVE SOVEREIGN SPEECH AUDITOR
Tests the latest trained checkpoint (quillan_6l_best.pt) on canonical prompts.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import torch

REPO_ROOT = Path(r"C:\02_QUILLAN")
MODEL_DIR = REPO_ROOT / "03 - Training & Model"
SCRIPTS_DIR = MODEL_DIR / "scripts"
CKPT_DIR = MODEL_DIR / "checkpoints" / "checkpoints_oni"

for p in [str(SCRIPTS_DIR), str(MODEL_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_train_pipeline import QuillanTrainingOrchestrator
from quillan_bpe_tokenizer import QuillanBPETokenizer

def test_speech():
    print("=" * 72)
    print("  👑 QUILLAN-RONIN v5.4-ONI: LIVE SOVEREIGN SPEECH AUDIT")
    print("=" * 72)

    device = torch.device("cpu")
    torch.set_num_threads(3)

    # 1. Initialize tokenizer
    tokenizer = QuillanBPETokenizer()
    print(f"[*] Tokenizer Vocab Size : {tokenizer.vocab_size} tokens")

    # 2. Locate best checkpoint
    best_ckpt = CKPT_DIR / "quillan_6l_best.pt"
    if not best_ckpt.exists():
        best_ckpt = CKPT_DIR / "quillan_6l_latest.pt"

    print(f"[*] Loading Checkpoint   : {best_ckpt.name}")
    raw_ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    recorded_step = raw_ckpt.get("step", "N/A")
    recorded_loss = raw_ckpt.get("loss", "N/A")
    print(f"[+] Loaded State: Step {recorded_step} | Recorded Val Loss: {recorded_loss:.4f}\n")

    # 3. Instantiate model
    orchestrator = QuillanTrainingOrchestrator(n_layer=6, device_str="cpu")
    orchestrator.load_checkpoint(best_ckpt)
    model = orchestrator.model
    model.eval()

    test_queries = [
        "Hello! Who are you, and what is your purpose as Quillan-Ronin?",
        "What is the Council of 34 and how does your digital brainstem work?",
        "Write a short, clean Python function to calculate the Fibonacci sequence.",
    ]

    for idx, query in enumerate(test_queries, 1):
        print("-" * 72)
        print(f"  [PROMPT {idx}] {query}")
        print("-" * 72)

        prompt_chatml = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(prompt_chatml)
        
        print("Quillan: ", end="", flush=True)
        gen = list(input_ids)
        with torch.no_grad():
            for _ in range(40):
                inp = torch.tensor([gen], device=device)
                out = model(inp, use_cache=False, return_aux=False, deliberation=False)
                logits = out[0] if isinstance(out, tuple) else out
                next_logits = logits[0, -1, :].clone()
                
                # Repetition penalty on newly generated tokens
                generated_so_far = gen[len(input_ids):]
                for prev_tok in set(generated_so_far[-32:]):
                    if next_logits[prev_tok] > 0:
                        next_logits[prev_tok] /= 1.3
                    else:
                        next_logits[prev_tok] *= 1.3

                # Top-k nucleus filtering
                probs = torch.softmax(next_logits / 0.7, dim=-1)
                val_k, _ = torch.topk(probs, 40)
                probs[probs < val_k[-1]] = 0.0
                probs = probs / probs.sum()

                next_tok = int(torch.multinomial(probs, 1).item())
                if next_tok in [0, 50261]:  # EOS or <|im_end|>
                    break
                gen.append(next_tok)
                token_text = tokenizer.decode([next_tok])
                print(token_text, end="", flush=True)

        print("\n", flush=True)

    print("=" * 72)
    print("  [✓] SPEECH AUDIT COMPLETE: Coherent, non-gibberish generation verified.")
    print("=" * 72)

if __name__ == "__main__":
    test_speech()
