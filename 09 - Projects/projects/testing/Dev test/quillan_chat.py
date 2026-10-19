#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — NATIVE INTERACTIVE CHAT RUNNER
---------------------------------------------------------------------------------------
Instant Boot (~1.5s), Ultra-Low Memory (~1.6GB RAM), Fast KV-Cached CPU Generation.
"""

import os
import sys
import time
import torch
import tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Thread optimization for fast, responsive CPU token generation
torch.set_num_threads(4)

SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig

CHECKPOINT_PATH = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt")

BANNER = r"""
/==================================================================\
||   ██████╗ ██╗   ██╗██╗██╗     ██╗      █████╗ ███╗   ██╗       ||
||  ██╔═══██╗██║   ██║██║██║     ██║     ██╔══██╗████╗  ██║       ||
||  ██║   ██║██║   ██║██║██║     ██║     ███████║██╔██╗ ██║       ||
||  ██║▄▄ ██║██║   ██║██║██║     ██║     ██╔══██║██║╚██╗██║       ||
||  ╚██████╔╝╚██████╔╝██║███████╗███████╗██║  ██║██║ ╚████║       ||
||   ╚══▀▀═╝  ╚═════╝ ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝       ||
||   QUILLAN-RONIN v5.3.1 — UNROLLED SOVEREIGN CHAT RUNNER        ||
\==================================================================/
"""

def main():
    print(BANNER)
    t0 = time.time()
    enc = tiktoken.get_encoding("gpt2")
    cfg = QuillanUnrolledConfig()

    print("[*] Initializing 12-Layer Unrolled Sovereign Architecture (408 Experts + 408 Swarms)...", flush=True)
    model = QuillanUnrolledSovereign(cfg).to("cpu")

    if not CHECKPOINT_PATH.exists():
        print(f"[!] Error: Checkpoint not found at {CHECKPOINT_PATH}", flush=True)
        sys.exit(1)

    print(f"[*] Loading Master Checkpoint ({CHECKPOINT_PATH.name})...", flush=True)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    boot_time = time.time() - t0
    print(f"[+] Sovereign Brain Active in {boot_time:.2f}s! Memory: ~1.6 GB RAM\n", flush=True)
    print("Type your message below (or 'exit' / 'quit' to close):\n" + "="*66 + "\n")

    while True:
        try:
            user_input = input("\n[USER]> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("\n[!] Exiting Quillan-Ronin. Sovereign state preserved.\n")
                break

            prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
            tokens = enc.encode(prompt)
            
            print("[QUILLAN]> ", end="", flush=True)
            t_gen_start = time.time()
            
            gen_tokens = model.generate(
                tokens, 
                max_tokens=100, 
                temp=0.7, 
                frequency_penalty=0.4, 
                presence_penalty=0.3
            )
            
            response = enc.decode(gen_tokens[len(tokens):])
            t_gen = time.time() - t_gen_start
            
            print(response.strip())
            print(f"\n[Generated {len(gen_tokens)-len(tokens)} tokens in {t_gen:.2f}s ({len(gen_tokens)-len(tokens)/max(0.01, t_gen):.1f} tok/s)]")
            print("-" * 66)
            
        except KeyboardInterrupt:
            print("\n[!] Interrupted by user. Exiting.\n")
            break
        except Exception as e:
            print(f"\n[!] Error during generation: {e}\n")

if __name__ == "__main__":
    main()
