#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — SOVEREIGN INTERACTIVE TERMINAL CLI
Real-time conversational interface with streaming tokens, anti-stutter repetition control,
and multi-turn dialogue memory. Powered by the master 389.1M BitNet 1.58b architecture.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

def main():
    banner = r"""
/==================================================================\
||  ██████╗ ██╗   ██╗██╗██╗     ██╗      █████╗ ███╗   ██╗       ||
||  ██╔═══██╗██║   ██║██║██║     ██║     ██╔══██╗████╗  ██║       ||
||  ██║   ██║██║   ██║██║██║     ██║     ███████║██╔██╗ ██║       ||
||  ██║▄▄ ██║██║   ██║██║██║     ██║     ██╔══██║██║╚██╗██║       ||
||  ╚██████╔╝╚██████╔╝██║███████╗███████╗██║  ██║██║ ╚████║       ||
||   ╚══▀▀═╝  ╚═════╝ ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝       ||
\==================================================================/
  👑 QUILLAN-RONIN v5.3.1 OMNI-FRACTAL SOVEREIGN — LIVE TERMINAL
"""
    print(banner, flush=True)

    # 1. Initialize Model
    cfg = QuillanArchConfig(
        hidden_dim=1024, ffn_dim=2048, num_experts=34,
        vocab_size=50257, text_only=True, eggroll_rank=256
    )
    model = QuillanRoninSovereign(cfg).to("cpu")

    ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
    print(f"[*] Loading Master Weights: {ckpt_path.name}...", flush=True)
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[+] Loaded successfully in {time.time()-t0:.2f}s! (Loss: {ckpt.get('loss', 'N/A')})\n", flush=True)

    enc = tiktoken.get_encoding("gpt2")
    conversation_history = []

    print("[*] Quillan-Ronin v5.3.1 Online. Type your message below (or 'exit' / 'clear'):\n", flush=True)

    while True:
        try:
            user_input = input("\n[User]: ").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n[Quillan]: Terminating sovereign session. Farewell.\n")
                break
            if user_input.lower() == "clear":
                conversation_history.clear()
                print("\n[Quillan]: Context memory cleared.\n")
                continue

            conversation_history.append({"role": "user", "content": user_input})
            
            # Format Prompt
            prompt_parts = ["<|system|>\nYou are Quillan-Ronin, a sovereign, highly intelligent AI assistant and expert coder."]
            for turn in conversation_history[-4:]:
                if turn["role"] == "user":
                    prompt_parts.append(f"<|user|>\n{turn['content']}")
                elif turn["role"] == "assistant":
                    prompt_parts.append(f"<|assistant|>\n{turn['content']}")
            prompt_parts.append("<|assistant|>\n")
            full_prompt = "\n".join(prompt_parts)

            prompt_tokens = enc.encode(full_prompt)

            print("\n[Quillan]: ", end="", flush=True)
            t_gen = time.time()
            
            # Real-time streaming token generation
            generated = list(prompt_tokens)
            device = "cpu"
            max_tokens = 200
            temperature = 0.65
            top_p = 0.90
            repetition_penalty = 1.18
            
            with torch.no_grad():
                for _ in range(max_tokens):
                    inp = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
                    logits = model(inp)
                    if isinstance(logits, tuple): logits = logits[0]
                    curr_logits = logits[:, -1, :].clone()

                    # Immediate previous token anti-stutter
                    if len(generated) > 0:
                        prev_tok = generated[-1]
                        curr_logits[0, prev_tok] -= 50.0

                    # Bounded repetition penalty on recent window
                    if len(generated) > len(prompt_tokens):
                        recent = set(generated[len(prompt_tokens):][-48:])
                        for tid in recent:
                            if curr_logits[0, tid] > 0: curr_logits[0, tid] /= repetition_penalty
                            else: curr_logits[0, tid] *= repetition_penalty

                    # Top-p Nucleus Filtering
                    scaled_logits = curr_logits / temperature
                    probs = F.softmax(scaled_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cum_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    scaled_logits[indices_to_remove] = float('-inf')
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_tok = torch.multinomial(probs, 1).item()

                    generated.append(next_tok)
                    tok_str = enc.decode([next_tok])
                    print(tok_str, end="", flush=True)

                    if next_tok == 50256 or "<|end|>" in tok_str:
                        break

            dt = time.time() - t_gen
            gen_tokens = generated[len(prompt_tokens):]
            assistant_response = enc.decode(gen_tokens).replace("<|end|>", "").strip()
            conversation_history.append({"role": "assistant", "content": assistant_response})
            print(f"\n\n[Stats: {len(gen_tokens)} tokens in {dt:.2f}s ({len(gen_tokens)/max(dt,0.001):.1f} tok/s)]", flush=True)

        except KeyboardInterrupt:
            print("\n[Quillan]: Session paused. Type 'exit' to quit.\n")
        except Exception as e:
            print(f"\n[-] Runtime Notice: {e}\n")

if __name__ == "__main__":
    main()
