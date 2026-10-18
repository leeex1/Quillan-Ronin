import os
import sys
import torch
import torch.nn.functional as F
import tiktoken
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = r"C:\02_QUILLAN"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

print("==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 — ULTRA-FAST KV-CACHE INFERENCE TEST")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

def test_checkpoint(ckpt_path, title):
    if not os.path.exists(ckpt_path):
        print(f"[!] Checkpoint missing: {ckpt_path}")
        return
        
    print(f"\n==================================================")
    print(f"[*] TESTING CHECKPOINT: {title}")
    print(f"    Path: {ckpt_path}")
    print(f"==================================================")
    
    cfg = QuillanArchConfig(hidden_dim=1024, ffn_dim=2048, num_experts=34, text_only=True, eggroll_rank=256)
    model = QuillanRoninSovereign(cfg).to("cpu")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model_sd = model.state_dict()
    copied = 0
    for k, v in sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)
            copied += 1
    model.load_state_dict(model_sd)
    step_val = ckpt.get('step', 'N/A') if isinstance(ckpt, dict) else 'N/A'
    loss_val = ckpt.get('loss', 'N/A') if isinstance(ckpt, dict) else 'N/A'
    print(f"[+] Loaded {copied}/{len(model_sd)} layers (Step: {step_val}, Loss: {loss_val})")
    model.eval()

    def generate_fast(prompt, max_new_tokens=80, temp=0.7, top_p=0.9, rep_penalty=1.2):
        tokens = enc.encode(prompt)
        generated = list(tokens)
        
        print(f"\nPROMPT: {prompt.strip()}")
        print("RESPONSE: ", end="", flush=True)
        
        t0 = time.time()
        with torch.no_grad():
            past_kv = None
            inp = torch.tensor([tokens], dtype=torch.long)
            out = model(inp, past_key_values=past_kv, use_cache=True)
            if isinstance(out, tuple):
                logits, past_kv = out[0], out[1]
            else:
                logits = out
            
            curr_logits = logits[:, -1, :].clone()
            
            for token_idx in range(max_new_tokens):
                # Repetition penalty
                if rep_penalty > 1.0:
                    for tid in set(generated[-40:]):
                        if curr_logits[0, tid] < 0:
                            curr_logits[0, tid] *= rep_penalty
                        else:
                            curr_logits[0, tid] /= rep_penalty

                if temp > 0:
                    scaled_logits = curr_logits / temp
                    probs = F.softmax(scaled_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cum_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    scaled_logits[indices_to_remove] = float('-inf')
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_tok = torch.argmax(curr_logits, dim=-1).item()

                generated.append(next_tok)
                decoded = enc.decode([next_tok])
                print(decoded, end="", flush=True)

                if next_tok == 50256 or "<|end|>" in decoded:
                    break

                # Autoregressive single-token step
                next_inp = torch.tensor([[next_tok]], dtype=torch.long)
                out = model(next_inp, past_key_values=past_kv, use_cache=True)
                if isinstance(out, tuple):
                    curr_logits, past_kv = out[0][:, -1, :].clone(), out[1]
                else:
                    curr_logits = out[:, -1, :].clone()

        t1 = time.time()
        tps = len(generated[len(tokens):]) / max(0.001, (t1 - t0))
        print(f"\n[{len(generated[len(tokens):])} tokens, {tps:.1f} tok/s]")

    prompts = [
        "<|user|>\nHello! Who are you?\n<|assistant|>\n",
        "<|user|>\nWhat can you help me with?\n<|assistant|>\n",
        "<|user|>\nExplain quantum physics in simple terms.\n<|assistant|>\n",
        "<|user|>\nWrite a python function to add two numbers.\n<|assistant|>\n"
    ]

    for p in prompts:
        generate_fast(p, max_new_tokens=60, temp=0.6, top_p=0.9, rep_penalty=1.2)

# Run tests on both checkpoints
test_checkpoint(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_best.pt", "BEST LOSS CHECKPOINT (Step 914, Loss 1.95)")
test_checkpoint(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_latest.pt", "LATEST CHECKPOINT (Step 3000, Complete)")
