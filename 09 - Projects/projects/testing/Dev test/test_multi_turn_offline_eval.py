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

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — MULTI-TURN STREAMING EVALUATION", flush=True)
print("==================================================================", flush=True)

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt"
print(f"[*] Loading Master Model Checkpoint: {ckpt_path}", flush=True)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
model.eval()
print(f"[+] Master Model Loaded Successfully!", flush=True)

def generate_multi_turn_reply_stream(conversation_history, max_tokens=80, temp=0.2, top_p=0.9):
    prompt_str = "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n"
    for turn in conversation_history:
        role = turn["role"]
        content = turn["content"]
        prompt_str += f"<|{role}|>\n{content}\n"
    prompt_str += "<|assistant|>\n"
    
    tokens = enc.encode(prompt_str)
    generated = list(tokens)
    out_tokens = []
    
    with torch.no_grad():
        for i in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr_logits = logits[:, -1, :].clone()
            
            # Zero-Stutter hard penalty on immediate previous token
            if len(generated) > 0:
                prev_tok = generated[-1]
                curr_logits[0, prev_tok] -= 50.0
                
            # Repetition window penalty
            recent_tokens = generated[-48:]
            for tid in set(recent_tokens):
                count = recent_tokens.count(tid)
                curr_logits[0, tid] -= (4.0 * count)

            if temp == 0.0:
                next_tok = torch.argmax(curr_logits, dim=-1).item()
            else:
                scaled_logits = curr_logits / max(temp, 0.01)
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
            out_tokens.append(next_tok)
            
            tok_text = enc.decode([next_tok])
            sys.stdout.write(tok_text)
            sys.stdout.flush()
            
            if next_tok == 50256:
                break
                
    sys.stdout.write("\n")
    sys.stdout.flush()
    return enc.decode(out_tokens).strip()

# Define 3 Multi-Turn Dialogue Scenarios
test_turns = [
    "Hello! Who are you and what are your core capabilities as Quillan-Ronin?",
    "Can you explain the difference between P and NP in computer science?",
    "Write a short Python function implementing an LRU cache."
]

history = []

for idx, user_input in enumerate(test_turns, 1):
    print(f"\n==================================================", flush=True)
    print(f"TURN {idx} USER:", flush=True)
    print(user_input, flush=True)
    print("==================================================", flush=True)
    print(f"TURN {idx} QUILLAN STREAMING RESPONSE:", flush=True)
    
    history.append({"role": "user", "content": user_input})
    reply = generate_multi_turn_reply_stream(history, max_tokens=80, temp=0.2)
    history.append({"role": "assistant", "content": reply})
    print("--------------------------------------------------", flush=True)

print("\n[+] Multi-Turn Real-Time Streaming Evaluation Complete!", flush=True)
