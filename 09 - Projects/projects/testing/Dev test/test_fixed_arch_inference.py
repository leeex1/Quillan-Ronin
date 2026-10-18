import os, sys, torch, torch.nn.functional as F, tiktoken, gc
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

print("==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 — FIXED ARCHITECTURE INFERENCE TEST")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

# Test the best existing checkpoint (Loss 0.06, 8048 steps)
ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_sft_v3_best.pt"
print(f"[*] Loading: {ckpt_path.name}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
loss_val = ckpt.get('loss', 'N/A')
step_val = ckpt.get('step', 'N/A')
print(f"[+] Loaded (Step: {step_val}, Loss: {loss_val})")

model.eval()
gc.collect()

prompts = [
    "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nHello! Who are you?\n<|assistant|>\n",
    "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nWhat is 2 + 2?\n<|assistant|>\n",
    "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nExplain photosynthesis in one paragraph.\n<|assistant|>\n",
]

for prompt in prompts:
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt.strip()[-60:]}")
    print(f"{'='*60}")
    print("RESPONSE: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(200):
            inp = torch.tensor([generated[-256:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr = logits[:, -1, :].clone()
            
            # Stutter block
            if len(generated) > 0:
                curr[0, generated[-1]] -= 50.0
            recent = generated[-48:]
            for tid in set(recent):
                curr[0, tid] -= 4.0 * recent.count(tid)
            
            # Sample with temp=0.3
            scaled = curr / 0.3
            probs = F.softmax(scaled, dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, dim=-1)
            remove = cum > 0.9
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0
            mask = remove.scatter(1, sorted_i, remove)
            scaled[mask] = float('-inf')
            probs = F.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            
            generated.append(next_tok)
            word = enc.decode_bytes([next_tok]).decode('utf-8', errors='ignore')
            print(word, end="", flush=True)
            
            if next_tok == 50256:
                break
    
    print(f"\n{'-'*60}")

print("\n[+] Inference test complete!")
