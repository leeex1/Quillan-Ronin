import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(r'C:\Users\Admin\Quillan-Ronin')
sys.path.insert(0, str(ROOT / '_dev'))
sys.path.insert(0, str(ROOT))

from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
from quillan_bpe_tokenizer import QuillanBPETokenizer

# Force UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

tok_path = ROOT / "quillan_bpe_tokenizer.pkl"
tokenizer = QuillanBPETokenizer(path=tok_path)

cfg = QuillanArchConfig(hidden_dim=1024, ffn_dim=2048, num_experts=34, vocab_size=tokenizer.vocab_size, text_only=True, device="cpu")
model = QuillanRoninSovereign(cfg)

ckpt_path = ROOT / "checkpoints_v2/quillan_sovereign_latest.pt"
sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
if "state_dict" in sd:
    sd = sd["state_dict"]

model.load_state_dict(sd, strict=False)
model.eval()

prompt = "The future of artificial intelligence is"
tokens = tokenizer.encode(prompt)
input_tensor = torch.tensor([tokens], dtype=torch.long)
past_key_values = None
generated_tokens = []

print(f"Prompt: {prompt}")
print("Generation (no EOS break):")

for step in range(60):
    with torch.no_grad():
        if past_key_values is None:
            out = model(input_tensor, past_key_values=None, use_cache=True)
        else:
            last_token = input_tensor[:, -1:]
            out = model(last_token, past_key_values=past_key_values, use_cache=True)
            
        logits = out["logits"][:, -1, :]
        past_key_values = out["past_key_values"]
        
        # Repetition penalty
        for token in set(generated_tokens):
            if logits[0, token] > 0:
                logits[0, token] /= 1.2
            else:
                logits[0, token] *= 1.2

        # Sampling (temp=0.7)
        logits = logits / 0.7
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        token_id = next_token.item()
        generated_tokens.append(token_id)
        input_tensor = torch.cat([input_tensor, next_token], dim=-1)
        
        token_text = tokenizer.decode([token_id])
        print(token_text, end='', flush=True)

print("\nDone.")
