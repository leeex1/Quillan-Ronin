import os
import sys
import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)

ckpt_dir = REPO_ROOT / "checkpoints" / "checkpoints_sft"
ckpts = list(ckpt_dir.glob("*.pt"))

prompt_str = "<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\nHello! Who are you?\n<|assistant|>\n"
tokens = enc.encode(prompt_str)

print("==================================================================")
print("   👑 AUDITING ALL CHECKPOINTS FOR ENGLISH DIALOGUE FLUENCY")
print("==================================================================")

for ckpt_path in ckpts:
    print(f"\n[*] Testing: {ckpt_path.name}")
    try:
        model = QuillanRoninSovereign(cfg).to("cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()
        
        generated = list(tokens)
        with torch.no_grad():
            for _ in range(40):
                inp = torch.tensor([generated[-128:]], dtype=torch.long)
                logits = model(inp)
                if isinstance(logits, tuple): logits = logits[0]
                curr_logits = logits[:, -1, :].clone()
                if len(generated) > 0:
                    curr_logits[0, generated[-1]] -= 50.0
                next_tok = torch.argmax(curr_logits, dim=-1).item()
                generated.append(next_tok)
                if next_tok == 50256: break
                
        resp = enc.decode(generated[len(tokens):]).strip()
        print(f"   [RESPONSE]: {resp}")
    except Exception as e:
        print(f"   [ERROR]: {e}")

print("\n[+] Audit Complete!")
