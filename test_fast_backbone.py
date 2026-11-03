import os
import sys
import time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
sys.path.insert(0, r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

llama_dir = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots"
llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
tok_path = os.path.join(llama_snap, "tokenizer.json")
tok = Tokenizer.from_file(tok_path)

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_1b_sovereign_wsp128k.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
raw_cfg = ckpt["config"]

import inspect
valid_keys = set(inspect.signature(QuillanOniConfig.__init__).parameters.keys())
filtered_cfg = {k: v for k, v in raw_cfg.items() if k in valid_keys}
cfg = QuillanOniConfig(**filtered_cfg)

model = QuillanRoninOni(cfg).cpu().eval()
model.load_state_dict(ckpt["model_state_dict"], strict=False)

def fast_generate(prompt: str, max_tokens: int = 20):
    enc = tok.encode(prompt).ids
    gen = list(enc)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([gen[-64:]], dtype=torch.long)
            # Direct Transformer Forward Pass:
            x = model.wte(inp)
            for block in model.h:
                x = block(x)[0]
            x = model.ln_f(x)
            logits = model.lm_head(x)[:, -1, :].clone().float()
            
            # Simple repetition penalty
            for prev in set(gen[-16:]):
                if logits[0, prev] > 0:
                    logits[0, prev] /= 1.2
                else:
                    logits[0, prev] *= 1.2
            
            top_v, top_i = torch.topk(logits[0] / 0.7, 40)
            probs = F.softmax(top_v, dim=-1)
            next_id = top_i[torch.multinomial(probs, 1).item()].item()
            gen.append(next_id)
            if next_id in [tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|im_end|>")]:
                break
    elapsed = time.time() - t0
    print(f"Generated {len(gen)-len(enc)} tokens in {elapsed:.2f}s ({elapsed/(len(gen)-len(enc)):.3f}s/tok)")
    print("OUTPUT:\n", tok.decode(gen))

print("=== TEST 1 ===")
fast_generate("Quillan is a sovereign cognitive agent. Its mission is to")
print("\n=== TEST 2 ===")
fast_generate("In the bushido code, honor and rectitude are")
