import os
import sys
import time
import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path
from tokenizers import Tokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

tik_enc = tiktoken.get_encoding("gpt2")
hf_tok_path = r"C:\02_QUILLAN\quillan_bpe_tokenizer_hf\tokenizer.json"
hf_tok = Tokenizer.from_file(hf_tok_path) if os.path.exists(hf_tok_path) else None

cfg = QuillanArchConfig(hidden_dim=1024, ffn_dim=2048, num_experts=34, text_only=True, eggroll_rank=256)

CKPT_DIR = r"C:\02_QUILLAN\checkpoints\checkpoints_sft"
ckpt_files = [
    "quillan_causal_aligned.pt",
    "quillan_dialogue_best.pt",
    "quillan_sft_v3_best.pt",
    "quillan_final_best.pt",
    "quillan_final_latest.pt"
]

def generate_kv(model, tokenizer_type, prompt, max_tokens=40, temp=0.3):
    if tokenizer_type == "tiktoken":
        enc_ids = tik_enc.encode(prompt)
        decode_fn = lambda ids: tik_enc.decode(ids)
    else:
        enc_ids = hf_tok.encode(prompt).ids
        decode_fn = lambda ids: hf_tok.decode(ids)

    tokens = list(enc_ids)
    generated = []
    
    with torch.no_grad():
        inp = torch.tensor([tokens], dtype=torch.long)
        out = model(inp, past_key_values=None, use_cache=True)
        logits = out[0] if isinstance(out, tuple) else out
        past_kv = out[1] if isinstance(out, tuple) else None
        curr_logits = logits[:, -1, :].clone()
        
        for _ in range(max_tokens):
            if temp > 0:
                probs = F.softmax(curr_logits / temp, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
            else:
                next_tok = torch.argmax(curr_logits, dim=-1).item()

            generated.append(next_tok)
            if next_tok in [50256, 0, 1, 2]:
                break
                
            next_inp = torch.tensor([[next_tok]], dtype=torch.long)
            out = model(next_inp, past_key_values=past_kv, use_cache=True)
            curr_logits = (out[0][:, -1, :] if isinstance(out, tuple) else out[:, -1, :]).clone()
            past_kv = out[1] if isinstance(out, tuple) else None

    return decode_fn(generated)

print("==================================================================")
print("   👑 HIGH-SPEED KV-CACHE CHECKPOINT SPEECH AUDIT")
print("==================================================================")

prompts = [
    ("GPT2-TAGS", "<|user|>\nHello! Who are you?\n<|assistant|>\n"),
    ("ROLE-TAGS", "user: Hello! Who are you?\nassistant: "),
]

for fname in ckpt_files:
    fpath = os.path.join(CKPT_DIR, fname)
    if not os.path.exists(fpath): continue
    
    print(f"\n" + "=" * 60)
    print(f"[*] CHECKPOINT: {fname}")
    print("=" * 60)
    
    try:
        model = QuillanRoninSovereign(cfg).to("cpu")
        ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        msd = model.state_dict()
        copied = 0
        for k, v in sd.items():
            if k in msd and v.shape == msd[k].shape:
                msd[k].copy_(v)
                copied += 1
        model.load_state_dict(msd)
        model.eval()
        print(f"[+] Loaded {copied}/{len(msd)} parameter layers.")
        
        for plabel, ptext in prompts:
            # Test tiktoken
            t_res = generate_kv(model, "tiktoken", ptext, max_tokens=35, temp=0.2)
            print(f"  [tiktoken | {plabel}] -> {repr(t_res.strip())}")
            
            # Test HF tokenizer if available
            if hf_tok:
                h_res = generate_kv(model, "hf", ptext, max_tokens=35, temp=0.2)
                print(f"  [HF BPE   | {plabel}] -> {repr(h_res.strip())}")
                
    except Exception as e:
        print(f"[!] Error auditing {fname}: {e}")

print("\n==================================================================")
