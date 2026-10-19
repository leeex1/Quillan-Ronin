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
sys.path.insert(0, str(REPO_ROOT / "_dev"))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
from _dev.quillan_bpe_tokenizer import QuillanBPETokenizer

print("=== TOKENIZER COMPARISON TEST ===")

prompt = "<|user|>\nHello! Who are you?\n<|assistant|>\n"

# 1. Tiktoken
tik_enc = tiktoken.get_encoding("gpt2")
tik_ids = tik_enc.encode(prompt)
print("\n[tiktoken gpt2]:")
print("Token IDs:", tik_ids)
print("Decoded :", tik_enc.decode(tik_ids))

# 2. QuillanBPETokenizer
tok_path = REPO_ROOT / "_dev" / "quillan_bpe_tokenizer_hf" / "tokenizer.json"
q_tok = QuillanBPETokenizer()
if tok_path.exists():
    q_tok.load(str(tok_path))
    q_ids = q_tok.encode(prompt)
    print("\n[QuillanBPETokenizer HF]:")
    print("Token IDs:", q_ids)
    print("Decoded :", q_tok.decode(q_ids))
else:
    print("[!] Tokenizer path not found:", tok_path)

# Test Generation with QuillanBPETokenizer
cfg = QuillanArchConfig(hidden_dim=1024, ffn_dim=2048, num_experts=34, text_only=True, eggroll_rank=256)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_best.pt"
if os.path.exists(ckpt_path):
    print(f"\n[*] Loading best model weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    msd = model.state_dict()
    for k, v in sd.items():
        if k in msd and v.shape == msd[k].shape:
            msd[k].copy_(v)
    model.load_state_dict(msd)
    model.eval()

    print("\n--- GENERATING WITH QuillanBPETokenizer (Temp=0.7, Top-P=0.9) ---")
    tokens = q_ids if tok_path.exists() else tik_ids
    generated = list(tokens)
    with torch.no_grad():
        for _ in range(60):
            inp = torch.tensor([generated[-256:]], dtype=torch.long)
            logits = model(inp)[:, -1, :] / 0.7
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            generated.append(next_tok)
            if q_tok._tok:
                tok_str = q_tok.decode([next_tok])
            else:
                tok_str = tik_enc.decode([next_tok])
            print(tok_str, end="", flush=True)
    print("\n-------------------------------------------------------------")
