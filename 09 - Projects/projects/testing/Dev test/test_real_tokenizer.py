import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

tok_path = r"C:\02_QUILLAN\quillan_bpe_tokenizer_hf\tokenizer.json"
print(f"[*] Loading tokenizer from: {tok_path}")
tok = Tokenizer.from_file(tok_path)

prompt = "<|user|>\nHello! Who are you?\n<|assistant|>\n"
encoded_ids = tok.encode(prompt).ids
print(f"Prompt Token IDs: {encoded_ids}")
print(f"Decoded Prompt : {tok.decode(encoded_ids)}")

# Check Dataset Sample
pt_sample_path = os.path.join(REPO_ROOT, "training_data", "code_train.pt")
if os.path.exists(pt_sample_path):
    pt_data = torch.load(pt_sample_path, map_location="cpu", weights_only=False)
    if isinstance(pt_data, torch.Tensor):
        pt_ids = pt_data[:50].tolist()
        print("\n[code_train.pt Sample IDs]:", pt_ids[:15])
        print("Decoded with quillan HF tokenizer:", tok.decode(pt_ids))

# Test generation with quillan_bpe_tokenizer_hf
cfg = QuillanArchConfig(hidden_dim=1024, ffn_dim=2048, num_experts=34, text_only=True, eggroll_rank=256)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_best.pt"
if os.path.exists(ckpt_path):
    print(f"\n[*] Loading model state from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model_sd = model.state_dict()
    for k, v in sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)
    model.load_state_dict(model_sd)
    model.eval()

    print("\n--- GENERATION WITH HF TOKENIZER (Temp=0.7, Top-P=0.9) ---")
    tokens = list(encoded_ids)
    with torch.no_grad():
        for _ in range(60):
            inp = torch.tensor([tokens[-256:]], dtype=torch.long)
            logits = model(inp)[:, -1, :] / 0.7
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            tokens.append(next_tok)
            print(tok.decode([next_tok]), end="", flush=True)
    print("\n-----------------------------------------------------------")
