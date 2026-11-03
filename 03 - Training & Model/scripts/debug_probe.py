import os
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

print(">>> Step 1: Loading Tokenizer...", flush=True)
tok_dir = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots"
tok_snap = os.path.join(tok_dir, os.listdir(tok_dir)[0])
tok = Tokenizer.from_file(os.path.join(tok_snap, "tokenizer.json"))
print("    Tokenizer loaded.", flush=True)

print(">>> Step 2: Loading Checkpoint weights...", flush=True)
t0 = time.time()
ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_1b_sovereign_wsp128k_qwen35.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg_dict = ckpt["config"]
state_dict = ckpt.pop("model_state_dict")
del ckpt
import gc
gc.collect()
print(f"    Checkpoint loaded in {time.time()-t0:.2f}s. State dict has {len(state_dict)} keys.", flush=True)

print(">>> Step 3: Instantiating QuillanRoninOni...", flush=True)
t0 = time.time()
cfg = QuillanOniConfig(
    vocab_size=cfg_dict.get("vocab_size", 128256),
    hidden_dim=cfg_dict.get("hidden_dim", 2048),
    n_layer=cfg_dict.get("n_layer", 16),
    n_head=cfg_dict.get("n_head", 32),
    head_dim=cfg_dict.get("head_dim", 64),
    ffn_dim=cfg_dict.get("ffn_dim", 8192),
    num_experts=cfg_dict.get("num_experts", 34),
    expert_rank=16,
    use_evo_moe=False
)
cfg.vectorized_experts = True

# Instantiate with meta device or directly
orig_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
model = QuillanRoninOni(cfg)
torch.set_default_dtype(orig_dtype)
print(f"    Instantiated in {time.time()-t0:.2f}s.", flush=True)

print(">>> Step 4: Loading state_dict into model...", flush=True)
t0 = time.time()
missing, unexpected = model.load_state_dict(state_dict, strict=False)
del state_dict
gc.collect()
print(f"    State dict loaded in {time.time()-t0:.2f}s. Missing: {len(missing)}, Unexpected: {len(unexpected)}", flush=True)

model.eval()

print(">>> Step 5: Test forward pass with single prompt...", flush=True)
prompt = "Explain the core difference between inductive and deductive reasoning."
encoded = tok.encode(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
input_ids = torch.tensor([encoded.ids], dtype=torch.long)
print(f"    Prompt encoded to {input_ids.shape[-1]} tokens. Forwarding...", flush=True)

t0 = time.time()
with torch.no_grad():
    logits = model(input_ids, compute_aux=False, return_aux=False)
    if isinstance(logits, tuple):
        logits = logits[0]
print(f"    Forward pass finished in {time.time()-t0:.2f}s! Logits shape: {logits.shape}", flush=True)

next_token_logits = logits[0, -1, :]
probs = F.softmax(next_token_logits / 0.6, dim=-1)
next_token = torch.multinomial(probs, num_samples=1).item()
print(f"    First predicted token: {next_token} -> '{tok.decode([next_token])}'", flush=True)
print(">>> ALL VERIFICATION STEPS PASSED!", flush=True)
