import os
import sys
import time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

p = r'C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_mini_0.6b_sovereign_wsp128k_qwen35.pt'
tok_path = r'C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\tokenizer.json'

print("Loading tokenizer...", flush=True)
tok = Tokenizer.from_file(tok_path)

print(f"Loading Mini 0.8B checkpoint from {p}...", flush=True)
ckpt = torch.load(p, map_location='cpu', weights_only=False)
cfg_dict = ckpt['config']
state_dict = ckpt.pop('model_state_dict')
del ckpt
import gc
gc.collect()

cfg = QuillanOniConfig(
    vocab_size=cfg_dict.get("vocab_size", 128256),
    hidden_dim=cfg_dict.get("hidden_dim", 2048),
    n_layer=cfg_dict.get("n_layer", 8),
    n_head=cfg_dict.get("n_head", 32),
    head_dim=cfg_dict.get("head_dim", 64),
    ffn_dim=cfg_dict.get("ffn_dim", 8192),
    num_experts=cfg_dict.get("num_experts", 34),
    expert_rank=16,
    use_evo_moe=False
)
cfg.vectorized_experts = True

print("Instantiating Mini model directly in bfloat16...", flush=True)
orig_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
model = QuillanRoninOni(cfg)
torch.set_default_dtype(orig_dtype)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
del state_dict
gc.collect()
print(f"Mini state dict loaded. Missing={len(missing)}, Unexpected={len(unexpected)}", flush=True)
model.eval()

prompts = [
    "Write a short poetic metaphor describing a quantum particle in a superposition.",
    "Explain the core invariant of the Paxos consensus algorithm in one concise sentence."
]

for p_idx, prompt in enumerate(prompts, 1):
    print(f"\n==================================================", flush=True)
    print(f"PROMPT {p_idx}: {prompt}", flush=True)
    print("==================================================", flush=True)
    print("RESPONSE: ", end="", flush=True)

    enc = tok.encode(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    cur_input = torch.tensor([enc.ids], dtype=torch.long)

    with torch.no_grad():
        for step in range(35):
            logits = model(cur_input, compute_aux=False, return_aux=False)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_logits = logits[0, -1, :]
            
            # Low temperature for 8-layer edge stability
            temp = 0.35
            top_k = 30
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[-1]] = -float('Inf')
            probs = F.softmax(next_logits / temp, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            
            if nxt in [128001, 128009]:
                break
            token_str = tok.decode([nxt])
            print(token_str, end="", flush=True)
            cur_input = torch.cat([cur_input, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

    print("\n", flush=True)

print("\n[MINI TEST COMPLETE]", flush=True)

