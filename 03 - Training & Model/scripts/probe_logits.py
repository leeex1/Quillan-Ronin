import os
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

tok_path = r'C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\tokenizer.json'
tok = Tokenizer.from_file(tok_path)

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

p = r'C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_mini_0.6b_sovereign_wsp128k_qwen35.pt'
print(f"Loading checkpoint {p}...", flush=True)
ckpt = torch.load(p, map_location='cpu', weights_only=False)
cfg_dict = ckpt['config']
state_dict = ckpt.pop('model_state_dict')

cfg = QuillanOniConfig(
    vocab_size=128256,
    hidden_dim=2048,
    n_layer=8,
    n_head=32,
    head_dim=64,
    ffn_dim=8192,
    num_experts=34,
    expert_rank=16,
    use_evo_moe=False
)
cfg.vectorized_experts = True

torch.set_default_dtype(torch.bfloat16)
model = QuillanRoninOni(cfg)
torch.set_default_dtype(torch.float32)

model.load_state_dict(state_dict, strict=False)
model.eval()

prompt = "Write a short poetic metaphor describing a quantum particle in a superposition."
enc = tok.encode(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
cur_input = torch.tensor([enc.ids], dtype=torch.long)

with torch.no_grad():
    logits = model(cur_input, compute_aux=False, return_aux=False)
    if isinstance(logits, tuple):
        logits = logits[0]
    next_logits = logits[0, -1, :].float()
    probs = F.softmax(next_logits, dim=-1)
    top_p, top_idx = torch.topk(probs, 10)
    
    print("\n--- TOP 10 PREDICTIONS FOR FIRST TOKEN ---", flush=True)
    for p, idx in zip(top_p, top_idx):
        token_str = tok.decode([idx.item()])
        print(f"Token: '{token_str}' (ID={idx.item()}): prob={p.item():.4f}", flush=True)

    print("\n--- GREEDY (ARGMAX) 15-TOKEN GENERATION ---", flush=True)
    greedy_input = cur_input.clone()
    for _ in range(15):
        l = model(greedy_input, compute_aux=False, return_aux=False)
        if isinstance(l, tuple): l = l[0]
        nxt = torch.argmax(l[0, -1, :]).item()
        if nxt in [128001, 128009]: break
        t_str = tok.decode([nxt])
        print(t_str, end="", flush=True)
        greedy_input = torch.cat([greedy_input, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    print("\n", flush=True)

