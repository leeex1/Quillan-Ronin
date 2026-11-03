import os
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

tok_path = r'C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\tokenizer.json'
tok = Tokenizer.from_file(tok_path)

ckpt_path = r'C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_mini_0.6b_sovereign_wsp128k_qwen35.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']

prompt = "The capital of France is"
enc = tok.encode(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
input_ids = torch.tensor([enc.ids], dtype=torch.long)
B, T = input_ids.shape

def rms_norm(x, weight, eps=1e-5):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

# 1. Direct Forward through checkpoint tensors (Raw Math)
print("=== Running Direct Raw Checkpoint Forward ===")
wte = sd["wte.weight"].float()
x = F.embedding(input_ids, wte)

# RoPE
def apply_rope(q, k, base=500000.0):
    dim = 64
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(T, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().view(1, 1, T, dim)
    sin = emb.sin().view(1, 1, T, dim)
    def rot(v):
        v1, v2 = v[..., :32], v[..., 32:]
        return torch.cat((v1 * cos[..., :32] - v2 * sin[..., :32], v1 * sin[..., :32] + v2 * cos[..., :32]), dim=-1)
    return rot(q), rot(k)

for l in range(8):
    ln1_w = sd[f"h.{l}.ln_1.weight"].float()
    ln2_w = sd[f"h.{l}.ln_2.weight"].float()
    c_attn_w = sd[f"h.{l}.attn.c_attn.weight"].float()
    c_proj_w = sd[f"h.{l}.attn.c_proj.weight"].float()
    c_fc_w = sd[f"h.{l}.moe.c_fc.weight"].float()
    c_down_w = sd[f"h.{l}.moe.c_proj.weight"].float()

    # Attention
    x_n = rms_norm(x, ln1_w)
    qkv = F.linear(x_n, c_attn_w)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, T, 32, 64).transpose(1, 2)
    k = k.view(B, T, 32, 64).transpose(1, 2)
    v = v.view(B, T, 32, 64).transpose(1, 2)
    q, k = apply_rope(q, k)
    scores = (q @ k.transpose(-1, -2)) / 8.0
    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    o_h = (attn @ v).transpose(1, 2).reshape(B, T, 2048)
    x = x + F.linear(o_h, c_proj_w)

    # MLP
    x_n2 = rms_norm(x, ln2_w)
    fc = F.linear(x_n2, c_fc_w)
    gate, act = fc.chunk(2, dim=-1)
    # Test WITHOUT rms_norm on SwiGLU vs WITH rms_norm
    mlp = F.linear(F.silu(gate) * act, c_down_w)
    x = x + mlp

ln_f_w = sd["ln_f.weight"].float()
h = rms_norm(x, ln_f_w)
logits = F.linear(h, wte)
next_token = logits[0, -1].argmax().item()
print("Raw Checkpoint (without swiglu rmsnorm) Top Token:", repr(tok.decode([next_token])))
top5 = logits[0, -1].topk(5)
for idx, prob in zip(top5.indices, top5.values):
    print(f"  {repr(tok.decode([idx.item()]))}: {prob.item():.2f}")
