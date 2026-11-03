import os, sys, torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

hf_cache = r'C:\Users\Admin\.cache\huggingface\hub'
llama_dir = os.path.join(hf_cache, 'models--unsloth--Llama-3.2-1B-Instruct', 'snapshots')
llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
tok = Tokenizer.from_file(os.path.join(llama_snap, 'tokenizer.json'))
st = load_file(os.path.join(llama_snap, 'model.safetensors'))

def to_bitnet_ternary(W, eps=1e-5):
    scale = 1.0 / W.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    W_scaled = W * scale
    W_q = torch.round(torch.clamp(W_scaled, -1.0, 1.0))
    return W_q / scale

def rms_norm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight

def apply_rope(q, k, t_len, base=500000.0):
    dim = 64
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(t_len).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().view(1, 1, t_len, dim)
    sin = emb.sin().view(1, 1, t_len, dim)
    def rot(v):
        v1, v2 = v[..., :32], v[..., 32:]
        return torch.cat((v1 * cos[..., :32] - v2 * sin[..., :32], v1 * sin[..., :32] + v2 * cos[..., :32]), dim=-1)
    return rot(q), rot(k)

wte = st['model.embed_tokens.weight'].float()
norm_f = st['model.norm.weight'].float()

layers = []
for l in range(16):
    p = f'model.layers.{l}'
    ln1 = st[f'{p}.input_layernorm.weight'].float()
    ln2 = st[f'{p}.post_attention_layernorm.weight'].float()
    q_w = st[f'{p}.self_attn.q_proj.weight'].float()
    k_w = st[f'{p}.self_attn.k_proj.weight'].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).float()
    v_w = st[f'{p}.self_attn.v_proj.weight'].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).float()
    o_w = st[f'{p}.self_attn.o_proj.weight'].float()
    gate_w = st[f'{p}.mlp.gate_proj.weight'].float()
    up_w = st[f'{p}.mlp.up_proj.weight'].float()
    down_w = st[f'{p}.mlp.down_proj.weight'].float()

    U, S, V = torch.svd_lowrank(down_w, q=16)
    w0_tern = to_bitnet_ternary(torch.diag(torch.sqrt(S)) @ V.t())
    w1_tern = to_bitnet_ternary(U @ torch.diag(torch.sqrt(S)))

    layers.append({
        'ln1': ln1, 'ln2': ln2, 'q': q_w, 'k': k_w, 'v': v_w, 'o': o_w,
        'gate': gate_w, 'up': up_w, 'down': down_w, 'w0_tern': w0_tern, 'w1_tern': w1_tern
    })

def forward(input_ids):
    B, T = input_ids.shape
    x = F.embedding(input_ids, wte)
    for l in layers:
        x_n = rms_norm(x, l['ln1'])
        q = F.linear(x_n, l['q']).view(B, T, 32, 64).transpose(1, 2)
        k = F.linear(x_n, l['k']).view(B, T, 32, 64).transpose(1, 2)
        v = F.linear(x_n, l['v']).view(B, T, 32, 64).transpose(1, 2)
        q, k = apply_rope(q, k, T)
        scores = (q @ k.transpose(-1, -2)) / 8.0
        scores = scores.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        x = x + F.linear((attn @ v).transpose(1, 2).reshape(B, T, 2048), l['o'])

        x_n2 = rms_norm(x, l['ln2'])
        swiglu = F.silu(F.linear(x_n2, l['gate'])) * F.linear(x_n2, l['up'])
        mlp_dense = F.linear(swiglu, l['down'])
        expert_delta = F.linear(swiglu, l['w0_tern'])
        expert_out = F.linear(expert_delta, l['w1_tern']) * 0.05
        x = x + mlp_dense + expert_out

    h = rms_norm(x, norm_f)
    return F.linear(h, wte)

prompts = [
    'Explain the core difference between inductive and deductive reasoning in two sentences.',
    'What is Bushido, and how does it guide honor and duty?'
]

for prompt in prompts:
    print(f"\n[Prompt]: {prompt}", flush=True)
    print(">>> Output: ", end="", flush=True)
    enc = tok.encode(f'<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    cur = torch.tensor([enc.ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(35):
            logits = forward(cur)
            nxt = logits[0, -1].argmax().item()
            if nxt in [128001, 128009]: break
            t_str = tok.decode([nxt])
            print(t_str, end="", flush=True)
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    print("\n", flush=True)
