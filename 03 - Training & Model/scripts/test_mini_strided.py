import os, sys, time, torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

hf_cache = r'C:\Users\Admin\.cache\huggingface\hub'
llama_dir = os.path.join(hf_cache, 'models--unsloth--Llama-3.2-1B-Instruct', 'snapshots')
llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
tok = Tokenizer.from_file(os.path.join(llama_snap, 'tokenizer.json'))
st_llama = load_file(os.path.join(llama_snap, 'model.safetensors'))

wte = st_llama['model.embed_tokens.weight'].to(device=device, dtype=dtype)
norm_f = st_llama['model.norm.weight'].to(device=device, dtype=dtype)

def rms_norm(x, weight, eps=1e-5):
    x_f = x.float()
    w_f = weight.float()
    norm = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (norm * w_f).to(x.dtype)

def apply_rope(q, k, t_len, base=500000.0, dev='cuda'):
    dim = 64
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=dev).float() / dim))
    t = torch.arange(t_len, device=dev).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().view(1, 1, t_len, dim)
    sin = emb.sin().view(1, 1, t_len, dim)
    def rot(v):
        v_f = v.float()
        v1, v2 = v_f[..., :32], v_f[..., 32:]
        r = torch.cat((v1 * cos[..., :32] - v2 * sin[..., :32], v1 * sin[..., :32] + v2 * cos[..., :32]), dim=-1)
        return r.to(v.dtype)
    return rot(q), rot(k)

def build_layers(indices):
    layers = []
    for l in indices:
        p = f'model.layers.{l}'
        ln1 = st_llama[f'{p}.input_layernorm.weight'].to(device=device, dtype=dtype)
        ln2 = st_llama[f'{p}.post_attention_layernorm.weight'].to(device=device, dtype=dtype)
        q_w = st_llama[f'{p}.self_attn.q_proj.weight'].to(device=device, dtype=dtype)
        k_w = st_llama[f'{p}.self_attn.k_proj.weight'].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).to(device=device, dtype=dtype)
        v_w = st_llama[f'{p}.self_attn.v_proj.weight'].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).to(device=device, dtype=dtype)
        o_w = st_llama[f'{p}.self_attn.o_proj.weight'].to(device=device, dtype=dtype)
        gate_w = st_llama[f'{p}.mlp.gate_proj.weight'].to(device=device, dtype=dtype)
        up_w = st_llama[f'{p}.mlp.up_proj.weight'].to(device=device, dtype=dtype)
        down_w = st_llama[f'{p}.mlp.down_proj.weight'].to(device=device, dtype=dtype)

        layers.append({
            'ln1': ln1, 'ln2': ln2, 'q': q_w, 'k': k_w, 'v': v_w, 'o': o_w,
            'gate': gate_w, 'up': up_w, 'down': down_w
        })
    return layers

def generate(layers, prompt, max_tokens=25):
    enc = tok.encode(f'<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    cur = torch.tensor([enc.ids], dtype=torch.long, device=device)
    output_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            B, T = cur.shape
            x = F.embedding(cur, wte)
            for l in layers:
                x_n = rms_norm(x, l['ln1'])
                q = F.linear(x_n, l['q']).view(B, T, 32, 64).transpose(1, 2)
                k = F.linear(x_n, l['k']).view(B, T, 32, 64).transpose(1, 2)
                v = F.linear(x_n, l['v']).view(B, T, 32, 64).transpose(1, 2)
                q, k = apply_rope(q, k, T, dev=device)
                
                scores = (q.float() @ k.float().transpose(-1, -2)) / 8.0
                scores = scores.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)), float('-inf'))
                attn = F.softmax(scores, dim=-1).to(dtype)
                x = x + F.linear((attn @ v).transpose(1, 2).reshape(B, T, 2048), l['o'])

                x_n2 = rms_norm(x, l['ln2'])
                swiglu = F.silu(F.linear(x_n2, l['gate'])) * F.linear(x_n2, l['up'])
                mlp_dense = F.linear(swiglu, l['down'])
                x = x + mlp_dense

            h = rms_norm(x, norm_f)
            logits = F.linear(h.float(), wte.float())
            nxt = logits[0, -1].argmax().item()
            if nxt in [128001, 128009]: break
            output_tokens.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
            
    return tok.decode(output_tokens)

print(">>> Testing strided 8-layer pruning configurations...")
configs = {
    "Strided 8L ([0, 2, 4, 6, 8, 10, 12, 15])": [0, 2, 4, 6, 8, 10, 12, 15],
    "Bottom-Top 8L ([0, 1, 2, 3, 12, 13, 14, 15])": [0, 1, 2, 3, 12, 13, 14, 15],
    "First 12L (12 Layers)": list(range(12)),
    "Full 16L (16 Layers)": list(range(16))
}

prompt = "The capital of France is"
for name, idxs in configs.items():
    l_set = build_layers(idxs)
    out = generate(l_set, prompt, max_tokens=10)
    print(f"[{name}]: {out.strip()}")
    del l_set
    torch.cuda.empty_cache()
