import os, sys, torch
import torch.nn.functional as F
from safetensors.torch import load_file
import gguf, gguf.quants
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

hf_cache = r'C:\Users\Admin\.cache\huggingface\hub'
ollama_blobs = r'C:\Users\Admin\.ollama\models\blobs'

print(">>> Initializing Quad-Donor Out-of-Domain Evaluation Harness...", flush=True)

# 1. Load Tokenizer & Llama
llama_dir = os.path.join(hf_cache, 'models--unsloth--Llama-3.2-1B-Instruct', 'snapshots')
llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
tok = Tokenizer.from_file(os.path.join(llama_snap, 'tokenizer.json'))
st_llama = load_file(os.path.join(llama_snap, 'model.safetensors'))

# 2. Load Falcon
falcon_blob = os.path.join(ollama_blobs, 'sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01')
r_falcon = gguf.GGUFReader(falcon_blob)
def dequant_gguf(t):
    if t.tensor_type != 0: arr = gguf.quants.dequantize(t.data, t.tensor_type)
    else: arr = t.data
    return torch.from_numpy(arr.copy()).float()
falcon_tensors = {t.name: dequant_gguf(t) for t in r_falcon.tensors if 'blk.0.ffn_down' in t.name}

# 3. Load BitNet
bitnet_dir = os.path.join(hf_cache, 'models--1bitLLM--bitnet_b1_58-large', 'snapshots')
bitnet_snap = os.path.join(bitnet_dir, os.listdir(bitnet_dir)[0])
st_bitnet = load_file(os.path.join(bitnet_snap, 'model.safetensors'))

# 4. Load Qwen 3.5
qwen35_blob = os.path.join(ollama_blobs, 'sha256-e8bf47d40c5ba68baf8f3e503ee36710da6e8641cb0a670a508cff5b3f0cfbf7')
r_qwen35 = gguf.GGUFReader(qwen35_blob)
qwen35_tensors = {t.name: dequant_gguf(t) for t in r_qwen35.tensors if 'blk.0.ffn_down' in t.name}

def to_bitnet_ternary(W, eps=1e-5):
    scale = 1.0 / W.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    W_scaled = W * scale
    W_q = torch.round(torch.clamp(W_scaled, -1.0, 1.0))
    return W_q / scale

def project_dim(W, target_out, target_in):
    r, c = W.shape
    if r == target_out and c == target_in: return W
    if r >= target_out: W = W[:target_out, :]
    else: W = torch.cat([W, W.repeat((target_out - r)//r + 1, 1)[:(target_out - r), :]], dim=0)
    if c >= target_in: W = W[:, :target_in]
    else: W = torch.cat([W, W.repeat(1, (target_in - c)//c + 1)[:, :(target_in - c)]], dim=1)
    return W

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

w_down_llama = st_llama["model.layers.0.mlp.down_proj.weight"].float()
w_down_falcon = falcon_tensors["blk.0.ffn_down.weight"].float()
w_down_bitnet = project_dim(st_bitnet["model.layers.0.mlp.down_proj.weight"].float(), 2048, 8192)
w_down_qwen = project_dim(qwen35_tensors["blk.0.ffn_down.weight"].float(), 2048, 8192)

def extract_svd_experts(W, n_exp, rank=16):
    U, S, V = torch.svd_lowrank(W, q=n_exp * 2)
    experts = []
    for e in range(n_exp):
        offset = (e * 2) % max(1, (U.size(1) - rank))
        u_sub = U[:, offset:offset+rank]
        s_sub = S[offset:offset+rank]
        v_sub = V[:, offset:offset+rank]
        w0 = torch.diag(torch.sqrt(s_sub)) @ v_sub.t()
        w1 = u_sub @ torch.diag(torch.sqrt(s_sub))
        experts.append((to_bitnet_ternary(w0), to_bitnet_ternary(w1)))
    return experts

exp_llama = extract_svd_experts(w_down_llama, 9)
exp_falcon = extract_svd_experts(w_down_falcon, 9)
exp_bitnet = extract_svd_experts(w_down_bitnet, 8)
exp_qwen = extract_svd_experts(w_down_qwen, 8)

council_34 = exp_llama + exp_falcon + exp_bitnet + exp_qwen

wte = st_llama['model.embed_tokens.weight'].float()
norm_f = st_llama['model.norm.weight'].float()

layers = []
for l in range(16):
    p = f'model.layers.{l}'
    ln1 = st_llama[f'{p}.input_layernorm.weight'].float()
    ln2 = st_llama[f'{p}.post_attention_layernorm.weight'].float()
    q_w = st_llama[f'{p}.self_attn.q_proj.weight'].float()
    k_w = st_llama[f'{p}.self_attn.k_proj.weight'].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).float()
    v_w = st_llama[f'{p}.self_attn.v_proj.weight'].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).float()
    o_w = st_llama[f'{p}.self_attn.o_proj.weight'].float()
    gate_w = st_llama[f'{p}.mlp.gate_proj.weight'].float()
    up_w = st_llama[f'{p}.mlp.up_proj.weight'].float()
    down_w = st_llama[f'{p}.mlp.down_proj.weight'].float()

    layers.append({
        'ln1': ln1, 'ln2': ln2, 'q': q_w, 'k': k_w, 'v': v_w, 'o': o_w,
        'gate': gate_w, 'up': up_w, 'down': down_w
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

        moe_out = torch.zeros_like(x)
        for (w0_t, w1_t) in council_34:
            e_mid = F.linear(swiglu, w0_t)
            e_out = F.linear(e_mid, w1_t)
            moe_out = moe_out + e_out * (1.0 / 34.0)
        x = x + mlp_dense + 0.05 * moe_out

    h = rms_norm(x, norm_f)
    return F.linear(h, wte)

out_of_domain_prompts = [
    ("Python Coding", "Write a Python function to check if a string has valid matching parentheses:"),
    ("Biomedical Science", "How do mRNA vaccines instruct human cells to produce antigens?"),
    ("Classical Physics", "Why does a pencil appear bent when partially submerged in water?"),
    ("Macroeconomics", "What is the primary objective of a central bank raising interest rates?"),
    ("Culinary Arts", "What are the essential steps to prepare a traditional Italian risotto?")
]

print(">>> Starting Out-of-Domain Generation Evaluation:\n" + "="*70, flush=True)

for category, prompt in out_of_domain_prompts:
    print(f"\n[Category]: {category}", flush=True)
    print(f"[Prompt]: {prompt}", flush=True)
    print(">>> [Generated Output]: ", end="", flush=True)
    enc = tok.encode(f'<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    cur = torch.tensor([enc.ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(50):
            logits = forward(cur)
            nxt = logits[0, -1].argmax().item()
            if nxt in [128001, 128009]: break
            t_str = tok.decode([nxt])
            print(t_str, end="", flush=True)
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    print("\n" + "-"*70, flush=True)

print("\n>>> Out-of-Domain Evaluation Complete!", flush=True)
