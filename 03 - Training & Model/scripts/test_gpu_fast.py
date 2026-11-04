import os, sys, time, torch
import torch.nn.functional as F
from safetensors.torch import load_file
import gguf, gguf.quants
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f">>> Initializing High-Performance KV-Cached GPU Engine on {device.upper()} ({torch.cuda.get_device_name(0)})...")

hf_cache = r'C:\Users\Admin\.cache\huggingface\hub'
ollama_blobs = r'C:\Users\Admin\.ollama\models\blobs'

# 1. Load Tokenizer & Llama
llama_dir = os.path.join(hf_cache, 'models--unsloth--Llama-3.2-1B-Instruct', 'snapshots')
llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
tok = Tokenizer.from_file(os.path.join(llama_snap, 'tokenizer.json'))
st_llama = load_file(os.path.join(llama_snap, 'model.safetensors'))

# 2. Falcon
falcon_blob = os.path.join(ollama_blobs, 'sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01')
r_falcon = gguf.GGUFReader(falcon_blob)
def dequant_gguf(t):
    if t.tensor_type != 0: arr = gguf.quants.dequantize(t.data, t.tensor_type)
    else: arr = t.data
    return torch.from_numpy(arr.copy()).float()
falcon_tensors = {t.name: dequant_gguf(t) for t in r_falcon.tensors if 'blk.0.ffn_down' in t.name}

# 3. BitNet
bitnet_dir = os.path.join(hf_cache, 'models--1bitLLM--bitnet_b1_58-large', 'snapshots')
bitnet_snap = os.path.join(bitnet_dir, os.listdir(bitnet_dir)[0])
st_bitnet = load_file(os.path.join(bitnet_snap, 'model.safetensors'))

# 4. Qwen
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
    x_f = x.float()
    w_f = weight.float()
    norm = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (norm * w_f).to(x.dtype)

def precompute_rope(max_len=2048, base=500000.0, dev='cuda'):
    dim = 64
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=dev).float() / dim))
    t = torch.arange(max_len, device=dev).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().view(1, 1, max_len, dim), emb.sin().view(1, 1, max_len, dim)

cos_table, sin_table = precompute_rope(2048, dev=device)

def apply_rope_cached(q, k, pos, seq_len=1):
    cos = cos_table[:, :, pos:pos+seq_len, :]
    sin = sin_table[:, :, pos:pos+seq_len, :]
    def rot(v):
        v_f = v.float()
        v1, v2 = v_f[..., :32], v_f[..., 32:]
        r = torch.cat((v1 * cos[..., :32] - v2 * sin[..., :32], v1 * sin[..., :32] + v2 * cos[..., :32]), dim=-1)
        return r.to(v.dtype)
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
        experts.append((
            to_bitnet_ternary(w0).to(device=device, dtype=dtype),
            to_bitnet_ternary(w1).to(device=device, dtype=dtype)
        ))
    return experts

print(">>> Building Batched 3D Tensor for 34 Council MoE Experts...")
council_34_list = (
    extract_svd_experts(w_down_llama, 9) +
    extract_svd_experts(w_down_falcon, 9) +
    extract_svd_experts(w_down_bitnet, 8) +
    extract_svd_experts(w_down_qwen, 8)
)

# Stacking all 34 experts into batched 3D tensors:
# W0_batched: [34, 16, 8192]
# W1_batched: [34, 2048, 16]
W0_batched = torch.stack([e[0] for e in council_34_list], dim=0)
W1_batched = torch.stack([e[1] for e in council_34_list], dim=0)

wte = st_llama['model.embed_tokens.weight'].to(device=device, dtype=dtype)
norm_f = st_llama['model.norm.weight'].to(device=device, dtype=dtype)

layers = []
for l in range(16):
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

print(f">>> VRAM Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB / 4096 MB")

def generate_fast(prompt, max_tokens=40):
    enc = tok.encode(f'<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    prompt_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
    B, prompt_len = prompt_ids.shape
    
    # Initialize KV cache for 16 layers
    kv_caches = [{'k': None, 'v': None} for _ in range(16)]
    
    start_time = time.time()
    
    # ── 1. PREFILL PHASE (Prompt) ──
    x = F.embedding(prompt_ids, wte)
    for idx, l in enumerate(layers):
        x_n = rms_norm(x, l['ln1'])
        q = F.linear(x_n, l['q']).view(B, prompt_len, 32, 64).transpose(1, 2)
        k = F.linear(x_n, l['k']).view(B, prompt_len, 32, 64).transpose(1, 2)
        v = F.linear(x_n, l['v']).view(B, prompt_len, 32, 64).transpose(1, 2)
        q, k = apply_rope_cached(q, k, pos=0, seq_len=prompt_len)
        
        kv_caches[idx]['k'] = k
        kv_caches[idx]['v'] = v
        
        scores = (q.float() @ k.float().transpose(-1, -2)) / 8.0
        scores = scores.masked_fill(~torch.tril(torch.ones(prompt_len, prompt_len, dtype=torch.bool, device=device)), float('-inf'))
        attn = F.softmax(scores, dim=-1).to(dtype)
        x = x + F.linear((attn @ v).transpose(1, 2).reshape(B, prompt_len, 2048), l['o'])

        x_n2 = rms_norm(x, l['ln2'])
        swiglu = F.silu(F.linear(x_n2, l['gate'])) * F.linear(x_n2, l['up'])
        mlp_dense = F.linear(swiglu, l['down'])

        # Batched 34 Council MoE
        # swiglu: [1, T, 8192] -> reshape for bmm
        swiglu_flat = swiglu.view(prompt_len, 8192).t().unsqueeze(0).repeat(34, 1, 1) # [34, 8192, T]
        e_mid = torch.bmm(W0_batched, swiglu_flat) # [34, 16, T]
        e_out = torch.bmm(W1_batched, e_mid)      # [34, 2048, T]
        moe_out = e_out.mean(dim=0).t().view(B, prompt_len, 2048)
        x = x + mlp_dense + 0.05 * moe_out

    h = rms_norm(x[:, -1:], norm_f)
    logits = F.linear(h.float(), wte.float())
    next_token = logits[0, -1].argmax().item()
    
    generated_tokens = [next_token]
    cur_pos = prompt_len
    
    # ── 2. DECODING PHASE (KV-Cached Token-by-Token) ──
    with torch.no_grad():
        for _ in range(max_tokens - 1):
            if next_token in [128001, 128009]: break
            
            x = F.embedding(torch.tensor([[next_token]], device=device), wte) # [1, 1, 2048]
            for idx, l in enumerate(layers):
                x_n = rms_norm(x, l['ln1'])
                q = F.linear(x_n, l['q']).view(B, 1, 32, 64).transpose(1, 2)
                k = F.linear(x_n, l['k']).view(B, 1, 32, 64).transpose(1, 2)
                v = F.linear(x_n, l['v']).view(B, 1, 32, 64).transpose(1, 2)
                q, k = apply_rope_cached(q, k, pos=cur_pos, seq_len=1)
                
                # Append to KV Cache
                k_all = torch.cat([kv_caches[idx]['k'], k], dim=2)
                v_all = torch.cat([kv_caches[idx]['v'], v], dim=2)
                kv_caches[idx]['k'] = k_all
                kv_caches[idx]['v'] = v_all
                
                scores = (q.float() @ k_all.float().transpose(-1, -2)) / 8.0
                attn = F.softmax(scores, dim=-1).to(dtype)
                x = x + F.linear((attn @ v_all).transpose(1, 2).reshape(B, 1, 2048), l['o'])

                x_n2 = rms_norm(x, l['ln2'])
                swiglu = F.silu(F.linear(x_n2, l['gate'])) * F.linear(x_n2, l['up'])
                mlp_dense = F.linear(swiglu, l['down'])

                # Batched 34 Council MoE (single step)
                swiglu_flat = swiglu.view(8192, 1).unsqueeze(0).repeat(34, 1, 1) # [34, 8192, 1]
                e_mid = torch.bmm(W0_batched, swiglu_flat) # [34, 16, 1]
                e_out = torch.bmm(W1_batched, e_mid)      # [34, 2048, 1]
                moe_out = e_out.mean(dim=0).t().view(B, 1, 2048)
                x = x + mlp_dense + 0.05 * moe_out

            h = rms_norm(x, norm_f)
            logits = F.linear(h.float(), wte.float())
            next_token = logits[0, -1].argmax().item()
            generated_tokens.append(next_token)
            cur_pos += 1
            
    elapsed = time.time() - start_time
    decoded = tok.decode(generated_tokens)
    return decoded, len(generated_tokens), elapsed

test_prompts = [
    "What is the capital of France and why is it culturally significant?",
    "Explain the core difference between inductive and deductive reasoning:",
    "What is Bushido, and how does it guide honor and duty?"
]

print("\n" + "="*80)
print("ACCELERATED GPU INFERENCE: BATCHED 34 COUNCIL MoE + KV-CACHE")
print("="*80)

for p in test_prompts:
    print(f"\n[Prompt]: {p}")
    text, n_tok, elap = generate_fast(p, max_tokens=35)
    speed = n_tok / max(1e-5, elap)
    print(f">>> Response: {text.strip()}")
    print(f"[Performance]: {n_tok} tokens in {elap:.2f}s -> {speed:.2f} tokens/sec")
    print("-" * 80)

print(f"\n>>> Final VRAM Peak: {torch.cuda.max_memory_allocated() / (1024**2):.1f} MB / 4096 MB")
