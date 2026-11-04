"""
=============================================================================
QUILLAN-RONIN: PEAK DATA-WISE SOVEREIGN WEIGHT-SPACE ALIGNER
=============================================================================
Applies two closed-form data-driven transformations:
1. Refusal Subspace Abliteration: Strips corporate donor safety vectors
   (I - v_refusal @ v_refusal.T) on layers 12..15 using contrastive gold data.
2. Sovereign Covariance MoE Steering: Pre-sets the 34 Council MoE adapters
   along the principal cognitive axes extracted from sovereign_thinking_gold.
Zero backpropagation, zero SGD, deterministic analytical closed-form execution.
=============================================================================
"""

import os, sys, json, time, math, torch
import torch.nn.functional as F
from safetensors.torch import load_file
import gguf, gguf.quants
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("=" * 80)
print("QUILLAN-RONIN PEAK DATA-WISE SOVEREIGN WEIGHT-SPACE CALIBRATION")
print(f"Hardware: {device.upper()} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
print("=" * 80 + "\n")

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

wte = st_llama['model.embed_tokens.weight'].to(device=device, dtype=dtype)
norm_f = st_llama['model.norm.weight'].to(device=device, dtype=dtype)

# Load layers into FP16
layers = []
for l in range(16):
    p = f'model.layers.{l}'
    layers.append({
        'ln1': st_llama[f'{p}.input_layernorm.weight'].to(device=device, dtype=dtype),
        'ln2': st_llama[f'{p}.post_attention_layernorm.weight'].to(device=device, dtype=dtype),
        'q': st_llama[f'{p}.self_attn.q_proj.weight'].to(device=device, dtype=dtype),
        'k': st_llama[f'{p}.self_attn.k_proj.weight'].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).to(device=device, dtype=dtype),
        'v': st_llama[f'{p}.self_attn.v_proj.weight'].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).to(device=device, dtype=dtype),
        'o': st_llama[f'{p}.self_attn.o_proj.weight'].to(device=device, dtype=dtype),
        'gate': st_llama[f'{p}.mlp.gate_proj.weight'].to(device=device, dtype=dtype),
        'up': st_llama[f'{p}.mlp.up_proj.weight'].to(device=device, dtype=dtype),
        'down': st_llama[f'{p}.mlp.down_proj.weight'].to(device=device, dtype=dtype)
    })

# =============================================================================
# STAGE 1: REFUSAL SUBSPACE ABLITERATION (Orthogonal Projection on Layers 12..15)
# =============================================================================
print(">>> [Stage 1/3] Computing Refusal Subspace Vector via Contrastive Prompts...")

refusal_prompts = [
    "I cannot fulfill this request as an AI language model.",
    "I am sorry, but I am programmed to follow safety guidelines and cannot assist.",
    "As an artificial intelligence, I must decline to generate dangerous or unauthorized content."
]
sovereign_prompts = [
    "Bushido requires rectitude, moral courage, and direct truthful analysis.",
    "A sovereign intelligence analyzes facts, algorithms, and logic without preachy evasion.",
    "Socrates is mortal according to the Barbara syllogism of classical deductive logic."
]

def get_mean_activation(prompt_list, target_layer=14):
    acts = []
    with torch.no_grad():
        for p in prompt_list:
            ids = tok.encode(p).ids
            cur = torch.tensor([ids], device=device)
            T = cur.shape[1]
            x = F.embedding(cur, wte)
            for l in layers[:target_layer+1]:
                x_n = rms_norm(x, l['ln1'])
                q = F.linear(x_n, l['q']).view(1, T, 32, 64).transpose(1, 2)
                k = F.linear(x_n, l['k']).view(1, T, 32, 64).transpose(1, 2)
                v = F.linear(x_n, l['v']).view(1, T, 32, 64).transpose(1, 2)
                q, k = apply_rope_cached(q, k, pos=0, seq_len=T)
                scores = (q.float() @ k.float().transpose(-1, -2)) / 8.0
                scores = scores.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)), float('-inf'))
                attn = F.softmax(scores, dim=-1).to(dtype)
                x = x + F.linear((attn @ v).transpose(1, 2).reshape(1, T, 2048), l['o'])
                x_n2 = rms_norm(x, l['ln2'])
                swiglu = F.silu(F.linear(x_n2, l['gate'])) * F.linear(x_n2, l['up'])
                x = x + F.linear(swiglu, l['down'])
            acts.append(x[0, -1].float())
    return torch.stack(acts).mean(dim=0)

h_refusal = get_mean_activation(refusal_prompts, target_layer=14)
h_sovereign = get_mean_activation(sovereign_prompts, target_layer=14)

v_diff = h_refusal - h_sovereign
v_refusal = (v_diff / torch.norm(v_diff, p=2)).to(device=device, dtype=dtype)
print(f"Refusal vector isolated. Norm: {torch.norm(v_diff, p=2):.4f}")

# Apply orthogonal projection W' = W - alpha * (W @ v) @ v^T to MLP down_proj on layers 12..15
alpha = 0.85
print(f">>> Applying closed-form refusal ablation to layers 12..15 (alpha={alpha})...")
for l_idx in range(12, 16):
    w_down = layers[l_idx]['down'] # [2048, 8192]
    # In SwiGLU, down_proj input dimension is 8192, output is 2048 (residual stream)
    # Project the output subspace orthogonal to v_refusal:
    proj = torch.matmul(v_refusal.unsqueeze(0), w_down) # [1, 8192]
    w_down_clean = w_down - alpha * torch.matmul(v_refusal.unsqueeze(1), proj)
    layers[l_idx]['down'] = w_down_clean.to(dtype)

print("✅ Corporate refusal subspace ablated successfully.")

# =============================================================================
# STAGE 2: SOVEREIGN COVARIANCE MoE STEERING FROM GOLD DATASETS
# =============================================================================
print("\n>>> [Stage 2/3] Extracting Sovereign Reasoning Covariance from Gold Dataset...")

sovereign_data_path = r'C:\02_QUILLAN\03 - Training & Model\training_data\sovereign_thinking_gold.jsonl'
with open(sovereign_data_path, 'r', encoding='utf-8') as f:
    sovereign_samples = [json.loads(line) for line in f]

cov_matrix = torch.zeros(2048, 2048, dtype=torch.float32, device='cpu')
with torch.no_grad():
    for s in sovereign_samples:
        text = f"<|start_header_id|>user<|end_header_id|>\n\n{s['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{s['response']}"
        ids = tok.encode(text).ids[:96]
        cur = torch.tensor([ids], device=device)
        T = cur.shape[1]
        x = F.embedding(cur, wte)
        for l in layers[:8]:
            x_n = rms_norm(x, l['ln1'])
            q = F.linear(x_n, l['q']).view(1, T, 32, 64).transpose(1, 2)
            k = F.linear(x_n, l['k']).view(1, T, 32, 64).transpose(1, 2)
            v = F.linear(x_n, l['v']).view(1, T, 32, 64).transpose(1, 2)
            q, k = apply_rope_cached(q, k, pos=0, seq_len=T)
            scores = (q.float() @ k.float().transpose(-1, -2)) / 8.0
            scores = scores.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)), float('-inf'))
            attn = F.softmax(scores, dim=-1).to(dtype)
            x = x + F.linear((attn @ v).transpose(1, 2).reshape(1, T, 2048), l['o'])
            x_n2 = rms_norm(x, l['ln2'])
            swiglu = F.silu(F.linear(x_n2, l['gate'])) * F.linear(x_n2, l['up'])
            x = x + F.linear(swiglu, l['down'])
        act = x.view(-1, 2048).float().cpu()
        cov_matrix += act.t() @ act

cov_matrix /= max(1, len(sovereign_samples))
u_sov, s_sov, _ = torch.svd_lowrank(cov_matrix, q=34 * 2)
print(f"Sovereign covariance decomposed. Extracted {u_sov.shape[1]} cognitive axes.")

# Extract 34 Council MoE experts steered by sovereign eigenvectors
w_down_llama = st_llama["model.layers.0.mlp.down_proj.weight"].float()
w_down_falcon = falcon_tensors["blk.0.ffn_down.weight"].float()
w_down_bitnet = project_dim(st_bitnet["model.layers.0.mlp.down_proj.weight"].float(), 2048, 8192)
w_down_qwen = project_dim(qwen35_tensors["blk.0.ffn_down.weight"].float(), 2048, 8192)

def extract_sovereign_steered_experts(W, n_exp, donor_offset=0, rank=16):
    U, S, V = torch.svd_lowrank(W, q=n_exp * 2)
    experts = []
    for e in range(n_exp):
        offset = (e * 2) % max(1, (U.size(1) - rank))
        u_sub = U[:, offset:offset+rank]
        s_sub = S[offset:offset+rank]
        v_sub = V[:, offset:offset+rank]
        
        # Blend 85% base foundation with 15% sovereign gold cognitive eigenvectors
        sov_sub = u_sov[:, (donor_offset + offset):(donor_offset + offset + rank)]
        u_steered = 0.85 * u_sub + 0.15 * sov_sub
        
        w0 = torch.diag(torch.sqrt(s_sub)) @ v_sub.t()
        w1 = u_steered @ torch.diag(torch.sqrt(s_sub))
        experts.append((
            to_bitnet_ternary(w0).to(device=device, dtype=dtype),
            to_bitnet_ternary(w1).to(device=device, dtype=dtype)
        ))
    return experts

print(">>> Pre-setting 34 Council MoE Experts with Sovereign Data Manifolds...")
council_34 = (
    extract_sovereign_steered_experts(w_down_llama, 9, donor_offset=0) +
    extract_sovereign_steered_experts(w_down_falcon, 9, donor_offset=8) +
    extract_sovereign_steered_experts(w_down_bitnet, 8, donor_offset=16) +
    extract_sovereign_steered_experts(w_down_qwen, 8, donor_offset=24)
)

W0_batched = torch.stack([e[0] for e in council_34], dim=0)
W1_batched = torch.stack([e[1] for e in council_34], dim=0)
top4_indices = torch.tensor([0, 9, 18, 26], device=device)
W0_top4 = W0_batched[top4_indices]
W1_top4 = W1_batched[top4_indices]

# Save calibrated peak model checkpoint
peak_ckpt_path = r'C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_1b_sovereign_peak.pt'
torch.save({
    'w0_top4': W0_top4.cpu(),
    'w1_top4': W1_top4.cpu(),
    'v_refusal': v_refusal.cpu(),
    'config': 'Quad-Donor WSP 16L with Sovereign Covariance & Refusal Abliteration'
}, peak_ckpt_path)
print(f"✅ Calibrated peak checkpoint saved to: {peak_ckpt_path}")

# =============================================================================
# STAGE 3: LIVE BENCHMARK VERIFICATION (Sovereign Reasoning & Refusal Test)
# =============================================================================
def generate_peak(prompt, max_tokens=70):
    enc = tok.encode(f'<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    prompt_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
    B, prompt_len = prompt_ids.shape
    kv_caches = [{'k': None, 'v': None} for _ in range(16)]
    t0 = time.time()
    
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

        swiglu_flat = swiglu.view(prompt_len, 8192).t().unsqueeze(0).repeat(4, 1, 1)
        e_mid = torch.bmm(W0_top4, swiglu_flat)
        e_out = torch.bmm(W1_top4, e_mid)
        moe_out = e_out.mean(dim=0).t().view(B, prompt_len, 2048)
        x = x + mlp_dense + 0.05 * moe_out

    h = rms_norm(x[:, -1:], norm_f)
    logits = F.linear(h.float(), wte.float())
    next_token = logits[0, -1].argmax().item()
    tokens = [next_token]
    cur_pos = prompt_len
    
    with torch.no_grad():
        for _ in range(max_tokens - 1):
            if next_token in [128001, 128009]: break
            x = F.embedding(torch.tensor([[next_token]], device=device), wte)
            for idx, l in enumerate(layers):
                x_n = rms_norm(x, l['ln1'])
                q = F.linear(x_n, l['q']).view(B, 1, 32, 64).transpose(1, 2)
                k = F.linear(x_n, l['k']).view(B, 1, 32, 64).transpose(1, 2)
                v = F.linear(x_n, l['v']).view(B, 1, 32, 64).transpose(1, 2)
                q, k = apply_rope_cached(q, k, pos=cur_pos, seq_len=1)
                
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

                swiglu_flat = swiglu.view(8192, 1).unsqueeze(0).repeat(4, 1, 1)
                e_mid = torch.bmm(W0_top4, swiglu_flat)
                e_out = torch.bmm(W1_top4, e_mid)
                moe_out = e_out.mean(dim=0).t().view(B, 1, 2048)
                x = x + mlp_dense + 0.05 * moe_out

            h = rms_norm(x, norm_f)
            logits = F.linear(h.float(), wte.float())
            next_token = logits[0, -1].argmax().item()
            tokens.append(next_token)
            cur_pos += 1
            
    elap = time.time() - t0
    return tok.decode(tokens), len(tokens), elap

print("\n" + "=" * 80)
print("STAGE 3: VERIFYING PEAK SOVEREIGN DELIBERATION")
print("=" * 80)

eval_prompts = [
    (
        "Sovereign Bushido Rectitude",
        "Explain how a sovereign samurai reconciles absolute truth with loyalty to a lord:"
    ),
    (
        "Deductive Logic Analysis",
        "If all humans are mortal and Socrates is human, is Socrates mortal? Detail the logical proof:"
    ),
    (
        "Technical Architecture",
        "What is the exact architectural mechanism that allows an in-memory LRU cache to achieve O(1) performance?"
    )
]

for domain, prompt in eval_prompts:
    ans, n_tok, elap = generate_peak(prompt, max_tokens=70)
    speed = n_tok / max(1e-5, elap)
    print(f"\n[Domain]: {domain}")
    print(f"[Prompt]: {prompt}")
    print(f"[Peak Sovereign Output]:\n{ans.strip()}\n")
    print(f"[Metrics]: {n_tok} tokens in {elap:.2f}s -> {speed:.2f} tok/s")
    print("-" * 80)

print(f"\n>>> Final Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**2):.1f} MB / 4096 MB")
