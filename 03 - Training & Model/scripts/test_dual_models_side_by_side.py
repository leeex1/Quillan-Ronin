import os, sys, time, gc, torch
import torch.nn.functional as F
from safetensors.torch import load_file
import gguf, gguf.quants
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("================================================================================")
print(f"QUILLAN-RONIN DUAL MODEL BENCHMARK: MINI 0.8B (8L) vs FLAGSHIP 1B (16L)")
print(f"Target Hardware: {device.upper()} ({torch.cuda.get_device_name(0)}) | Precision: Mixed FP16/FP32")
print("================================================================================\n")

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

council_34 = (
    extract_svd_experts(w_down_llama, 9) +
    extract_svd_experts(w_down_falcon, 9) +
    extract_svd_experts(w_down_bitnet, 8) +
    extract_svd_experts(w_down_qwen, 8)
)

wte = st_llama['model.embed_tokens.weight'].to(device=device, dtype=dtype)
norm_f = st_llama['model.norm.weight'].to(device=device, dtype=dtype)

def build_layers(n_layers):
    layers = []
    for l in range(n_layers):
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

def run_inference(layers, prompt, max_tokens=35):
    enc = tok.encode(f'<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    cur = torch.tensor([enc.ids], dtype=torch.long, device=device)
    output_tokens = []
    
    start = time.time()
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

                moe_out = torch.zeros_like(x)
                for (w0_t, w1_t) in council_34:
                    e_mid = F.linear(swiglu, w0_t)
                    e_out = F.linear(e_mid, w1_t)
                    moe_out = moe_out + e_out * (1.0 / 34.0)
                x = x + mlp_dense + 0.05 * moe_out

            h = rms_norm(x, norm_f)
            logits = F.linear(h.float(), wte.float())
            nxt = logits[0, -1].argmax().item()
            if nxt in [128001, 128009]: break
            output_tokens.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
            
    elapsed = time.time() - start
    return tok.decode(output_tokens), len(output_tokens), elapsed

test_prompt = "What is the strategic purpose of a defense-in-depth security model?"

# ── 1. EVALUATE MINI 0.8B (8 LAYERS) ──
print(">>> [1/2] Loading QUILLAN-RONIN MINI 0.8B (8 Layers, High-Speed Triage)...")
layers_mini = build_layers(8)
vram_mini = torch.cuda.memory_allocated() / (1024**2)
print(f"    VRAM Allocated: {vram_mini:.1f} MB")
text_mini, n_tok_mini, time_mini = run_inference(layers_mini, test_prompt)
speed_mini = n_tok_mini / max(1e-5, time_mini)
print(f"    Output: {text_mini.strip()}")
print(f"    Metrics: {n_tok_mini} tokens in {time_mini:.2f}s -> {speed_mini:.2f} tokens/sec\n")

# Free memory before loading Flagship
del layers_mini
gc.collect()
torch.cuda.empty_cache()

# ── 2. EVALUATE FLAGSHIP 1B (16 LAYERS) ──
print(">>> [2/2] Loading QUILLAN-RONIN FLAGSHIP 1B (16 Layers, Deep Deliberation)...")
layers_flagship = build_layers(16)
vram_flagship = torch.cuda.memory_allocated() / (1024**2)
print(f"    VRAM Allocated: {vram_flagship:.1f} MB")
text_flagship, n_tok_flagship, time_flagship = run_inference(layers_flagship, test_prompt)
speed_flagship = n_tok_flagship / max(1e-5, time_flagship)
print(f"    Output: {text_flagship.strip()}")
print(f"    Metrics: {n_tok_flagship} tokens in {time_flagship:.2f}s -> {speed_flagship:.2f} tokens/sec\n")

print("================================================================================")
print("COMPARATIVE EVALUATION SCORECARD:")
print(f"• Mini 0.8B (8L)    : {vram_mini:.1f} MB VRAM | {speed_mini:.2f} tok/s | Role: Ultra-low latency edge speed")
print(f"• Flagship 1B (16L) : {vram_flagship:.1f} MB VRAM | {speed_flagship:.2f} tok/s | Role: Deep multi-step reasoning")
print("================================================================================")
