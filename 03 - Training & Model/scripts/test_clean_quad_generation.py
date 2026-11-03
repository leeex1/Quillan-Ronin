import os
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import gguf, gguf.quants
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("================================================================================", flush=True)
print("TESTING CLEAN QUAD-DONOR GENERATION (EXACT RMSNORM + RAW DONOR WEIGHTS)", flush=True)
print("================================================================================", flush=True)

hf_cache = r'C:\Users\Admin\.cache\huggingface\hub'
ollama_blobs = r'C:\Users\Admin\.ollama\models\blobs'

# 1. Load Tokenizer & Llama
llama_dir = os.path.join(hf_cache, 'models--unsloth--Llama-3.2-1B-Instruct', 'snapshots')
llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
tok = Tokenizer.from_file(os.path.join(llama_snap, 'tokenizer.json'))
print("Loading Llama 3.2 1B...", flush=True)
st_llama = load_file(os.path.join(llama_snap, 'model.safetensors'))

# 2. Load Falcon
print("Loading Falcon 3 1B...", flush=True)
falcon_blob = os.path.join(ollama_blobs, 'sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01')
r_falcon = gguf.GGUFReader(falcon_blob)
def dequant_gguf(t):
    if t.tensor_type != 0: arr = gguf.quants.dequantize(t.data, t.tensor_type)
    else: arr = t.data
    return torch.from_numpy(arr.copy()).float()
falcon_tensors = {t.name: dequant_gguf(t) for t in r_falcon.tensors if 'blk.0.' in t.name}
del r_falcon

# 3. Load BitNet
print("Loading BitNet 1.58b...", flush=True)
bitnet_dir = os.path.join(hf_cache, 'models--1bitLLM--bitnet_b1_58-large', 'snapshots')
bitnet_snap = os.path.join(bitnet_dir, os.listdir(bitnet_dir)[0])
st_bitnet = load_file(os.path.join(bitnet_snap, 'model.safetensors'))

# 4. Load Qwen 3.5 2B
print("Loading Qwen 3.5 2B...", flush=True)
qwen35_blob = os.path.join(ollama_blobs, 'sha256-e8bf47d40c5ba68baf8f3e503ee36710da6e8641cb0a670a508cff5b3f0cfbf7')
r_qwen35 = gguf.GGUFReader(qwen35_blob)
qwen35_tensors = {t.name: dequant_gguf(t) for t in r_qwen35.tensors if 'blk.0.' in t.name}
del r_qwen35

print("All 4 donors loaded! Building clean quad-donor pipeline...", flush=True)

def rms_norm(x, weight, eps=1e-5):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

def project_dim(W, target_out, target_in):
    r, c = W.shape
    if r == target_out and c == target_in: return W
    if r >= target_out: W = W[:target_out, :]
    else: W = torch.cat([W, W.repeat((target_out - r)//r + 1, 1)[:(target_out - r), :]], dim=0)
    if c >= target_in: W = W[:, :target_in]
    else: W = torch.cat([W, W.repeat(1, (target_in - c)//c + 1)[:, :(target_in - c)]], dim=1)
    return W

wte = st_llama["model.embed_tokens.weight"].float()
norm_f = st_llama["model.norm.weight"].float()

# Build 4 clean layers: Llama (L0), Falcon (L1), BitNet (L2), Qwen (L3)
# Exact raw weights, zero mean-shifting!
layers = []

# Layer 0: Llama 3.2 1B
layers.append({
    "donor": "Llama 3.2 1B",
    "ln1": st_llama["model.layers.0.input_layernorm.weight"], # RAW mean=0.14
    "ln2": st_llama["model.layers.0.post_attention_layernorm.weight"],
    "q": st_llama["model.layers.0.self_attn.q_proj.weight"],
    "k": st_llama["model.layers.0.self_attn.k_proj.weight"].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048),
    "v": st_llama["model.layers.0.self_attn.v_proj.weight"].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048),
    "o": st_llama["model.layers.0.self_attn.o_proj.weight"],
    "gate": st_llama["model.layers.0.mlp.gate_proj.weight"],
    "up": st_llama["model.layers.0.mlp.up_proj.weight"],
    "down": st_llama["model.layers.0.mlp.down_proj.weight"],
})

# Layer 1: Falcon 3 1B
layers.append({
    "donor": "Falcon 3 1B",
    "ln1": falcon_tensors["blk.0.attn_norm.weight"],
    "ln2": falcon_tensors["blk.0.ffn_norm.weight"],
    "q": falcon_tensors["blk.0.attn_q.weight"],
    "k": falcon_tensors["blk.0.attn_k.weight"].view(16, 64, 2048).repeat_interleave(2, dim=0).view(2048, 2048),
    "v": falcon_tensors["blk.0.attn_v.weight"].view(16, 64, 2048).repeat_interleave(2, dim=0).view(2048, 2048),
    "o": falcon_tensors["blk.0.attn_output.weight"],
    "gate": falcon_tensors["blk.0.ffn_gate.weight"],
    "up": falcon_tensors["blk.0.ffn_up.weight"],
    "down": falcon_tensors["blk.0.ffn_down.weight"],
})

# Layer 2: BitNet 1.58b Large
layers.append({
    "donor": "BitNet 1.58b",
    "ln1": project_dim(st_bitnet["model.layers.0.input_layernorm.weight"].unsqueeze(0), 1, 2048).squeeze(0),
    "ln2": project_dim(st_bitnet["model.layers.0.post_attention_layernorm.weight"].unsqueeze(0), 1, 2048).squeeze(0),
    "q": project_dim(st_bitnet["model.layers.0.self_attn.q_proj.weight"], 2048, 2048),
    "k": project_dim(st_bitnet["model.layers.0.self_attn.k_proj.weight"], 2048, 2048),
    "v": project_dim(st_bitnet["model.layers.0.self_attn.v_proj.weight"], 2048, 2048),
    "o": project_dim(st_bitnet["model.layers.0.self_attn.o_proj.weight"], 2048, 2048),
    "gate": project_dim(st_bitnet["model.layers.0.mlp.gate_proj.weight"], 8192, 2048),
    "up": project_dim(st_bitnet["model.layers.0.mlp.up_proj.weight"], 8192, 2048),
    "down": project_dim(st_bitnet["model.layers.0.mlp.down_proj.weight"], 2048, 8192),
})

# Layer 3: Qwen 3.5 2B
if "blk.0.attn_qkv.weight" in qwen35_tensors:
    qkv = qwen35_tensors["blk.0.attn_qkv.weight"]
    q_raw = qkv[:2048, :]
    k_raw = qkv[2048:4096, :]
    v_raw = qkv[4096:6144, :]
    o_raw = qwen35_tensors.get("blk.0.ssm_out.weight", qwen35_tensors.get("blk.0.attn_gate.weight", qwen35_tensors.get("blk.0.attn_output.weight")))
else:
    q_raw = qwen35_tensors["blk.0.attn_q.weight"]
    k_raw = qwen35_tensors["blk.0.attn_k.weight"]
    v_raw = qwen35_tensors["blk.0.attn_v.weight"]
    o_raw = qwen35_tensors["blk.0.attn_output.weight"]

layers.append({
    "donor": "Qwen 3.5 2B",
    "ln1": qwen35_tensors["blk.0.attn_norm.weight"],
    "ln2": qwen35_tensors["blk.0.post_attention_norm.weight"],
    "q": project_dim(q_raw, 2048, 2048),
    "k": project_dim(k_raw, 2048, 2048),
    "v": project_dim(v_raw, 2048, 2048),
    "o": project_dim(o_raw, 2048, 2048),
    "gate": project_dim(qwen35_tensors["blk.0.ffn_gate.weight"], 8192, 2048),
    "up": project_dim(qwen35_tensors["blk.0.ffn_up.weight"], 8192, 2048),
    "down": project_dim(qwen35_tensors["blk.0.ffn_down.weight"], 2048, 8192),
})

# Ensure all layer weights are float
for l in layers:
    for k, v in l.items():
        if isinstance(v, torch.Tensor):
            l[k] = v.float()

def forward_quad(input_ids):
    x = F.embedding(input_ids, wte)
    B, T, C = x.shape
    for l_idx, l in enumerate(layers):
        # Attention
        x_norm = rms_norm(x, l["ln1"])
        q = F.linear(x_norm, l["q"]).view(B, T, 32, 64).transpose(1, 2)
        k = F.linear(x_norm, l["k"]).view(B, T, 32, 64).transpose(1, 2)
        v = F.linear(x_norm, l["v"]).view(B, T, 32, 64).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / 8.0
        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        o_h = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = x + F.linear(o_h, l["o"])
        
        # MLP
        x_norm2 = rms_norm(x, l["ln2"])
        gate = F.linear(x_norm2, l["gate"])
        up = F.linear(x_norm2, l["up"])
        mlp_out = F.linear(F.silu(gate) * up, l["down"])
        x = x + mlp_out

    h = rms_norm(x, norm_f)
    logits = F.linear(h, wte)
    return logits

prompts = [
    "The capital of France is",
    "In mathematics, 2 + 2 =",
    "Bushido is the way of the"
]

print("\n" + "=" * 80, flush=True)
print("TEST GENERATION ON 4 PROMPTS", flush=True)
print("=" * 80, flush=True)

with torch.no_grad():
    for p in prompts:
        print(f"\nPROMPT: {repr(p)}", flush=True)
        print("CONTINUATION: ", end="", flush=True)
        enc = tok.encode(p)
        cur_ids = torch.tensor([enc.ids], dtype=torch.long)
        for step in range(12):
            logits = forward_quad(cur_ids)
            next_logits = logits[0, -1, :].float()
            # Low temperature
            probs = F.softmax(next_logits / 0.5, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt in [128001, 128009]: break
            t_str = tok.decode([nxt])
            print(t_str, end="", flush=True)
            cur_ids = torch.cat([cur_ids, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        print("\n", flush=True)

print("=" * 80, flush=True)
print("TEST COMPLETE", flush=True)
