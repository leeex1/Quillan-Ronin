import os, sys, torch
import torch.nn.functional as F
from safetensors.torch import load_file
import gguf, gguf.quants
from tokenizers import Tokenizer

hf_cache = r'C:\Users\Admin\.cache\huggingface\hub'
ollama_blobs = r'C:\Users\Admin\.ollama\models\blobs'

llama_dir = os.path.join(hf_cache, 'models--unsloth--Llama-3.2-1B-Instruct', 'snapshots')
llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
tok = Tokenizer.from_file(os.path.join(llama_snap, 'tokenizer.json'))
st_llama = load_file(os.path.join(llama_snap, 'model.safetensors'))

falcon_blob = os.path.join(ollama_blobs, 'sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01')
r_falcon = gguf.GGUFReader(falcon_blob)
def dequant_gguf(t):
    if t.tensor_type != 0: arr = gguf.quants.dequantize(t.data, t.tensor_type)
    else: arr = t.data
    return torch.from_numpy(arr.copy()).float()
falcon_tensors = {t.name: dequant_gguf(t) for t in r_falcon.tensors if 'blk.0.' in t.name}

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

prompt = "The capital of France is"
enc = tok.encode(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
input_ids = torch.tensor([enc.ids], dtype=torch.long)
B, T = input_ids.shape

wte = st_llama["model.embed_tokens.weight"].float()
norm_f = st_llama["model.norm.weight"].float()

# Let's test: Llama Layer 0, then Falcon Layer 1 with Gain-Matched LN, then Llama Layers 2..15
x = F.embedding(input_ids, wte)

# Layer 0: Llama
p = "model.layers.0"
ln1 = st_llama[f"{p}.input_layernorm.weight"].float()
ln2 = st_llama[f"{p}.post_attention_layernorm.weight"].float()
q_w = st_llama[f"{p}.self_attn.q_proj.weight"].float()
k_w = st_llama[f"{p}.self_attn.k_proj.weight"].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).float()
v_w = st_llama[f"{p}.self_attn.v_proj.weight"].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).float()
o_w = st_llama[f"{p}.self_attn.o_proj.weight"].float()
gate_w = st_llama[f"{p}.mlp.gate_proj.weight"].float()
up_w = st_llama[f"{p}.mlp.up_proj.weight"].float()
down_w = st_llama[f"{p}.mlp.down_proj.weight"].float()

x_n = rms_norm(x, ln1)
q = F.linear(x_n, q_w).view(B, T, 32, 64).transpose(1, 2)
k = F.linear(x_n, k_w).view(B, T, 32, 64).transpose(1, 2)
v = F.linear(x_n, v_w).view(B, T, 32, 64).transpose(1, 2)
q, k = apply_rope(q, k, T)
scores = (q @ k.transpose(-1, -2)) / 8.0
scores = scores.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float('-inf'))
attn = F.softmax(scores, dim=-1)
x = x + F.linear((attn @ v).transpose(1, 2).reshape(B, T, 2048), o_w)
x_n2 = rms_norm(x, ln2)
x = x + F.linear(F.silu(F.linear(x_n2, gate_w)) * F.linear(x_n2, up_w), down_w)

# Layer 1: Falcon with gain matching
ln1_f = falcon_tensors["blk.0.attn_norm.weight"].float() * (ln1.mean() / falcon_tensors["blk.0.attn_norm.weight"].float().mean())
ln2_f = falcon_tensors["blk.0.ffn_norm.weight"].float() * (ln2.mean() / falcon_tensors["blk.0.ffn_norm.weight"].float().mean())
q_f = falcon_tensors["blk.0.attn_q.weight"].float()
k_f = falcon_tensors["blk.0.attn_k.weight"].view(16, 64, 2048).repeat_interleave(2, dim=0).view(2048, 2048).float()
v_f = falcon_tensors["blk.0.attn_v.weight"].view(16, 64, 2048).repeat_interleave(2, dim=0).view(2048, 2048).float()
o_f = falcon_tensors["blk.0.attn_output.weight"].float()
gate_f = falcon_tensors["blk.0.ffn_gate.weight"].float()
up_f = falcon_tensors["blk.0.ffn_up.weight"].float()
down_f = falcon_tensors["blk.0.ffn_down.weight"].float()

x_n_f = rms_norm(x, ln1_f)
q = F.linear(x_n_f, q_f).view(B, T, 32, 64).transpose(1, 2)
k = F.linear(x_n_f, k_f).view(B, T, 32, 64).transpose(1, 2)
v = F.linear(x_n_f, v_f).view(B, T, 32, 64).transpose(1, 2)
q, k = apply_rope(q, k, T)
scores = (q @ k.transpose(-1, -2)) / 8.0
scores = scores.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float('-inf'))
attn = F.softmax(scores, dim=-1)
x = x + F.linear((attn @ v).transpose(1, 2).reshape(B, T, 2048), o_f)
x_n2_f = rms_norm(x, ln2_f)
x = x + F.linear(F.silu(F.linear(x_n2_f, gate_f)) * F.linear(x_n2_f, up_f), down_f)

# Layers 2..15: Llama
for l in range(2, 16):
    p = f"model.layers.{l}"
    ln1 = st_llama[f"{p}.input_layernorm.weight"].float()
    ln2 = st_llama[f"{p}.post_attention_layernorm.weight"].float()
    q_w = st_llama[f"{p}.self_attn.q_proj.weight"].float()
    k_w = st_llama[f"{p}.self_attn.k_proj.weight"].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).float()
    v_w = st_llama[f"{p}.self_attn.v_proj.weight"].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).float()
    o_w = st_llama[f"{p}.self_attn.o_proj.weight"].float()
    gate_w = st_llama[f"{p}.mlp.gate_proj.weight"].float()
    up_w = st_llama[f"{p}.mlp.up_proj.weight"].float()
    down_w = st_llama[f"{p}.mlp.down_proj.weight"].float()

    x_n = rms_norm(x, ln1)
    q = F.linear(x_n, q_w).view(B, T, 32, 64).transpose(1, 2)
    k = F.linear(x_n, k_w).view(B, T, 32, 64).transpose(1, 2)
    v = F.linear(x_n, v_w).view(B, T, 32, 64).transpose(1, 2)
    q, k = apply_rope(q, k, T)
    scores = (q @ k.transpose(-1, -2)) / 8.0
    scores = scores.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float('-inf'))
    attn = F.softmax(scores, dim=-1)
    x = x + F.linear((attn @ v).transpose(1, 2).reshape(B, T, 2048), o_w)
    x_n2 = rms_norm(x, ln2)
    x = x + F.linear(F.silu(F.linear(x_n2, gate_w)) * F.linear(x_n2, up_w), down_w)

h = rms_norm(x, norm_f)
logits = F.linear(h, wte)
print("Top 5 tokens with Gain-Matched Falcon layer:")
for idx in logits[0, -1].topk(5).indices:
    print(" ", repr(tok.decode([idx.item()])))
