import os, sys, torch
import torch.nn.functional as F
from safetensors.torch import load_file
import gguf, gguf.quants
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

hf_cache = r'C:\Users\Admin\.cache\huggingface\hub'
ollama_blobs = r'C:\Users\Admin\.ollama\models\blobs'

# 1. Load Tokenizer & Donors
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
falcon_tensors = {t.name: dequant_gguf(t) for t in r_falcon.tensors if 'blk.0.ffn_down' in t.name}

bitnet_dir = os.path.join(hf_cache, 'models--1bitLLM--bitnet_b1_58-large', 'snapshots')
bitnet_snap = os.path.join(bitnet_dir, os.listdir(bitnet_dir)[0])
st_bitnet = load_file(os.path.join(bitnet_snap, 'model.safetensors'))

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
        # w0 is down_proj input mapping [rank, 2048]
        # In Quillan expert: delta = (x @ lora_A) @ lora_B
        # where x is [B, T, 2048], lora_A is [2048, 16], lora_B is [16, 2048]
        w_A = project_dim(u_sub, 2048, rank)
        w_B = project_dim(v_sub.t()[:rank, :], rank, 2048)
        experts.append((to_bitnet_ternary(w_A), to_bitnet_ternary(w_B)))
    return experts

exp_llama = extract_svd_experts(w_down_llama, 9)
exp_falcon = extract_svd_experts(w_down_falcon, 9)
exp_bitnet = extract_svd_experts(w_down_bitnet, 8)
exp_qwen = extract_svd_experts(w_down_qwen, 8)
council_34 = exp_llama + exp_falcon + exp_bitnet + exp_qwen

print(f"Council 34 BitNet ternary experts built: 9L/9F/8B/8Q = 34", flush=True)

# Build Flagship 1B QuillanRoninOni
cfg = QuillanOniConfig(
    vocab_size=128256,
    hidden_dim=2048,
    n_layer=16,
    n_head=32,
    head_dim=64,
    ffn_dim=8192,
    num_experts=34,
    expert_rank=16,
    use_evo_moe=False
)
cfg.vectorized_experts = True
model = QuillanRoninOni(cfg)

# Load Llama backbone into model
wte = st_llama["model.embed_tokens.weight"].to(torch.bfloat16)
model.wte.weight.data.copy_(wte)
model.lm_head.weight.data.copy_(wte)
model.ln_f.weight.data.copy_(st_llama["model.norm.weight"].to(torch.bfloat16))

for l in range(16):
    p = f"model.layers.{l}"
    blk = model.h[l]
    blk.ln_1.weight.data.copy_(st_llama[f"{p}.input_layernorm.weight"].to(torch.bfloat16))
    blk.ln_2.weight.data.copy_(st_llama[f"{p}.post_attention_layernorm.weight"].to(torch.bfloat16))

    q = st_llama[f"{p}.self_attn.q_proj.weight"].float()
    k = st_llama[f"{p}.self_attn.k_proj.weight"].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).float()
    v = st_llama[f"{p}.self_attn.v_proj.weight"].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).float()
    o = st_llama[f"{p}.self_attn.o_proj.weight"].float()

    c_attn = torch.cat([q, k, v], dim=0)
    blk.attn.c_attn.weight.data.copy_(c_attn.to(torch.bfloat16))
    blk.attn.c_attn.bias.data.zero_()
    blk.attn.c_proj.weight.data.copy_(o.to(torch.bfloat16))
    blk.attn.c_proj.bias.data.zero_()
    blk.attn.prism.w_gate.weight.data.zero_()

    gate = st_llama[f"{p}.mlp.gate_proj.weight"].float()
    up = st_llama[f"{p}.mlp.up_proj.weight"].float()
    down = st_llama[f"{p}.mlp.down_proj.weight"].float()

    c_fc = torch.cat([gate, up], dim=0)
    blk.moe.c_fc.weight.data.copy_(c_fc.to(torch.bfloat16))
    blk.moe.c_fc.bias.data.zero_()
    blk.moe.c_proj.weight.data.copy_(down.to(torch.bfloat16))
    blk.moe.c_proj.bias.data.zero_()

    # Load 34 BitNet ternary experts into blk.moe.experts
    for e_idx, (wA, wB) in enumerate(council_34):
        exp = blk.moe.experts[e_idx]
        exp.lora_A.data.copy_(wA.to(torch.bfloat16))
        exp.lora_B.data.copy_(wB.to(torch.bfloat16))
        exp.swarm.A.data.zero_()
        exp.swarm.B.data.zero_()
        exp.swarm.C.data.zero_()
        exp.swarm.D.data.zero_()
        exp.swarm.clone_diversity.data.zero_()
        exp.swarm.clone_coupling.data.zero_()

    blk.moe.router.weight.data.zero_()
    blk.moe.pull_gate.gate.weight.data.zero_()

model.eval()
print("Model initialized and loaded cleanly into QuillanRoninOni!", flush=True)

prompts = [
    "The capital of France is",
    "Explain the core difference between inductive and deductive reasoning in two sentences."
]

for prompt in prompts:
    print(f"\n[Prompt]: {prompt}", flush=True)
    print(">>> Quillan-Ronin ONI Output: ", end="", flush=True)
    enc = tok.encode(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    cur = torch.tensor([enc.ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(30):
            logits = model(cur, compute_aux=False, return_aux=False)
            if isinstance(logits, tuple): logits = logits[0]
            nxt = logits[0, -1].argmax().item()
            if nxt in [128001, 128009]: break
            t_str = tok.decode([nxt])
            print(t_str, end="", flush=True)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    print("\n", flush=True)
