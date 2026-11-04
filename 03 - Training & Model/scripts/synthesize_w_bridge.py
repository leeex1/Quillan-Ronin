import os, sys, json, time, torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32

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

def precompute_rope(max_len=2048, base=500000.0, dev='cuda'):
    dim = 64
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=dev).float() / dim))
    t = torch.arange(max_len, device=dev).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().view(1, 1, max_len, dim), emb.sin().view(1, 1, max_len, dim)

cos_table, sin_table = precompute_rope(2048, dev=device)

def apply_rope(q, k, seq_len):
    cos = cos_table[:, :, :seq_len, :]
    sin = sin_table[:, :, :seq_len, :]
    def rot(v):
        v_f = v.float()
        v1, v2 = v_f[..., :32], v_f[..., 32:]
        return torch.cat((v1 * cos[..., :32] - v2 * sin[..., :32], v1 * sin[..., :32] + v2 * cos[..., :32]), dim=-1).to(v.dtype)
    return rot(q), rot(k)

# Load all 16 layers
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

def forward_layer(x, l, seq_len):
    B = x.shape[0]
    x_n = rms_norm(x, l['ln1'])
    q = F.linear(x_n, l['q']).view(B, seq_len, 32, 64).transpose(1, 2)
    k = F.linear(x_n, l['k']).view(B, seq_len, 32, 64).transpose(1, 2)
    v = F.linear(x_n, l['v']).view(B, seq_len, 32, 64).transpose(1, 2)
    q, k = apply_rope(q, k, seq_len)
    scores = (q.float() @ k.float().transpose(-1, -2)) / 8.0
    scores = scores.masked_fill(~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)), float('-inf'))
    attn = F.softmax(scores, dim=-1).to(dtype)
    x = x + F.linear((attn @ v).transpose(1, 2).reshape(B, seq_len, 2048), l['o'])
    x_n2 = rms_norm(x, l['ln2'])
    swiglu = F.silu(F.linear(x_n2, l['gate'])) * F.linear(x_n2, l['up'])
    x = x + F.linear(swiglu, l['down'])
    return x

# Step 1: Collect H7 and H15 on dataset samples
data_path = r'C:\02_QUILLAN\03 - Training & Model\training_data\Quillan_Universal_Sovereign_Gold_1000.jsonl'
with open(data_path, 'r', encoding='utf-8') as f:
    samples = [json.loads(line) for line in f][:40]

print('>>> Computing closed-form H7 -> H15 bridge covariance across 40 sovereign gold samples...')
S77 = torch.zeros(2048, 2048, dtype=torch.float32, device='cpu')
S7_15 = torch.zeros(2048, 2048, dtype=torch.float32, device='cpu')

with torch.no_grad():
    for idx, s in enumerate(samples):
        q = s['question']
        r = s['response']
        text = f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{r}"
        ids = tok.encode(text).ids[:128]
        if len(ids) < 4: continue
        cur_ids = torch.tensor([ids], device=device)
        T = cur_ids.shape[1]
        x = F.embedding(cur_ids, wte)
        
        # Pass through layers 0..7
        for l in layers[:8]:
            x = forward_layer(x, l, T)
        h7 = x.view(-1, 2048).float().cpu()
        
        # Continue through layers 8..15
        for l in layers[8:]:
            x = forward_layer(x, l, T)
        h15 = x.view(-1, 2048).float().cpu()
        
        S77 += h7.t() @ h7
        S7_15 += h7.t() @ h15

# Solve W_bridge = (S77 + ridge * I)^-1 @ S7_15
ridge = 1e-2 * torch.trace(S77) / 2048.0
W_bridge = torch.linalg.solve(S77 + ridge * torch.eye(2048), S7_15).to(device=device, dtype=dtype)
print(f'W_bridge solved successfully! Shape: {W_bridge.shape}')

bridge_path = r'C:\02_QUILLAN\checkpoints\checkpoints_oni\w_bridge_8L_to_15L.pt'
torch.save(W_bridge.cpu(), bridge_path)
print(f'Saved bridge weights to: {bridge_path}')

# Test Generation with W_bridge on Mini 8-layer model
test_prompts = [
    "The capital of France is",
    "What is the speed of light in a vacuum?",
    "Explain what an in-memory LRU cache does:"
]

print("\n" + "="*80)
print("TESTING MINI 8-LAYER MODEL WITH CLOSED-FORM SOVEREIGN BRIDGE")
print("="*80)

for p in test_prompts:
    enc = tok.encode(f"<|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    cur = torch.tensor([enc.ids], device=device)
    tokens = []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(25):
            T = cur.shape[1]
            x = F.embedding(cur, wte)
            for l in layers[:8]:
                x = forward_layer(x, l, T)
            # Apply closed-form bridge!
            x_bridged = x @ W_bridge
            h = rms_norm(x_bridged, norm_f)
            logits = F.linear(h.float(), wte.float())
            nxt = logits[0, -1].argmax().item()
            if nxt in [128001, 128009]: break
            tokens.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
    elap = time.time() - t0
    decoded = tok.decode(tokens)
    tps = len(tokens) / max(1e-5, elap)
    print(f"\n[Prompt]: {p}")
    print(f"[Mini 8L Bridged Output]: {decoded.strip()}")
    print(f"[Speed]: {len(tokens)} tokens in {elap:.2f}s -> {tps:.2f} tok/s")

