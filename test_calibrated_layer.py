import torch
import torch.nn.functional as F
import inspect, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
sys.path.insert(0, r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from tokenizers import Tokenizer
from safetensors import safe_open

p_llama = r'C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\model.safetensors'
p_bitnet = r'C:\Users\Admin\.cache\huggingface\hub\models--1bitLLM--bitnet_b1_58-large\snapshots\85d047191dcb224f0e04f20d26110caaf8dc1a47\model.safetensors'
p_qwen = r'C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors'

tok_q_path = r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json'
tok_l_path = r'C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\tokenizer.json'
tok_b_path = r'C:\Users\Admin\.cache\huggingface\hub\models--1bitLLM--bitnet_b1_58-large\snapshots\85d047191dcb224f0e04f20d26110caaf8dc1a47\tokenizer.json'
tok_qw_path = r'C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\tokenizer.json'

t_q = Tokenizer.from_file(tok_q_path)
t_l = Tokenizer.from_file(tok_l_path)
t_b = Tokenizer.from_file(tok_b_path)
t_qw = Tokenizer.from_file(tok_qw_path)

vocab_size = 50262
hidden_dim = 1152
n_layer = 6
ffn_dim = 3072

def get_best_donor_id(donor_tok, text, raw_token):
    if raw_token:
        d_id = donor_tok.token_to_id(raw_token)
        if d_id is not None:
            return d_id
        cand = raw_token.replace('Ġ', ' ').replace(' ', ' ')
        d_id = donor_tok.token_to_id(cand)
        if d_id is not None:
            return d_id
        cand2 = raw_token.replace('Ġ', '')
        d_id = donor_tok.token_to_id(cand2)
        if d_id is not None:
            return d_id
    if text and len(text.strip()) > 0:
        enc = donor_tok.encode(text, add_special_tokens=False).ids
        if enc:
            if len(enc) > 1 and enc[0] in [29871, 259]:
                return enc[1]
            return enc[0]
    return 0

map_l = torch.zeros(vocab_size, dtype=torch.long)
map_b = torch.zeros(vocab_size, dtype=torch.long)
map_qw = torch.zeros(vocab_size, dtype=torch.long)

for q_id in range(vocab_size):
    s = t_q.id_to_token(q_id)
    text = t_q.decode([q_id])
    map_l[q_id] = get_best_donor_id(t_l, text, s)
    map_b[q_id] = get_best_donor_id(t_b, text, s)
    map_qw[q_id] = get_best_donor_id(t_qw, text, s)

def project_weight(W: torch.Tensor, target_rows: int, target_cols: int) -> torch.Tensor:
    W = W.float().cpu()
    r, c = W.shape
    if r == target_rows and c == target_cols:
        return W
    return F.interpolate(W.unsqueeze(0).unsqueeze(0), size=(target_rows, target_cols), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

def to_bitnet_ternary(W: torch.Tensor, target_std: float = None, eps: float = 1e-5) -> torch.Tensor:
    W_f = W.float().cpu()
    scale = 1.0 / W_f.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    W_scaled = W_f * scale
    W_q = torch.round(torch.clamp(W_scaled, -1.0, 1.0))
    W_dequant = (W_q / scale).cpu()
    if target_std is not None:
        curr_std = W_dequant.std().clamp(min=1e-6)
        W_dequant = W_dequant * (target_std / curr_std)
    return W_dequant

print("Building calibrated model test...")
with safe_open(p_llama, framework="pt", device="cpu") as f_l, \
     safe_open(p_bitnet, framework="pt", device="cpu") as f_b, \
     safe_open(p_qwen, framework="pt", device="cpu") as f_qw:

    emb_l = project_weight(f_l.get_tensor("model.embed_tokens.weight")[map_l], vocab_size, hidden_dim)
    emb_b = project_weight(f_b.get_tensor("model.embed_tokens.weight")[map_b], vocab_size, hidden_dim)
    emb_qw = project_weight(f_qw.get_tensor("model.embed_tokens.weight")[map_qw], vocab_size, hidden_dim)

    # Unit variance matching
    emb_l = emb_l * (0.025 / emb_l.std().clamp(min=1e-6))
    emb_b = emb_b * (0.025 / emb_b.std().clamp(min=1e-6))
    emb_qw = emb_qw * (0.025 / emb_qw.std().clamp(min=1e-6))

    wte = (emb_l * (1.0/3.0) + emb_b * (1.0/3.0) + emb_qw * (1.0/3.0)).to(torch.bfloat16)

prompt = "Quillan is a sovereign cognitive agent. Its mission is"
enc = t_q.encode(prompt).ids
x_ids = torch.tensor([enc], dtype=torch.long)
h = F.embedding(x_ids, wte.float())
print(f"Embedding shape: {h.shape}, norm: {h.norm().item():.2f}, std: {h.std().item():.4f}")

# Test dot product with LM head (which is wte)
logits = h @ wte.float().t()
last_logits = logits[0, -1]
probs = F.softmax(last_logits, dim=-1)
topk = torch.topk(probs, 5)
print("Embedding direct vocabulary match top-5 next tokens:")
for p, idx in zip(topk.values, topk.indices):
    tok_str = t_q.decode([idx.item()])
    print(f"  token {idx.item():5d} ('{tok_str}'): prob={p.item():.4f}")

