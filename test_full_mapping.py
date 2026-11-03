import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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

print(f"Token Map Built: Llama unq={len(torch.unique(map_l))}, BitNet unq={len(torch.unique(map_b))}, Qwen unq={len(torch.unique(map_qw))}")
print(f"Llama BOS counts: {(map_l == 128000).sum().item()}, BitNet BOS counts: {(map_b == 1).sum().item()}")
