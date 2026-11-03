from tokenizers import Tokenizer
import torch
import sys

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from build_true_tripartite_models import build_donor_token_map

t_quillan = Tokenizer.from_file(r"C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json")
t_llama = Tokenizer.from_file(r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\tokenizer.json")
t_qwen = Tokenizer.from_file(r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\tokenizer.json")
t_bitnet = Tokenizer.from_file(r"C:\Users\Admin\.cache\huggingface\hub\models--1bitLLM--bitnet_b1_58-large\snapshots\85d047191dcb224f0e04f20d26110caaf8dc1a47\tokenizer.json")

vocab_size = 50262
map_l = build_donor_token_map(t_quillan, t_llama, vocab_size)
map_q = build_donor_token_map(t_quillan, t_qwen, vocab_size)
map_b = build_donor_token_map(t_quillan, t_bitnet, vocab_size)

print(f"Unique tokens mapped in Llama : {len(set(map_l.tolist()))} / {vocab_size} ({len(set(map_l.tolist()))/vocab_size*100:.1f}%)")
print(f"Unique tokens mapped in Qwen  : {len(set(map_q.tolist()))} / {vocab_size} ({len(set(map_q.tolist()))/vocab_size*100:.1f}%)")
print(f"Unique tokens mapped in BitNet: {len(set(map_b.tolist()))} / {vocab_size} ({len(set(map_b.tolist()))/vocab_size*100:.1f}%)")

# Sample 10 tokens from Quillan and show what they mapped to
print("\n=== SAMPLE MAPPINGS ===")
for q_id in [10, 100, 1000, 5000, 10000, 20000, 30000, 40000, 50000]:
    q_str = t_quillan.id_to_token(q_id)
    l_id = map_l[q_id].item()
    qwen_id = map_q[q_id].item()
    b_id = map_b[q_id].item()
    print(f"ID {q_id:5d} ({repr(q_str):15s}) -> Llama: {repr(t_llama.id_to_token(l_id)):15s} | Qwen: {repr(t_qwen.id_to_token(qwen_id)):15s} | BitNet: {repr(t_bitnet.id_to_token(b_id)):15s}")

