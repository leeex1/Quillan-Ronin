from tokenizers import Tokenizer

t_quillan = Tokenizer.from_file(r"C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json")
t_llama = Tokenizer.from_file(r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\tokenizer.json")
t_qwen = Tokenizer.from_file(r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\tokenizer.json")
t_bitnet = Tokenizer.from_file(r"C:\Users\Admin\.cache\huggingface\hub\models--1bitLLM--bitnet_b1_58-large\snapshots\85d047191dcb224f0e04f20d26110caaf8dc1a47\tokenizer.json")

print("=== TOKENIZER INVENTORY ===")
print(f"Quillan Vocab Size: {t_quillan.get_vocab_size()}")
print(f"Llama Vocab Size  : {t_llama.get_vocab_size()}")
print(f"Qwen Vocab Size   : {t_qwen.get_vocab_size()}")
print(f"BitNet Vocab Size : {t_bitnet.get_vocab_size()}")

words = ["human", "Socrates", "mortal", "right", "triangle", "honor", "rectitude", "Quillan", "code", "1", "2"]
print("\n=== TOKEN ID COMPARISON FOR KEY WORDS ===")
for w in words:
    q_enc = t_quillan.encode(w).ids
    l_enc = t_llama.encode(w).ids
    w_enc = t_qwen.encode(w).ids
    b_enc = t_bitnet.encode(w).ids
    print(f"Word: {w:12s} | Quillan: {str(q_enc):12s} | Llama: {str(l_enc):12s} | Qwen: {str(w_enc):12s} | BitNet: {str(b_enc):12s}")

