import os
from tokenizers import Tokenizer

snap_qwen = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
tok_qwen = Tokenizer.from_file(os.path.join(snap_qwen, "tokenizer.json"))
for i in [50335, 16948, 28260]:
    print("Qwen:", i, repr(tok_qwen.decode([i])))

snap_llama = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c"
tok_llama = Tokenizer.from_file(os.path.join(snap_llama, "tokenizer.json"))
for i in [51435, 5109, 15836]:
    print("Llama:", i, repr(tok_llama.decode([i])))
