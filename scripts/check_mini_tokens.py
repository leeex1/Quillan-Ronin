from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(r'C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775')
for w in [' Crash', 'Crash', ' CrashOverrideX', 'OverrideX', 'Qwen', 'wen', ' Alibaba', 'alibaba', 'Alibaba']:
    ids = tok.encode(w, add_special_tokens=False)
    print(f'{repr(w)} -> {ids} -> {[tok.decode([i]) for i in ids]}')
