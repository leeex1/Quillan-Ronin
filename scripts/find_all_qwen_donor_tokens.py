from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(r'C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775')
vocab = tok.get_vocab()
matches = {}
for word, idx in vocab.items():
    low = word.lower()
    if 'qwen' in low or 'alibaba' in low or 'wen' == low or 'ibaba' in low:
        matches[idx] = word
for idx in sorted(matches.keys()):
    print(f'Token {idx}: {repr(matches[idx])}')
