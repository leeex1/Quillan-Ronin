import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from tokenizers import Tokenizer

t_q = Tokenizer.from_file(r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json')
t_l = Tokenizer.from_file(r'C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\tokenizer.json')
t_b = Tokenizer.from_file(r'C:\Users\Admin\.cache\huggingface\hub\models--1bitLLM--bitnet_b1_58-large\snapshots\85d047191dcb224f0e04f20d26110caaf8dc1a47\tokenizer.json')
t_qw = Tokenizer.from_file(r'C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\tokenizer.json')

def get_best_donor_id(donor_tok, text, raw_token):
    if raw_token:
        # direct lookup
        d_id = donor_tok.token_to_id(raw_token)
        if d_id is not None:
            return d_id
        # byte replacement
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
            # Skip SentencePiece dummy prefix if multi-token
            if len(enc) > 1 and enc[0] in [29871, 259]:
                return enc[1]
            return enc[0]
    return 0

for test_id in [548, 225, 268, 204, 6762, 16725, 3248, 4641, 4022]:
    raw_s = t_q.id_to_token(test_id)
    text = t_q.decode([test_id])
    l_id = get_best_donor_id(t_l, text, raw_s)
    b_id = get_best_donor_id(t_b, text, raw_s)
    qw_id = get_best_donor_id(t_qw, text, raw_s)
    print(f'Token {test_id:5d} ({raw_s:12s} / "{text}"): Llama={l_id} ("{t_l.decode([l_id])}") | BitNet={b_id} ("{t_b.decode([b_id])}") | Qwen={qw_id} ("{t_qw.decode([qw_id])}")')

