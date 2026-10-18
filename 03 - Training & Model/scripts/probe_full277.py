"""Probe mini_full_best.pt (val 2.77) with MATCHED legacy BPE, replicating gateway generation."""
import sys
from pathlib import Path
REPO = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "03 - Training & Model"))
sys.path.insert(0, str(REPO))
import torch
import torch.nn.functional as F
from quillan_bpe_tokenizer import QuillanBPETokenizer
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

CKPT = REPO / "checkpoints" / "hf_restore" / "mini_full_best.pt"
tok = QuillanBPETokenizer()
cfg = QuillanOniConfig(vocab_size=50262, hidden_dim=1024, ffn_dim=2048,
                       n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
m = QuillanRoninOni(cfg)
bd = torch.load(CKPT, map_location="cpu", weights_only=True)
missing, unexp = m.load_state_dict(bd.get("model_state_dict", bd), strict=False)
print(f"bound missing={len(missing)} unexp={len(unexp)}", flush=True)
m.eval()

PROMPT = "Say hello in one short sentence."
MAX_TOKENS, TEMP, TOP_K = 60, 0.65, 40
tokens = tok.encode(PROMPT)
if not tokens:
    tokens = [50256]
generated = list(tokens)
past_key_values = None
with torch.no_grad():
    pt = torch.tensor([tokens[-256:]], dtype=torch.long)
    out, past_key_values = m(pt, use_cache=True)
    curr_logits = out[:, -1, :].clone()
    for _ in range(MAX_TOKENS):
        gen_only = generated[len(tokens):]
        if gen_only:
            for tid in set(gen_only[-32:]):
                if curr_logits[0, tid] > 0:
                    curr_logits[0, tid] /= 1.15
                else:
                    curr_logits[0, tid] *= 1.15
        if len(gen_only) >= 3:
            last_2 = tuple(gen_only[-2:])
            for i in range(len(gen_only) - 2):
                if tuple(gen_only[i:i + 2]) == last_2:
                    banned = gen_only[i + 2]
                    curr_logits[0, banned] = float("-inf")
        if TEMP <= 0.05:
            next_tok = torch.argmax(curr_logits, dim=-1).item()
        else:
            v, _ = torch.topk(curr_logits, min(TOP_K, curr_logits.size(-1)))
            curr_logits[curr_logits < v[..., [-1]]] = float("-inf")
            probs = F.softmax(curr_logits / max(TEMP, 0.05), dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
        generated.append(next_tok)
        if next_tok in [50256, 0, 50261]:
            break
        ni = torch.tensor([[next_tok]], dtype=torch.long)
        out, past_key_values = m(ni, past_key_values=past_key_values, use_cache=True)
        curr_logits = out[:, -1, :].clone()
comp = generated[len(tokens):]
text = tok.decode(comp).strip()
for stop in ["<|end|>", "<|endoftext|>", "</assistant_response>", "<|user|>", "<|start|>", "<|im_end|>", "<|im_start|>"]:
    if stop in text:
        text = text.split(stop)[0].strip()
print("PROMPT:", PROMPT, flush=True)
print("OUTPUT:", repr(text), flush=True)
print("tokens:", len(tokens), "completion:", len(comp), flush=True)