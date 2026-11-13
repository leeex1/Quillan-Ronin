import os
import torch
import torch.nn.functional as F
import transformers
from tokenizers import Tokenizer
import wsp_v8_closed_form_sota_compiler as v8

snap_qwen = v8.MINI_SNAP
tok_qwen = Tokenizer.from_file(os.path.join(snap_qwen, "tokenizer.json"))
model_qwen = transformers.AutoModelForCausalLM.from_pretrained(snap_qwen, dtype=torch.float32).eval()
W = model_qwen.model.embed_tokens.weight.detach()

print("Raw options:")
for c in "ABCDE":
    tids = tok_qwen.encode(c, add_special_tokens=False).ids
    print(c, tids, [torch.norm(W[i]).item() for i in tids])

print("\nSpace options:")
for c in "ABCDE":
    tids = tok_qwen.encode(" " + c, add_special_tokens=False).ids
    print(" " + c, tids, [torch.norm(W[i]).item() for i in tids])
