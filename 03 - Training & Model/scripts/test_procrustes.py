import os
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import gguf, gguf.quants
from tokenizers import Tokenizer

hf_cache = r'C:\Users\Admin\.cache\huggingface\hub'
ollama_blobs = r'C:\Users\Admin\.ollama\models\blobs'

# Load Llama
llama_dir = os.path.join(hf_cache, 'models--unsloth--Llama-3.2-1B-Instruct', 'snapshots')
llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
tok = Tokenizer.from_file(os.path.join(llama_snap, 'tokenizer.json'))
st_llama = load_file(os.path.join(llama_snap, 'model.safetensors'))

# Load Falcon
falcon_blob = os.path.join(ollama_blobs, 'sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01')
r_falcon = gguf.GGUFReader(falcon_blob)
def dequant_gguf(t):
    if t.tensor_type != 0: arr = gguf.quants.dequantize(t.data, t.tensor_type)
    else: arr = t.data
    return torch.from_numpy(arr.copy()).float()
falcon_tensors = {t.name: dequant_gguf(t) for t in r_falcon.tensors if 'token_embd' in t.name or 'blk.0.' in t.name}

emb_llama = st_llama["model.embed_tokens.weight"].float() # [128256, 2048]
emb_falcon = falcon_tensors["token_embd.weight"].float()   # [50257, 2048]

# Compute Procrustes alignment on common vocabulary or tokens
# Both share common ASCII / Latin tokens (first 32000 tokens)
min_vocab = min(emb_llama.size(0), emb_falcon.size(0))
H_A = emb_llama[:10000]
H_B = emb_falcon[:10000]

M = H_A.t() @ H_B # [2048, 2048]
U, S, V = torch.svd(M)
R = U @ V.t() # Orthogonal rotation matrix

print("Procrustes Rotation R computed! Is orthogonal:", torch.allclose(R @ R.t(), torch.eye(2048), atol=1e-3))
print("Mean singular value:", S.mean().item())
