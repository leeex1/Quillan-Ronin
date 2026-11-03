import os
import torch
from safetensors.torch import load_file
import gguf

hf_cache = r"C:\Users\Admin\.cache\huggingface\hub"
ollama_blobs = r"C:\Users\Admin\.ollama\models\blobs"

# 1. Llama 3.2 1B
llama_path = os.path.join(hf_cache, "models--unsloth--Llama-3.2-1B-Instruct", "snapshots")
if os.path.exists(llama_path):
    snap = os.listdir(llama_path)[0]
    st = load_file(os.path.join(llama_path, snap, "model.safetensors"))
    print("Llama 3.2 1B loaded:", len(st), "tensors. embed:", st["model.embed_tokens.weight"].shape)

# 2. Falcon 3 1B GGUF
falcon_blob = os.path.join(ollama_blobs, "sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01")
if os.path.exists(falcon_blob):
    r = gguf.GGUFReader(falcon_blob)
    print("Falcon 3 1B GGUF loaded:", len(r.tensors), "tensors.")

# 3. BitNet 1.58b Large
bitnet_path = os.path.join(hf_cache, "models--1bitLLM--bitnet_b1_58-large", "snapshots")
if os.path.exists(bitnet_path):
    snap = os.listdir(bitnet_path)[0]
    st_b = load_file(os.path.join(bitnet_path, snap, "model.safetensors"))
    print("BitNet 1.58b Large loaded:", len(st_b), "tensors. embed:", st_b["model.embed_tokens.weight"].shape)

# 4. Qwen 2.5 0.5B
qwen_path = os.path.join(hf_cache, "models--Qwen--Qwen2.5-0.5B-Instruct", "snapshots")
if os.path.exists(qwen_path):
    snap = os.listdir(qwen_path)[0]
    st_q = load_file(os.path.join(qwen_path, snap, "model.safetensors"))
    print("Qwen 2.5 0.5B loaded:", len(st_q), "tensors. embed:", st_q["model.embed_tokens.weight"].shape)
