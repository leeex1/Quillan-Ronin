from safetensors import safe_open
import torch

d_llama = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\model.safetensors"
d_bitnet = r"C:\Users\Admin\.cache\huggingface\hub\models--1bitLLM--bitnet_b1_58-large\snapshots\85d047191dcb224f0e04f20d26110caaf8dc1a47\model.safetensors"
d_qwen = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors"

print("=== RAW DONOR BITNET WEIGHTS ===")
with safe_open(d_bitnet, framework="pt", device="cpu") as f:
    for k in list(f.keys())[:15]:
        t = f.get_tensor(k).float()
        print(f"{k:45s} | shape: {str(list(t.shape)):15s} | mean: {t.mean().item():+.5f} | std: {t.std().item():.5f} | min: {t.min().item():+.4f} | max: {t.max().item():+.4f}")

print("\n=== RAW DONOR LLAMA WEIGHTS ===")
with safe_open(d_llama, framework="pt", device="cpu") as f:
    for k in ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.o_proj.weight", "model.norm.weight", "lm_head.weight"]:
        if k in f.keys():
            t = f.get_tensor(k).float()
            print(f"{k:45s} | shape: {str(list(t.shape)):15s} | mean: {t.mean().item():+.5f} | std: {t.std().item():.5f} | min: {t.min().item():+.4f} | max: {t.max().item():+.4f}")

print("\n=== RAW DONOR QWEN WEIGHTS ===")
with safe_open(d_qwen, framework="pt", device="cpu") as f:
    for k in ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.o_proj.weight", "model.norm.weight", "lm_head.weight"]:
        if k in f.keys():
            t = f.get_tensor(k).float()
            print(f"{k:45s} | shape: {str(list(t.shape)):15s} | mean: {t.mean().item():+.5f} | std: {t.std().item():.5f} | min: {t.min().item():+.4f} | max: {t.max().item():+.4f}")
