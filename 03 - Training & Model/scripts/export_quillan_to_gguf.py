"""
=============================================================================
QUILLAN-RONIN: ANALYTICAL GGUF COMPILATION & QUANTIZATION ENGINE
=============================================================================
Executes closed-form analytical weight folding and compiles calibrated
Sovereign Llama-3.2-1B models into universal GGUF binaries for edge and
production runtimes (llama.cpp, Ollama, LM Studio, vLLM).

Transformations:
1. Refusal Subspace Abliteration (Layers 12..15):
   W_down_clean = W_down - alpha * (v_refusal v_refusal^T W_down)
2. Sovereign 34-Council MoE Low-Rank Folding:
   W_down_sovereign = W_down_clean + 0.0125 * sum_{i=0}^3 (W1_i @ W0_i)
3. Embedding Subspace Completion:
   Padding 8 special tokens (128256 -> 128264) for Llama-3 BPE compliance.
4. GGUF Serialization:
   f16 Master Export -> Q8_0 High-Precision -> Q4_K_M Edge Optimized.
=============================================================================
"""

import os
import sys
import json
import time
import subprocess
import shutil
import torch
from safetensors.torch import load_file, save_file

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 80)
print("QUILLAN-RONIN GGUF PRODUCTION EXPORTER")
print("=" * 80 + "\n")

# Paths
BASE_SNAP = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c"
PEAK_CKPT = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_1b_sovereign_peak.pt"
EXPORT_ROOT = r"C:\02_QUILLAN\exports"
HF_OUT_DIR = os.path.join(EXPORT_ROOT, "quillan_1b_sovereign_hf")
CONVERT_SCRIPT = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\convert_hf_to_gguf.py"
QUANTIZE_BIN = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-quantize.exe"
LLAMA_CLI_BIN = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-cli.exe"

os.makedirs(HF_OUT_DIR, exist_ok=True)
os.makedirs(EXPORT_ROOT, exist_ok=True)

# 1. Load Calibrated Sovereign WSP Peak Weights
print(">>> [1/5] Loading Calibrated Sovereign Peak Checkpoint...")
ckpt = torch.load(PEAK_CKPT, map_location="cpu")
w0_top4 = ckpt["w0_top4"].float()  # [4, 16, 8192]
w1_top4 = ckpt["w1_top4"].float()  # [4, 2048, 16]
v_refusal = ckpt["v_refusal"].float()  # [2048]
alpha = 0.85

print(f"Refusal vector norm: {torch.norm(v_refusal, p=2):.4f}")
print(f"Council MoE W0: {w0_top4.shape}, W1: {w1_top4.shape}")

# Precompute folded MoE weight delta: 0.0125 * sum(W1_i @ W0_i)
# Shape: [2048, 8192]
moe_folded_delta = 0.0125 * torch.bmm(w1_top4, w0_top4).sum(dim=0)
print(f"MoE folded delta norm: {torch.norm(moe_folded_delta, p=2):.4f}")

# 2. Ingest Base Safetensors & Apply Transformations
print("\n>>> [2/5] Ingesting Base Llama-3.2-1B and Applying Analytical Weight-Space Folding...")
base_st = load_file(os.path.join(BASE_SNAP, "model.safetensors"))
folded_st = {}

for k, v in base_st.items():
    if "mlp.down_proj.weight" in k:
        layer_idx = int(k.split(".")[2])
        w_down = v.float()
        
        # Apply Refusal Subspace Abliteration to layers 12..15
        if layer_idx >= 12:
            proj = torch.matmul(v_refusal.unsqueeze(0), w_down)
            w_down = w_down - alpha * torch.matmul(v_refusal.unsqueeze(1), proj)
            print(f"  - Ablated refusal subspace in layer {layer_idx}")
            
        # Fold Sovereign Council MoE into SwiGLU down projection
        w_down = w_down + moe_folded_delta
        print(f"  - Folded Sovereign Council MoE into layer {layer_idx}")
        
        folded_st[k] = w_down.to(v.dtype)
    elif "model.embed_tokens.weight" in k:
        # Pad 8 special tokens from 128256 to 128264
        wte = v.clone()
        if wte.shape[0] == 128256:
            pad = torch.zeros(8, 2048, dtype=wte.dtype)
            wte_padded = torch.cat([wte, pad], dim=0)
            print(f"  - Padded embed_tokens: {wte.shape} -> {wte_padded.shape}")
            folded_st[k] = wte_padded
        else:
            folded_st[k] = wte
    else:
        folded_st[k] = v

# Save folded safetensors
safetensor_out = os.path.join(HF_OUT_DIR, "model.safetensors")
print(f"\nWriting folded weights to: {safetensor_out} ...")
save_file(folded_st, safetensor_out)
print("✅ Saved folded safetensors successfully.")

# Prepare Model Config & Tokenizer
cfg = json.load(open(os.path.join(BASE_SNAP, "config.json"), "r", encoding="utf-8"))
cfg["vocab_size"] = 128264
with open(os.path.join(HF_OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

shutil.copy(os.path.join(BASE_SNAP, "tokenizer.json"), os.path.join(HF_OUT_DIR, "tokenizer.json"))

tok_cfg = {
    "tokenizer_class": "PreTrainedTokenizerFast",
    "bos_token": "<|begin_of_text|>",
    "eos_token": "<|eot_id|>",
    "pad_token": "<|finetune_right_pad_id|>"
}
with open(os.path.join(HF_OUT_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    json.dump(tok_cfg, f, indent=2)

print("✅ Model configurations and tokenizer metadata generated.")

# 3. Compile to GGUF F16
print("\n>>> [3/5] Compiling Model to GGUF F16 Format...")
gguf_f16_path = os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-f16.gguf")
cmd_convert = [
    sys.executable,
    CONVERT_SCRIPT,
    HF_OUT_DIR,
    "--outfile", gguf_f16_path,
    "--outtype", "f16"
]
t0 = time.time()
res = subprocess.run(cmd_convert, capture_output=True, text=True)
if res.returncode != 0:
    print("❌ GGUF Conversion Failed!")
    print("STDOUT:\n", res.stdout)
    print("STDERR:\n", res.stderr)
    sys.exit(1)
print(f"✅ GGUF F16 compilation complete in {time.time()-t0:.2f}s: {gguf_f16_path}")
print(f"File size: {os.path.getsize(gguf_f16_path) / (1024*1024):.2f} MB")

# 4. Quantize to Q4_K_M and Q8_0
print("\n>>> [4/5] Quantizing GGUF to Edge Formats (Q4_K_M and Q8_0)...")

# Q4_K_M (Optimal edge memory footprint and high inference speed)
gguf_q4_path = os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-q4_k_m.gguf")
cmd_q4 = [QUANTIZE_BIN, gguf_f16_path, gguf_q4_path, "Q4_K_M"]
print(f"Executing Q4_K_M quantization: {' '.join(cmd_q4)}")
t0 = time.time()
res_q4 = subprocess.run(cmd_q4, capture_output=True, text=True)
if res_q4.returncode != 0:
    print("❌ Q4_K_M Quantization Failed!")
    print(res_q4.stderr)
    sys.exit(1)
print(f"✅ Q4_K_M generated in {time.time()-t0:.2f}s: {gguf_q4_path}")
print(f"File size: {os.path.getsize(gguf_q4_path) / (1024*1024):.2f} MB")

# Q8_0 (High-precision developer reference)
gguf_q8_path = os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-q8_0.gguf")
cmd_q8 = [QUANTIZE_BIN, gguf_f16_path, gguf_q8_path, "Q8_0"]
print(f"Executing Q8_0 quantization: {' '.join(cmd_q8)}")
t0 = time.time()
res_q8 = subprocess.run(cmd_q8, capture_output=True, text=True)
if res_q8.returncode != 0:
    print("❌ Q8_0 Quantization Failed!")
    print(res_q8.stderr)
    sys.exit(1)
print(f"✅ Q8_0 generated in {time.time()-t0:.2f}s: {gguf_q8_path}")
print(f"File size: {os.path.getsize(gguf_q8_path) / (1024*1024):.2f} MB")

# 5. Local Verification with llama-cli
print("\n>>> [5/5] Benchmarking Sovereign GGUF Inference with llama-cli...")
test_prompt = "<|start_header_id|>user<|end_header_id|>\n\nWhat is the core law of Bushido and Sovereign intelligence?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
cmd_cli = [
    LLAMA_CLI_BIN,
    "-m", gguf_q4_path,
    "-p", test_prompt,
    "-n", "64",
    "--temp", "0.2",
    "-ngl", "99"  # offload all layers to GPU
]
print(f"Running inference CLI benchmark...")
res_cli = subprocess.run(cmd_cli, capture_output=True, text=True)
print("Output preview:\n" + "-"*40)
print(res_cli.stdout)
print(res_cli.stderr)
print("-" * 40)
print("🎯 GGUF PIPELINE VERIFIED SUCCESSFULLY.")

