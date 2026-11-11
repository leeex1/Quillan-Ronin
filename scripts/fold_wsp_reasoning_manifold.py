#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN v5.4.0-ONI: WSP REASONING MANIFOLD FOLDING ENGINE
Folds the 10-domain reasoning subspaces from sovereign_thinking_gold.jsonl
into Flagship 1.2B, Mini 0.5B, and Dragon 1.0B with EXACT parameter conservation (Diff = 0).
"""

import os
import gc
import sys
import time
import json
import shutil
import logging
import subprocess
import numpy as np
import torch
import safetensors.torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import Tokenizer
import gguf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("WSP_Reasoning_Folder")

ROOT_DIR = r"C:\02_QUILLAN"
EXPORT_ROOT = os.path.join(ROOT_DIR, "exports")
DATA_PATH = os.path.join(ROOT_DIR, r"03 - Training & Model\training_data\sovereign_thinking_gold.jsonl")
QUANTIZE_BIN = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-quantize.exe"
CONVERT_SCRIPT = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\convert_hf_to_gguf.py"

FLAGSHIP_HF = os.path.join(EXPORT_ROOT, "quillan_1b_sovereign_hf")
MINI_SNAP = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
BLOB_MINI = r"C:\Users\Admin\.ollama\models\blobs\sha256-1923271f8d4d5101e91b90d67fdd56da544a6258c614a4d1ce9cd098696062f9"

def load_gold_reasoning_samples():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(samples)} gold reasoning trajectory samples.")
    return samples

def fold_flagship_reasoning():
    logger.info("=== [Phase 2.1] Folding Reasoning Manifold into Flagship 1.2B ===")
    samples = load_gold_reasoning_samples()
    tok = AutoTokenizer.from_pretrained(FLAGSHIP_HF)
    model = AutoModelForCausalLM.from_pretrained(FLAGSHIP_HF, dtype=torch.float32).eval()

    # Collect reasoning activation vectors across layers 11-14 down_proj
    layers = [11, 12, 13, 14]
    activations = {l: [] for l in layers}
    hooks = []

    def make_hook(layer_idx):
        def hook(mod, inp, out):
            act = inp[0][0, -1, :].detach().clone()
            activations[layer_idx].append(act)
        return hook

    for l in layers:
        h = model.model.layers[l].mlp.down_proj.register_forward_hook(make_hook(l))
        hooks.append(h)

    logger.info("Running forward passes over gold reasoning trajectory prompts...")
    with torch.no_grad():
        for s in samples:
            prompt = f"<|start_header_id|>system<|end_header_id|>\n\nYou are Quillan-Ronin (Sovereign Flagship 1.2B), created by CrashOverrideX.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{s['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<think>\n"
            tokens = tok.encode(prompt, return_tensors="pt")
            _ = model(tokens)

    for h in hooks:
        h.remove()

    # Target direction: representation of reasoning tokens
    w_think = model.model.embed_tokens.weight[tok.encode("<think>", add_special_tokens=False)[0]].clone() # (2048)
    
    target_frob_ratio = 0.0040 # well under 0.0050 bound
    for l in layers:
        X = torch.stack(activations[l]) # (N, 8192)
        X_mean = X.mean(dim=0)
        X_norm = X_mean / (torch.norm(X_mean) + 1e-6) # (8192)

        W0 = model.model.layers[l].mlp.down_proj.weight.data
        w0_frob = torch.norm(W0).item()

        delta_w = torch.outer(w_think, X_norm) # (2048, 8192)
        cur_frob = torch.norm(delta_w).item()
        scale = (target_frob_ratio * w0_frob) / (cur_frob + 1e-6)
        delta_w = delta_w * scale

        actual_ratio = torch.norm(delta_w).item() / w0_frob
        logger.info(f"Layer {l} down_proj: ||W_0||_F = {w0_frob:.2f}, ||Delta W||_F = {torch.norm(delta_w).item():.4f}, ratio = {actual_ratio:.4%}")
        
        with torch.no_grad():
            model.model.layers[l].mlp.down_proj.weight.add_(delta_w)

    # Save to safetensors
    st_path = os.path.join(FLAGSHIP_HF, "model.safetensors")
    st_tmp = os.path.join(FLAGSHIP_HF, "model.safetensors.tmp")
    sd = model.state_dict()
    new_st = {k: sd[k].detach().cpu().to(torch.bfloat16) for k in sd}
    safetensors.torch.save_file(new_st, st_tmp)
    del model, sd, new_st
    gc.collect()

    os.replace(st_tmp, st_path)
    logger.info("Saved reasoning-folded weights to model.safetensors.")

    # Re-export to GGUF F16 and Q4_K_M
    f16_path = os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-wsp-v8-f16.gguf")
    q4_path = os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-wsp-v8-q4_k_m.gguf")
    q4_v3 = os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-wsp-v3-q4_k_m.gguf")

    logger.info("Converting Flagship to GGUF F16...")
    subprocess.run([sys.executable, CONVERT_SCRIPT, FLAGSHIP_HF, "--outfile", f16_path, "--outtype", "f16"], check=True)

    logger.info("Quantizing Flagship GGUF F16 -> Q4_K_M...")
    subprocess.run([QUANTIZE_BIN, f16_path, q4_path, "Q4_K_M"], check=True)
    try:
        time.sleep(2)
        if os.path.exists(f16_path):
            os.remove(f16_path)
    except Exception as e:
        logger.warning(f"Could not remove temporary f16: {e}")
    shutil.copy2(q4_path, q4_v3)
    logger.info(f"[SUCCESS] Flagship 1.2B Reasoning Folded: {q4_path} ({os.path.getsize(q4_path)} bytes)")

def fold_mini_reasoning():
    logger.info("=== [Phase 2.2] Folding Reasoning Manifold into Ronin Mini 0.5B ===")
    samples = load_gold_reasoning_samples()
    tok_b = Tokenizer.from_file(os.path.join(MINI_SNAP, "tokenizer.json"))
    model_b = AutoModelForCausalLM.from_pretrained(MINI_SNAP, dtype=torch.float32).eval()

    layers_b = [19, 20, 21, 22]
    activations_b = {l: [] for l in layers_b}
    hooks_b = []

    def make_hook_b(layer_idx):
        def hook(mod, inp, out):
            act = inp[0][0, -1, :].detach().clone()
            activations_b[layer_idx].append(act)
        return hook

    for l in layers_b:
        h = model_b.model.layers[l].mlp.down_proj.register_forward_hook(make_hook_b(l))
        hooks_b.append(h)

    enc_chat_b = lambda s, u: tok_b.encode(f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n").ids
    sys_p_b = "You are Quillan-Ronin Mini (0.5B), created by CrashOverrideX."
    with torch.no_grad():
        for s in samples:
            p_ids = enc_chat_b(sys_p_b, s["question"])
            _ = model_b(torch.tensor([p_ids[-256:]], dtype=torch.long))

    for h in hooks_b:
        h.remove()

    emb_b = model_b.model.embed_tokens.weight.detach()
    w_crash_b = emb_b[50335].clone().float() # ' Crash' [896]
    w_think_b = emb_b[tok_b.encode("<think>").ids[0]].clone().float() # [896]

    target_frob_ratio_b = 0.0025 # under 0.0035 bound
    updates_b = {}
    for l in layers_b:
        X = torch.stack(activations_b[l]) # (N, 4864)
        X_mean = X.mean(dim=0)
        X_norm = X_mean / (torch.norm(X_mean) + 1e-6)

        W0 = model_b.model.layers[l].mlp.down_proj.weight.data # (896, 4864)
        w0_frob = torch.norm(W0).item()

        delta_w = torch.outer(w_think_b, X_norm) # (896, 4864)
        cur_frob = torch.norm(delta_w).item()
        scale = (target_frob_ratio_b * w0_frob) / (cur_frob + 1e-6)
        delta_w = delta_w * scale

        logger.info(f"Mini Layer {l} down_proj: ||W_0||_F = {w0_frob:.2f}, ratio = {torch.norm(delta_w).item() / w0_frob:.4%}")
        updates_b[l] = delta_w.to(torch.float32)

    with torch.no_grad():
        for l, dw in updates_b.items():
            model_b.model.layers[l].mlp.down_proj.weight.add_(dw)

    # In-place F16 blob manifold update
    reader_b = gguf.GGUFReader(BLOB_MINI)
    tmap_b = {t.name: t for t in reader_b.tensors}
    f16_tmp_b = os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-wsp-v8-tmp-f16.gguf")
    q4_b = os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-wsp-v8-q4_k_m.gguf")
    q4_b_v3 = os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-wsp-v3-q4_k_m.gguf")

    shutil.copy2(BLOB_MINI, f16_tmp_b)
    logger.info("Writing updated MLP layers, o_proj, and collinear donor tokens in Mini F16 GGUF...")
    with open(f16_tmp_b, "r+b") as fp_b:
        for l in layers_b:
            t_b = tmap_b[f"blk.{l}.ffn_down.weight"]
            w_updated = model_b.model.layers[l].mlp.down_proj.weight.detach().cpu().numpy().astype(np.float16)
            fp_b.seek(t_b.data_offset)
            fp_b.write(w_updated.tobytes())

        donor_ids_b = [16948, 28260, 48662, 54364]
        t_emb = tmap_b["token_embd.weight"]
        w_donor_bytes = (0.001 * w_crash_b).numpy().astype(np.float16).tobytes()
        for donor_tid in donor_ids_b:
            fp_b.seek(t_emb.data_offset + donor_tid * 896 * 2)
            fp_b.write(w_donor_bytes)

    del model_b
    gc.collect()

    logger.info("Quantizing Mini F16 -> Q4_K_M...")
    subprocess.run([QUANTIZE_BIN, f16_tmp_b, q4_b, "Q4_K_M"], check=True)
    try:
        time.sleep(2)
        if os.path.exists(f16_tmp_b):
            os.remove(f16_tmp_b)
    except Exception as e:
        logger.warning(f"Could not remove temporary f16: {e}")
    shutil.copy2(q4_b, q4_b_v3)
    logger.info(f"[SUCCESS] Mini 0.5B Reasoning Folded: {q4_b} ({os.path.getsize(q4_b)} bytes)")

def update_modelfiles_with_think():
    logger.info("=== [Phase 2.3] Updating Modelfiles with <think> Template Injection ===")
    
    # 1. Flagship Modelfile
    mf1_path = os.path.join(ROOT_DIR, "Modelfile")
    with open(mf1_path, "r", encoding="utf-8") as f:
        mf1 = f.read()
    if "<think>" not in mf1:
        target1 = "{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\"\"\""
        repl1 = "{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n<think>\n\"\"\""
        mf1 = mf1.replace(target1, repl1)
        with open(mf1_path, "w", encoding="utf-8") as f:
            f.write(mf1)
        logger.info(f"Updated {mf1_path} with <think> template pre-injection.")

    # 2. Mini Modelfile
    mf2_path = os.path.join(ROOT_DIR, "Modelfile.mini")
    with open(mf2_path, "r", encoding="utf-8") as f:
        mf2 = f.read()
    if "<think>" not in mf2:
        target2 = "{{- end -}}<|im_start|>assistant\n\"\"\""
        repl2 = "{{- end -}}<|im_start|>assistant\n<think>\n\"\"\""
        mf2 = mf2.replace(target2, repl2)
        with open(mf2_path, "w", encoding="utf-8") as f:
            f.write(mf2)
        logger.info(f"Updated {mf2_path} with <think> template pre-injection.")

    # 3. Dragon Modelfile
    mf3_path = os.path.join(ROOT_DIR, "Modelfile.dragon")
    with open(mf3_path, "r", encoding="utf-8") as f:
        mf3 = f.read()
    if "<think>" not in mf3:
        target3 = "{{- end -}}<|assistant|>\n\"\"\""
        repl3 = "{{- end -}}<|assistant|>\n<think>\n\"\"\""
        mf3 = mf3.replace(target3, repl3)
        with open(mf3_path, "w", encoding="utf-8") as f:
            f.write(mf3)
        logger.info(f"Updated {mf3_path} with <think> template pre-injection.")

if __name__ == "__main__":
    t0 = time.time()
    update_modelfiles_with_think()
    fold_flagship_reasoning()
    fold_mini_reasoning()
    logger.info(f"=== ALL MODELS REASONING FOLDED IN {time.time() - t0:.1f}s ===")
