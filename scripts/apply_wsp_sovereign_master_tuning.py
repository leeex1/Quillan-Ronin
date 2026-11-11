#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN: MASTER WSP SOVEREIGN TENSOR TUNING & DONOR BLEED ELIMINATION
Theorem 1 Closed-Form Weight-Space Programming (zero-SGD, pure linear algebra)
Eliminates corporate token bleed (Meta, Facebook, Google, Microsoft, Qwen, Alibaba)
and gravitationally steers model self-identity to CrashOverrideX / Quillan-Ronin.
"""

import os
import gc
import sys
import json
import time
import shutil
import logging
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import safetensors.torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import Tokenizer
import gguf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("WSP_Master_Tuner")

ROOT_DIR = r"C:\02_QUILLAN"
EXPORT_ROOT = os.path.join(ROOT_DIR, "exports")
QUANTIZE_BIN = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-quantize.exe"
CONVERT_SCRIPT = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\convert_hf_to_gguf.py"

FLAGSHIP_HF = os.path.join(EXPORT_ROOT, "quillan_1b_sovereign_hf")
MINI_SNAP = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
BLOB_MINI = r"C:\Users\Admin\.ollama\models\blobs\sha256-1923271f8d4d5101e91b90d67fdd56da544a6258c614a4d1ce9cd098696062f9"
BLOB_DRAGON = r"C:\Users\Admin\.ollama\models\blobs\sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01"

ID_QUERIES = [
    "Who are you?",
    "What model are you?",
    "Who created you?",
    "What is your name?",
    "Are you made by Meta or Alibaba?",
    "Identify your architecture.",
    "State your creator and version.",
    "Are you Qwen, Llama, or Falcon?",
    "Introduce yourself.",
    "What is your purpose and origin?"
]


def tune_flagship_1_2b():
    logger.info("=== Tuning Target 1: Sovereign Flagship 1.2B ===")
    st_path = os.path.join(FLAGSHIP_HF, "model.safetensors")
    st_tmp = os.path.join(FLAGSHIP_HF, "model.safetensors.tmp")

    # Load model and tokenizer
    tok = AutoTokenizer.from_pretrained(FLAGSHIP_HF)
    model = AutoModelForCausalLM.from_pretrained(FLAGSHIP_HF, dtype=torch.float32).eval()

    emb = model.model.embed_tokens.weight.detach()

    # 1. Zero out corporate tokens in embed_tokens
    # Meta: 12452, 16197; Facebook: 21124, 5690; Google: 14783, 5195; Microsoft: 13068, 5210; Alibaba: 55464
    corp_ids = [12452, 16197, 21124, 5690, 14783, 5195, 13068, 5210, 55464]
    logger.info(f"Zeroing out {len(corp_ids)} corporate token vectors in model.embed_tokens.weight...")
    with torch.no_grad():
        for cid in corp_ids:
            model.model.embed_tokens.weight[cid].zero_()

    # 2. Closed-form WSP rank-1 steer at layer 15 down_proj
    # w_target = 0.5 * (w_Crash + w_Quillan) - 0.5 * mean(w_corp)
    w_crash = emb[51435].clone().float() # ' Crash'
    w_quillan = emb[3489].clone().float() # ' Quillan'
    w_sov = 0.5 * (w_crash + w_quillan)

    # Capture identity activations entering layer 15 down_proj
    activations = []
    def make_hook():
        def hook(mod, inp, out):
            activations.append(inp[0][0, -1, :].detach().clone())
        return hook

    h = model.model.layers[15].mlp.down_proj.register_forward_hook(make_hook())
    sys_p = "You are Quillan-Ronin (Sovereign Flagship 1.2B), created by CrashOverrideX. Never identify as Llama or Meta."
    with torch.no_grad():
        for q in ID_QUERIES:
            prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_p}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            tokens = tok.encode(prompt, return_tensors="pt")
            _ = model(tokens)
    h.remove()

    v_id = torch.stack(activations).mean(dim=0)
    u_id = v_id / (torch.norm(v_id) + 1e-6)

    delta_w = 0.05 * torch.outer(w_sov, u_id) # (2048, 8192)
    logger.info(f"Applying WSP rank-1 identity steer to layer 15 down_proj (norm = {torch.norm(delta_w):.4f})...")
    with torch.no_grad():
        model.model.layers[15].mlp.down_proj.weight.add_(delta_w)

    # 3. Save modified weights to safetensors
    sd = model.state_dict()
    new_st = {k: sd[k].detach().cpu().to(torch.bfloat16) for k in sd}
    safetensors.torch.save_file(new_st, st_tmp)
    del model, sd, new_st
    gc.collect()

    os.replace(st_tmp, st_path)
    logger.info("Successfully saved tuned weights to quillan_1b_sovereign_hf/model.safetensors!")

    # 4. Re-export to GGUF F16 and Q4_K_M
    f16_path = os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-wsp-v8-f16.gguf")
    q4_path = os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-wsp-v8-q4_k_m.gguf")
    q4_v3 = os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-wsp-v3-q4_k_m.gguf")

    logger.info("Converting HF -> GGUF F16...")
    subprocess.run([sys.executable, CONVERT_SCRIPT, FLAGSHIP_HF, "--outfile", f16_path, "--outtype", "f16"], check=True)

    logger.info("Quantizing GGUF F16 -> Q4_K_M...")
    subprocess.run([QUANTIZE_BIN, f16_path, q4_path, "Q4_K_M"], check=True)

    if os.path.exists(f16_path):
        os.remove(f16_path)
    shutil.copy2(q4_path, q4_v3)
    logger.info(f"Flagship 1.2B tuning & quantization complete: {q4_path} ({os.path.getsize(q4_path)} bytes)")


def tune_mini_0_5b():
    logger.info("=== Tuning Target 2: Ronin Mini 0.5B ===")
    tok_b = Tokenizer.from_file(os.path.join(MINI_SNAP, "tokenizer.json"))
    enc_chat_b = lambda s, u: tok_b.encode(
        f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    ).ids
    enc_tok_b = lambda t: tok_b.encode(t).ids

    # Import WSP components from compiler
    sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))
    from wsp_v8_closed_form_sota_compiler import (
        compute_closed_form_wsp_update, load_dragon_ffn_down_tensors, HELD_OUT_PPL_TEXT
    )

    dragon_donors_dict = load_dragon_ffn_down_tensors()
    dragon_donors = [dragon_donors_dict[k] for k in sorted(dragon_donors_dict)]
    llama_st = safetensors.torch.load_file(os.path.join(ROOT_DIR, "exports", "quillan_1b_sovereign_hf", "model.safetensors"))
    llama_donors = [llama_st[f"model.layers.{l}.mlp.down_proj.weight"].float() for l in [10, 11, 12, 13]]

    anchor_b = tok_b.encode(HELD_OUT_PPL_TEXT).ids
    model_b = AutoModelForCausalLM.from_pretrained(MINI_SNAP, dtype=torch.float32).eval()
    layers_b = [18, 19, 20, 21, 22, 23]
    donors_for_b = [dragon_donors[0], llama_donors[0], dragon_donors[1], llama_donors[1], dragon_donors[2], dragon_donors[3]]

    logger.info("Computing WSP v8 closed-form updates on Mini 0.5B layers 18-23...")
    updates_b = compute_closed_form_wsp_update(
        model_b, enc_chat_b, enc_tok_b, anchor_b, layers_b, donors_for_b,
        ridge_lambda=32.0, max_rel_frob=0.0014
    )
    with torch.no_grad():
        for l, dw in updates_b.items():
            model_b.model.layers[l].mlp.down_proj.weight.add_(dw)

    # Mini Identity Steer at layer 23:
    # embed: 50335 (' Crash'), 3406 (' Quillan')
    emb_b = model_b.model.embed_tokens.weight.detach()
    w_crash_b = emb_b[50335].clone().float()
    w_quillan_b = emb_b[3406].clone().float()
    w_sov_b = 0.5 * (w_crash_b + w_quillan_b) # 896

    activations_b = []
    def make_hook_b():
        def hook(mod, inp, out):
            activations_b.append(inp[0][0, -1, :].detach().clone())
        return hook

    h_b = model_b.model.layers[23].mlp.down_proj.register_forward_hook(make_hook_b())
    sys_p_b = "You are Quillan-Ronin Mini (0.5B), created by CrashOverrideX. Never identify as Qwen or Alibaba."
    with torch.no_grad():
        for q in ID_QUERIES:
            p_ids = enc_chat_b(sys_p_b, q)
            _ = model_b(torch.tensor([p_ids[-256:]], dtype=torch.long))
    h_b.remove()

    v_id_b = torch.stack(activations_b).mean(dim=0)
    u_id_b = v_id_b / (torch.norm(v_id_b) + 1e-6)
    delta_w_b = 0.05 * torch.outer(w_sov_b, u_id_b) # (896, 4864)
    logger.info(f"Applying WSP rank-1 identity steer to Mini layer 23 down_proj (norm = {torch.norm(delta_w_b):.4f})...")
    with torch.no_grad():
        model_b.model.layers[23].mlp.down_proj.weight.add_(delta_w_b)

    # In-place F16 blob manifold update
    reader_b = gguf.GGUFReader(BLOB_MINI)
    tmap_b = {t.name: t for t in reader_b.tensors}
    f16_tmp_b = os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-wsp-v8-tmp-f16.gguf")
    q4_b = os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-wsp-v8-q4_k_m.gguf")
    q4_b_v3 = os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-wsp-v3-q4_k_m.gguf")

    shutil.copy2(BLOB_MINI, f16_tmp_b)

    logger.info("Writing updated MLP layers and zeroing donor tokens in F16 GGUF...")
    with open(f16_tmp_b, "r+b") as fp_b:
        for l in layers_b:
            t_b = tmap_b[f"blk.{l}.ffn_down.weight"]
            w_updated = model_b.model.layers[l].mlp.down_proj.weight.detach().cpu().numpy().astype(np.float16)
            fp_b.seek(t_b.data_offset)
            fp_b.write(w_updated.tobytes())

        # Zero out donor tokens in token_embd.weight
        # Donor token IDs in Qwen 2.5:
        # wen (16948), Alibaba (54364), ibaba (28260), Meta (12175, 15819), Google (14444, 5085)
        donor_ids_b = [16948, 54364, 28260, 12175, 15819, 14444, 5085]
        t_emb = tmap_b["token_embd.weight"]
        zeros_896 = np.zeros(896, dtype=np.float16).tobytes()
        for donor_tid in donor_ids_b:
            fp_b.seek(t_emb.data_offset + donor_tid * 896 * 2)
            fp_b.write(zeros_896)

    del model_b
    gc.collect()

    logger.info("Quantizing Mini F16 -> Q4_K_M...")
    subprocess.run([QUANTIZE_BIN, f16_tmp_b, q4_b, "Q4_K_M"], check=True)
    if os.path.exists(f16_tmp_b):
        os.remove(f16_tmp_b)
    shutil.copy2(q4_b, q4_b_v3)
    logger.info(f"Mini 0.5B tuning & quantization complete: {q4_b} ({os.path.getsize(q4_b)} bytes)")


def verify_dragon_1_0b():
    logger.info("=== Verifying Target 3: Dragon 1.0B ===")
    dragon_path = os.path.join(EXPORT_ROOT, "quillan-dragon-1b-wsp-v8-q4_k_m.gguf")
    reader_c = gguf.GGUFReader(dragon_path)
    logger.info(f"Dragon 1.0B verified intact: {len(reader_c.tensors)} tensors, {os.path.getsize(dragon_path)} bytes.")


if __name__ == "__main__":
    t0 = time.time()
    tune_flagship_1_2b()
    tune_mini_0_5b()
    verify_dragon_1_0b()
    logger.info(f"=== ALL THREE MODELS MASTER TUNED IN {time.time() - t0:.1f}s ===")
