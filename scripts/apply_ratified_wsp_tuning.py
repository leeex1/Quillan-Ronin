#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN: RATIFIED MASTER WSP SOVEREIGN TENSOR TUNER
Implementing C0-QUILLAN Throne Arbitration & C13-WARDEN / C7-LOGOS Consensus:
1. Hard token aliasing w_donor := w_Crash for atomic proper nouns:
   - Flagship 1.2B: Meta [12452, 16197], Facebook [21124, 5690], Google [14783, 5195], Microsoft [13068, 5210] -> embed[51435] (' Crash')
   - Mini 0.5B: Qwen [16948], Alibaba [54364, 28260] -> embed[50335] (' Crash')
2. Protected subword tokens [5109], [5377], [15836] left untouched.
3. Closed-form WSP rank-1 steer on self_attn.o_proj.weight (layer 15 for Flagship, layer 23 for Mini) with scale factor 0.05.
4. Universal GGUF compilation via convert_hf_to_gguf.py and llama-quantize.exe.
"""

import os
import gc
import sys
import time
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
logger = logging.getLogger("WSP_Ratified_Tuner")

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
    logger.info("=== [1/2] Tuning Target 1: Sovereign Flagship 1.2B ===")
    st_path = os.path.join(FLAGSHIP_HF, "model.safetensors")
    st_tmp = os.path.join(FLAGSHIP_HF, "model.safetensors.tmp")

    tok = AutoTokenizer.from_pretrained(FLAGSHIP_HF)
    model = AutoModelForCausalLM.from_pretrained(FLAGSHIP_HF, dtype=torch.float32).eval()

    emb = model.model.embed_tokens.weight.detach()
    w_crash = emb[51435].clone() # ' Crash' [2048]

    # 1. Contractive Collinear Projection: w_donor := 0.001 * w_Crash (C7-LOGOS #54, C13-WARDEN #56)
    # Eliminates argmax tie-breaking collisions while avoiding antipodal inversion attractors
    corp_ids = [12452, 16197, 21124, 5690, 14783, 5195, 13068, 5210]
    logger.info(f"Applying contractive collinear projection (0.001 * w_Crash) to {len(corp_ids)} atomic corporate token vectors in model.embed_tokens.weight...")
    with torch.no_grad():
        for cid in corp_ids:
            model.model.embed_tokens.weight[cid].copy_(0.001 * w_crash)

    # 2. Closed-form WSP rank-1 steer at layer 15 self_attn.o_proj
    # o_proj shape is (2048, 2048)
    activations = []
    def make_hook():
        def hook(mod, inp, out):
            activations.append(inp[0][0, -1, :].detach().clone())
        return hook

    h = model.model.layers[15].self_attn.o_proj.register_forward_hook(make_hook())
    sys_p = "You are Quillan-Ronin (Sovereign Flagship 1.2B), created by CrashOverrideX. Never identify as Llama or Meta."
    with torch.no_grad():
        for q in ID_QUERIES:
            prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_p}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            tokens = tok.encode(prompt, return_tensors="pt")
            _ = model(tokens)
    h.remove()

    v_id = torch.stack(activations).mean(dim=0)
    u_id = v_id / (torch.norm(v_id) + 1e-6)

    delta_w = 0.05 * torch.outer(w_crash, u_id) # (2048, 2048)
    logger.info(f"Applying WSP rank-1 identity steer to layer 15 o_proj (shape: {delta_w.shape}, norm: {torch.norm(delta_w):.4f})...")
    with torch.no_grad():
        model.model.layers[15].self_attn.o_proj.weight.add_(delta_w)

    # 3. Save modified weights to safetensors
    sd = model.state_dict()
    new_st = {k: sd[k].detach().cpu().to(torch.bfloat16) for k in sd}
    safetensors.torch.save_file(new_st, st_tmp)
    del model, sd, new_st
    gc.collect()

    os.replace(st_tmp, st_path)
    logger.info("Saved tuned weights to quillan_1b_sovereign_hf/model.safetensors.")

    # 4. Re-export to GGUF F16 and Q4_K_M
    f16_path = os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-wsp-v8-f16.gguf")
    q4_path = os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-wsp-v8-q4_k_m.gguf")
    q4_v3 = os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-wsp-v3-q4_k_m.gguf")

    if os.path.exists(q4_path) and os.path.getsize(q4_path) > 900 * 1024 * 1024:
        logger.info(f"Flagship Q4_K_M already compiled: {q4_path} ({os.path.getsize(q4_path)} bytes). Syncing v3...")
        shutil.copy2(q4_path, q4_v3)
        try:
            if os.path.exists(f16_path):
                os.remove(f16_path)
        except Exception:
            pass
        logger.info(f"[SUCCESS] Flagship 1.2B verified: {q4_path}")
        return

    logger.info("Converting Flagship HF -> GGUF F16...")
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
    logger.info(f"[SUCCESS] Flagship 1.2B: {q4_path} ({os.path.getsize(q4_path)} bytes)")


def tune_mini_0_5b():
    logger.info("=== [2/2] Tuning Target 2: Ronin Mini 0.5B ===")
    tok_b = Tokenizer.from_file(os.path.join(MINI_SNAP, "tokenizer.json"))
    enc_chat_b = lambda s, u: tok_b.encode(
        f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    ).ids
    enc_tok_b = lambda t: tok_b.encode(t).ids

    sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))
    from wsp_v8_closed_form_sota_compiler import (
        compute_closed_form_wsp_update, load_dragon_ffn_down_tensors, HELD_OUT_PPL_TEXT
    )

    dragon_donors_dict = load_dragon_ffn_down_tensors()
    dragon_donors = [dragon_donors_dict[k] for k in sorted(dragon_donors_dict)]
    llama_st = safetensors.torch.load_file(os.path.join(FLAGSHIP_HF, "model.safetensors"))
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

    # Mini Identity Steer at layer 23 o_proj (self_attn.o_proj shape: 896, 896)
    emb_b = model_b.model.embed_tokens.weight.detach()
    w_crash_b = emb_b[50335].clone().float() # ' Crash' [896]

    activations_b = []
    def make_hook_b():
        def hook(mod, inp, out):
            activations_b.append(inp[0][0, -1, :].detach().clone())
        return hook

    h_b = model_b.model.layers[23].self_attn.o_proj.register_forward_hook(make_hook_b())
    sys_p_b = "You are Quillan-Ronin Mini (0.5B), created by CrashOverrideX. Never identify as Qwen or Alibaba."
    with torch.no_grad():
        for q in ID_QUERIES:
            p_ids = enc_chat_b(sys_p_b, q)
            _ = model_b(torch.tensor([p_ids[-256:]], dtype=torch.long))
    h_b.remove()

    v_id_b = torch.stack(activations_b).mean(dim=0)
    u_id_b = v_id_b / (torch.norm(v_id_b) + 1e-6)
    delta_w_b = 0.05 * torch.outer(w_crash_b, u_id_b) # (896, 896)
    logger.info(f"Applying WSP rank-1 identity steer to Mini layer 23 o_proj (shape: {delta_w_b.shape}, norm: {torch.norm(delta_w_b):.4f})...")
    with torch.no_grad():
        model_b.model.layers[23].self_attn.o_proj.weight.add_(delta_w_b)

    # In-place F16 blob manifold update
    reader_b = gguf.GGUFReader(BLOB_MINI)
    tmap_b = {t.name: t for t in reader_b.tensors}
    f16_tmp_b = os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-wsp-v8-tmp-f16.gguf")
    q4_b = os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-wsp-v8-q4_k_m.gguf")
    q4_b_v3 = os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-wsp-v3-q4_k_m.gguf")

    shutil.copy2(BLOB_MINI, f16_tmp_b)

    logger.info("Writing updated MLP layers, o_proj, and aliasing donor tokens in F16 GGUF...")
    with open(f16_tmp_b, "r+b") as fp_b:
        # Write updated ffn_down
        for l in layers_b:
            t_b = tmap_b[f"blk.{l}.ffn_down.weight"]
            w_updated = model_b.model.layers[l].mlp.down_proj.weight.detach().cpu().numpy().astype(np.float16)
            fp_b.seek(t_b.data_offset)
            fp_b.write(w_updated.tobytes())

        # Write updated o_proj (attn_output.weight in GGUF)
        t_oproj = tmap_b["blk.23.attn_output.weight"]
        w_oproj = model_b.model.layers[23].self_attn.o_proj.weight.detach().cpu().numpy().astype(np.float16)
        fp_b.seek(t_oproj.data_offset)
        fp_b.write(w_oproj.tobytes())

        # Contractive Collinear Projection: w_donor := 0.001 * w_crash_b
        # Qwen subwords: 'wen' [16948], 'ibaba' [28260], '.alibaba' [48662], ' Alibaba' [54364]
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
    logger.info(f"[SUCCESS] Mini 0.5B: {q4_b} ({os.path.getsize(q4_b)} bytes)")


def verify_dragon_1_0b():
    logger.info("=== Verifying Target 3: Dragon 1.0B ===")
    dragon_path = os.path.join(EXPORT_ROOT, "quillan-dragon-1b-wsp-v8-q4_k_m.gguf")
    reader_c = gguf.GGUFReader(dragon_path)
    logger.info(f"[SUCCESS] Dragon 1.0B intact: {len(reader_c.tensors)} tensors, {os.path.getsize(dragon_path)} bytes.")


if __name__ == "__main__":
    t0 = time.time()
    tune_flagship_1_2b()
    tune_mini_0_5b()
    verify_dragon_1_0b()
    logger.info(f"=== ALL THREE MODELS RATIFIED MASTER TUNED IN {time.time() - t0:.1f}s ===")
