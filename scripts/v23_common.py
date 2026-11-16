#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared model, tokenizer, GGUF adapter, Procrustes alignment, and validation
utilities for Quillan-Ronin WSP v23 (Flagship 1B, Mini 0.5B, Dragon 1B).
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import gguf
import gguf.quants
import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import wsp_v8_closed_form_sota_compiler as v8  # noqa: E402

logger = logging.getLogger("V23_Common")

PYTHON_EXE = Path(sys.executable)
EXPORT_ROOT = Path(r"C:\02_QUILLAN\exports")
DONOR_CACHE_DIR = EXPORT_ROOT / "donor_cache_v23"
CLEAN_DATA_PATH = EXPORT_ROOT / "v23_clean_train_data.json"
PPL_BIN = Path(r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-perplexity.exe")
HELD_OUT_PPL_FILE = EXPORT_ROOT / "held_out_ppl_final_verify.txt"

BITNET_2B4T_DIR = Path(r"E:\source_models\bitnet_2b_4t")
BITNET_LARGE_HUB = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--1bitLLM--bitnet_b1_58-large"
)

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "flagship": {
        "display": "Flagship 1.2B (Llama-3.2)",
        "base_hf": Path(v8.FLAGSHIP_SNAP),
        "clean_gguf": EXPORT_ROOT / "clean-llama-1b-q8_0.gguf",
        "prev_gguf": EXPORT_ROOT / "quillan-1b-sovereign-wsp-v20-q8_0.gguf",
        "stageb_hf": EXPORT_ROOT / "v23_stageb_flagship_hf",
        "stageb_gguf": EXPORT_ROOT / "quillan-1b-sovereign-stageb-v23-q8_0.gguf",
        "stagec_hf": EXPORT_ROOT / "v23_stagec_flagship_hf",
        "stagec_gguf": EXPORT_ROOT / "quillan-1b-sovereign-wsp-v23-q8_0.gguf",
        "num_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "target_layers": [4, 7, 10, 12, 14, 15],
        "self_donor": "llama_1b",
    },
    "mini": {
        "display": "Mini 0.5B (Qwen2.5)",
        "base_hf": Path(v8.MINI_SNAP),
        "clean_gguf": EXPORT_ROOT / "clean-qwen-0.5b-q8_0.gguf",
        "prev_gguf": EXPORT_ROOT / "quillan-mini-0.5b-wsp-v15-q8_0.gguf",
        "stageb_hf": EXPORT_ROOT / "v23_stageb_mini_hf",
        "stageb_gguf": EXPORT_ROOT / "quillan-mini-0.5b-stageb-v23-q8_0.gguf",
        "stagec_hf": EXPORT_ROOT / "v23_stagec_mini_hf",
        "stagec_gguf": EXPORT_ROOT / "quillan-mini-0.5b-wsp-v23-q8_0.gguf",
        "num_layers": 24,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "target_layers": [6, 10, 14, 18, 21, 23],
        "self_donor": "qwen_0.5b",
    },
    "dragon": {
        "display": "Dragon 1.0B (Falcon3)",
        "base_gguf": Path(v8.BLOB_DRAGON),
        "clean_gguf": Path(v8.BLOB_DRAGON),
        "prev_gguf": EXPORT_ROOT / "quillan-dragon-1b-wsp-v18b-q8_0.gguf",
        "stageb_hf": EXPORT_ROOT / "v23_stageb_dragon_hf",
        "stageb_gguf": EXPORT_ROOT / "quillan-dragon-1b-stageb-v23-q8_0.gguf",
        "stagec_hf": EXPORT_ROOT / "v23_stagec_dragon_hf",
        "stagec_gguf": EXPORT_ROOT / "quillan-dragon-1b-wsp-v23-q8_0.gguf",
        "num_layers": 18,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "target_layers": [4, 7, 10, 13, 15, 17],
        "self_donor": "dragon_1b",
    },
}


def ensure_dragon_tokenizer_cache() -> Tuple[Path, Path]:
    """Extracts and caches Dragon GGUF vocab/merges once for sub-second reloads."""
    DONOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tok_cache = DONOR_CACHE_DIR / "dragon_tokens.json"
    mrg_cache = DONOR_CACHE_DIR / "dragon_merges.json"
    if tok_cache.exists() and mrg_cache.exists():
        return tok_cache, mrg_cache
    logger.info("Caching Dragon GGUF tokenizer from %s...", v8.BLOB_DRAGON)
    r = gguf.GGUFReader(str(v8.BLOB_DRAGON))
    toks = r.fields["tokenizer.ggml.tokens"].contents()
    merges = r.fields["tokenizer.ggml.merges"].contents()
    with open(tok_cache, "w", encoding="utf-8") as f:
        json.dump(toks, f)
    with open(mrg_cache, "w", encoding="utf-8") as f:
        json.dump(merges, f)
    return tok_cache, mrg_cache


def load_dragon_tokenizer() -> PreTrainedTokenizerFast:
    """Reconstructs Falcon3-1B BPE tokenizer matching BLOB_DRAGON token IDs."""
    tok_cache, mrg_cache = ensure_dragon_tokenizer_cache()
    with open(tok_cache, "r", encoding="utf-8") as f:
        toks: List[str] = json.load(f)
    with open(mrg_cache, "r", encoding="utf-8") as f:
        merges_raw: List[str] = json.load(f)
    vocab = {t: i for i, t in enumerate(toks)}
    merges = [tuple(m.split(" ")) for m in merges_raw]
    btok = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=False))
    btok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    btok.decoder = decoders.ByteLevel()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=btok,
        eos_token=toks[11],
        pad_token=toks[2023],
    )
    fast_tok.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '<|endoftext|>' }}"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
    )
    return fast_tok


def load_donor_vocab(donor_name: str) -> Dict[str, int]:
    """Loads token->id vocabulary mapping quickly for any cached donor."""
    if donor_name == "dragon_1b":
        tok_cache, _ = ensure_dragon_tokenizer_cache()
        with open(tok_cache, "r", encoding="utf-8") as f:
            toks = json.load(f)
        return {t: i for i, t in enumerate(toks)}
    if donor_name == "llama_1b":
        tok_file = Path(v8.FLAGSHIP_SNAP) / "tokenizer.json"
    elif donor_name == "qwen_0.5b":
        tok_file = Path(v8.MINI_SNAP) / "tokenizer.json"
    elif donor_name == "bitnet_2b4t":
        tok_file = BITNET_2B4T_DIR / "tokenizer.json"
    elif donor_name == "bitnet_large":
        matches = list(BITNET_LARGE_HUB.rglob("tokenizer.json"))
        if not matches:
            raise FileNotFoundError("BitNet-Large tokenizer.json not found in cache")
        tok_file = matches[0]
    else:
        raise ValueError(f"Unknown donor_name: {donor_name}")

    with open(tok_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    vocab: Dict[str, int] = dict(data.get("model", {}).get("vocab", {}))
    for item in data.get("added_tokens", []):
        content = item.get("content")
        tid = item.get("id")
        if isinstance(content, str) and isinstance(tid, int) and content not in vocab:
            vocab[content] = tid
    return vocab


def _gguf_to_hf_qk(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Converts llama.cpp interleaved RoPE Q/K weight rows into HuggingFace half-split layout."""
    in_dim = w.shape[1]
    return w.view(n_heads, head_dim // 2, 2, in_dim).transpose(1, 2).reshape(n_heads * head_dim, in_dim).contiguous()


def _hf_to_gguf_qk(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Converts HuggingFace half-split Q/K weight rows back into llama.cpp interleaved RoPE layout."""
    in_dim = w.shape[1]
    return w.view(n_heads, 2, head_dim // 2, in_dim).transpose(1, 2).reshape(n_heads * head_dim, in_dim).contiguous()


def load_clean_host(
    model_key: str, dtype: torch.dtype = torch.float32
) -> Tuple[torch.nn.Module, PreTrainedTokenizerFast]:
    """Loads untouched clean base HF model + tokenizer for flagship, mini, or dragon."""
    spec = MODEL_SPECS[model_key]
    if model_key in ("flagship", "mini"):
        hf_dir = str(spec["base_hf"])
        tok = AutoTokenizer.from_pretrained(hf_dir, local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            hf_dir,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model, tok

    # Dragon 1.0B (Falcon3): dequantize BLOB_DRAGON with RoPE un-interleave
    tok = load_dragon_tokenizer()
    r = gguf.GGUFReader(str(spec["base_gguf"]))
    tmap = {t.name: t for t in r.tensors}

    def dq(name: str) -> torch.Tensor:
        t = tmap[name]
        return torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).to(dtype)

    cfg = LlamaConfig(
        vocab_size=131072,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=4,
        rms_norm_eps=1e-6,
        rope_theta=1000042.0,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
        bos_token_id=11,
        eos_token_id=11,
        pad_token_id=2023,
    )
    sd: Dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": dq("token_embd.weight"),
        "lm_head.weight": dq("output.weight"),
        "model.norm.weight": dq("output_norm.weight"),
    }
    for l in range(18):
        sd[f"model.layers.{l}.input_layernorm.weight"] = dq(f"blk.{l}.attn_norm.weight")
        sd[f"model.layers.{l}.post_attention_layernorm.weight"] = dq(f"blk.{l}.ffn_norm.weight")
        sd[f"model.layers.{l}.self_attn.q_proj.weight"] = _gguf_to_hf_qk(dq(f"blk.{l}.attn_q.weight"), 8, 256)
        sd[f"model.layers.{l}.self_attn.k_proj.weight"] = _gguf_to_hf_qk(dq(f"blk.{l}.attn_k.weight"), 4, 256)
        sd[f"model.layers.{l}.self_attn.v_proj.weight"] = dq(f"blk.{l}.attn_v.weight")
        sd[f"model.layers.{l}.self_attn.o_proj.weight"] = dq(f"blk.{l}.attn_output.weight")
        sd[f"model.layers.{l}.mlp.gate_proj.weight"] = dq(f"blk.{l}.ffn_gate.weight")
        sd[f"model.layers.{l}.mlp.up_proj.weight"] = dq(f"blk.{l}.ffn_up.weight")
        sd[f"model.layers.{l}.mlp.down_proj.weight"] = dq(f"blk.{l}.ffn_down.weight")

    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)
    model.load_state_dict(sd, strict=True, assign=True)
    model.model.rotary_emb = LlamaRotaryEmbedding(cfg, device="cpu")
    model.eval()
    return model, tok


def audit_model_integrity(model: torch.nn.Module, expected_params: Optional[int] = None) -> Dict[str, Any]:
    """Asserts every parameter/buffer is finite and checks exact parameter count."""
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if not torch.isfinite(param.float()).all():
            raise ValueError(f"Non-finite weights detected in {name}")
    if expected_params is not None and total_params != expected_params:
        raise AssertionError(f"Parameter count mismatch: got {total_params}, expected {expected_params}")
    return {"total_params": total_params, "finite": True}


def save_hf_checkpoint(model: torch.nn.Module, tokenizer: Any, out_dir: Path) -> None:
    """Saves an HF checkpoint cleanly with safetensors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))


def convert_hf_dir_to_gguf(hf_dir: Path, out_gguf: Path, outtype: str = "q8_0") -> None:
    """Runs llama.cpp convert_hf_to_gguf.py in an isolated subprocess."""
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON_EXE),
        str(v8.CONVERT_SCRIPT),
        str(hf_dir),
        "--outfile",
        str(out_gguf),
        "--outtype",
        outtype,
    ]
    logger.info("Converting HF %s -> GGUF %s (%s)...", hf_dir.name, out_gguf.name, outtype)
    subprocess.run(cmd, check=True)


def copy_gguf_field(writer: gguf.GGUFWriter, key: str, field: Any) -> None:
    """Copies any GGUF KV metadata field faithfully, preserving exact scalar/array types."""
    val = field.contents()
    ftype = field.types[0] if field.types else None
    if ftype == gguf.GGUFValueType.ARRAY:
        writer.add_array(key, val)
    elif ftype == gguf.GGUFValueType.STRING:
        writer.add_string(key, str(val))
    elif ftype == gguf.GGUFValueType.BOOL:
        writer.add_bool(key, bool(val))
    elif ftype == gguf.GGUFValueType.UINT8:
        writer.add_uint8(key, int(val))
    elif ftype == gguf.GGUFValueType.INT8:
        writer.add_int8(key, int(val))
    elif ftype == gguf.GGUFValueType.UINT16:
        writer.add_uint16(key, int(val))
    elif ftype == gguf.GGUFValueType.INT16:
        writer.add_int16(key, int(val))
    elif ftype == gguf.GGUFValueType.UINT32:
        writer.add_uint32(key, int(val))
    elif ftype == gguf.GGUFValueType.INT32:
        writer.add_int32(key, int(val))
    elif ftype == gguf.GGUFValueType.FLOAT32:
        writer.add_float32(key, float(val))
    elif ftype == gguf.GGUFValueType.UINT64:
        writer.add_uint64(key, int(val))
    elif ftype == gguf.GGUFValueType.INT64:
        writer.add_int64(key, int(val))
    elif ftype == gguf.GGUFValueType.FLOAT64:
        writer.add_float64(key, float(val))


def export_dragon_gguf_from_hf(model: torch.nn.Module, out_gguf: Path) -> None:
    """
    Writes a patched Dragon Q8_0 GGUF preserving all 38 original BLOB_DRAGON metadata fields,
    Falcon3 BPE tokenizer, and tensor quantization types while re-quantizing HF weights
    (including hf_to_gguf_qk RoPE restoration on q_proj/k_proj).
    """
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    reader = gguf.GGUFReader(str(v8.BLOB_DRAGON))
    sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}

    # Build reverse mapping: GGUF tensor name -> modified float32 tensor
    hf_to_gguf_map: Dict[str, torch.Tensor] = {
        "token_embd.weight": sd["model.embed_tokens.weight"],
        "output.weight": sd["lm_head.weight"],
        "output_norm.weight": sd["model.norm.weight"],
    }
    for l in range(18):
        hf_to_gguf_map[f"blk.{l}.attn_norm.weight"] = sd[f"model.layers.{l}.input_layernorm.weight"]
        hf_to_gguf_map[f"blk.{l}.ffn_norm.weight"] = sd[f"model.layers.{l}.post_attention_layernorm.weight"]
        hf_to_gguf_map[f"blk.{l}.attn_q.weight"] = _hf_to_gguf_qk(
            sd[f"model.layers.{l}.self_attn.q_proj.weight"], 8, 256
        )
        hf_to_gguf_map[f"blk.{l}.attn_k.weight"] = _hf_to_gguf_qk(
            sd[f"model.layers.{l}.self_attn.k_proj.weight"], 4, 256
        )
        hf_to_gguf_map[f"blk.{l}.attn_v.weight"] = sd[f"model.layers.{l}.self_attn.v_proj.weight"]
        hf_to_gguf_map[f"blk.{l}.attn_output.weight"] = sd[f"model.layers.{l}.self_attn.o_proj.weight"]
        hf_to_gguf_map[f"blk.{l}.ffn_gate.weight"] = sd[f"model.layers.{l}.mlp.gate_proj.weight"]
        hf_to_gguf_map[f"blk.{l}.ffn_up.weight"] = sd[f"model.layers.{l}.mlp.up_proj.weight"]
        hf_to_gguf_map[f"blk.{l}.ffn_down.weight"] = sd[f"model.layers.{l}.mlp.down_proj.weight"]

    writer = gguf.GGUFWriter(str(out_gguf), arch="llama")
    skip_keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for key, field in reader.fields.items():
        if key in skip_keys:
            continue
        copy_gguf_field(writer, key, field)

    for t in reader.tensors:
        if t.name in hf_to_gguf_map:
            arr = np.ascontiguousarray(hf_to_gguf_map[t.name].numpy(), dtype=np.float32)
            if t.tensor_type == gguf.GGMLQuantizationType.F32:
                qdata = arr
            else:
                qdata = gguf.quants.quantize(arr, t.tensor_type)
            writer.add_tensor(t.name, qdata, raw_dtype=t.tensor_type)
        else:
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    del reader, sd, hf_to_gguf_map
    gc.collect()
    logger.info("Wrote patched Dragon GGUF -> %s", out_gguf)


def compute_shared_vocab_procrustes(
    host_vocab: Dict[str, int],
    donor_vocab: Dict[str, int],
    host_emb: torch.Tensor,
    donor_emb: torch.Tensor,
    max_pairs: int = 40000,
) -> Tuple[torch.Tensor, int]:
    """
    Computes isometric Orthogonal Procrustes alignment P_d2h (d_donor, d_host)
    from shared token pairs such that E_donor_shared @ P_d2h ~ E_host_shared.
    Handles SentencePiece '_' vs ByteLevel 'Ġ' normalization when needed.
    """
    h_norm: Dict[str, int] = {}
    for tok, tid in host_vocab.items():
        if tok not in h_norm:
            h_norm[tok] = tid

    pairs: List[Tuple[int, int]] = []
    # Exact token match first
    for tok, d_id in donor_vocab.items():
        if d_id >= donor_emb.shape[0]:
            continue
        h_id = h_norm.get(tok)
        if h_id is not None and h_id < host_emb.shape[0]:
            pairs.append((h_id, d_id))

    # Fallback/augment with SentencePiece '▁' <-> 'Ġ' normalization if < 10k pairs
    if len(pairs) < 10000:
        seen_h = {h for h, _ in pairs}
        h_sp = {k.replace("Ġ", "▁"): v for k, v in host_vocab.items() if v < host_emb.shape[0]}
        for tok, d_id in donor_vocab.items():
            if d_id >= donor_emb.shape[0]:
                continue
            h_id = h_sp.get(tok.replace("Ġ", "▁"))
            if h_id is not None and h_id not in seen_h:
                pairs.append((h_id, d_id))
                seen_h.add(h_id)

    if len(pairs) < 512:
        raise ValueError(f"Insufficient shared tokens for Procrustes: {len(pairs)}")

    if len(pairs) > max_pairs:
        step = len(pairs) / float(max_pairs)
        pairs = [pairs[int(i * step)] for i in range(max_pairs)]

    h_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    d_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    e_h = host_emb[h_idx].float()
    e_d = donor_emb[d_idx].float()
    # Center and normalize row-wise to emphasize directional geometry
    e_h = e_h - e_h.mean(dim=0, keepdim=True)
    e_d = e_d - e_d.mean(dim=0, keepdim=True)

    cross = e_d.T @ e_h  # (d_donor, d_host)
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    p_d2h = u @ vh  # (d_donor, d_host) partial isometry / orthogonal matrix
    return p_d2h, len(pairs)


@torch.no_grad()
def evaluate_texts_nll(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Iterable[str],
    max_len: int = 192,
    device: str = "cpu",
) -> float:
    """Computes token-weighted average cross-entropy NLL across a list of non-test texts."""
    model.eval()
    total_nll = 0.0
    total_tok = 0
    for txt in texts:
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc["input_ids"].to(device)
        if ids.shape[1] < 4:
            continue
        out = model(ids, labels=ids)
        n_pred = ids.shape[1] - 1
        total_nll += float(out.loss.item()) * n_pred
        total_tok += n_pred
    if total_tok == 0:
        raise ValueError("No valid validation tokens evaluated")
    return total_nll / float(total_tok)


def measure_gguf_ppl_cli(
    gguf_path: Path, ppl_file: Path = HELD_OUT_PPL_FILE, chunks: Optional[int] = None, ctx: int = 128
) -> float:
    """Runs CPU llama-perplexity.exe on gguf_path and parses Final estimate PPL."""
    cmd = [str(PPL_BIN), "-m", str(gguf_path), "-f", str(ppl_file), "-ngl", "0", "-c", str(ctx)]
    if chunks is not None:
        cmd.extend(["--chunks", str(chunks)])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in reversed(combined.splitlines()):
        if "Final estimate: PPL =" in line:
            val = line.split("Final estimate: PPL =", 1)[1].strip().split()[0]
            return float(val)
    raise RuntimeError(f"Failed to parse PPL from {gguf_path}:\n{combined[-1000:]}")
