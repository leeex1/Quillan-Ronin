#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[EXPORT] QUILLAN-RONIN v5.4.0-ONI: UNIFIED SAFETENSORS & GGUF EXPORTER
======================================================================
Exports trained checkpoints (quillan_oni_latest.pt) into:
1. Hugging Face SafeTensors (`model.safetensors` + `config.json`)
2. GGUF format (`quillan_oni_f16.gguf` / `quillan_oni_q4_k_m.gguf`) for Ollama/llama.cpp
"""

import os
import shutil
import sys
import json
import torch
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# SafeTensors export
try:
    from safetensors.torch import save_file as save_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# GGUF export
try:
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[3]
ONI_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "09 - Projects"
    / "projects"
    / "05_Training"
    / "checkpoints"
    / "checkpoints_oni"
    / "quillan_oni_latest.pt"
)
DEFAULT_EXPORT_DIR = REPO_ROOT / "exports" / "v5_4_oni"

# These files are copied into the Hub package so trust_remote_code has a
# complete, repository-relative import surface.  The checkpoint remains a
# separate artifact and is only emitted when it exists locally.
HF_RUNTIME_FILES = (
    "configuration_quillan_oni.py",
    "modeling_quillan_oni.py",
    "quillan_v5_4_oni.py",
    "quillan_tokenizer_unified.py",
    "tokenizer.json",
    "evo_moe.py",
    "mamba_block.py",
    "flash_attn_wrapper.py",
    "swarm_real.py",
    "world_model_oni.py",
    "speculative_decode.py",
    "nitro_pocket.py",
    "es_at_scale.py",
    "protrian_memo.py",
    "wiki_skill.py",
    "reasoning_engine_oni.py",
)


def _cfg_as_dict(cfg):
    """Return checkpoint configuration as a JSON-safe dictionary."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        values = dict(cfg)
    elif hasattr(cfg, "__dataclass_fields__"):
        from dataclasses import asdict

        values = asdict(cfg)
    else:
        values = {
            key: value
            for key, value in vars(cfg).items()
            if not key.startswith("_")
        }
    return {
        key: value
        for key, value in values.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _write_hf_package(out_dir: Path, cfg: dict) -> None:
    """Write custom-code, tokenizer, and config files for HF loading."""
    runtime_files = set(HF_RUNTIME_FILES)
    runtime_files.update(path.name for path in ONI_DIR.glob("paper_*.py"))
    for filename in sorted(runtime_files):
        source = ONI_DIR / filename
        if source.is_file():
            shutil.copy2(source, out_dir / filename)

    max_seq_len = int(cfg.get("max_seq_len", 512))
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_input_names": ["input_ids", "attention_mask"],
        "model_max_length": max_seq_len,
        "bos_token": "<|bos|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "clean_up_tokenization_spaces": False,
    }
    with (out_dir / "tokenizer_config.json").open("w", encoding="utf-8") as handle:
        json.dump(tokenizer_config, handle, indent=2)
    with (out_dir / "special_tokens_map.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "bos_token": "<|bos|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
            },
            handle,
            indent=2,
        )


def export_to_safetensors(checkpoint_path: str, output_dir: str):
    print(f"\n[1/2] Exporting to SafeTensors...")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    cfg = _cfg_as_dict(ckpt.get("cfg", {}))
    
    # Save config.json
    config_data = {
        "architectures": ["QuillanOniForCausalLM"],
        "model_type": "quillan_ronin_oni",
        "auto_map": {
            "AutoConfig": "configuration_quillan_oni.QuillanOniConfig",
            "AutoModelForCausalLM": "modeling_quillan_oni.QuillanOniForCausalLM",
        },
        "version": "5.4.0-oni",
        "hidden_size": cfg.get("hidden_dim", 1024),
        "intermediate_size": cfg.get("ffn_dim", 2048),
        "num_hidden_layers": cfg.get("n_layer", 12),
        "num_attention_heads": cfg.get("n_head", 16),
        "vocab_size": cfg.get("vocab_size", 50257),
        "max_position_embeddings": cfg.get("max_seq_len", 512),
        "eos_token_id": cfg.get("eos_token_id", 0),
        "bos_token_id": cfg.get("bos_token_id", 2),
        "pad_token_id": cfg.get("pad_token_id", 1),
        "num_experts": cfg.get("num_experts", 34),
        "router_mode": cfg.get("router_mode", "dense_pull"),
        "expert_rank": cfg.get("expert_rank", 8),
        "swarm_rank": cfg.get("swarm_rank", 8),
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "transformers_version": "4.45.0"
    }
    config_data.update(cfg)
    
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"  Saved config.json to {out_dir / 'config.json'}")
    
    # Ensure clone/contiguous tensors to handle shared pointers cleanly
    clean_sd = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            # modeling_quillan_oni.QuillanOniForCausalLM wraps the canonical
            # model under ``model`` so Transformers can own the config/generation
            # contract without changing the native Quillan implementation.
            hf_key = k if k.startswith("model.") else f"model.{k}"
            clean_sd[hf_key] = v.detach().clone().contiguous()
            
    if SAFETENSORS_AVAILABLE:
        st_path = out_dir / "model.safetensors"
        save_safetensors(clean_sd, str(st_path))
        print(f"  Successfully saved {len(clean_sd)} tensors to {st_path} ({st_path.stat().st_size / 1e6:.1f} MB)")
    else:
        pt_path = out_dir / "pytorch_model.bin"
        torch.save(clean_sd, str(pt_path))
        print(f"  SafeTensors library not found. Saved fallback PyTorch bin to {pt_path}")

    _write_hf_package(out_dir, cfg)
    print(f"  Wrote HF custom-code package to {out_dir}")


def export_to_gguf(checkpoint_path: str, output_file: str):
    print(f"\n[2/2] Exporting to GGUF format...")
    if not GGUF_AVAILABLE:
        print("  gguf package not installed. Skipping GGUF export (install with `pip install gguf`).")
        return
        
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    cfg = _cfg_as_dict(ckpt.get("cfg", {}))
    
    # NOTE: arch must be "llama" for Ollama/llama.cpp compatibility.
    # llama-quantize validates against a known-arch allowlist; "quillan" causes exit 1.
    # We store the real architecture identity in general.source_arch metadata.
    writer = GGUFWriter(str(out_path), "llama")
    
    # Metadata
    writer.add_name("Quillan-Ronin-v5.4.0-ONI")
    writer.add_string("general.source_arch", "quillan_ronin_v540_oni")
    writer.add_context_length(cfg.get("max_seq_len", 2048))
    writer.add_embedding_length(cfg.get("hidden_dim", 1024))
    writer.add_block_count(cfg.get("n_layer", 12))
    writer.add_head_count(cfg.get("n_head", 16))
    writer.add_expert_count(cfg.get("num_experts", 34))
    
    tensor_count = 0
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        t_data = tensor.detach().float().numpy()
        writer.add_tensor(name, t_data)
        tensor_count += 1
        
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"  Successfully saved {tensor_count} tensors to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main():
    ckpt_path = DEFAULT_CHECKPOINT
    if not ckpt_path.exists():
        legacy_candidates = (
            ONI_DIR / "checkpoint_step_500.pt",
            REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_oni_latest.pt",
        )
        ckpt_path = next((candidate for candidate in legacy_candidates if candidate.exists()), ckpt_path)

    export_dir = DEFAULT_EXPORT_DIR
    gguf_out = export_dir / "quillan_oni_f16.gguf"
    
    print("=" * 75)
    print("[EXPORT] QUILLAN-RONIN v5.4.0-ONI EXPORT PIPELINE")
    print(f"Source Checkpoint: {ckpt_path}")
    print("=" * 75)
    
    if not ckpt_path.exists():
        raise FileNotFoundError(
            "No ONI checkpoint found. Train or place one at "
            f"{DEFAULT_CHECKPOINT} before exporting model weights."
        )

    export_to_safetensors(str(ckpt_path), str(export_dir))
    export_to_gguf(str(ckpt_path), str(gguf_out))
    
    print("\n" + "=" * 75)
    print("[SUCCESS] EXPORT PIPELINE COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    main()
