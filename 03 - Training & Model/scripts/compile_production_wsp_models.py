import os
import sys
import time
import math
import logging
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import gguf, gguf.quants
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProductionWSPCompiler")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

def to_bitnet_ternary_wsp(W: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Canonical BitNet 1.58b ternary quantization: {-1, 0, 1}."""
    W_f = W.float().cpu()
    scale = 1.0 / W_f.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    W_scaled = W_f * scale
    W_q = torch.round(torch.clamp(W_scaled, -1.0, 1.0))
    return W_q / scale

def project_dim(W: torch.Tensor, target_out: int, target_in: int) -> torch.Tensor:
    W_f = W.float().cpu()
    r, c = W_f.shape
    if r == target_out and c == target_in: return W_f
    if r >= target_out: W_f = W_f[:target_out, :]
    else:
        pad = target_out - r
        W_f = torch.cat([W_f, W_f.repeat((pad // r) + 1, 1)[:pad, :]], dim=0)
    if c >= target_in: W_f = W_f[:, :target_in]
    else:
        pad = target_in - c
        W_f = torch.cat([W_f, W_f.repeat(1, (pad // c) + 1)[:, :pad]], dim=1)
    return W_f

def extract_svd_experts(W: torch.Tensor, n_exp: int, target_dim: int = 2048, rank: int = 16) -> list:
    """Extracts n_exp low-rank experts and quantizes them to BitNet ternary."""
    W_f = W.float().cpu()
    U, S, V = torch.svd_lowrank(W_f, q=max(32, n_exp * 2))
    experts = []
    for e in range(n_exp):
        offset = (e * 2) % max(1, (U.size(1) - rank))
        u_sub = U[:, offset:offset+rank]
        s_sub = S[offset:offset+rank]
        v_sub = V[:, offset:offset+rank]
        w_A = project_dim(u_sub, target_dim, rank)
        w_B = project_dim(v_sub.t()[:rank, :], rank, target_dim)
        w_A_tern = to_bitnet_ternary_wsp(w_A)
        w_B_tern = to_bitnet_ternary_wsp(w_B)
        experts.append((w_A_tern.to(torch.bfloat16), w_B_tern.to(torch.bfloat16)))
    return experts

def compile_sovereign_model(n_layers: int, model_name: str, out_filename: str):
    logger.info("=" * 80)
    logger.info(f"COMPILING {model_name.upper()} (Layers={n_layers}, D=2048, Experts=34 BitNet Ternary)")
    logger.info("=" * 80)
    t0 = time.time()

    hf_cache = r"C:\Users\Admin\.cache\huggingface\hub"
    ollama_blobs = r"C:\Users\Admin\.ollama\models\blobs"

    # 1. Load Donors
    logger.info(">>> [1/5] Ingesting Quad Donors (Llama 3.2, Falcon 3, BitNet 1.58b, Qwen 3.5)...")
    llama_dir = os.path.join(hf_cache, "models--unsloth--Llama-3.2-1B-Instruct", "snapshots")
    llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
    st_llama = load_file(os.path.join(llama_snap, "model.safetensors"))

    falcon_blob = os.path.join(ollama_blobs, "sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01")
    r_falcon = gguf.GGUFReader(falcon_blob)
    def dequant_gguf(t):
        if t.tensor_type != 0: arr = gguf.quants.dequantize(t.data, t.tensor_type)
        else: arr = t.data
        return torch.from_numpy(arr.copy()).float()
    falcon_tensors = {t.name: dequant_gguf(t) for t in r_falcon.tensors if 'blk.0.ffn_down' in t.name}
    del r_falcon

    bitnet_dir = os.path.join(hf_cache, "models--1bitLLM--bitnet_b1_58-large", "snapshots")
    bitnet_snap = os.path.join(bitnet_dir, os.listdir(bitnet_dir)[0])
    st_bitnet = load_file(os.path.join(bitnet_snap, "model.safetensors"))

    qwen35_blob = os.path.join(ollama_blobs, "sha256-e8bf47d40c5ba68baf8f3e503ee36710da6e8641cb0a670a508cff5b3f0cfbf7")
    r_qwen35 = gguf.GGUFReader(qwen35_blob)
    qwen35_tensors = {t.name: dequant_gguf(t) for t in r_qwen35.tensors if 'blk.0.ffn_down' in t.name}
    del r_qwen35

    # 2. Extract Balanced 34 MoE Council Experts (25% Each)
    logger.info(">>> [2/5] Synthesizing 34 Balanced Council MoE Experts in BitNet Ternary (9L / 9F / 8B / 8Q)...")
    w_down_llama = st_llama["model.layers.0.mlp.down_proj.weight"].float()
    w_down_falcon = falcon_tensors["blk.0.ffn_down.weight"].float()
    w_down_bitnet = project_dim(st_bitnet["model.layers.0.mlp.down_proj.weight"].float(), 2048, 8192)
    w_down_qwen = project_dim(qwen35_tensors["blk.0.ffn_down.weight"].float(), 2048, 8192)

    exp_llama = extract_svd_experts(w_down_llama, 9, target_dim=2048, rank=16)
    exp_falcon = extract_svd_experts(w_down_falcon, 9, target_dim=2048, rank=16)
    exp_bitnet = extract_svd_experts(w_down_bitnet, 8, target_dim=2048, rank=16)
    exp_qwen = extract_svd_experts(w_down_qwen, 8, target_dim=2048, rank=16)

    council_34 = exp_llama + exp_falcon + exp_bitnet + exp_qwen
    assert len(council_34) == 34
    logger.info(f"    Council Distribution: Llama={len(exp_llama)} (26.5%), Falcon={len(exp_falcon)} (26.5%), BitNet={len(exp_bitnet)} (23.5%), Qwen={len(exp_qwen)} (23.5%) - Balanced Quad 25.0%")

    # 3. Instantiate Architecture
    logger.info(f">>> [3/5] Instantiating Canonical QuillanRoninOni ({model_name})...")
    cfg = QuillanOniConfig(
        vocab_size=128256,
        hidden_dim=2048,
        n_layer=n_layers,
        n_head=32,
        head_dim=64,
        ffn_dim=8192,
        num_experts=34,
        expert_rank=16,
        use_evo_moe=False
    )
    cfg.vectorized_experts = True
    model = QuillanRoninOni(cfg)

    # 4. Ingest Weights
    logger.info(">>> [4/5] Populating 128k Embeddings and Transformer Backbone...")
    wte = st_llama["model.embed_tokens.weight"].to(torch.bfloat16)
    model.wte.weight.data.copy_(wte)
    model.lm_head.weight.data.copy_(wte)
    model.ln_f.weight.data.copy_(st_llama["model.norm.weight"].to(torch.bfloat16))

    model.q1_bridge.weight.data.copy_(torch.eye(2048, dtype=torch.bfloat16))
    model.q2_bridge.weight.data.copy_(torch.eye(2048, dtype=torch.bfloat16))
    model.ingest_gate.weight.data.zero_()
    model.ingest_gate.bias.data.zero_()
    model.quillan_finalizer_q1.weight.data.copy_(torch.eye(2048, dtype=torch.bfloat16))
    model.quillan_finalizer_q2.weight.data.copy_(torch.eye(2048, dtype=torch.bfloat16))
    model.quillan_comm_gate.weight.data.zero_()
    model.quillan_comm_gate.bias.data.zero_()

    for l in range(n_layers):
        p = f"model.layers.{l}"
        blk = model.h[l]
        blk.ln_1.weight.data.copy_(st_llama[f"{p}.input_layernorm.weight"].to(torch.bfloat16))
        blk.ln_2.weight.data.copy_(st_llama[f"{p}.post_attention_layernorm.weight"].to(torch.bfloat16))

        q = st_llama[f"{p}.self_attn.q_proj.weight"].float()
        k = st_llama[f"{p}.self_attn.k_proj.weight"].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).float()
        v = st_llama[f"{p}.self_attn.v_proj.weight"].view(8, 64, 2048).repeat_interleave(4, dim=0).view(2048, 2048).float()
        o = st_llama[f"{p}.self_attn.o_proj.weight"].float()

        c_attn = torch.cat([q, k, v], dim=0)
        blk.attn.c_attn.weight.data.copy_(c_attn.to(torch.bfloat16))
        blk.attn.c_attn.bias.data.zero_()
        blk.attn.c_proj.weight.data.copy_(o.to(torch.bfloat16))
        blk.attn.c_proj.bias.data.zero_()
        blk.attn.prism.w_gate.weight.data.zero_()

        gate = st_llama[f"{p}.mlp.gate_proj.weight"].float()
        up = st_llama[f"{p}.mlp.up_proj.weight"].float()
        down = st_llama[f"{p}.mlp.down_proj.weight"].float()

        c_fc = torch.cat([gate, up], dim=0)
        blk.moe.c_fc.weight.data.copy_(c_fc.to(torch.bfloat16))
        blk.moe.c_fc.bias.data.zero_()
        blk.moe.c_proj.weight.data.copy_(down.to(torch.bfloat16))
        blk.moe.c_proj.bias.data.zero_()

        # Load 34 Council MoE experts in BitNet ternary
        for e_idx, (wA, wB) in enumerate(council_34):
            exp = blk.moe.experts[e_idx]
            exp.lora_A.data.copy_(wA.to(torch.bfloat16))
            exp.lora_B.data.copy_(wB.to(torch.bfloat16))
            exp.swarm.A.data.zero_()
            exp.swarm.B.data.zero_()
            exp.swarm.C.data.zero_()
            exp.swarm.D.data.zero_()
            exp.swarm.clone_diversity.data.zero_()
            exp.swarm.clone_coupling.data.zero_()

        blk.moe.moe_gate.weight.data.zero_()
        blk.moe.moe_gate.bias.data.zero_()
        blk.moe.router.weight.data.zero_()
        blk.moe.pull_gate.gate.weight.data.zero_()

    # 5. Save Checkpoint
    out_dir = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_filename)
    logger.info(f">>> [5/5] Serializing Model Checkpoint to {out_path}...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "compilation_time_s": time.time() - t0,
        "donor_distribution": {
            "Llama-3.2-1B": 0.265,
            "Falcon-3-1B": 0.265,
            "BitNet-1.58b": 0.235,
            "Qwen-3.5-2B": 0.235
        },
        "quantization": "BitNet 1.58b Ternary {-1, 0, 1} with scale-factor preservation"
    }, out_path)
    logger.info(f"SUCCESS: {model_name} compiled in {time.time()-t0:.1f}s ({os.path.getsize(out_path)/(1024*1024):.1f} MB)")
    del model
    import gc
    gc.collect()

if __name__ == "__main__":
    # Compile Mini 0.8B (8 layers)
    compile_sovereign_model(n_layers=8, model_name="Quillan-Ronin Mini 0.8B Sovereign", out_filename="quillan_mini_0.6b_sovereign_wsp128k_qwen35.pt")

    # Compile Flagship 1B (16 layers)
    compile_sovereign_model(n_layers=16, model_name="Quillan-Ronin Flagship 1B Sovereign", out_filename="quillan_1b_sovereign_wsp128k_qwen35.pt")

    logger.info("=" * 80)
    logger.info("ALL SOVEREIGN QUAD-DONOR MODELS COMPILED SUCCESSFULLY!")
    logger.info("=" * 80)
