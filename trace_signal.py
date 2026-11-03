import torch
from tokenizers import Tokenizer
import sys, inspect

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
sys.path.insert(0, r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

tok_path = r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json'
tok = Tokenizer.from_file(tok_path)

ckpt_path = r'C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_mini_0.6b_tripartite.pt'
ckpt = torch.load(ckpt_path, map_location="cpu")

valid_keys = set(inspect.signature(QuillanOniConfig.__init__).parameters.keys())
cfg_dict = {k: v for k, v in ckpt["config"].items() if k in valid_keys}
cfg = QuillanOniConfig(**cfg_dict)
model = QuillanRoninOni(cfg).cpu().eval()
model.load_state_dict(ckpt["model_state_dict"], strict=False)

prompt = "Quillan is a sovereign cognitive agent. Its mission is"
enc = tok.encode(prompt).ids
x = torch.tensor([enc], dtype=torch.long)

with torch.no_grad():
    # 1. Embedding
    h = model.wte(x)
    print(f"Step 1: wte(x) -> shape: {h.shape}, norm: {h.norm().item():.2f}, std: {h.std().item():.4f}")
    
    # 2. Ingestion
    q1 = model.q1_bridge(h)
    q2 = model.q2_bridge(h)
    g_ingest = torch.sigmoid(model.ingest_gate(torch.cat([q1, q2], dim=-1)))
    h = h + 0.05 * (g_ingest * q1 + (1.0 - g_ingest) * q2)
    print(f"Step 2: after ingestion -> norm: {h.norm().item():.2f}, std: {h.std().item():.4f}")
    
    # 3. Each Layer
    for i, layer in enumerate(model.h):
        # Attention
        normed1 = layer.ln_1(h)
        attn_out, _ = layer.attn(normed1)
        print(f"  Layer {i} ln_1(h) std: {normed1.std().item():.4f} | attn_out norm: {attn_out.norm().item():.2f}, std: {attn_out.std().item():.4f}")
        h = h + attn_out
        
        # MoE
        normed2 = layer.ln_2(h)
        moe_out, _, _, _, _ = layer.moe(normed2)
        print(f"  Layer {i} ln_2(h) std: {normed2.std().item():.4f} | moe_out norm: {moe_out.norm().item():.2f}, std: {moe_out.std().item():.4f}")
        h = h + moe_out
        print(f"  Layer {i} end h norm: {h.norm().item():.2f}, std: {h.std().item():.4f}")
        
    # 4. Final Norm & LM Head
    h_final = model.ln_f(h)
    logits = model.lm_head(h_final)
    print(f"Final ln_f(h) norm: {h_final.norm().item():.2f}, std: {h_final.std().item():.4f}")
    print(f"Logits shape: {logits.shape}, min: {logits.min().item():.2f}, max: {logits.max().item():.2f}, std: {logits.std().item():.4f}")
