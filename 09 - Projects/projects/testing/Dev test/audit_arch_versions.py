import sys, torch
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
ckpt_dir = REPO_ROOT / "checkpoints" / "checkpoints_sft"

# Audit architecture shapes across the best checkpoints
targets = [
    "quillan_sft_v3_best.pt",
    "quillan_dialogue_best.pt", 
    "quillan_hyper_tuned_v531.pt",
    "quillan_final_best.pt",
]

for name in targets:
    path = ckpt_dir / name
    if not path.exists():
        continue
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    step = ckpt.get('step', 'N/A')
    loss = ckpt.get('loss', 'N/A')
    
    # Check key architecture indicators
    txt_emb = sd.get('ingestion.txt_emb.weight', None)
    txt_dec = sd.get('txt_dec.weight', None)
    
    # Check attention structure
    has_qkv = 'diffusion_core.couil_attn.qkv.weight' in sd
    has_qproj = 'diffusion_core.couil_attn.q_proj.weight' in sd
    
    # Check FFN structure
    ffn0_key = 'diffusion_core.ffn.0.weight'
    ffn0 = sd.get(ffn0_key, None)
    
    print(f"\n{'='*60}")
    print(f"  {name} (Step={step}, Loss={loss})")
    print(f"  Tensors: {len(sd)}")
    print(f"  txt_emb: {txt_emb.shape if txt_emb is not None else 'MISSING'}")
    print(f"  txt_dec: {txt_dec.shape if txt_dec is not None else 'MISSING'}")
    print(f"  Attention: {'fused qkv' if has_qkv else 'split q/k/v' if has_qproj else 'UNKNOWN'}")
    print(f"  FFN[0]: {ffn0.shape if ffn0 is not None else 'MISSING'}")
    
    # Check for Q1/Q2 ingestion
    has_q1 = 'ingestion.q1_ingest.weight' in sd
    print(f"  Q1/Q2 Ingestion: {'YES' if has_q1 else 'NO'}")

print(f"\n{'='*60}")
print("[+] Architecture audit complete.")
