#!/usr/bin/env python3
"""Head-only tune on CLEAN slice with FIXED tokenizer (owner's data+vocab).
Thin driver over existing QuillanSFTCalibrator: vocab 50262, base = mapped 6L,
data = clean 20k slice, lr 1.5e-5, 150 steps, batch 2, accum 8 (eff 16).
Saves NEW file quillan_head_v62_clean.pt (nothing overwritten)."""
import sys
from pathlib import Path
import torch

REPO = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO / "scripts"))
from quillan_sft_calibrator import QuillanSFTCalibrator

CKPT = REPO / "checkpoints" / "checkpoints_sft" / "quillan_6l_vocab62_mapped.pt"
DATA = REPO / "training_data" / "canonical_standardized" / "quillan_clean_v62_cpu20k.pt"
OUT = REPO / "checkpoints" / "checkpoints_sft" / "quillan_head_v62_clean.pt"

# Bypass __init__ (it hardcodes vocab 50257 and shape-crashes on mapped ckpt).
# Same fields, model built at fixed 50262, mapped weights load exactly.
import torch as _t
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from quillan_bpe_tokenizer import QuillanBPETokenizer
cal = QuillanSFTCalibrator.__new__(QuillanSFTCalibrator)
cal.device = _t.device("cpu")
cal.checkpoint_path, cal.dataset_path, cal.lr = CKPT, DATA, 1.5e-5
cal.tokenizer = QuillanBPETokenizer()
cal.cfg = QuillanOniConfig(vocab_size=50262, hidden_dim=1024, ffn_dim=2048,
                           n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
cal.model = QuillanRoninOni(cal.cfg).to(cal.device)
_m = _t.load(CKPT, map_location=cal.device, weights_only=True)
_sd = _m.get("model", _m.get("model_state_dict", _m))
_missing, _unexpected = cal.model.load_state_dict(_sd, strict=False)
print(f"built vocab=50262 missing={len(_missing)} unexpected={len(_unexpected)}",
      flush=True)
print("driver: vocab=50262 base=mapped slice=cpu20k lr=1.5e-5 steps=150 eff=16",
      flush=True)
cal.run_calibration(num_steps=150, batch_size=2, grad_accum_steps=8,
                    calibrate_head_only=True, save_path=OUT)
print(f"saved {OUT.name}", flush=True)
