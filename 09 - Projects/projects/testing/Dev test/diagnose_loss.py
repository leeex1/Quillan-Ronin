#!/usr/bin/env python3
"""
Deep diagnostic: Find exactly WHY loss is 54+ instead of ~10.82
"""
import os, sys, json, torch, math
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = r"C:\02_QUILLAN"
sys.path.insert(0, REPO_ROOT)

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

# 1. Create model with matching config
cfg = QuillanArchConfig(hidden_dim=1024, ffn_dim=2048, num_experts=34, text_only=True)
model = QuillanRoninSovereign(cfg).to("cpu")
print(f"[1] Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# 2. Load base checkpoint
base_ckpt = r"C:\02_QUILLAN\checkpoints\checkpoints_v2\quillan_full_base_final.pt"
ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)

# Check what keys are in checkpoint vs model
model_keys = set(model.state_dict().keys())
ckpt_keys = set(sd.keys())
print(f"\n[2] Checkpoint keys: {len(ckpt_keys)}")
print(f"    Model keys: {len(model_keys)}")
print(f"    Matched (same name): {len(model_keys & ckpt_keys)}")
print(f"    In ckpt but not model: {len(ckpt_keys - model_keys)}")
print(f"    In model but not ckpt: {len(model_keys - ckpt_keys)}")

# Show shape mismatches
model_sd = model.state_dict()
shape_match = 0
shape_mismatch = 0
for k in model_keys & ckpt_keys:
    if model_sd[k].shape == sd[k].shape:
        shape_match += 1
    else:
        shape_mismatch += 1
        print(f"    SHAPE MISMATCH: {k}: model={model_sd[k].shape} vs ckpt={sd[k].shape}")

print(f"    Shape matches: {shape_match}")
print(f"    Shape mismatches: {shape_mismatch}")

# Load matched weights
copied = 0
for k, v in sd.items():
    if k in model_sd and v.shape == model_sd[k].shape:
        model_sd[k].copy_(v)
        copied += 1
model.load_state_dict(model_sd)
print(f"[3] Copied {copied} weight tensors into model")

# Show what was NOT loaded (these are randomly initialized)
not_loaded = model_keys - set(k for k in ckpt_keys if k in model_sd and sd[k].shape == model_sd[k].shape)
print(f"\n[4] {len(not_loaded)} model layers are RANDOMLY INITIALIZED (not in checkpoint):")
for k in sorted(not_loaded)[:30]:
    print(f"    - {k}: {model_sd[k].shape}")
if len(not_loaded) > 30:
    print(f"    ... and {len(not_loaded) - 30} more")

# 3. Test forward pass
model.eval()

# Load 1 real sample
corpus_path = r"C:\02_QUILLAN\training_data\unified_tokenized_corpus.jsonl"
with open(corpus_path, "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())

tokens = sample["input_ids"][:512]
labels = sample["labels"][:512]

txt_in = torch.tensor([tokens], dtype=torch.long)
target_in = torch.tensor([labels], dtype=torch.long)

with torch.no_grad():
    # Check what forward returns
    out = model(txt_in)
    print(f"\n[5] Forward pass output type: {type(out)}")
    
    if isinstance(out, dict):
        print(f"    Dict keys: {out.keys()}")
        logits = out.get("logits", out.get("output", None))
        if logits is None:
            print("    ERROR: No 'logits' key in output dict!")
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: shape={v.shape}")
    elif isinstance(out, tuple):
        print(f"    Tuple length: {len(out)}")
        logits = out[0]
    elif isinstance(out, torch.Tensor):
        logits = out
    else:
        print(f"    UNEXPECTED TYPE: {type(out)}")
        logits = None
    
    if logits is not None:
        print(f"    Logits shape: {logits.shape}")
        print(f"    Logits dtype: {logits.dtype}")
        print(f"    Logits min/max/mean: {logits.min().item():.4f} / {logits.max().item():.4f} / {logits.mean().item():.4f}")
        print(f"    Logits std: {logits.std().item():.4f}")
        
        # Check if logits are reasonable
        # For vocab_size=50257, random logits should give loss ~= ln(50257) = 10.82
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_in[..., 1:].contiguous()
        
        # Count non-masked labels
        valid_labels = (shift_labels != -100).sum().item()
        total_labels = shift_labels.numel()
        print(f"\n[6] Labels: {valid_labels} valid out of {total_labels} total ({100*valid_labels/total_labels:.1f}%)")
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, cfg.vocab_size), 
            shift_labels.view(-1), 
            ignore_index=-100
        )
        print(f"    Cross-entropy loss: {ce_loss.item():.4f}")
        print(f"    Expected random baseline: {math.log(cfg.vocab_size):.4f}")
        
        if ce_loss.item() > 15.0:
            print(f"\n[7] LOSS IS ABNORMALLY HIGH ({ce_loss.item():.1f} >> {math.log(cfg.vocab_size):.1f})")
            print(f"    This means the model's logit outputs are BADLY SCALED.")
            
            # Check logit magnitude per-token
            sample_logits = shift_logits[0, :5, :]
            for i in range(min(5, sample_logits.shape[0])):
                row = sample_logits[i]
                print(f"    Token {i}: min={row.min().item():.2f} max={row.max().item():.2f} std={row.std().item():.2f} mean={row.mean().item():.4f}")
                # What would softmax look like?
                probs = F.softmax(row, dim=-1)
                top5_p, top5_i = torch.topk(probs, 5)
                print(f"             top5 probs: {[f'{p:.4f}' for p in top5_p.tolist()]} indices: {top5_i.tolist()}")
