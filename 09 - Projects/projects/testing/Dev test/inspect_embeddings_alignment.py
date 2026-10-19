#!/usr/bin/env python3
import torch
from pathlib import Path

ckpt_path = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)

print("Checkpoint keys summary:")
for k in ["ingestion.txt_emb.weight", "txt_dec.weight", "moe.w1", "moe.w2"]:
    if k in sd:
        t = sd[k]
        print(f"  {k}: shape={list(t.shape)}, mean={t.mean():.5f}, std={t.std():.5f}, norm={t.norm():.2f}")

if "ingestion.txt_emb.weight" in sd and "txt_dec.weight" in sd:
    emb = sd["ingestion.txt_emb.weight"]
    dec = sd["txt_dec.weight"]
    sim = torch.cosine_similarity(emb, dec).mean().item()
    print(f"\nMean Cosine Similarity between txt_emb and txt_dec: {sim:.4f}")
