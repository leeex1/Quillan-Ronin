import torch

ckpt = torch.load(r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_mini_0.6b_tripartite.pt", map_location="cpu")
sd = ckpt["model_state_dict"]

print("=== INSPECTING WEIGHT SPACE OF MINI 0.6B ===")
for k in ["wte.weight", "ln_f.weight", "lm_head.weight"]:
    if k in sd:
        t = sd[k].float()
        print(f"{k:20s} | shape: {str(list(t.shape)):15s} | mean: {t.mean().item():+.5f} | std: {t.std().item():.5f} | norm: {t.norm().item():.2f}")

print("\n=== LAYER BY LAYER RESIDUAL GAIN & FROBENIUS NORMS ===")
for l in range(6):
    ln1 = sd[f"h.{l}.ln_1.weight"].float()
    ln2 = sd[f"h.{l}.ln_2.weight"].float()
    c_attn = sd[f"h.{l}.attn.c_attn.weight"].float()
    c_proj = sd[f"h.{l}.attn.c_proj.weight"].float()
    print(f"Layer {l} | ln1_mean: {ln1.mean().item():.3f} (std: {ln1.std().item():.3f}) | ln2_mean: {ln2.mean().item():.3f} (std: {ln2.std().item():.3f}) | c_attn_std: {c_attn.std().item():.5f} (norm: {c_attn.norm().item():.2f}) | c_proj_std: {c_proj.std().item():.5f} (norm: {c_proj.norm().item():.2f})")
