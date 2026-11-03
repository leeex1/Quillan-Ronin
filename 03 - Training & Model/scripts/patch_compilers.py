import os

def update_compilers():
    # 1. Update Flagship 1B compiler
    p1 = r"C:\02_QUILLAN\03 - Training & Model\scripts\compile_wsp_quad_sovereign_qwen35.py"
    with open(p1, "r", encoding="utf-8") as f:
        c1 = f.read()

    # ln_f
    t_lnf = """    norm_l = st_llama["model.norm.weight"].float()
    norm_f = falcon_tensors["blk.0.attn_norm.weight"].float()
    norm_unified = 0.70 * norm_l + 0.30 * norm_f
    model.ln_f.weight.data.copy_(normalize_layernorm_vector(norm_unified, hidden_dim, target_mean=1.0).to(torch.bfloat16))
    model.ln_f.bias.data.zero_()"""
    
    r_lnf = """    norm_l = st_llama["model.norm.weight"].float()
    model.ln_f.weight.data.copy_(norm_l.to(torch.bfloat16))
    if hasattr(model.ln_f, "bias") and model.ln_f.bias is not None:
        model.ln_f.bias.data.zero_()"""
    assert t_lnf in c1, "Failed to find ln_f in 1B compiler"
    c1 = c1.replace(t_lnf, r_lnf, 1)

    # ln_1 and ln_2
    t_ln12 = """        # 1. Gain-Matched LayerNorms
        blk.ln_1.weight.data.copy_(normalize_layernorm_vector(ln1, hidden_dim, target_mean=1.0).to(torch.bfloat16))
        blk.ln_1.bias.data.zero_()
        blk.ln_2.weight.data.copy_(normalize_layernorm_vector(ln2, hidden_dim, target_mean=1.0).to(torch.bfloat16))
        blk.ln_2.bias.data.zero_()"""
    
    r_ln12 = """        # 1. Exact Raw LayerNorms (Scale-Only RMSNorm, No 7x Distortion)
        ln1_proj = project_dim(ln1.unsqueeze(0), 1, hidden_dim).squeeze(0)
        ln2_proj = project_dim(ln2.unsqueeze(0), 1, hidden_dim).squeeze(0)
        blk.ln_1.weight.data.copy_(ln1_proj.to(torch.bfloat16))
        if hasattr(blk.ln_1, "bias") and blk.ln_1.bias is not None:
            blk.ln_1.bias.data.zero_()
        blk.ln_2.weight.data.copy_(ln2_proj.to(torch.bfloat16))
        if hasattr(blk.ln_2, "bias") and blk.ln_2.bias is not None:
            blk.ln_2.bias.data.zero_()"""
    assert t_ln12 in c1, "Failed to find ln1/ln2 in 1B compiler"
    c1 = c1.replace(t_ln12, r_ln12, 1)

    with open(p1, "w", encoding="utf-8") as f:
        f.write(c1)
    print("Flagship 1B compiler updated successfully!", flush=True)

    # 2. Update Mini 0.8B compiler
    p2 = r"C:\02_QUILLAN\03 - Training & Model\scripts\compile_wsp_mini_quad_sovereign.py"
    with open(p2, "r", encoding="utf-8") as f:
        c2 = f.read()

    t_mini_lnf = """    norm_l = st_llama["model.norm.weight"].float()
    model.ln_f.weight.data.copy_(normalize_layernorm_vector(norm_l, hidden_dim, target_mean=1.0).to(torch.bfloat16))
    model.ln_f.bias.data.zero_()"""

    r_mini_lnf = """    norm_l = st_llama["model.norm.weight"].float()
    model.ln_f.weight.data.copy_(norm_l.to(torch.bfloat16))
    if hasattr(model.ln_f, "bias") and model.ln_f.bias is not None:
        model.ln_f.bias.data.zero_()"""
    assert t_mini_lnf in c2, "Failed to find ln_f in Mini compiler"
    c2 = c2.replace(t_mini_lnf, r_mini_lnf, 1)

    t_mini_ln12 = """        blk.ln_1.weight.data.copy_(normalize_layernorm_vector(ln1, hidden_dim, target_mean=1.0).to(torch.bfloat16))
        blk.ln_1.bias.data.zero_()
        blk.ln_2.weight.data.copy_(normalize_layernorm_vector(ln2, hidden_dim, target_mean=1.0).to(torch.bfloat16))
        blk.ln_2.bias.data.zero_()"""

    r_mini_ln12 = """        ln1_proj = project_dim(ln1.unsqueeze(0), 1, hidden_dim).squeeze(0)
        ln2_proj = project_dim(ln2.unsqueeze(0), 1, hidden_dim).squeeze(0)
        blk.ln_1.weight.data.copy_(ln1_proj.to(torch.bfloat16))
        if hasattr(blk.ln_1, "bias") and blk.ln_1.bias is not None:
            blk.ln_1.bias.data.zero_()
        blk.ln_2.weight.data.copy_(ln2_proj.to(torch.bfloat16))
        if hasattr(blk.ln_2, "bias") and blk.ln_2.bias is not None:
            blk.ln_2.bias.data.zero_()"""
    assert t_mini_ln12 in c2, "Failed to find ln1/ln2 in Mini compiler"
    c2 = c2.replace(t_mini_ln12, r_mini_ln12, 1)

    with open(p2, "w", encoding="utf-8") as f:
        f.write(c2)
    print("Mini 0.8B compiler updated successfully!", flush=True)

if __name__ == "__main__":
    update_compilers()

