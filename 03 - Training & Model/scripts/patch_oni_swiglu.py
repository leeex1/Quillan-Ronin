with open(r'C:\02_QUILLAN\03 - Training & Model\scripts\quillan_v5_4_oni.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix SwiGLU RMSNorm
old_swiglu = "h_swiglu = F.rms_norm(F.silu(gate) * act, (act.size(-1),), eps=1e-5)"
new_swiglu = "h_swiglu = F.silu(gate) * act"
assert old_swiglu in content, "old_swiglu not found"
content = content.replace(old_swiglu, new_swiglu)

# 2. Fix moe_gate inference bypass
old_moe_out = """        g = torch.tanh(self.moe_gate(flat_x))
        out = h_dense + (moe_out * g).view(B, T, C)"""
new_moe_out = """        if self.training:
            g = torch.tanh(self.moe_gate(flat_x))
            out = h_dense + (moe_out * g).view(B, T, C)
        else:
            out = h_dense + 0.05 * moe_out.view(B, T, C)"""
assert old_moe_out in content, "old_moe_out not found"
content = content.replace(old_moe_out, new_moe_out)

# 3. Fix flow & dqso modulation at inference
old_mod = """            # Apply modulation outside torch.no_grad() to preserve gradient backprop graph for transformer backbone
            x = x * (0.9 + 0.1 * flow.unsqueeze(-1)) + 0.05 * dqso_delta"""
new_mod = """            # Apply modulation outside torch.no_grad() to preserve gradient backprop graph for transformer backbone
            if self.training:
                x = x * (0.9 + 0.1 * flow.unsqueeze(-1)) + 0.05 * dqso_delta"""
assert old_mod in content, "old_mod not found"
content = content.replace(old_mod, new_mod)

with open(r'C:\02_QUILLAN\03 - Training & Model\scripts\quillan_v5_4_oni.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully applied SwiGLU and clean inference patches to quillan_v5_4_oni.py!")
